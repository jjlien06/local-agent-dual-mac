"""Source-bound native CPU bridge tests; stdlib unittest, no model loading."""
import ast
import hashlib
import json
import resource
import socket
import sys
import time
import importlib
import types
import unittest
from unittest import mock
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/mlx-vlm"
CONFIG_PATH = Path("/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/config.json")
ROPE_PATH = UPSTREAM / "mlx_vlm/models/rope_utils.py"
LANGUAGE_PATH = UPSTREAM / "mlx_vlm/models/qwen3_5/language.py"

import numpy as np


def sample(tokens=3, dtype=np.float32):
    rng = np.random.default_rng(195)
    return (rng.normal(size=(1, 24, tokens, 256)).astype(dtype),
            rng.normal(size=(1, 4, tokens, 256)).astype(dtype),
            np.tile(np.arange(tokens, dtype=np.int32), (3, 1, 1)))


def no_mlx_import(name, *args, **kwargs):
    if name == "mlx" or name.startswith("mlx."):
        raise AssertionError("MLX imported before admission")
    return _real_import(name, *args, **kwargs)


_real_import = __import__


def native_oracle(q, k, positions):
    """Independent load and calls: actual class AND actual language wrapper."""
    import mlx.core as mx
    expected_hashes = (
        (ROPE_PATH, '6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b'),
        (LANGUAGE_PATH, '6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7'))
    data = []
    for path, expected in expected_hashes:
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == expected
        data.append(blob)
    with mx.stream(mx.cpu):
        module = types.ModuleType('_independent_pinned_cpu_oracle')
        exec(compile(data[0], str(ROPE_PATH), 'exec'), module.__dict__)
        module._HAS_METAL = False
        nodes = [n for n in ast.parse(data[1]).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in {'Qwen3_5RotaryEmbedding', 'apply_multimodal_rotary_pos_emb'}]
        assert len(nodes) == 2
        scope = {'mx': mx, 'MRoPERotaryEmbedding': module.MRoPERotaryEmbedding,
                 '_apply_mrope': module.apply_multimodal_rotary_pos_emb}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LANGUAGE_PATH), 'exec'), scope)
        rotary = scope['Qwen3_5RotaryEmbedding'](
            64, max_position_embeddings=262144, base=10000000, mrope_section=[11, 11, 10])
        assert rotary.fused_apply is False
        assert np.array(rotary.position_selector).tolist() == [i % 3 for i in range(32)]
        # MLX rejects negative strides; independent oracle packs its own inputs.
        nq, nk, p = (mx.array(a.copy(order="C")) for a in (q, k, positions))
        class_q, class_k = rotary.apply_rotary(nq, nk, p)
        cos, sin = rotary(nk, p)
        wrapper_q, wrapper_k = scope['apply_multimodal_rotary_pos_emb'](nq, nk, cos, sin)
        mx.eval(class_q, class_k, wrapper_q, wrapper_k)
        first = (np.array(class_q), np.array(class_k))
        second = (np.array(wrapper_q), np.array(wrapper_k))
        for a, b in zip(first, second):
            assert a.tobytes() == b.tobytes()
        return first


def bridge():
    try:
        module = importlib.import_module("multimodal_native_cpu")
    except ModuleNotFoundError as exc:
        if exc.name != "multimodal_native_cpu":
            raise
        module = None
    function = getattr(module, "rotate_reference_cpu", None)
    assert callable(function), "rotate_reference_cpu callable is missing"
    return function


def golden_cases():
    """Authoritative explicit nonsquare coordinates, not production position math."""
    yield 'text-boundaries', np.tile(np.array([0, 17, 1023, 1055], np.int32), (3, 1, 1))
    yield 'image-4x6', np.array([
        [[0, 1, 2, 2, 2, 2, 2, 2, 5, 6]],
        [[0, 1, 2, 2, 2, 3, 3, 3, 5, 6]],
        [[0, 1, 2, 3, 4, 2, 3, 4, 5, 6]]], np.int32)
    yield 'decode-4x6', np.full((3, 1, 1), 7, np.int32)
    yield 'image-6x4', np.array([
        [[0, 1, 2, 2, 2, 2, 2, 2, 5, 6]],
        [[0, 1, 2, 2, 3, 3, 4, 4, 5, 6]],
        [[0, 1, 2, 3, 2, 3, 2, 3, 5, 6]]], np.int32)
    yield 'decode-6x4', np.full((3, 1, 1), 7, np.int32)
    # Literal merged 8x16 grid: independently enumerated row-major h/w axes.
    horizontal = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    vertical = [v for v in (2, 3, 4, 5, 6, 7, 8, 9) for _ in range(16)]
    yield 'image-16x32', np.array([
        [[0, 1] + [2] * 128 + [18, 19]],
        [[0, 1] + vertical + [18, 19]],
        [[0, 1] + horizontal * 8 + [18, 19]]], np.int32)
    yield 'decode-16x32', np.full((3, 1, 1), 20, np.int32)


def bounded_probe():
    """Real native calls only; unchanged original strict tolerance, no NumPy math."""
    started = time.monotonic()
    import importlib.metadata
    import mlx.core as mx
    function = bridge()
    rows, controls, devices = [], [], []
    previous = mx.default_device()
    real_eval = mx.eval
    max_payload = total_payload = 0
    def cpu_eval(*args, **kwargs):
        assert mx.default_device() == mx.cpu
        devices.append(str(mx.default_device()))
        return real_eval(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('network/Metal/allocator/NumPy rotary forbidden')
    with mock.patch.object(mx.fast, 'metal_kernel', side_effect=forbidden) as metal, mock.patch.object(
            mx, 'eval', side_effect=cpu_eval), mock.patch.object(
            mx, 'set_memory_limit', side_effect=forbidden), mock.patch.object(
            mx, 'set_cache_limit', side_effect=forbidden), mock.patch.object(
            socket, 'socket', side_effect=forbidden), mock.patch.object(
            socket, 'create_connection', side_effect=forbidden), mock.patch(
            'multimodal_rotary.rotate_numpy', side_effect=forbidden):
        with mx.stream(mx.cpu):
            for name, positions in golden_cases():
                for dtype in (np.float32, np.float16):
                    q, k, _ = sample(positions.shape[-1], dtype)
                    originals = [a.tobytes() for a in (q, k, positions)]
                    payload = q.nbytes + k.nbytes + positions.nbytes
                    assert payload < 16 * 1024 * 1024
                    max_payload = max(max_payload, payload)
                    total_payload += payload
                    actual = function(q, k, positions)
                    assert mx.default_device() == mx.cpu
                    reference = native_oracle(q, k, positions)
                    assert originals == [a.tobytes() for a in (q, k, positions)]
                    for role, value, expected, original in zip(('q', 'k'), actual, reference, (q, k)):
                        error = np.abs(value.astype(np.float64) - expected.astype(np.float64))
                        tolerance = 1e-6 + 1e-5 * np.abs(expected.astype(np.float64))
                        finite = bool(np.isfinite(value).all() and np.isfinite(expected).all())
                        tail_exact = value[..., 64:].tobytes() == original[..., 64:].tobytes()
                        rows.append(dict(case=name, dtype=str(np.dtype(dtype)), role=role,
                                         shape=list(value.shape), max_abs_error=float(error.max()),
                                         max_error_ratio=float((error / tolerance).max()),
                                         violations=int(np.count_nonzero(error > tolerance)),
                                         strict_pass=finite and bool(np.all(error <= tolerance)),
                                         native_exact=value.tobytes() == expected.tobytes(),
                                         native_wrapper_exact=True, unchanged_tail_exact=tail_exact,
                                         finite=finite, input_unchanged=True))
                    if name.startswith('image-'):
                        for control, wrong in (('t-only', np.repeat(positions[:1], 3, axis=0)),
                                               ('hw-swap', positions[[0, 2, 1]])):
                            wrong_result = function(q, k, wrong)
                            for role, expected, mutant in zip(('q', 'k'), reference, wrong_result):
                                error = np.abs(expected.astype(np.float64) - mutant.astype(np.float64))
                                tolerance = 1e-6 + 1e-5 * np.abs(expected.astype(np.float64))
                                controls.append(dict(case=name, dtype=str(np.dtype(dtype)), role=role,
                                                     control=control,
                                                     strict_violations=int(np.count_nonzero(error > tolerance))))
        assert metal.call_count == 0
    restored = mx.default_device() == previous
    assert restored
    passed = (len(rows) == 28 and all(r['strict_pass'] and r['native_exact'] and r['unchanged_tail_exact'] for r in rows)
              and len(controls) == 24 and all(r['strict_violations'] > 0 for r in controls))
    return dict(status='passed_native_cpu_reference' if passed else 'native_cpu_reference_rejection',
                passed=passed, atol=1e-6, rtol=1e-5, row_count=len(rows), rows=rows,
                negative_control_count=len(controls), negative_controls=controls,
                eval_count=len(devices), eval_devices=sorted(set(devices)),
                default_before=str(previous), default_after=str(mx.default_device()),
                default_restored=restored, allocator_limits_changed=False,
                metal_kernel_call_count=metal.call_count, gpu_kernels_executed=False,
                full_model_loaded=False, network_used=False,
                max_input_payload_bytes=max_payload, cumulative_input_payload_bytes=total_payload,
                maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                elapsed_seconds=time.monotonic() - started,
                versions={n: importlib.metadata.version(n) for n in ('mlx', 'mlx-metal', 'numpy')},
                mlx_imports=sorted(n for n in sys.modules if n == 'mlx' or n.startswith('mlx.')),
                forbidden_model_imports=sorted(n for n in sys.modules if n == 'mlx_vlm' or n.startswith('mlx_vlm.')))


class NativeCPUBridgeTests(unittest.TestCase):
    def test_callable_exists(self):
        self.assertTrue(callable(bridge()))

    def test_invalid_arrays_fail_before_mlx_import(self):
        function = bridge()
        q, k, p = sample()
        class SubArray(np.ndarray):
            pass
        cases = [(q.tolist(), k, p), (q, k.tolist(), p), (q, k, p.tolist()),
                 (q.view(SubArray), k, p), (q, k.view(SubArray), p),
                 (q, k, p.view(SubArray)), (q.astype(np.float64), k, p),
                 (q, k.astype(np.float64), p), (q, k.astype(np.float16), p),
                 (q.astype('>f4'), k, p), (q, k, p.astype(np.int64)),
                 (q, k, p.astype('>i4')), (q[0], k, p), (q, k[0], p),
                 (q[:, :23], k, p), (q, k[:, :3], p),
                 (q[..., :255], k, p), (q, k[..., :255], p),
                 (np.repeat(q, 2, axis=0), k, p),
                 (q, np.repeat(k, 2, axis=0), p), (q, k[:, :, :2], p),
                 (q, k, p[:2]), (q, k, p[:, 0]), (q, k, p[:, :, :2]),
                 (q, k, np.full_like(p, -1)), (q, k, np.full_like(p, 1056))]
        for tokens in (0, 257):
            cases.append((np.zeros((1, 24, tokens, 256), np.float32),
                          np.zeros((1, 4, tokens, 256), np.float32),
                          np.zeros((3, 1, tokens), np.int32)))
        for role in (0, 1):
            for value in (np.nan, np.inf, -np.inf):
                values = [q.copy(), k.copy(), p]
                values[role][0, 0, 0, -1] = value
                cases.append(tuple(values))
        with mock.patch('builtins.__import__', side_effect=no_mlx_import):
            for index, values in enumerate(cases):
                with self.subTest(index=index):
                    try:
                        function(*values)
                    except Exception as exc:
                        self.assertIsInstance(exc, ValueError)
                    else:
                        self.fail('invalid input accepted')

    def test_exact_immutable_config_rejected_before_arrays_or_mlx(self):
        from dataclasses import FrozenInstanceError, replace
        from multimodal_rotary import TARGET, TargetRotaryConfig
        function = bridge()
        module = importlib.import_module('multimodal_native_cpu')
        q, k, p = sample()
        with self.assertRaises(FrozenInstanceError):
            TARGET.theta = 1
        fields = ('model_id', 'revision', 'config_sha256', 'query_heads', 'kv_heads',
                  'head_dim', 'rotary_dim', 'partial_rotary_factor', 'theta',
                  'mrope_interleaved', 'mrope_section', 'pairing')
        invalid = [None, dict(vars(type(TARGET))), object()]
        for field in fields:
            value = getattr(TARGET, field)
            wrong = (value + '-wrong' if type(value) is str else
                     False if type(value) is bool else
                     (11, 10, 11) if type(value) is tuple else value + 1)
            invalid.append(replace(TARGET, **{field: wrong}))
        for field, value in [('query_heads', 24.0), ('kv_heads', np.int32(4)),
                             ('head_dim', np.int64(256)), ('theta', 10000000.0),
                             ('partial_rotary_factor', np.float64(.25)),
                             ('mrope_interleaved', 1), ('mrope_section', [11, 11, 10]),
                             ('mrope_section', (np.int32(11), 11, 10))]:
            invalid.append(replace(TARGET, **{field: value}))
        class SubConfig(TargetRotaryConfig):
            pass
        invalid.append(SubConfig(*(getattr(TARGET, f) for f in fields)))
        with mock.patch('builtins.__import__', side_effect=no_mlx_import), mock.patch.object(
                module, '_validate_arrays', side_effect=AssertionError('arrays before metadata')):
            for index, config in enumerate(invalid):
                with self.subTest(index=index):
                    try:
                        function(q, k, p, config=config)
                    except Exception as exc:
                        self.assertIsInstance(exc, ValueError)
                        self.assertIn('config', str(exc))
                    else:
                        self.fail('invalid config accepted')


    def test_tampered_or_missing_sources_fail_before_mlx(self):
        function = bridge()
        arrays = sample()
        original = Path.read_bytes
        for target in (CONFIG_PATH, ROPE_PATH, LANGUAGE_PATH):
            for missing in (False, True):
                def read(path, *, target=target, missing=missing):
                    if path == target:
                        if missing:
                            raise FileNotFoundError(str(path))
                        return original(path) + b'\n'
                    return original(path)
                with self.subTest(target=str(target), missing=missing), mock.patch(
                        'builtins.__import__', side_effect=no_mlx_import), mock.patch.object(
                        Path, 'read_bytes', autospec=True, side_effect=read):
                    try:
                        function(*arrays)
                    except Exception as exc:
                        self.assertIsInstance(exc, ValueError)
                        self.assertRegex(str(exc), 'source|config')
                    else:
                        self.fail('unbound source admitted')
        wrong = subprocess.CompletedProcess([], 0, '0' * 40 + '\n', '')
        with mock.patch('builtins.__import__', side_effect=no_mlx_import), mock.patch(
                'subprocess.run', return_value=wrong):
            try:
                function(*arrays)
            except Exception as exc:
                self.assertIsInstance(exc, ValueError)
                self.assertIn('source', str(exc))
            else:
                self.fail('wrong checkout admitted')


    def test_unpinned_runtime_rejected_before_mlx_import(self):
        import importlib.metadata
        function = bridge()
        arrays = sample()
        original = importlib.metadata.version
        for name in ('mlx', 'mlx-metal', 'numpy'):
            def version(package, *, name=name):
                return '0.invalid' if package == name else original(package)
            with self.subTest(package=name), mock.patch(
                    'builtins.__import__', side_effect=no_mlx_import), mock.patch(
                    'importlib.metadata.version', side_effect=version):
                try:
                    function(*arrays)
                except Exception as exc:
                    self.assertIsInstance(exc, ValueError)
                    self.assertIn('runtime', str(exc))
                else:
                    self.fail('unpinned runtime admitted')


    def test_native_eager_cpu_matches_independent_calls_restores_default_no_metal(self):
        import mlx.core as mx
        from multimodal_rotary import TARGET
        from dataclasses import replace
        function = bridge()
        q, k, _ = sample()
        positions = np.tile(np.array([0, 17, 1055], np.int32), (3, 1, 1))
        originals = [a.tobytes() for a in (q, k, positions)]
        previous = mx.default_device()
        devices = []
        real_eval = mx.eval
        def cpu_eval(*args, **kwargs):
            devices.append(str(mx.default_device()))
            self.assertEqual(mx.default_device(), mx.cpu)
            return real_eval(*args, **kwargs)
        with mock.patch.object(mx.fast, 'metal_kernel', side_effect=AssertionError('Metal forbidden')) as metal, mock.patch.object(
                mx, 'eval', side_effect=cpu_eval), mock.patch.object(
                mx, 'set_memory_limit', side_effect=AssertionError('allocator change forbidden')), mock.patch.object(
                mx, 'set_cache_limit', side_effect=AssertionError('allocator change forbidden')), mock.patch(
                'multimodal_rotary.rotate_numpy', side_effect=AssertionError('NumPy rotary forbidden')):
            try:
                actual = function(q, k, positions, config=replace(TARGET))
            except NotImplementedError as exc:
                self.fail(str(exc))
            expected = native_oracle(q, k, positions)
            metal.assert_not_called()
        self.assertEqual(mx.default_device(), previous)
        self.assertTrue(devices)
        self.assertIs(type(actual), tuple)
        self.assertEqual(len(actual), 2)
        for value, reference, original in zip(actual, expected, (q, k)):
            self.assertIs(type(value), np.ndarray)
            self.assertEqual(value.shape, original.shape)
            self.assertEqual(value.dtype, original.dtype)
            self.assertTrue(np.isfinite(value).all())
            np.testing.assert_allclose(value, reference, atol=1e-6, rtol=1e-5)
            self.assertEqual(value.tobytes(), reference.tobytes())
            self.assertEqual(value[..., 64:].tobytes(), original[..., 64:].tobytes())
        self.assertEqual(originals, [a.tobytes() for a in (q, k, positions)])


    def test_missing_cpu_api_fails_closed_before_device_change(self):
        import mlx.core as mx
        function = bridge()
        arrays = sample()
        previous = mx.default_device()
        for name in ('cpu', 'array', 'eval'):
            with self.subTest(api=name), mock.patch.object(mx, name, None), mock.patch.object(
                    mx, 'set_default_device', side_effect=AssertionError('device changed before API admission')) as setter:
                try:
                    function(*arrays)
                except Exception as exc:
                    self.assertIsInstance(exc, ValueError)
                    self.assertIn('API', str(exc))
                else:
                    self.fail('missing CPU API admitted')
                setter.assert_not_called()
        self.assertEqual(mx.default_device(), previous)


    def test_verification_full_explicit_fixture_probe(self):
        result = bounded_probe()
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['forbidden_model_imports'], [])

    def test_verification_import_is_mlx_lazy_in_fresh_interpreter(self):
        code = """import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'mlx' or name.startswith('mlx.') or name == 'mlx_vlm' or name.startswith('mlx_vlm.'):
        raise AssertionError('eager heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import multimodal_native_cpu
assert callable(multimodal_native_cpu.rotate_reference_cpu)
assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
print('lazy import verified; no MLX/VLM imports')
"""
        completed = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('lazy import verified', completed.stdout)

    def test_verification_finally_restores_default_after_eval_or_conversion_failure(self):
        import mlx.core as mx
        module = importlib.import_module('multimodal_native_cpu')
        q, k, p = sample()
        previous = mx.default_device()
        real_eval = mx.eval
        for fail_at in (1, 2, 3):
            count = 0
            def failing_eval(*args, **kwargs):
                nonlocal count
                count += 1
                self.assertEqual(mx.default_device(), mx.cpu)
                if count == fail_at:
                    raise RuntimeError('injected evaluation failure')
                return real_eval(*args, **kwargs)
            with self.subTest(fail_at=fail_at), mock.patch.object(
                    mx.fast, 'metal_kernel', side_effect=AssertionError('Metal forbidden')), mock.patch.object(
                    mx, 'eval', side_effect=failing_eval):
                with self.assertRaisesRegex(RuntimeError, 'injected evaluation failure'):
                    module.rotate_reference_cpu(q, k, p)
            self.assertEqual(count, fail_at)
            self.assertEqual(mx.default_device(), previous)
        with mock.patch.object(mx.fast, 'metal_kernel', side_effect=AssertionError('Metal forbidden')), mock.patch.object(
                module.np, 'array', side_effect=RuntimeError('injected conversion failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected conversion failure'):
                module.rotate_reference_cpu(q, k, p)
        self.assertEqual(mx.default_device(), previous)

    def test_verification_token_boundaries_readonly_views_and_tail_bits(self):
        import mlx.core as mx
        function = bridge()
        for tokens in (1, 256):
            for dtype in (np.float16, np.float32):
                q, k, _ = sample(tokens, dtype)
                positions = np.tile(np.resize(np.array([0, 17, 1023, 1055], np.int32), tokens), (3, 1, 1))
                # Positive/negative zeros and subnormals must survive the tail byte-for-byte.
                tiny = np.nextafter(dtype(0), dtype(1), dtype=dtype)
                for array in (q, k):
                    array[..., 64] = dtype(-0.0)
                    array[..., 65] = tiny
                    array[..., 66] = -tiny
                    array.setflags(write=False)
                positions.setflags(write=False)
                # A genuine strided, read-only ndarray remains admitted.
                q, k, positions = q[:, ::-1], k[:, ::-1], positions[:, :, ::-1]
                originals = [a.tobytes() for a in (q, k, positions)]
                previous = mx.default_device()
                with mock.patch.object(mx.fast, 'metal_kernel', side_effect=AssertionError('Metal forbidden')):
                    try:
                        actual = function(q, k, positions)
                    except ValueError as exc:
                        self.fail(f"legal read-only strided input rejected: {exc}")
                    expected = native_oracle(q, k, positions)
                self.assertEqual(mx.default_device(), previous)
                self.assertEqual(originals, [a.tobytes() for a in (q, k, positions)])
                for value, reference, original in zip(actual, expected, (q, k)):
                    np.testing.assert_allclose(value, reference, atol=1e-6, rtol=1e-5)
                    self.assertEqual(value.tobytes(), reference.tobytes())
                    self.assertEqual(value[..., 64:].tobytes(), original[..., 64:].tobytes())


    def test_uninitialized_exact_config_fails_closed_with_value_error(self):
        from multimodal_rotary import TargetRotaryConfig
        malformed = object.__new__(TargetRotaryConfig)
        function = bridge()
        with mock.patch('builtins.__import__', side_effect=no_mlx_import):
            try:
                function(*sample(), config=malformed)
            except Exception as exc:
                self.assertIsInstance(exc, ValueError)
                self.assertIn('config', str(exc))
            else:
                self.fail('missing config fields accepted')


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        result = bounded_probe()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        raise SystemExit(0 if result["passed"] else 2)
    unittest.main()
