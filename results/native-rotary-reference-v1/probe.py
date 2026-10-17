"""CPU-only native mRoPE diagnostic; no loaded models or attention integration."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import resource
import socket
import sys
import time
from pathlib import Path

WORK = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise RuntimeError('network is forbidden in the native CPU reference probe')


def main():
    started = time.monotonic()
    socket.socket = forbidden
    socket.create_connection = forbidden
    import numpy as np
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    assert mx.default_device() == mx.cpu
    # Advisory only; the small fixture arrays are bounded independently below.
    mx.set_memory_limit(128 * 1024 * 1024)
    mx.set_cache_limit(8 * 1024 * 1024)
    from multimodal_contract import prepare_request
    from multimodal_rotary import rotate_numpy

    base = WORK / 'upstream/mlx-vlm/mlx_vlm/models'
    rope_path = base / 'rope_utils.py'
    language_path = base / 'qwen3_5/language.py'
    assert digest(rope_path) == '6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b'
    assert digest(language_path) == '6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7'
    spec = importlib.util.spec_from_file_location('_pinned_cpu_rope_reference', rope_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Force the upstream documented eager fallback, not a Metal custom kernel.
    # No source bytes or function bodies are modified.
    module._HAS_METAL = False
    language_tree = ast.parse(language_path.read_text())
    names = {'Qwen3_5RotaryEmbedding', 'apply_multimodal_rotary_pos_emb'}
    chosen = [n for n in language_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(chosen) == 2
    scope = {'mx': mx, 'MRoPERotaryEmbedding': module.MRoPERotaryEmbedding,
             '_apply_mrope': module.apply_multimodal_rotary_pos_emb}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(language_path), 'exec'), scope)
    rotary = scope['Qwen3_5RotaryEmbedding'](64, max_position_embeddings=262144,
                                          base=10000000, mrope_section=[11, 11, 10])
    assert rotary.fused_apply is False
    selector = np.array(rotary.position_selector)
    assert selector.tolist() == [i % 3 for i in range(32)]
    cases = []
    text_positions = np.tile(np.array([0, 17, 1023, 1055], dtype=np.int32), (3, 1, 1))
    cases.append(('text-boundaries', text_positions))
    for grid in [(1, 4, 6), (1, 6, 4), (1, 16, 32)]:
        count = grid[1] * grid[2] // 4
        request = prepare_request([7, 248053] + [248056] * count + [248054, 8],
                                  request_id='native-' + str(grid[1]) + '-' + str(grid[2]), grid=grid)
        cases.append(('image-' + str(grid[1]) + 'x' + str(grid[2]), request.position_ids))
        cases.append(('decode-' + str(grid[1]) + 'x' + str(grid[2]),
                      request.continuation(request.prompt_length)))
    rows = []
    rng = np.random.default_rng(195)
    total_payload_bytes = 0
    for name, positions in cases:
        for dtype, mdtype in [(np.float32, mx.float32), (np.float16, mx.float16)]:
            q = rng.normal(size=(1, 24, positions.shape[-1], 256)).astype(dtype)
            k = rng.normal(size=(1, 4, positions.shape[-1], 256)).astype(dtype)
            assert q.nbytes + k.nbytes + positions.nbytes < 16 * 1024 * 1024
            total_payload_bytes += q.nbytes + k.nbytes + positions.nbytes
            nq, nk, pos = mx.array(q, dtype=mdtype), mx.array(k, dtype=mdtype), mx.array(positions, dtype=mx.int32)
            original_q, original_k = rotary.apply_rotary(nq, nk, pos)
            mx.eval(original_q, original_k)
            cos, sin = rotary(nk, pos)
            wrapper_q, wrapper_k = scope['apply_multimodal_rotary_pos_emb'](nq, nk, cos, sin)
            mx.eval(wrapper_q, wrapper_k)
            assert np.array_equal(np.array(original_q), np.array(wrapper_q))
            assert np.array_equal(np.array(original_k), np.array(wrapper_k))
            for role, original, reference in [('q', q, original_q), ('k', k, original_k)]:
                expected = np.array(reference)
                before = original.tobytes()
                actual = rotate_numpy(original, positions)
                assert before == original.tobytes()
                assert np.array_equal(actual[..., 64:], original[..., 64:])
                assert np.array_equal(expected[..., 64:], original[..., 64:])
                error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                tolerance = 1e-6 + 1e-5 * np.abs(expected.astype(np.float64))
                finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))
                bad = int(np.count_nonzero(error > tolerance))
                rows.append(dict(case=name, dtype=str(np.dtype(dtype)), role=role,
                                 shape=list(q.shape if role == 'q' else k.shape),
                                 max_abs_error=float(error.max()),
                                 max_error_ratio=float(np.max(error / tolerance)),
                                 violations=bad, finite=finite,
                                 strict_pass=finite and bad == 0,
                                 native_wrapper_exact=True, unchanged_tail_exact=True,
                                 input_unchanged=True))
            del nq, nk, pos, original_q, original_k, wrapper_q, wrapper_k, cos, sin
            mx.clear_cache()
    result = dict(schema='native-mlx-cpu-rotary-reference-v1',
                  status='passed' if all(r['strict_pass'] for r in rows) else 'numerical_rejection',
                  passed=all(r['strict_pass'] for r in rows),
                  device=str(mx.default_device()), gpu_kernels_executed=False,
                  model_loaded=False, network_used=False, upstream_fused_metal_disabled=True,
                  atol=1e-6, rtol=1e-5, rows=rows, row_count=len(rows),
                  cumulative_input_payload_bytes=total_payload_bytes,
                  maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  elapsed_seconds=time.monotonic() - started,
                  package_versions={name:importlib.metadata.version(name) for name in ['mlx', 'mlx-metal', 'numpy']},
                  sources={str(p.relative_to(WORK)):digest(p) for p in
                           [rope_path, language_path, WORK/'multimodal_rotary/__init__.py',
                            WORK/'multimodal_contract/__init__.py', Path(__file__)]},
                  scope='Native MLX CPU eager mRoPE versus existing NumPy primitive only. No Metal fused kernel, BF16, attention/cache/encoder, model logits, full-model capacity or screenshot qualification. A numerical rejection is retained without loosening tolerance.')
    with (OUT / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status', 'passed', 'device', 'row_count', 'elapsed_seconds', 'maxrss_bytes']}))
    print(json.dumps({'failed_rows':[r for r in rows if not r['strict_pass']]}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
