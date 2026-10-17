"""Actual pinned VLM package import plus tiny CPU rotary smoke; no models."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import socket
import traceback

WORK = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
OUT = Path(__file__).resolve().parent
SOURCE = WORK/'upstream/mlx-vlm'


def main():
    started = time.monotonic()
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(SOURCE))
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    assert mx.default_device() == mx.cpu
    blocked = []
    kernel_factories = []
    kernel_invocations = []
    socket.has_ipv6 = False  # Owned offline probe only: skip urllib3 loopback bind capability probe.
    def audit(event, args):
        if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo',
                     'subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.fork', 'os.kill'}:
            blocked.append(event)
            raise RuntimeError('forbidden external operation: '+event)
        if event == 'open' and args and isinstance(args[0], str):
            if args[0].lower().endswith(('.safetensors', '.gguf', '.npz')):
                blocked.append('model-weight-file-open')
                raise RuntimeError('model weight reads forbidden in import probe')
    sys.addaudithook(audit)
    def forbidden(*args, **kwargs):
        raise RuntimeError('Metal custom kernels and model loads forbidden in import probe')
    original_factory = mx.fast.metal_kernel
    def guarded_factory(*args, **kwargs):
        native_callable = original_factory(*args, **kwargs)
        assert callable(native_callable)
        name = kwargs.get('name', args[0] if args else '<unnamed>')
        kernel_factories.append(name)
        def blocked_launch(*inputs, **options):
            kernel_invocations.append(name)
            raise RuntimeError('actual custom Metal kernel invocation is forbidden')
        return blocked_launch
    mx.fast.metal_kernel = guarded_factory
    mx.load = forbidden
    status = 'failed'
    detail = {}
    try:
        import mlx_vlm
        import mlx_vlm.models.rope_utils as rope
        import mlx_vlm.models.qwen3_5.language as language
        assert Path(mlx_vlm.__file__).resolve() == (SOURCE/'mlx_vlm/__init__.py').resolve()
        language_path = Path(language.__file__).resolve()
        rope_path = Path(rope.__file__).resolve()
        language_hash = hashlib.sha256(language_path.read_bytes()).hexdigest()
        rope_hash = hashlib.sha256(rope_path.read_bytes()).hexdigest()
        assert language_hash == '6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7'
        assert rope_hash == '6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b'
        rope._HAS_METAL = False
        rotary = language.Qwen3_5RotaryEmbedding(64, max_position_embeddings=262144,
                                               base=10000000, mrope_section=[11,11,10])
        assert not rotary.fused_apply
        q, k = mx.ones((1,24,1,256)), mx.ones((1,4,1,256))
        positions = mx.zeros((3,1,1), dtype=mx.int32)
        rq, rk = rotary.apply_rotary(q, k, positions)
        mx.eval(rq, rk)
        assert np.array_equal(np.array(rq), np.array(q))
        assert np.array_equal(np.array(rk), np.array(k))
        assert mx.default_device() == mx.cpu
        detail = dict(package_version=mlx_vlm.__version__, package_path=str(Path(mlx_vlm.__file__).resolve()),
                      language_sha256=language_hash, rope_sha256=rope_hash,
                      actual_pinned_package_imported=True, q_shape=list(q.shape), k_shape=list(k.shape),
                      tiny_zero_position_native_rotary_exact=True, fused_metal_disabled=True)
        status = 'passed'
    except Exception as exc:
        detail = dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(), actual_pinned_package_imported=False)
    result = dict(schema='actual-pinned-vlm-source-import-cpu-smoke-v1', status=status,
                  passed=status=='passed', detail=detail, blocked_operations=blocked,
                  native_kernel_factories_registered=kernel_factories,
                  native_custom_kernel_invocations=kernel_invocations,
                  urllib3_ipv6_probe_disabled_in_child_only=True,
                  device=str(mx.default_device()), model_loaded=False,
                  gpu_kernel_execution=False, elapsed_seconds=time.monotonic()-started,
                  maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  numeric_versions={n:importlib.metadata.version(n) for n in ['mlx','mlx-metal','numpy']},
                  scope='Actual complete VLM package imports and one tiny all-zero-position CPU rotary smoke only. No encoder/model construction, weights, real image, Metal, full-attention/cache or screenshot admission.')
    with (OUT/'import-result-v2.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
