"""Native CPU synthetic attention seam witness; no checkpoint/model runtime."""
import __future__
import ast
import hashlib
import importlib.util
import json
import resource
import socket
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
LM = Path('/Users/jeremylien/Projects/paretoquant-metal/.venv/lib/python3.13/site-packages/mlx_lm/models')
VLM = ROOT / 'upstream/mlx-vlm/mlx_vlm/models'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute_nodes(path, names, scope):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), scope)


def main():
    started = time.monotonic()
    def forbidden(*args, **kwargs):
        raise RuntimeError('network or Metal custom kernel forbidden in CPU witness')
    socket.socket = forbidden
    socket.create_connection = forbidden
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_default_device(mx.cpu)
    assert mx.default_device() == mx.cpu
    mx.fast.metal_kernel = forbidden
    mx.set_memory_limit(192 * 1024 * 1024)  # Advisory, not a hard residency bound.
    mx.set_cache_limit(8 * 1024 * 1024)
    from multimodal_contract import prepare_request
    from witness import BoundRope

    expected = {
        LM/'qwen3_next.py': '67ca08a4d08cb7be124965e95399735e556f1eaeb21a16bc23cb88d1ac035e11',
        VLM/'qwen3_5/language.py': '6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7',
        VLM/'rope_utils.py': '6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b',
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    package = types.ModuleType('_witness_lm')
    package.__path__ = [str(LM)]
    sys.modules[package.__name__] = package
    base = load('_witness_lm.base', LM/'base.py')
    cache_module = load('_witness_lm.cache', LM/'cache.py')
    native_rope = load('_witness_lm.rope_utils', LM/'rope_utils.py')
    vlm_rope = load('_witness_vlm_rope', VLM/'rope_utils.py')
    vlm_rope._HAS_METAL = False
    language_scope = {'mx': mx, 'nn': nn,
                      'MRoPERotaryEmbedding': vlm_rope.MRoPERotaryEmbedding,
                      '_apply_mrope': vlm_rope.apply_multimodal_rotary_pos_emb,
                      'scaled_dot_product_attention': base.scaled_dot_product_attention}
    execute_nodes(VLM/'qwen3_5/language.py',
                  {'Qwen3_5RotaryEmbedding', 'apply_multimodal_rotary_pos_emb', 'Qwen3_5Attention'}, language_scope)
    lm_scope = {'mx': mx, 'nn': nn, 'initialize_rope': native_rope.initialize_rope,
                'scaled_dot_product_attention': base.scaled_dot_product_attention}
    execute_nodes(LM/'qwen3_next.py', {'Qwen3NextAttention'}, lm_scope)
    # The witness intentionally shares the original ordinary-KV SDPA helper.
    # Verify that the VLM's ordinary unquantized terminal call is identical.
    trees = [ast.parse(p.read_text()) for p in [LM/'base.py', VLM/'base.py']]
    sdpa_calls = []
    for tree in trees:
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scaled_dot_product_attention')
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'mx.fast.scaled_dot_product_attention']
        sdpa_calls.append(calls[-1])
    assert ast.dump(sdpa_calls[0], include_attributes=False) == ast.dump(sdpa_calls[1], include_attributes=False)
    args = SimpleNamespace(hidden_size=32, num_attention_heads=24, num_key_value_heads=4,
                           head_dim=256, attention_bias=False, rms_norm_eps=1e-6,
                           partial_rotary_factor=0.25, rope_theta=10000000,
                           max_position_embeddings=262144, rope_scaling=None,
                           rope_parameters={'partial_rotary_factor':0.25, 'rope_theta':10000000,
                                            'mrope_section':[11,11,10]})
    fixtures = [('text', prepare_request([7,8,9], request_id='attention-text'))]
    for h, width, prefix in [(4,6,1), (6,4,1), (16,32,1), (4,6,252)]:
        count = h * width // 4
        ids = [7]*prefix + [248053] + [248056]*count + [248054,8]
        name = f'image-{h}x{width}-prefix{prefix}'
        fixtures.append((name, prepare_request(ids, request_id=name, grid=(1,h,width))))
    rows = []
    scalar_rows = []
    rng = np.random.default_rng(905)
    shared = ['q_proj','k_proj','v_proj','o_proj','q_norm','k_norm']
    for dtype in [mx.float32, mx.float16]:
        for name, request in fixtures:
            mx.random.seed(905)
            reference = language_scope['Qwen3_5Attention'](args)
            candidate = lm_scope['Qwen3NextAttention'](args)
            scalar = lm_scope['Qwen3NextAttention'](args) if name == 'text' else None
            for attr in shared:
                module = getattr(reference, attr)
                module.weight = module.weight.astype(dtype)
                setattr(candidate, attr, module)
                if scalar is not None:
                    setattr(scalar, attr, module)
            rotary = language_scope['Qwen3_5RotaryEmbedding'](64, max_position_embeddings=262144,
                                                          base=10000000, mrope_section=[11,11,10])
            assert not rotary.fused_apply and not reference.rotary_emb.fused_apply
            hook = BoundRope(rotary)
            candidate.rope = hook
            rc, cc, sc = cache_module.KVCache(), cache_module.KVCache(), cache_module.KVCache()
            intervals = [(start, min(start+256, request.prompt_length-1))
                         for start in range(0, request.prompt_length-1, 256)]
            intervals += [(request.prompt_length-1, request.prompt_length)]
            intervals += [(request.prompt_length+i, request.prompt_length+i+1) for i in range(2)]
            for start, end in intervals:
                assert rc.offset == cc.offset == start
                if start < request.prompt_length:
                    positions = request.chunk(start, end).position_ids
                else:
                    positions = request.continuation(start)
                data = rng.normal(size=(1, end-start, 32)).astype(np.float32)
                x = mx.array(data, dtype=dtype)
                pos = mx.array(positions, dtype=mx.int32)
                mask = base.create_attention_mask(x, cc)
                hook.bind(pos, physical_offset=cc.offset, tokens=end-start)
                expected_output = reference(x, mask=mask, cache=rc, position_ids=pos)
                actual_output = candidate(x, mask=mask, cache=cc)
                mx.eval(expected_output, actual_output, rc.state, cc.state)
                assert rc.offset == cc.offset == end
                metrics = [('output', expected_output, actual_output),
                           ('live_keys', rc.keys_and_values()[0], cc.keys_and_values()[0]),
                           ('live_values', rc.keys_and_values()[1], cc.keys_and_values()[1])]
                for part, exp, actual in metrics:
                    e, v = np.array(exp).astype(np.float64), np.array(actual).astype(np.float64)
                    error = np.abs(v-e)
                    tolerance = 1e-6 + 1e-5*np.abs(e)
                    finite = bool(np.isfinite(v).all() and np.isfinite(e).all())
                    rows.append(dict(case=name, dtype=str(dtype), interval=[start,end], part=part,
                                     max_abs_error=float(error.max()), violations=int(np.count_nonzero(error>tolerance)),
                                     finite=finite, strict_pass=finite and bool(np.all(error<=tolerance)),
                                     physical_offset=end, maximum_position=int(positions.max())))
                if scalar is not None:
                    stock = scalar(x, mask=mask, cache=sc)
                    mx.eval(stock, sc.state)
                    e, v = np.array(expected_output).astype(np.float64), np.array(stock).astype(np.float64)
                    error = np.abs(v-e)
                    tolerance = 1e-6+1e-5*np.abs(e)
                    scalar_rows.append(dict(dtype=str(dtype), interval=[start,end],
                                            max_abs_error=float(error.max()),
                                            violations=int(np.count_nonzero(error>tolerance)),
                                            strict_pass=bool(np.all(error<=tolerance))))
            del reference, candidate, scalar, rc, cc, sc, hook, rotary
            mx.clear_cache()
    passed = all(v['strict_pass'] for v in rows)
    result = dict(schema='native-cpu-attention-rope-seam-witness-v1', passed=passed,
                  status='passed' if passed else 'numerical_rejection', device=str(mx.default_device()),
                  atol=1e-6, rtol=1e-5, rows=rows, row_count=len(rows),
                  stock_scalar_text_control=scalar_rows,
                  stock_scalar_text_control_passed=all(v['strict_pass'] for v in scalar_rows),
                  hidden_size=32, query_heads=24, kv_heads=4, head_dim=256,
                  original_attention_forward_modified=False,
                  original_cache_update_used=True, checkpoint_loaded=False,
                  gpu_kernels_executed=False, native_fused_metal_disabled=True,
                  ordinary_unquantized_sdpa_call_ast_equal=True,
                  elapsed_seconds=time.monotonic()-started,
                  maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [LM/'qwen3_next.py', LM/'base.py', LM/'cache.py', LM/'rope_utils.py',
                            VLM/'qwen3_5/language.py', VLM/'base.py', VLM/'rope_utils.py',
                            OUT/'witness.py', OUT/'test_binding.py', Path(__file__)]},
                  scope='Synthetic unquantized hidden32 attention with target24Q/4KV/256head geometry on native CPU, not loaded checkpoint or actual16-layer integration. Original LM/VLM attention classes assembled by AST; both use original LM ordinary-KV SDPA helper whose native terminal call matches VLM. No quantized KV, padding, TurboQuant, BF16, fused Metal, encoder, full-model logits, distributed, MTP or screenshot admission.')
    with (OUT/'result.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','passed','row_count','stock_scalar_text_control_passed','elapsed_seconds','maxrss_bytes']}))
    print(json.dumps({'failed_rows':[v for v in rows if not v['strict_pass']],
                      'failed_stock_scalar_control':[v for v in scalar_rows if not v['strict_pass']]}))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
