"""Bounded native MLX CPU rotary reference; no model adapter."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import types

from multimodal_rotary import TARGET, TargetRotaryConfig

import numpy as np

_CONFIG_FIELDS = (
    ('model_id', 'mlx-community/Qwen3.8-27B-4bit'),
    ('revision', '10c35caafbb80f7dc6a7a432cdd11af10a6d4818'),
    ('config_sha256', '14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85'),
    ('query_heads', 24), ('kv_heads', 4), ('head_dim', 256), ('rotary_dim', 64),
    ('partial_rotary_factor', 0.25), ('theta', 10000000),
    ('mrope_interleaved', True), ('mrope_section', (11, 11, 10)),
    ('pairing', 'half_split'),
)


_ROOT = Path(__file__).resolve().parents[1]
_UPSTREAM = _ROOT / 'upstream/mlx-vlm'
_CONFIG_PATH = Path('/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/config.json')
_ROPE_PATH = _UPSTREAM / 'mlx_vlm/models/rope_utils.py'
_LANGUAGE_PATH = _UPSTREAM / 'mlx_vlm/models/qwen3_5/language.py'
_UPSTREAM_REVISION = 'b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0'
_SOURCE_DIGESTS = (
    (_CONFIG_PATH, '14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85'),
    (_ROPE_PATH, '6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b'),
    (_LANGUAGE_PATH, '6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7'),
)


def _admit_sources():
    """Read once, verify before execution; never import the model package."""
    admitted = []
    for path, expected in _SOURCE_DIGESTS:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ValueError(f'source/config unavailable: {path}') from exc
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'source/config SHA256 mismatch: {path}')
        admitted.append(data)
    try:
        head = subprocess.run(['git', '-C', str(_UPSTREAM), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('source checkout revision unavailable') from exc
    if head != _UPSTREAM_REVISION:
        raise ValueError('source checkout revision mismatch')
    return json.loads(admitted[0])['text_config'], admitted[1], admitted[2]


def _admit_runtime():
    for name, expected in (('mlx', '0.32.4'), ('mlx-metal', '0.32.4'), ('numpy', '2.5.3')):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f'pinned runtime unavailable: {name}') from exc
        if version != expected:
            raise ValueError(f'pinned runtime version mismatch: {name}')


def _validate_config(config):
    if type(config) is not TargetRotaryConfig:
        raise ValueError('config requires exact immutable TargetRotaryConfig')
    for field, expected in _CONFIG_FIELDS:
        value = getattr(config, field, None)
        if type(value) is not type(expected) or value != expected:
            raise ValueError('config must match pinned metadata and exact types')
        if field == 'mrope_section' and any(type(v) is not int for v in value):
            raise ValueError('config sections require exact Python integers')


def _validate_arrays(q, k, positions):
    for array, heads in ((q, 24), (k, 4)):
        if type(array) is not np.ndarray or array.dtype not in (np.dtype('float16'), np.dtype('float32')):
            raise ValueError('Q/K require exact native NumPy float16/float32 arrays')
        if (array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != heads
                or array.shape[3] != 256 or not 1 <= array.shape[2] <= 256):
            raise ValueError('Q/K shapes must be [1,24/4,T,256], T=1..256')
        if not np.isfinite(array).all():
            raise ValueError('Q/K inputs must be finite')
    if q.dtype != k.dtype or q.shape[2] != k.shape[2]:
        raise ValueError('Q/K must have identical token count and dtype')
    if type(positions) is not np.ndarray or positions.dtype != np.dtype('int32'):
        raise ValueError('positions require an exact native NumPy int32 array')
    if positions.shape != (3, 1, q.shape[2]):
        raise ValueError('positions require shape [3,1,T]')
    if np.any(positions < 0) or np.any(positions > 1055):
        raise ValueError('positions must be in 0..1055')


def rotate_reference_cpu(q_numpy, k_numpy, position_ids_numpy, config=TARGET):
    """Return Q/K from the pinned upstream eager MLX CPU rotary implementation."""
    _validate_config(config)
    _validate_arrays(q_numpy, k_numpy, position_ids_numpy)
    text, rope_bytes, language_bytes = _admit_sources()
    _admit_runtime()
    import mlx.core as mx

    required = ('default_device', 'set_default_device', 'array', 'eval', 'compile')
    if getattr(mx, 'cpu', None) is None or not all(callable(getattr(mx, name, None)) for name in required):
        raise ValueError('pinned MLX CPU API unavailable; no device substitution is permitted')
    previous_device = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        if mx.default_device() != mx.cpu:
            raise RuntimeError('CPU default device was not established')
        # Execute the complete verified rope_utils bytes, not a rewritten subset.
        # Decorated helpers remain upstream; this route never selects fused Metal.
        module = types.ModuleType('_source_bound_native_cpu_rope')
        module.__file__ = str(_ROPE_PATH)
        exec(compile(rope_bytes, str(_ROPE_PATH), 'exec'), module.__dict__)
        module._HAS_METAL = False
        names = {'Qwen3_5RotaryEmbedding', 'apply_multimodal_rotary_pos_emb'}
        nodes = [node for node in ast.parse(language_bytes).body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if len(nodes) != 2:
            raise RuntimeError('verified upstream rotary AST definitions unavailable')
        scope = {'mx': mx, 'MRoPERotaryEmbedding': module.MRoPERotaryEmbedding,
                 '_apply_mrope': module.apply_multimodal_rotary_pos_emb}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_LANGUAGE_PATH), 'exec'), scope)
        rotary = scope['Qwen3_5RotaryEmbedding'](
            int(text['head_dim'] * text['partial_rotary_factor']),
            max_position_embeddings=text['max_position_embeddings'],
            base=text['rope_parameters']['rope_theta'],
            mrope_section=text['rope_parameters']['mrope_section'])
        if rotary.fused_apply is not False:
            raise RuntimeError('fused Metal must be disabled before rotary construction')
        # Explicit copies also remove negative strides on singleton axes (which
        # ascontiguousarray may retain). Transport only, not replacement math.
        nq, nk, positions = (mx.array(a.copy(order="C"))
                             for a in (q_numpy, k_numpy, position_ids_numpy))
        q, k = rotary.apply_rotary(nq, nk, positions)
        mx.eval(q, k)
        return np.array(q), np.array(k)
    finally:
        # Allocator limits are deliberately never changed by this bridge.
        mx.set_default_device(previous_device)
