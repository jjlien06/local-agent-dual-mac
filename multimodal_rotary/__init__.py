"""Opt-in bounded target rotary math; CPU NumPy only, no attention integration."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class TargetRotaryConfig:
    """Pinned identity metadata, not an override surface."""

    model_id: str
    revision: str
    config_sha256: str
    query_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    partial_rotary_factor: float
    theta: int
    mrope_interleaved: bool
    mrope_section: tuple
    pairing: str


_CONFIG_FIELDS = (
    ("model_id", "mlx-community/Qwen3.8-27B-4bit"),
    ("revision", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"),
    ("config_sha256", "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85"),
    ("query_heads", 24), ("kv_heads", 4), ("head_dim", 256), ("rotary_dim", 64),
    ("partial_rotary_factor", 0.25), ("theta", 10000000),
    ("mrope_interleaved", True), ("mrope_section", (11, 11, 10)),
    ("pairing", "half_split"),
)
TARGET = TargetRotaryConfig(*(value for _, value in _CONFIG_FIELDS))


def _validate_config(config):
    if type(config) is not TargetRotaryConfig:
        raise ValueError("config requires exact TargetRotaryConfig metadata")
    for name, expected in _CONFIG_FIELDS:
        value = getattr(config, name)
        if type(value) is not type(expected):
            raise ValueError("config fields require exact immutable types")
        if name == "mrope_section" and any(type(part) is not int for part in value):
            raise ValueError("config sections require exact Python integers")
        if value != expected:
            raise ValueError("config must match the pinned target")

# Axis of the position used at each of the target's 32 complex frequencies.
AXIS_SELECTION = (
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
    1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
)


def rotate_numpy(x, position_ids, *, config=TARGET):
    """Rotate a single Q or K array; authoritative positions are never derived."""
    _validate_config(config)
    if type(x) is not np.ndarray or x.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("x must be an exact native NumPy float16/float32 array")
    if (
        x.ndim != 4 or x.shape[0] != 1 or x.shape[3] != 256
        or not 1 <= x.shape[1] <= 24 or not 1 <= x.shape[2] <= 256
    ):
        raise ValueError("x shape must be [1,H,T,256], H=1..24, T=1..256")
    if type(position_ids) is not np.ndarray or position_ids.dtype != np.dtype("int32"):
        raise ValueError("positions must be an exact native NumPy int32 array")
    if position_ids.shape != (3, 1, x.shape[2]):
        raise ValueError("positions must have authoritative shape [3,1,T]")
    if np.any(position_ids < 0) or np.any(position_ids > 1055):
        raise ValueError("positions must be within CPU prompt/decode range 0..1055")
    inv_freq = 1.0 / (10000000 ** (np.arange(0, 64, 2).astype(np.float32) / 64))
    selected = np.take(position_ids, AXIS_SELECTION, axis=0).transpose(1, 2, 0)
    freqs = selected.astype(np.float32) * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(x.dtype)[:, None]
    sin = np.sin(emb).astype(x.dtype)[:, None]
    prefix = x[..., :64]
    rotated = np.concatenate([-prefix[..., 32:], prefix[..., :32]], axis=-1)
    output = (prefix * cos) + (rotated * sin)
    return np.concatenate([output.astype(x.dtype), x[..., 64:]], axis=-1)
