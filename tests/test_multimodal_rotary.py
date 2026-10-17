"""Bounded rotary tests; original pinned AST runs with NumPy, never MLX."""

import ast
import importlib
from dataclasses import FrozenInstanceError, replace

import pytest
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from multimodal_contract import prepare_request

# Frequency j=0..31 selects t=0, h=1, w=2; NOT adjacent-channel pairing.
GOLDEN_AXIS_SELECTION = (
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
    1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
)

from multimodal_rotary._pinned_source import LANGUAGE_SOURCE, ROPE_UTILS_SOURCE


def original_namespace():
    """Execute original eager AST: mx=NumPy, eval/compile=no-op, no Metal."""
    mx = SimpleNamespace(
        **{name: getattr(np, name) for name in (
            "array", "arange", "float32", "int32", "take", "concatenate",
            "cos", "sin", "expand_dims", "repeat",
        )},
        eval=lambda *args: None,
        compile=lambda function: function,
    )
    scope = {
        "mx": mx, "nn": SimpleNamespace(Module=object),
        "Optional": Optional, "Sequence": Sequence,
        "_HAS_METAL": False, "_HALF_SPLIT": 0, "_EVEN_ODD": 1,
    }
    exec(compile(ast.parse(ROPE_UTILS_SOURCE), "<pinned-rope-utils>", "exec"), scope)
    # Use separate globals so the language wrapper cannot shadow shared apply.
    language = dict(scope, _apply_mrope=scope["apply_multimodal_rotary_pos_emb"])
    exec(compile(ast.parse(LANGUAGE_SOURCE), "<pinned-language>", "exec"), language)
    return language


def oracle(x, positions):
    scope = original_namespace()
    rotary = scope["Qwen3_5RotaryEmbedding"](
        64, max_position_embeddings=262144, base=10000000,
        mrope_section=[11, 11, 10],
    )
    q, k = rotary.apply_rotary(x, x, positions)
    np.testing.assert_array_equal(q, k, strict=True)
    cos, sin = rotary(x, positions)
    q_wrapper, _ = scope["apply_multimodal_rotary_pos_emb"](x, x, cos, sin)
    np.testing.assert_array_equal(q, q_wrapper, strict=True)
    return q


def primitive():
    module = importlib.import_module("multimodal_rotary")
    assert callable(getattr(module, "rotate_numpy", None)), "rotate_numpy is missing"
    return module.rotate_numpy


def sample(heads=1, tokens=3, dtype=np.float32):
    return np.random.default_rng(19).normal(size=(1, heads, tokens, 256)).astype(dtype)


def test_scalar_text_matches_original_ast_and_independent_half_split_math():
    x = sample()
    positions = np.tile(np.array([0, 17, 1055], dtype=np.int32), (3, 1, 1))
    expected = oracle(x, positions)
    # Independent scalar RoPE derivation: complex pair j is (x_j, x_{j+32}).
    manual = x.copy()
    for token, position in enumerate((0, 17, 1055)):
        for j in range(32):
            angle = position / (10000000 ** (2 * j / 64))
            a, b = x[0, 0, token, j], x[0, 0, token, j + 32]
            manual[0, 0, token, j] = a * np.cos(angle) - b * np.sin(angle)
            manual[0, 0, token, j + 32] = b * np.cos(angle) + a * np.sin(angle)
    np.testing.assert_allclose(expected, manual, atol=2e-5, rtol=2e-4)
    actual = primitive()(x, positions)
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_non_square_image_golden_axis_selection_and_pairing_mutants():
    request = prepare_request(
        [7, 248053] + [248056] * 6 + [248054, 8],
        request_id="rotary-nonsquare", grid=[1, 4, 6],
    )
    positions = request.position_ids
    x = sample(heads=4, tokens=request.prompt_length)
    scope = original_namespace()
    selector = scope["mrope_position_selector"]("interleaved", [11, 11, 10], 32)
    assert tuple(selector.tolist()) == GOLDEN_AXIS_SELECTION
    expected = oracle(x, positions)
    np.testing.assert_array_equal(primitive()(x, positions), expected, strict=True)
    module = importlib.import_module("multimodal_rotary")
    assert module.AXIS_SELECTION == GOLDEN_AXIS_SELECTION
    # Negative controls: a t-only or h/w-swapped source cannot pass this fixture.
    assert not np.array_equal(expected, oracle(x, positions[[0, 2, 1]]))
    assert not np.array_equal(expected, oracle(x, np.repeat(positions[:1], 3, axis=0)))
    inv = scope["compute_inv_freq"](64, 10000000)
    freq = scope["compute_mrope_frequencies"](positions, inv, [11, 11, 10])
    c = np.repeat(np.cos(freq), 2, axis=-1)[:, None]
    s = np.repeat(np.sin(freq), 2, axis=-1)[:, None]
    prefix = x[..., :64]
    adjacent = np.stack((-prefix[..., 1::2], prefix[..., ::2]), axis=-1).reshape(prefix.shape)
    mutant = prefix * c + adjacent * s
    assert not np.allclose(expected[..., :64], mutant, atol=1e-6, rtol=1e-5)


def test_metadata_is_pinned_immutable_and_validated_by_exact_field_types(monkeypatch):
    module = importlib.import_module("multimodal_rotary")
    assert hasattr(module, "TARGET"), "immutable target metadata is missing"
    target = module.TARGET
    expected = {
        "model_id": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "config_sha256": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
        "query_heads": 24, "kv_heads": 4, "head_dim": 256, "rotary_dim": 64,
        "partial_rotary_factor": 0.25, "theta": 10000000,
        "mrope_interleaved": True, "mrope_section": (11, 11, 10),
        "pairing": "half_split",
    }
    for name, value in expected.items():
        assert type(getattr(target, name)) is type(value)
        assert getattr(target, name) == value
    with pytest.raises(FrozenInstanceError):
        target.theta = 1
    x = sample()
    positions = np.zeros((3, 1, 3), dtype=np.int32)
    np.testing.assert_array_equal(module.rotate_numpy(x, positions, config=replace(target)), x)

    def forbidden(*args, **kwargs):
        pytest.fail("rotary backend reached for invalid metadata")

    monkeypatch.setattr(module.np, "arange", forbidden)
    for name, value in expected.items():
        if type(value) is str:
            wrong = value + "-wrong"
        elif type(value) is bool:
            wrong = False
        elif type(value) is tuple:
            wrong = (11, 10, 11)
        else:
            wrong = value + 1
        with pytest.raises(ValueError, match="config"):
            module.rotate_numpy(x, positions, config=replace(target, **{name: wrong}))
    # Equality-compatible scalar and tuple values must still be rejected.
    impostors = {
        "theta": 10000000.0, "head_dim": np.int64(256),
        "partial_rotary_factor": np.float64(0.25), "mrope_interleaved": 1,
        "mrope_section": [11, 11, 10],
    }
    for name, value in impostors.items():
        with pytest.raises(ValueError, match="config"):
            module.rotate_numpy(x, positions, config=replace(target, **{name: value}))
    for section in ((11.0, 11, 10), (11, np.int64(11), 10)):
        with pytest.raises(ValueError, match="config"):
            module.rotate_numpy(x, positions, config=replace(target, mrope_section=section))
    for foreign in (None, expected, SimpleNamespace(**expected)):
        with pytest.raises(ValueError, match="config"):
            module.rotate_numpy(x, positions, config=foreign)


@pytest.mark.parametrize("kind", [
    "list", "subclass", "float64", "int32", "bool", "complex64", "object",
    "big_endian", "scalar", "rank3", "rank5", "batch0", "batch2", "heads0",
    "heads25", "tokens0", "tokens257", "width64", "width255", "width257",
])
def test_rejects_unsupported_x_before_rotary_math(kind, monkeypatch):
    module = importlib.import_module("multimodal_rotary")
    x = sample()
    if kind == "list":
        x = x.tolist()
    elif kind == "subclass":
        class Subarray(np.ndarray):
            pass
        x = x.view(Subarray)
    elif kind in ("float64", "int32", "bool", "complex64", "object"):
        x = x.astype(kind)
    elif kind == "big_endian":
        x = x.astype(">f4")
    else:
        shapes = {
            "scalar": (), "rank3": (1, 3, 256), "rank5": (1, 1, 1, 3, 256),
            "batch0": (0, 1, 3, 256), "batch2": (2, 1, 3, 256),
            "heads0": (1, 0, 3, 256), "heads25": (1, 25, 3, 256),
            "tokens0": (1, 1, 0, 256), "tokens257": (1, 1, 257, 256),
            "width64": (1, 1, 3, 64), "width255": (1, 1, 3, 255),
            "width257": (1, 1, 3, 257),
        }
        x = np.zeros(shapes[kind], dtype=np.float32)
    positions = np.zeros((3, 1, 3), dtype=np.int32)

    def forbidden(*args, **kwargs):
        pytest.fail("rotary backend reached for invalid x")

    monkeypatch.setattr(module.np, "arange", forbidden)
    with pytest.raises(ValueError, match="x"):
        module.rotate_numpy(x, positions)


@pytest.mark.parametrize("kind", [
    "list", "subclass", "float32", "float64", "int64", "uint32", "bool",
    "object", "complex64", "big_endian", "rank2", "rank4", "axes1",
    "axes4", "batch2", "tokens2", "tokens4", "negative", "over_cap",
    "int32_min", "int32_max", "nan", "inf",
])
def test_rejects_malformed_authoritative_positions_before_rotary_math(kind, monkeypatch):
    module = importlib.import_module("multimodal_rotary")
    x = sample()
    positions = np.zeros((3, 1, 3), dtype=np.int32)
    if kind == "list":
        positions = positions.tolist()
    elif kind == "subclass":
        class Subarray(np.ndarray):
            pass
        positions = positions.view(Subarray)
    elif kind in ("float32", "float64", "int64", "uint32", "bool", "object", "complex64"):
        positions = positions.astype(kind)
    elif kind == "big_endian":
        positions = positions.astype(">i4")
    elif kind in ("negative", "over_cap", "int32_min", "int32_max"):
        positions[1, 0, 1] = {
            "negative": -1, "over_cap": 1056, "int32_min": -2147483648,
            "int32_max": 2147483647,
        }[kind]
    elif kind in ("nan", "inf"):
        positions = positions.astype(np.float32)
        positions[2, 0, 2] = float(kind)
    else:
        shapes = {
            "rank2": (3, 3), "rank4": (3, 1, 1, 3), "axes1": (1, 1, 3),
            "axes4": (4, 1, 3), "batch2": (3, 2, 3),
            "tokens2": (3, 1, 2), "tokens4": (3, 1, 4),
        }
        positions = np.zeros(shapes[kind], dtype=np.int32)

    def forbidden(*args, **kwargs):
        pytest.fail("rotary backend reached for invalid positions")

    monkeypatch.setattr(module.np, "arange", forbidden)
    with pytest.raises(ValueError, match="position"):
        module.rotate_numpy(x, positions)
