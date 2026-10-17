"""CPU-only request/position contracts; never import a model runtime."""

import hashlib
import importlib
import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest

IMAGE = 248056
START = 248053
END = 248054


def image_ids(prefix=1):
    return [7] * prefix + [START] + [IMAGE] * 6 + [END, 9]


def golden(prefix=1):
    # Independently enumerated row-major merged 2x3 grid, not production math.
    assert prefix == 1
    return np.array(
        [
            [[0, 1, 2, 2, 2, 2, 2, 2, 5, 6]],
            [[0, 1, 2, 2, 2, 3, 3, 3, 5, 6]],
            [[0, 1, 2, 3, 4, 2, 3, 4, 5, 6]],
        ],
        dtype="<i4",
    )


@pytest.mark.parametrize(
    "grid,expected",
    [
        ((1, 4, 6), golden()),
        (
            (1, 6, 4),
            np.array(
                [
                    [[0, 1, 2, 2, 2, 2, 2, 2, 5, 6]],
                    [[0, 1, 2, 2, 3, 3, 4, 4, 5, 6]],
                    [[0, 1, 2, 3, 2, 3, 2, 3, 5, 6]],
                ],
                dtype="<i4",
            ),
        ),
    ],
)
def test_non_square_image_exact_coordinates_and_delta(grid, expected):
    request = api().prepare_request(image_ids(), request_id="image-1", grid=grid)
    np.testing.assert_array_equal(request.position_ids, expected)
    assert request.delta == -3
    assert request.prompt_length == 10  # actual cache length, NOT spatial max+1=7
    np.testing.assert_array_equal(request.continuation(10), [[[7]]] * 3)
    np.testing.assert_array_equal(request.continuation(11), [[[8]]] * 3)
    # Poison witnesses: these must fail exact independent expectations.
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(request.position_ids[[0, 2, 1]], expected)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(np.full((3, 1, 1), 10), request.continuation(10))


def test_bounded_immutable_chunks_cover_boundaries_and_last_token():
    for length in (255, 256, 257, 270):
        request = api().prepare_request(
            image_ids(length - 9), request_id=f"chunks-{length}", grid=(1, 4, 6)
        )
        chunks = [
            request.chunk(start, min(start + 256, length))
            for start in range(0, length, 256)
        ]
        np.testing.assert_array_equal(
            np.concatenate([chunk.position_ids for chunk in chunks], axis=2),
            request.position_ids,
        )
        assert (
            tuple(token for chunk in chunks for token in chunk.token_ids)
            == request.token_ids
        )
        assert chunks[-1].stop == length
        assert chunks[-1].token_ids[-1] == 9
        for chunk in chunks:
            assert chunk.delta == -3
            assert chunk.position_ids.dtype == np.dtype("<i4")
            with pytest.raises(ValueError):
                chunk.position_ids.setflags(write=True)
            with pytest.raises(AttributeError):
                chunk.start = 999
    request = api().prepare_request(image_ids(252), request_id="cross", grid=(1, 4, 6))
    for boundary in (255, 256, 257):
        chunk = request.chunk(boundary - 1, boundary + 1)
        np.testing.assert_array_equal(
            chunk.position_ids, request.position_ids[:, :, boundary - 1 : boundary + 1]
        )
    for start, stop in (
        (0, 257),
        (-1, 1),
        (1, 1),
        (2, 1),
        (0, 999),
        (True, 2),
        (0, 1.5),
    ):
        with pytest.raises(ValueError):
            request.chunk(start, stop)


def test_versioned_manifest_binds_exact_request_and_positions():
    request = api().prepare_request(image_ids(), request_id="manifest", grid=(1, 4, 6))
    expected = {
        "version": 1,
        "request_id": "manifest",
        "token_ids": image_ids(),
        "grid_thw": [1, 4, 6],
        "delta": -3,
        "config_sha256": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
        "position_dtype": "<i4",
        "position_shape": [3, 1, 10],
        "limits": {"prompt_tokens": 1024, "image_tokens": 256, "decode_tokens": 32},
        "position_sha256": hashlib.sha256(golden().tobytes(order="C")).hexdigest(),
    }
    expected_bytes = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert request.request_bytes == expected_bytes
    assert (
        request.manifest_hash
        == hashlib.sha256(b"qwen38-cpu-request-v1\0" + expected_bytes).hexdigest()
    )
    assert (
        api().prepare_request(image_ids(), request_id="manifest", grid=(1, 4, 6))
        == request
    )
    for changed in (
        api().prepare_request(
            [8] + image_ids()[1:], request_id="manifest", grid=(1, 4, 6)
        ),
        api().prepare_request(image_ids(), request_id="other", grid=(1, 4, 6)),
        api().prepare_request(image_ids(), request_id="manifest", grid=(1, 6, 4)),
        replace(request, delta=0),
        replace(request, position_bytes=request.position_ids[[0, 2, 1]].tobytes()),
        replace(request, config_sha256="0" * 64),
    ):
        assert changed.manifest_hash != request.manifest_hash
    assert request.chunk(0, 5).request_hash == request.manifest_hash


def test_lifecycle_physical_cursor_rejects_replay_poison_and_resets():
    request = api().prepare_request(image_ids(), request_id="lifecycle", grid=(1, 4, 6))
    session = api().Session(request)
    assert session.cursor == 0
    session.consume(request.chunk(0, 9), actual_cache_offset=0)
    assert session.cursor == 9
    session.consume(request.chunk(9, 10), actual_cache_offset=9)
    assert session.cursor == 10
    np.testing.assert_array_equal(session.decode(actual_cache_offset=10), [[[7]]] * 3)
    np.testing.assert_array_equal(session.decode(actual_cache_offset=11), [[[8]]] * 3)
    assert session.cursor == 12
    session.finish()
    assert session.request is None
    assert session.cursor == 0
    for terminal in ("finish", "cancel", "reset"):
        current = api().Session(request)
        if terminal == "finish":
            current.consume(request.chunk(0, 10), actual_cache_offset=0)
        getattr(current, terminal)()
        assert current.request is None
        assert current.cursor == 0
        with pytest.raises(ValueError):
            current.consume(request.chunk(0, 10), actual_cache_offset=0)
    text = api().Session(api().prepare_request([1, 2], request_id="fresh-text"))
    text.consume(text.request.chunk(0, 2), actual_cache_offset=0)
    np.testing.assert_array_equal(text.decode(actual_cache_offset=2), [[[2]]] * 3)
    chunk = request.chunk(0, 5)
    poisons = [
        replace(chunk, request_hash="0" * 64),
        replace(chunk, delta=0),
        replace(chunk, token_ids=(99,) + chunk.token_ids[1:]),
        replace(chunk, position_bytes=chunk.position_ids[[0, 2, 1]].tobytes()),
        replace(chunk, position_bytes=chunk.position_ids[:2].tobytes()),
        request.chunk(5, 10),
    ]
    for poison in poisons:
        current = api().Session(request)
        with pytest.raises(ValueError):
            current.consume(poison, actual_cache_offset=0)
        assert current.state == "failed"
        assert current.request is None
    current = api().Session(request)
    current.consume(chunk, actual_cache_offset=0)
    with pytest.raises(ValueError):
        current.consume(chunk, actual_cache_offset=5)  # duplicate
    for operation in (
        lambda s: s.consume(chunk, actual_cache_offset=7),  # spatial cursor misuse
        lambda s: s.decode(actual_cache_offset=0),  # premature decode
        lambda s: s.finish(),  # missing last prompt token
    ):
        current = api().Session(request)
        with pytest.raises(ValueError):
            operation(current)
        assert current.state == "failed"
    for forged in (
        replace(request, delta=0),
        replace(request, position_bytes=request.position_ids[[0, 2, 1]].tobytes()),
        replace(request, config_sha256="0" * 64),
    ):
        with pytest.raises(ValueError):
            api().Session(forged)


@pytest.mark.parametrize(
    "ids,grid",
    [
        ([IMAGE] * 6, (1, 4, 6)),  # missing both boundaries
        ([START] + [IMAGE] * 6, (1, 4, 6)),
        ([IMAGE] * 6 + [END], (1, 4, 6)),
        ([START, 7] + [IMAGE] * 6 + [END], (1, 4, 6)),
        ([START] + [IMAGE] * 5 + [END], (1, 4, 6)),
        ([START] + [IMAGE] * 7 + [END], (1, 4, 6)),
        ([START, IMAGE, 7] + [IMAGE] * 5 + [END], (1, 4, 6)),
        (image_ids() + [START, IMAGE, END], (1, 4, 6)),
        (image_ids() + [END], (1, 4, 6)),
        ([START] + image_ids(), (1, 4, 6)),
        (image_ids(), None),
        ([START, END], None),
        ([248057], None),
        ([START, 248057, END], (1, 4, 6)),
        (image_ids(), (2, 4, 6)),
        (image_ids(), (1, 3, 6)),
        (image_ids(), (1, 4, 0)),
        (image_ids(), (1, -4, 6)),
        (image_ids(), (1, 4)),
        (image_ids(), ((1, 4, 6), (1, 4, 6))),
        (image_ids(), (True, 4, 6)),
        (image_ids(), (1, 4.0, 6)),
        ([7, 8], (1, 4, 6)),
    ],
)
def test_image_admission_rejects_malformed_or_unsupported_blocks(ids, grid):
    with pytest.raises(ValueError, match="image|video|grid"):
        api().prepare_request(ids, request_id="reject-image", grid=grid)


@pytest.mark.parametrize(
    "ids",
    [
        [True],
        [1.5],
        ["7"],
        [-1],
        [248320],
        [2**80],
        [],
        [[1, 2]],
        "12",
        b"12",
        {1: 2},
        {1, 2},
        None,
        np.array([1, 2]),
        iter([1, 2]),
        [248044],  # configured generation padding ID is conservatively reserved
    ],
)
def test_flat_batch_one_token_admission(ids):
    with pytest.raises(ValueError, match="token|padding"):
        api().prepare_request(ids, request_id="invalid-tokens")


@pytest.mark.parametrize("identity", ["", "x" * 65, "a b", "a\n", "é", 7, None])
def test_request_identity_admission(identity):
    with pytest.raises(ValueError, match="request_id"):
        api().prepare_request([7], request_id=identity)


def test_declared_caps_and_continuation_admission():
    request = api().prepare_request([1, 2, 3], request_id="caps")
    assert api().MAX_PROMPT_TOKENS == 1024
    assert api().MAX_IMAGE_TOKENS == 256
    assert api().MAX_DECODE_TOKENS == 32
    assert json.loads(request.request_bytes)["limits"] == {
        "prompt_tokens": 1024,
        "image_tokens": 256,
        "decode_tokens": 32,
    }
    with pytest.raises(ValueError, match="cap"):
        api().prepare_request([7] * 1025, request_id="too-long")
    assert api().prepare_request([7] * 1024, request_id="at-cap").prompt_length == 1024
    with pytest.raises(ValueError, match="cap"):
        api().prepare_request(
            [START] + [IMAGE] * 272 + [END], request_id="image-cap", grid=(1, 34, 32)
        )
    assert (
        api()
        .prepare_request(
            [START] + [IMAGE] * 256 + [END], request_id="image-at-cap", grid=(1, 32, 32)
        )
        .delta
        == -240
    )
    for offset in (-1, 0, 2, True, 3.0, 35, 2**80):
        with pytest.raises(ValueError, match="offset|cap"):
            request.continuation(offset)
    session = api().Session(request)
    session.consume(request.chunk(0, 3), actual_cache_offset=0)
    for offset in range(3, 35):
        np.testing.assert_array_equal(
            session.decode(actual_cache_offset=offset),
            np.full((3, 1, 1), offset, dtype="<i4"),
        )
    with pytest.raises(ValueError, match="cap"):
        session.decode(actual_cache_offset=35)
    assert session.state == "failed"
    assert session.request is None


@pytest.mark.parametrize(
    "options",
    [
        {"attention_mask": [1]},
        {"padding": True},
        {"video_grid_thw": [1, 2, 2]},
        {"images": [b"image"]},
        {"prefix_reuse": True},
        {"mtp": True},
        {"config_sha256": "0" * 64},
        {"position_ids": [[[1]]] * 3},
        {"max_prompt_tokens": float("inf")},
    ],
)
def test_unsupported_input_options_are_rejected(options):
    with pytest.raises(TypeError):
        api().prepare_request([7], request_id="unsupported", **options)


def test_config_sha256_mutable_zero_dimensional_array_rejected_at_admission():
    request = api().prepare_request(
        [7, 248053] + [248056] * 6 + [248054, 9],
        request_id="x",
        grid=(1, 4, 6),
    )
    digest = np.array(api().CONFIG_SHA256)
    assert digest.ndim == 0
    assert digest == request.config_sha256
    forged = replace(request, config_sha256=digest)
    with pytest.raises(ValueError):
        api().Session(forged)
    digest[()] = "0" * 64
    with pytest.raises(ValueError):
        api().Session(forged)


def test_session_retains_fresh_canonical_request_without_changing_positions_or_cursor():
    request = api().prepare_request(image_ids(), request_id="canonical", grid=(1, 4, 6))
    original_bytes = request.request_bytes
    original_hash = request.manifest_hash
    session = api().Session(request)
    assert session.request is not request
    assert type(session.request) is api().Request
    assert session.request == request
    assert session.request.request_bytes == original_bytes
    assert session.request.manifest_hash == original_hash
    assert type(session.request.config_sha256) is str
    assert session.request.config_sha256 == api().CONFIG_SHA256
    np.testing.assert_array_equal(session.request.position_ids, golden())
    with pytest.raises(AttributeError):
        session.request.config_sha256 = "0" * 64
    with pytest.raises(ValueError):
        session.request.position_ids.setflags(write=True)
    assert session.state == "active"
    assert session.cursor == 0
    session.consume(request.chunk(0, 9), actual_cache_offset=0)
    assert session.cursor == 9
    session.consume(request.chunk(9, 10), actual_cache_offset=9)
    assert session.cursor == 10
    np.testing.assert_array_equal(session.decode(actual_cache_offset=10), [[[7]]] * 3)
    assert session.cursor == 11
    session.finish()
    assert session.state == "finished"
    assert session.request is None
    assert session.cursor == 0
    assert request.request_bytes == original_bytes
    assert request.manifest_hash == original_hash


@pytest.mark.parametrize(
    "kind",
    [
        "pinned-str-subclass",
        "lying-wrong-str-subclass",
        "comparison-trap-str-subclass",
        "pinned-0d-array",
        "pinned-singleton-array",
        "numpy-string",
    ],
)
def test_config_sha256_nonexact_types_rejected_before_comparison(kind):
    class DigestSubclass(str):
        pass

    class LyingDigest(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    class ComparisonTrapDigest(str):
        def __eq__(self, other):
            raise AssertionError("untrusted digest equality must not run")

        def __ne__(self, other):
            raise AssertionError("untrusted digest inequality must not run")

    request = api().prepare_request(
        image_ids(), request_id="config-types", grid=(1, 4, 6)
    )
    payloads = {
        "pinned-str-subclass": DigestSubclass(api().CONFIG_SHA256),
        "lying-wrong-str-subclass": LyingDigest("0" * 64),
        "comparison-trap-str-subclass": ComparisonTrapDigest(api().CONFIG_SHA256),
        "pinned-0d-array": np.array(api().CONFIG_SHA256),
        "pinned-singleton-array": np.array([api().CONFIG_SHA256]),
        "numpy-string": np.str_(api().CONFIG_SHA256),
    }
    with pytest.raises(ValueError, match="exact immutable types"):
        api().Session(replace(request, config_sha256=payloads[kind]))


def test_config_sha256_wrong_exact_string_rejected_by_pinned_value_check():
    request = api().prepare_request([1, 2, 3], request_id="config-wrong")
    wrong_digest = "0" * 64
    assert type(wrong_digest) is str
    with pytest.raises(ValueError, match="configuration must match pinned digest"):
        api().Session(replace(request, config_sha256=wrong_digest))


def test_mutable_forgery_and_wrong_exact_field_types_fail_closed():
    request = api().prepare_request([1, 2, 3], request_id="immutable")
    for forged in (
        replace(request, token_ids=[1, 2, 3]),
        replace(request, position_bytes=bytearray(request.position_bytes)),
        replace(request, delta=False),
    ):
        with pytest.raises(ValueError):
            api().Session(forged)
    chunk = request.chunk(0, 1)
    for forged in (
        replace(chunk, position_bytes=bytearray(chunk.position_bytes)),
        replace(chunk, start=False),
        replace(chunk, stop=True),
        replace(chunk, delta=False),
        replace(chunk, token_ids=(True,)),
        replace(chunk, token_ids=(np.int32(1),)),
    ):
        session = api().Session(request)
        with pytest.raises(ValueError):
            session.consume(forged, actual_cache_offset=0)
        assert session.state == "failed"
    ids, grid = image_ids(), [1, 4, 6]
    image = api().prepare_request(ids, request_id="input-copy", grid=grid)
    ids[0], grid[1] = 99, 6
    assert image.token_ids[0] == 7
    assert image.grid == (1, 4, 6)
    with pytest.raises(ValueError):
        image.position_ids.setflags(write=True)
    session = api().Session(image)
    session.consume(image.chunk(0, image.prompt_length), actual_cache_offset=0)
    with pytest.raises(ValueError):
        session.decode(
            actual_cache_offset=7
        )  # NEVER max spatial position+1 as KV offset
    assert session.request is None
    wrong_request = api().prepare_request([1, 2, 3], request_id="different-identity")
    session = api().Session(request)
    with pytest.raises(ValueError):
        session.consume(wrong_request.chunk(0, 3), actual_cache_offset=0)


def api():
    assert importlib.util.find_spec("multimodal_contract") is not None, (
        "missing CPU position contract"
    )
    return importlib.import_module("multimodal_contract")


def test_text_positions_and_continuation_tracer():
    request = api().prepare_request([10, 11, 12], request_id="text-1")
    np.testing.assert_array_equal(request.position_ids, [[[0, 1, 2]]] * 3)
    assert request.position_ids.dtype == np.dtype("<i4")
    assert request.delta == 0
    assert request.prompt_length == 3
    np.testing.assert_array_equal(request.continuation(3), [[[3]]] * 3)
    np.testing.assert_array_equal(request.continuation(4), [[[4]]] * 3)
