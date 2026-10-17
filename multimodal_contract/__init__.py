"""Pure CPU multimodal positions; not a model or GPU adapter."""

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

CONFIG_SHA256 = "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85"
MAX_PROMPT_TOKENS = 1024
MAX_IMAGE_TOKENS = 256
MAX_DECODE_TOKENS = 32


def _array(data, shape):
    return np.frombuffer(data, dtype="<i4").reshape(shape)


@dataclass(frozen=True)
class Chunk:
    start: int
    stop: int
    token_ids: tuple
    position_bytes: bytes
    delta: int
    request_hash: str

    @property
    def position_ids(self):
        return _array(self.position_bytes, (3, 1, self.stop - self.start))


@dataclass(frozen=True)
class Request:
    request_id: str
    token_ids: tuple
    position_bytes: bytes
    delta: int
    grid: tuple | None
    config_sha256: str = CONFIG_SHA256

    @property
    def request_bytes(self):
        manifest = {
            "version": 1,
            "request_id": self.request_id,
            "token_ids": self.token_ids,
            "grid_thw": self.grid,
            "delta": self.delta,
            "config_sha256": self.config_sha256,
            "position_dtype": "<i4",
            "position_shape": [3, 1, self.prompt_length],
            "position_sha256": hashlib.sha256(self.position_bytes).hexdigest(),
            "limits": {
                "prompt_tokens": MAX_PROMPT_TOKENS,
                "image_tokens": MAX_IMAGE_TOKENS,
                "decode_tokens": MAX_DECODE_TOKENS,
            },
        }
        return json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    @property
    def manifest_hash(self):
        return hashlib.sha256(
            b"qwen38-cpu-request-v1\0" + self.request_bytes
        ).hexdigest()

    @property
    def prompt_length(self):
        return len(self.token_ids)

    @property
    def position_ids(self):
        return _array(self.position_bytes, (3, 1, self.prompt_length))

    def chunk(self, start, stop):
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= self.prompt_length
            or stop - start > 256
        ):
            raise ValueError("invalid physical token slice (maximum 256)")
        return Chunk(
            start,
            stop,
            self.token_ids[start:stop],
            self.position_ids[:, :, start:stop].tobytes(),
            self.delta,
            self.manifest_hash,
        )

    def continuation(self, actual_cache_offset):
        if (
            type(actual_cache_offset) is not int
            or not self.prompt_length
            <= actual_cache_offset
            < self.prompt_length + MAX_DECODE_TOKENS
        ):
            raise ValueError("continuation offset outside prompt/decode cap")
        return _array(
            np.full((3, 1, 1), actual_cache_offset + self.delta, dtype="<i4").tobytes(),
            (3, 1, 1),
        )


class Session:
    """One-shot CPU cursor tracer; never owns or reads real model caches."""

    def __init__(self, request):
        if (
            type(request) is not Request
            or type(request.token_ids) is not tuple
            or type(request.position_bytes) is not bytes
            or type(request.delta) is not int
            or type(request.config_sha256) is not str
            or (request.grid is not None and type(request.grid) is not tuple)
        ):
            raise ValueError("request fields must have exact immutable types")
        if request.config_sha256 != CONFIG_SHA256:
            raise ValueError("request configuration must match pinned digest")
        canonical = prepare_request(
            request.token_ids, request_id=request.request_id, grid=request.grid
        )
        if request != canonical:
            raise ValueError("request does not match canonical configuration/positions")
        self.request = canonical
        self.cursor = 0
        self.state = "active"

    def _close(self, state):
        self.request = None
        self.cursor = 0
        self.state = state

    def _fail(self, message):
        self._close("failed")
        raise ValueError(message)

    def _active(self):
        if self.state != "active":
            self._fail("request is terminal; create a fresh Session")

    def _offset(self, actual_cache_offset):
        self._active()
        if type(actual_cache_offset) is not int or actual_cache_offset != self.cursor:
            self._fail("actual cache offset does not match physical token cursor")

    def consume(self, chunk, *, actual_cache_offset):
        self._offset(actual_cache_offset)
        if (
            type(chunk) is not Chunk
            or type(chunk.start) is not int
            or type(chunk.stop) is not int
            or type(chunk.delta) is not int
            or type(chunk.position_bytes) is not bytes
            or type(chunk.token_ids) is not tuple
            or any(type(token) is not int for token in chunk.token_ids)
            or type(chunk.request_hash) is not str
        ):
            self._fail("invalid chunk")
        try:
            expected = self.request.chunk(self.cursor, chunk.stop)
        except ValueError:
            self._fail("invalid or out-of-order chunk range")
        if chunk != expected:
            self._fail("chunk identity/tokens/positions/delta mismatch")
        self.cursor = chunk.stop

    def decode(self, *, actual_cache_offset):
        self._offset(actual_cache_offset)
        if self.cursor < self.request.prompt_length:
            self._fail("prompt incomplete")
        try:
            positions = self.request.continuation(self.cursor)
        except ValueError:
            self._fail("decode token cap exceeded")
        self.cursor += 1
        return positions

    def finish(self):
        self._active()
        if self.cursor < self.request.prompt_length:
            self._fail("prompt incomplete")
        self._close("finished")

    def cancel(self):
        self._active()
        self._close("cancelled")

    def reset(self):
        self._active()
        self._close("reset")


def _image_grid(ids, grid):
    if 248057 in ids:
        raise ValueError("video is unsupported")
    if grid is None:
        if any(marker in ids for marker in (248053, 248054, 248056)):
            raise ValueError("image markers require one grid")
        return None
    if (
        type(grid) not in (list, tuple)
        or len(grid) != 3
        or any(type(value) is not int or value <= 0 for value in grid)
        or grid[0] != 1
        or grid[1] % 2
        or grid[2] % 2
    ):
        raise ValueError("grid must be one flat [1,H,W], H/W positive multiples of 2")
    count = (grid[1] // 2) * (grid[2] // 2)
    if count > MAX_IMAGE_TOKENS:
        raise ValueError("image token cap exceeded")
    if ids.count(248053) != 1 or ids.count(248054) != 1 or ids.count(248056) != count:
        raise ValueError("image marker/count mismatch")
    start = ids.index(248053)
    stop = ids.index(248054)
    if stop != start + count + 1 or ids[start + 1 : stop] != (248056,) * count:
        raise ValueError("image placeholders must form one contiguous bounded block")
    return tuple(grid)


def prepare_request(token_ids, *, request_id, grid=None):
    if (
        type(request_id) is not str
        or re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", request_id) is None
    ):
        raise ValueError("request_id must be 1..64 ASCII identity characters")
    if type(token_ids) not in (list, tuple) or not token_ids:
        raise ValueError("token IDs must be a nonempty flat list/tuple, implicit B=1")
    if len(token_ids) > MAX_PROMPT_TOKENS:
        raise ValueError("prompt token cap exceeded")
    if any(type(token) is not int or not 0 <= token < 248320 for token in token_ids):
        raise ValueError(
            "token IDs must be exact Python integers inside target vocabulary"
        )
    if 248044 in token_ids:
        raise ValueError("padding ID is unsupported in this bounded prompt contract")
    ids = tuple(token_ids)
    grid = _image_grid(ids, grid)
    positions = np.tile(np.arange(len(ids), dtype="<i4"), (3, 1, 1))
    if grid is not None:
        _, height, width = grid
        height, width = height // 2, width // 2
        start = ids.index(248056)
        count = height * width
        positions[:, 0, start : start + count] = np.array(
            [
                [start] * count,
                [start + row for row in range(height) for _ in range(width)],
                [start + col for _ in range(height) for col in range(width)],
            ],
            dtype="<i4",
        )
        resume = start + max(height, width)
        positions[:, 0, start + count :] = np.arange(
            resume, resume + len(ids) - start - count, dtype="<i4"
        )
    delta = int(positions.max()) + 1 - len(ids)
    return Request(
        request_id,
        ids,
        positions.tobytes(),
        delta,
        tuple(grid) if grid is not None else None,
    )
