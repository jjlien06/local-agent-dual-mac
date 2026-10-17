# CPU single-image request / position contract

**This is a bounded NumPy contract, not a visual model runtime or numerical admission. The initial independent review failed on CPU-CONTRACT-001. Its repair is implemented and locally tested; independent re-review is pending. Not approved.** It performs no image processing, feature replacement, cache I/O, MLX import, GPU execution, RPC, SSH, serving or MTP.

## Accepted inputs and fixed bounds

```python
from multimodal_contract import Session, prepare_request

request = prepare_request(
    [7, 248053] + [248056] * 6 + [248054, 9],
    request_id="fresh-image-001",
    grid=(1, 4, 6),
)
session = Session(request)
session.consume(request.chunk(0, 9), actual_cache_offset=0)
session.consume(request.chunk(9, 10), actual_cache_offset=9)  # last prompt token
positions = session.decode(actual_cache_offset=10)  # three axes all equal 7
session.finish()
```

- Implicit **B=1**, nonempty **flat exact Python `list`/`tuple` of exact `int`** token IDs in `[0,248320)`. Nested `[1,T]` arrays, NumPy arrays/scalars, bools, floats, strings, generators, masks and unknown keyword arguments are rejected rather than coerced.
- Prompt length <= **1,024** tokens, merged image placeholders <= **256**, continuation calls <= **32** tokens, chunk length **1..256**. These are predeclared CPU-contract caps, not measured physical memory/context admission. Overrides are unsupported. Image tokens count toward prompt length.
- `request_id` must fully match `[A-Za-z0-9._:-]{1,64}`. The caller must give each fresh physical request a fresh identity; there is no global identity registry, authentication or prefix-cache reuse.
- `grid=None` admits ordinary unpadded text **without any vision markers**. Otherwise accept exactly one flat grid `[1,H,W]` of positive exact integers, H/W divisible by **2**. These are raw patch-grid dimensions, not pixels; merged grid is `[1,H/2,W/2]`.
- Exactly one start **248053**, exactly one end **248054**, and one contiguous immediately enclosed image-placeholder **248056** run of length `(H/2)*(W/2)`. All duplicates, missing boundaries, text inside the block, surplus/missing placeholders or grids are rejected. Video **248057**, temporal grids, multiple images, padding, prefix reuse, user-provided positions/config overrides and MTP are unsupported.
- Conservative padding policy: token **248044** is rejected in prompts because it is the configured generation padding ID (also a BOS/EOS alias). This deliberately rejects even a legitimate use of that alias; it does **not** infer padding from token counts or silently strip it. Token **248046** is not reclassified as padding. Relaxing this policy or supporting token masks is a separate contract change.

## Positions versus physical cache length

Positions are `[3,1,T]`, axis order **temporal, height, width**. Text is scalar `arange` replicated on three axes. Image coordinates traverse merged height rows then width columns. The image origin is the physical index of its first placeholder; temporal coordinates are constant for a still image. Text after the image resumes at **maximum image coordinate + 1**, including the vision-end marker as text.

Independent explicit golden for IDs `[7,start,image*6,end,9]`, grid `[1,4,6]`:

```text
t: 0 1 2 2 2 2 2 2 5 6
h: 0 1 2 2 2 3 3 3 5 6
w: 0 1 2 3 4 2 3 4 5 6
```

A second `[1,6,4]` golden enumerates merged rows `0,0,1,1,2,2` and columns `0,1,0,1,0,1` independently. Both have delta **-3**. Delta is the Python integer `max(position_ids)+1-valid_token_length`; no padding is admitted, so valid length equals prompt length.

**Actual prompt cache length is 10, not 7.** First two continuation input-token offsets are physical **10 and 11**; positions are **7 and 8**, on all axes. `Request.continuation(offset)` is a stateless bounded position lookup; only `Session` checks progression. Never use maximum spatial position as a KV offset.

`Request.chunk(start,stop)` is a half-open **physical-token** slice. It includes IDs, three-axis positions, delta, range and request hash. Frozen records contain tuple IDs and immutable `bytes`; returned arrays are read-only views backed by those bytes and cannot be made writable. A sliced chunk may contain only part of the image run: admission applies to the whole request, never independently to a chunk. Every final prompt token must be consumed before decode/finish.

## Exact bytes and hashing (version 1)

- Positions use **little-endian signed 32-bit integers**, NumPy dtype string **`<i4`**, C-order contiguous serialization. Shape is explicitly `[3,1,T]`. Chunk positions use the same format `[3,1,t]`. Delta is a signed JSON integer, not an array payload.
- Token IDs use **JSON integer literals**, not a binary token dtype. Grid is a JSON three-integer list or `null`; identity is exact ASCII with no normalization. The supplied Python list/grid are copied to tuples, not retained as mutable inputs.
- `position_sha256 = SHA256(position_bytes).hexdigest()` (lowercase hex).
- `request_bytes` is exactly `json.dumps(manifest, sort_keys=True, separators=(",",":"), ensure_ascii=True, allow_nan=False).encode("ascii")`, **no BOM, newline, trailing whitespace or float values**.
- Manifest keys: `version`, `request_id`, `token_ids`, `grid_thw`, `delta`, `config_sha256`, `position_dtype`, `position_shape`, `position_sha256`, `limits`. Limits contain `prompt_tokens`, `image_tokens`, `decode_tokens`. The full exact local target config digest binds its marker/merge/mRoPE semantics; accepting different configs is unsupported, not an optional caller assertion.
- `manifest_hash = SHA256(b"qwen38-cpu-request-v1\0" + request_bytes).hexdigest()`. The domain separator ends in one NUL byte. Changing identity, IDs, grid, config digest, positions or delta changes the binding. This hash is integrity binding, **not authentication or image-byte/weight attestation**.
- A chunk carries the full request manifest hash; there is no invented wire format or GPU protocol. `Session.consume` compares its **exact fields** with the authoritative slice (including exact immutable field types), not merely a claimed hash.

For the above IDs/grid with `request_id="manifest"`, the observed manifest hash is `6f07cc8f26fca445ca42408ce4eb0eb2130ed3e4d18ce4e6edba55b74f3f3a44`. Exact request ASCII and position hex are in `results/mlx-vision-contract-cpu.json`.

## One-shot lifecycle

`Request` and `Chunk` are data containers, not validated wire parsers. `Session(request)` is the authoritative admission boundary: it requires an exact Python `str` configuration digest with the exact pinned value before canonical comparison, rejects array/scalar/string-subclass forgeries, independently reconstructs and validates the canonical request, and retains only that fresh immutable canonical object, never the supplied request object. Its physical cursor begins at **0**. `consume` requires the current observed physical cache offset, exactly contiguous next slice and matching binding; duplicates, skips, out-of-order, stale identities, wrong tokens, poisoned axes/delta and missing axes fail closed. Invalid calls on an admitted session drop its request reference, clear cursor to 0, set `state="failed"` and raise `ValueError`; invalid construction raises `ValueError` without admitting a request.

`decode(actual_cache_offset=...)` checks full prompt coverage and the physical offset, returns one `[3,1,1]` position array, and advances the **CPU tracer** cursor by one. This is not a commit/ACK for actual model execution, and it does not ingest generated token IDs. A future adapter must observe real caches and coordinate execution/commit failures itself.

`finish()` requires full prompt consumption; `cancel()` and `reset()` can terminate early. All release session metadata and clear its tracer cursor; all are **terminal**. `reset()` invalidates this session rather than reopening it. Create a new request and new `Session` for the next fresh-cache request. Calling any operation on a terminal session fails closed. The module does not clear, restore, inspect or certify physical KV/GDN state. External references to immutable requests/chunks remain valid data, not authorization to resume a terminal session.

## Reference evidence and limits

The required architecture review was read and hash-checked:
`results/full-target-load-v3/mlx-vision-architecture-review.md`, SHA-256 `5cf36d9896a6a838a450de36e980aad0ad9055f8e24df0bf8fc2c96ba70b1496`.

Exact local target config:
`/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/config.json`, SHA-256 `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85`.

Primary source: MLX-VLM commit **`b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0`**, `mlx_vlm/models/qwen3_5/language.py`, SHA-256 **`6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7`**; `get_rope_index`, lines 1459–1642; exact extracted function source SHA-256 **`7941460bac85b33c0f59abda8dbf6f47c33f92fd829a3428688be4b6a32e8a51`**. Retrieved only this small pinned source over HTTPS, verified its full source digest before extracting/executing the function AST. The complete executed reference command is retained in the JSON artifact.

**Reference limitation:** executed the original function body with `mx` mapped to NumPy and annotations removed, not the MLX implementation or a loaded VLM. Ten exact-array/value comparisons passed: both non-square grids with prefixes 0/1/252, and text lengths 3/255/256/257. Upstream text returns `[1,T]`; comparison explicitly lifts it to the contract's three axes. MLX array/lazy execution, GPU dtype arithmetic, rotary frequencies/interleaving, encoder/scatter, logits, model weights, every-layer KV/GDN caches and distributed transport were **not exercised**. The focused pytest tests are offline; the pinned-reference comparison is a separately recorded execution, not silently replaced with a reimplementation in tests.

Existing scalar-offset language RoPE is still incorrect for images. A future physical integration must carry exact position slices through **every owned full-attention layer on both ranks**, preserve actual token cache offsets and native GDN arithmetic, and retain hidden transport `[1,t,5120]`. This CPU contract does not change any of those components. No vision/physical arithmetic equivalence, memory capacity, throughput, server readiness or approval is established.

## Verification

From `/Users/jeremylien/Projects/local-agent-dual-mac`:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m ruff check --no-cache multimodal_contract tests/test_multimodal_contract.py
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m ruff format --check --no-cache multimodal_contract tests/test_multimodal_contract.py
```

Historical implementation evidence: **63 focused tests passed**, Ruff check and format passed, runtime-import guard passed, in-memory axis-swap and dropped-delta mutants were caught. Missing-marker/duplicate/video/padding/cap/cursor/lifecycle tests reject unsupported inputs. Those historical receipts are unchanged and do not override the independent FAIL.

CPU-CONTRACT-001 repair evidence: **72 focused tests passed**, including **9 repair regression cases**. Two new vertical RED→GREEN cycles demonstrated acceptance of the mutable zero-dimensional digest and retention of the caller's object before their respective minimal fixes. Added post-GREEN coverage rejects string subclasses (including forged equality and comparison traps), NumPy digest arrays/scalars, and an exact wrong string; canonical immutable requests retain their manifest, positions, and physical cursor/decode behavior. See the appended repair section in `MULTIMODAL-TDD.md` and the new `results/mlx-vision-contract-fix-verification.json` for exact commands, exits, outputs and before/after hashes. Original qualification tests were not edited or rerun by this repair. Fresh independent re-review remains a separate gate; no physical-cache, native mRoPE, encoder, autonomy or 24x7 approval is implied.
