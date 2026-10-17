# Bounded multimodal HTTP qualification client

Python 3.10+ standard library only; no model download, GPU, SSH, shell tools, or action execution. Run from the repository root. This client speaks a small OpenAI-compatible chat subset to an **already-running loopback** server. It does not start or configure that server.

## Run the same frozen inputs against each physical configuration

```sh
python3 -m qualification.client --help

# Server started independently in single-host configuration:
python3 -m qualification.client \
  --endpoint http://127.0.0.1:8080 \
  --manifest qualification/fixtures/public-cases.json \
  --model YOUR_SERVER_MODEL_ALIAS \
  --output qualification/single-host-run-001 \
  --timeout 60 --deadline 240 --max-response-bytes 1048576 \
  --max-tokens 32

# Restart/reconfigure server independently for two-host private RPC, then:
python3 -m qualification.client \
  --endpoint http://127.0.0.1:8080 \
  --manifest qualification/fixtures/public-cases.json \
  --model YOUR_SERVER_MODEL_ALIAS \
  --output qualification/two-host-run-001 \
  --timeout 60 --deadline 240 --max-response-bytes 1048576 \
  --max-tokens 32
```

Use the **same model weights, vision projector, alias, server revision/template and sampling configuration** on both runs. Compare `manifest_sha256`, `settings`, each `request_sha256`, extracted `output`, `usage`, `finish_reason`, and response bodies. Endpoint and output directory are not embedded in the JSON request body, so identical configuration produces byte-identical chat request bodies. A different HTTP Host header can change `request_wire_sha256` without changing model inputs.

For a separate one-token-budget control, run each configuration into another new output directory with `--max-tokens 1`. This is **not a streaming TTFT measurement**, and a single token need not be a word. A reasoning-only/empty-content response fails the text-output check; configure model thinking/template settings independently and identically if needed. This client does not silently change a template or disable thinking.

The initial small-model architecture control is **Qwen3.5-0.8B VLM**, not a substitute for the target Qwen3.8-27B. No physical model/GPU qualification is asserted by these CPU tests.

## Frozen manifest

`qualification/fixtures/public-cases.json` contains three original public synthetic 128x128 RGB PNGs: solid red, solid blue, red-left/blue-right split. Prompts and exact lowercase SHA-256 hashes are frozen in the existing file. `reference_output` is for human/independent semantic checking; **exit 0 means transport + validated text-output success, not reference-answer correctness**.

Minimal format:

```json
{"version":1,"cases":[{"id":"red","prompt":"Name the color.","image":"red.png","sha256":"EXACT_64_LOWERCASE_HEX_CHARACTERS"}]}
```

The placeholder above is intentionally not runnable. Use the real fixture manifest. Never regenerate a manifest to make a mismatched input pass. Optional reproduction, to a new directory only:

```sh
python3 -m qualification.make_fixtures --output qualification/reproduced-fixtures
```

The generator uses a deterministic stored DEFLATE block, so it does not depend on zlib compression-version choices. Client validation never invokes it.

## Safety and bounds

- Only `http://127.0.0.1[:port]` or `http://[::1][:port]`, optionally a trailing slash. Rejects DNS names, other IPs, credentials, paths, query/fragment values and alternate schemes. Uses direct `http.client` sockets; ignores proxy environment settings. Redirects are retained as failed responses, never followed. No retries.
- Validates all cases before any network request. Manifest <=1 MiB, 1..64 cases; IDs unique and restricted to 1..64 ASCII letters/digits/underscore/hyphen; prompts 1..16384 characters. Image path must be relative to the manifest directory, without `..` or symlinks. Exact hash verification is mandatory. JSON duplicate keys, nonfinite numbers, invalid UTF-8/lone surrogates fail.
- PNG validation checks signature, chunk bounds/order/CRC, IHDR dimensions, pixel limit, bounded DEFLATE expansion, exact raster length and row filter bytes. Deliberately accepts only **8-bit noninterlaced RGB/RGBA, IHDR + consecutive IDAT + IEND**, with no extra chunks or trailing bytes. Other valid PNG formats are rejected rather than partially decoded. No Pillow dependency.
- Defaults: 4 MiB/image, 1,048,576 pixels/image, 1 MiB HTTP entity body, 32 output tokens, 30s/exchange, 300s shared run budget. Hard caps: image/body/pixels each 16,777,216; tokens 32,768; timeout 3600s; deadline 86400s. All limits must be finite and positive. Case count and image bounds also bound aggregate memory/disk consumption; these caps are not a promise of small peak memory at maximum settings.
- `--timeout` is an absolute HTTP exchange budget, not an inactivity timeout: a socket shutdown watchdog interrupts trickled headers/bodies. The shared monotonic deadline starts after argument parsing and is consumed by validation, probes and chat requests. Network work receives only its remaining budget; finite local validation/serialization/filesystem cleanup is not forcibly killed at the deadline.
- `--max-response-bytes` bounds **entity-body bytes retained**, including chunked bodies; oversize bodies retain only a bounded prefix and fail with `complete:false`. HTTP headers have the standard library's separate fixed line/count bounds, not this body limit. No content decoding; encoded bodies, conflicting Content-Length, or ambiguous transfer framing fail. Chunked responses use a narrow `HTTPResponse` subclass: plain hexadecimal sizes (either case, leading zeros allowed), strict CRLF on size lines and after every data chunk, and a zero-size chunk followed by the complete empty trailer-section CRLF. Chunk extensions and nonempty trailers are unsupported and rejected, not ignored or guessed. Size lines are capped at 1024 bytes including CRLF; aggregate chunk framing (size lines, data separators and terminal CRLF) is capped at 65536 bytes per response, separately from the entity cap, with at most one overflow sentinel byte read. Trailer validation reads only two bytes and fails unless they are CRLF; it never scans an unbounded trailer section. All reads remain under the same absolute watchdog/shared deadline; standard-library status/header parsing and bounded entity reads are retained.
- Output must not exist. Existing files/directories/symlinks are never overwritten. Parent path must already exist and contain no symlinks. Inputs are opened component-by-component with `O_NOFOLLOW`; artifacts are fd-relative to the exclusively created directory, so replacing its pathname with a symlink cannot redirect writes. This targets POSIX/macOS/Linux, not Windows. No protection is claimed against an attacker with root or arbitrary same-process memory access.

## Receipts and failure semantics

`receipt.json` is atomically checkpointed at initialization, after probes, before each chat request, and after each successful case; final receipt includes any caught error. Failure stops further cases and exits 1; argparse errors exit 2. One failure can never yield overall success. Earlier successful cases remain visibly distinct from failed/not-attempted cases. A process kill may leave an earlier **unsuccessful** checkpoint, not a completed run; durability across power loss is not asserted.

Each run retains:

- Exact original `manifest.json`, its hash, settings, requested model alias, timestamps, elapsed times, total/completed counts.
- `health.*` / `props.*`: actual GET request serialization, bounded response bodies, headers/status/errors and parsed observations. Complete 404/405/501 means unsupported and is recorded, not invented; other HTTP/transport/JSON failures stop the run.
- `NNNN.request.json`: exact bytes handed to the chat request; real `data:image/png;base64,...` content, explicit `model`, `temperature:0`, `max_tokens`, `stream:false`.
- `NNNN.request.http`: exact serialized bytes handed to the socket send path (headers + body), with SHA-256. On send failure this records attempted bytes; it cannot prove the peer received every byte.
- `NNNN.response.bin`: full entity-body bytes on success, or bounded partial bytes on failure. HTTP status/headers, body hash/count/completeness and errors live in the receipt. Transfer framing is decoded by the HTTP library; this is not a raw packet capture.
- Per-case output string, observed model field (null if absent), usage (null if absent), finish reason, input image hash/dimensions and request hash. Requires exactly one choice with nonempty text; rejects action-call responses and malformed usage/identity types. Model metadata is observed, not cryptographically verified.

`attestation:false` is unconditional. Neither HTTP output nor matching hashes proves GPU execution, allocation, RPC use, both hosts' participation, model file identity, or performance equivalence. Keep independent server/RPC/device-allocation logs and model/projector file hashes. The client never interprets output as a command or executes requested actions.

## CPU verification

```sh
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest tests/test_qualification.py -q
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m ruff check qualification tests
```

Tests spawn real temporary IPv4/IPv6 HTTP servers, not a fake network layer or model. See root `TDD.md` for observed RED/GREEN progression and `qualification/verification.json` for final executed command results.
