# Qualification client: observed vertical TDD evidence

Implemented with Python `/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python` (pytest 9.1.1, Ruff 0.16.10). Real temporary local HTTP servers only: no GPU, model download, SSH, external server, or model-generated answers. Run evidence dates were 2026-10-17 UTC. This document summarizes actual tool runs; it is not a fabricated full terminal transcript.

Repository root: `/Users/jeremylien/Projects/local-agent-dual-mac`.

Default loop, run after each tracer bullet:

```sh
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest tests/test_qualification.py -q --tb=short
```

For focused RED runs, replace the file argument with the named `::test_...` selector(s). Parameter cases were added only for the current boundary, not as an upfront imagined suite.

| Vertical feature | Actual RED observation | Actual GREEN observation |
|---|---|---|
| Real data-URI chat request, exact serialized JSON, output/usage receipt | `test_real_image_payload_and_exact_receipt`: 1 failed, module `qualification.client` absent | 1 passed |
| Literal-loopback endpoint boundary and exclusive output ownership | Full then-current suite: 9 failed, 1 passed; disallowed endpoints created output and an existing receipt was overwritten | 10 passed |
| Frozen manifest/image validation boundary | Full suite: 10 failed, 11 passed; wrong hash, duplicate IDs, symlink, empty/version, bad PNG CRC/signature/raster accepted; image bound options absent | 21 passed |
| Bounded HTTP error/partial-response receipt | Full suite: 9 failed, 21 passed; missing per-case failure state/bytes/wire, null content incorrectly successful | 30 passed |
| Actual health/props observations, unsupported vs unavailable, no attestation | Focused `test_health_props_observations_not_attestation`: 3 failed, absent attestation/probes | 33 passed full suite |
| Numeric budgets and total/partial success accounting | Focused `test_numeric_bounds_rejected_before_output` + `test_deadline_spans_requests_and_partial_success_is_failure`: 8 failed; invalid limits created output; aggregate count fields absent | 41 passed full suite |
| Output symlink-parent rejection and fd-pinned writes | Focused `test_symlink_escape_rejected_or_pinned`: 2 failed, 2 passed; writes escaped through symlink parent or pathname replacement | 45 passed full suite |
| Reproducible frozen red/blue/split fixtures, one-token request budget | Focused `test_frozen_synthetic_fixtures_are_reproducible`: 1 failed, generator module absent | Generator actually run into `qualification/fixtures`; 46 passed full suite |
| Checkpointed receipt and manifest/image/request bindings | Focused `test_receipt_checkpoints_bind_inputs_and_requests`: 1 failed, no receipt visible to first request handler | 47 passed full suite |
| Malformed/action response rejection | Focused `test_malformed_or_action_responses_are_not_success`: 6 failed; action calls, wrong role, negative usage, duplicate JSON key, error object and nonstring model all incorrectly successful | 53 passed full suite |
| Ambiguous HTTP framing boundary | Focused `test_invalid_http_framing_is_failure`: 2 failed, 3 passed; conflicting Content-Length and length+chunked accepted | 58 passed full suite |
| Unicode malformed-response receipt preservation | Focused Unicode + supplementary loopback chunk-success tests: 1 failed, 2 passed; final receipt lacked response/error after a lone surrogate | 61 passed full suite |
| Every HTTP exchange timestamped; serialized HTTP request hash | Extended Unicode receipt test: 1 failed, probe `started_at` absent | Final full-suite execution recorded in `qualification/verification.json` |

## Test/implementation refinements

- An early traversal example referred to a nonexistent outside file and therefore already failed for missing-file reasons. Strengthened it to a real reachable outside PNG with the correct hash; traversal still rejects before any network request.
- Initial deadline test delayed both requests at a subsecond shared budget and could lose the first request under scheduling load. It now replies immediately to the first request and delays only the second; it verifies first success, second failure, overall nonzero exit and correct totals.
- Supplementary already-supported behavior tests exercise valid chunked JSON over **both real 127.0.0.1 and ::1** listeners while proxy environment variables point elsewhere. No production behavior was added to make those checks pass.
- Fixture checks independently inspect 128x128 IHDR dimensions and decompressed first-row red/blue/split RGB bytes, in addition to hashes and byte-for-byte reproduction.
- Ruff first reported import ordering, two intentional broad fail-closed exception boundaries, one exception type preference, and explicit subprocess `check` arguments. Imports were sorted, remaining findings addressed, and Ruff subsequently returned `All checks passed!`.
- Initial `git status` discovery failed because this workspace is not a Git repository. No commit was made, and no upstream/build files were changed by this task.

## Final reproducible commands

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest tests/test_qualification.py -q -p no:cacheprovider
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m ruff check --no-cache qualification tests
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m qualification.client --help
```

`qualification/verification.json` captures actual final command outputs/exit codes plus fixture hashes. These results qualify the client CPU/HTTP path only. Physical single-host/dual-private-RPC runs, real model outputs, GPU placement and independent attestation remain with the parent operator.

## HTTP-CHUNK-FRAMING blocking repair (2026-10-17 UTC)

Scope: only the blocking chunk-framing issue; no manifest-version or finish-reason policy changes. The failed independent review and original `qualification/verification.json` are historical evidence, not rewritten to claim approval. This section records the new execution results separately.

### Root cause and constrained repair

Read the exact interpreter's `/opt/homebrew/Cellar/python@3.13/3.13.2/Frameworks/Python.framework/Versions/3.13/lib/python3.13/http/client.py` using `inspect.getsource(http.client.HTTPResponse)`. Ranked, falsifiable findings:

1. `_get_chunk_left()` calls `_safe_read(2)` and discards the result without checking CRLF: predicts valid JSON followed by `XX` will incorrectly succeed.
2. `_read_and_discard_trailer()` explicitly accepts EOF and bare LF: predicts EOF immediately after `0\r\n` will incorrectly succeed even after fixing the data separator.
3. `_read_next_chunk_size()` strips extensions and uses permissive `int(..., 16)`; trailer processing lacks an aggregate framing budget: predicts acceptance of unsupported size spellings/extensions and excessive framing for a small entity.

The first two predictions were reproduced separately with real `127.0.0.1` HTTP fixtures and the subprocess CLI, each returning exit 0, `success:true`, `complete:true` before its respective fix. Neither regression uses invalid JSON as a substitute for invalid HTTP. Each rejects with nonzero exit, unsuccessful/incomplete receipts, and the exact original entity bytes/hash after its fix.

Added a narrow `HTTPResponse` subclass through `HTTPConnection.response_class`. It overrides only chunk-framing seams, retaining stdlib status/header parsing, `read1` entity reads, the existing body cap/prefix retention, and the existing socket-shutdown watchdog/shared deadline. No whole-HTTP parser or unbounded read was introduced. Size syntax is plain hex + CRLF; extensions/nonempty trailers fail closed. Size-line limit is 1024 bytes including CRLF; total per-response chunk framing is 65536 bytes, including separators and the terminal empty-trailer CRLF, with one sentinel byte permitted to detect overflow. Trailer checking consumes at most two bytes instead of scanning trailer fields.

### Observed vertical RED → GREEN execution

Every production increment below followed its own failing wire test, not an upfront batch. Each exact selector command below was run once for RED and again for GREEN, with the same arguments. All runs used `PYTHONDONTWRITEBYTECODE=1`, `-B`, disabled pytest caches, and scratch-local fixtures.

| Increment | Actual RED | Actual focused GREEN | Full focused suite immediately afterward |
|---|---|---|---|
| Require data-chunk CRLF | 1 failed in 0.64s: old client exit 0 / successful complete JSON receipt despite `XX` | 1 passed in 0.61s | 62 passed in 26.92s |
| Require complete empty trailer section | 1 failed in 0.67s: after separator fix, inherited old terminal handling still returned exit 0 for EOF after `0\r\n` | 1 passed in 0.62s | 63 passed in 27.70s |
| Require plain hex size + CRLF; reject extensions | 6 failed in 3.49s: `+2b`, `0x2b`, leading space, bare LF, well-formed extension and unterminated quoted extension all succeeded | 6 passed in 3.53s | 69 passed in 31.31s |
| Bound individual size lines | 1 failed in 0.67s: 1025-byte size line accepted | 1 passed in 0.61s | 70 passed in 32.07s |
| Bound aggregate framing separately from entity | 1 failed in 0.65s: 100-byte valid JSON entity with excessive framing accepted | 1 passed in 0.60s | 71 passed in 32.51s |

Exact commands used for both RED and GREEN, from `/Users/jeremylien/Projects/local-agent-dual-mac`:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m pytest tests/test_qualification.py::test_chunk_data_requires_crlf_retaining_exact_entity -q --tb=short -p no:cacheprovider --basetemp /Users/jeremylien/.hermes/cache/scratch/chunk-framing-xva05035/pytest
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m pytest tests/test_qualification.py::test_last_chunk_requires_complete_empty_trailer_section -q --tb=short -p no:cacheprovider --basetemp /Users/jeremylien/.hermes/cache/scratch/chunk-framing-xva05035/pytest
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m pytest tests/test_qualification.py::test_chunk_size_requires_plain_hex_and_crlf -q --tb=short -p no:cacheprovider --basetemp /Users/jeremylien/.hermes/cache/scratch/chunk-framing-xva05035/pytest
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m pytest tests/test_qualification.py::test_chunk_size_line_has_separate_byte_bound -q --tb=short -p no:cacheprovider --basetemp /Users/jeremylien/.hermes/cache/scratch/chunk-framing-xva05035/pytest
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m pytest tests/test_qualification.py::test_chunk_framing_has_aggregate_bound_separate_from_entity -q --tb=short -p no:cacheprovider --basetemp /Users/jeremylien/.hermes/cache/scratch/chunk-framing-xva05035/pytest
```

Supplementary checks exercise behavior already implemented by those tracer bullets; no additional production changes were made for these tests:

- Exact aggregate limit 65536 succeeds; 65537 fails at the terminal CRLF and retains the full entity. Valid leading-zero sizes, a multi-zero last chunk, uppercase sizes and 1024-byte size lines are exercised.
- Bare LF/partial CR terminal delimiters, nonempty and oversized trailers, terminal extensions, and a non-CRLF data separator all fail while retaining the entity, never trailer bytes.
- Trickle chunk-size lines and a stalled partial terminal CRLF still hit the absolute watchdog, with `TimeoutError`, incomplete receipts and the expected entity prefix.
- Exact body cap succeeds; one extra entity byte and truncated chunk data fail with the exact capped/received entity bytes, counts and SHA-256.
- Existing valid chunked IPv4/IPv6/proxy-environment controls and the previous deadline, body-limit, fd-pinned output and other client regressions remain green.

Executed supplementary/final commands:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m pytest tests/test_qualification.py::test_chunk_framing_budget_includes_terminal_crlf tests/test_qualification.py::test_unsupported_chunk_trailers_and_partial_delimiters_fail tests/test_qualification.py::test_chunk_framing_reads_keep_absolute_watchdog -q --tb=short -p no:cacheprovider --basetemp /Users/jeremylien/.hermes/cache/scratch/chunk-framing-xva05035/pytest
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m pytest tests/test_qualification.py::test_chunked_entity_prefix_and_body_cap_are_unchanged -q --tb=short -p no:cacheprovider --basetemp /Users/jeremylien/.hermes/cache/scratch/chunk-framing-xva05035/pytest
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m pytest tests/test_qualification.py -q --tb=short -p no:cacheprovider --basetemp /Users/jeremylien/.hermes/cache/scratch/chunk-framing-xva05035/pytest
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -B -m ruff check --no-cache qualification tests/test_qualification.py
```

Observed results, respectively: **10 passed in 5.89s**, **3 passed in 1.80s**, **84 passed in 40.17s**, and **All checks passed!** (all exit 0). The full-suite command above is also the exact command used after each earlier GREEN.

Historical SHA-256 bindings preserved:

- `results/qualification-client-independent-review.json`: `90d735e8e50a9e64fa2759a42ac7253a7dc9320bc5d0fe79791ec3a07b77114c`.
- `qualification/verification.json`: `3d0fd3178e6115c73af75fdffb9032877da30f9364486950f28486b0a6f0dbfe`.

Limitations: this is CPU/local-fixture verification, not independent approval or GPU/model/physical-host attestation. The strict reader deliberately rejects chunk extensions/nonempty trailers and otherwise valid chunked messages over its framing budgets. It uses stdlib private chunk hooks, so interpreter upgrades require rerunning these wire regressions; only the specified Python 3.13.2 interpreter was executed here. Workspace discovery again confirmed there is no Git repository, so the change is compared against scratch snapshots, not a Git commit. No SSH, GPU work, downloads, installs, live-model calls, legacy-project tests or commits were performed.

