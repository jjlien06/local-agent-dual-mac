# CPU multimodal vertical TDD evidence

**Not approved. Independent review follows. CPU-only tests cannot approve physical arithmetic equivalence.**

Workdir: `/Users/jeremylien/Projects/local-agent-dual-mac`. No model runtime imports, weights, installs, SSH, deployment, live application operations or commits. All Python/pytest/Ruff execution used the existing baseline venv; `PYTHONDONTWRITEBYTECODE=1`, pytest cache disabled, Ruff cache disabled.

Each tracer was completed RED → minimal GREEN before the next tracer. The initial single test was the text-only positions plus two continuation steps. Tests grew only after each preceding tracer was green. Stage 7 extended the unreleased manifest fixture with the declared limits it now binds. A second explicit tall-grid golden was added after implementation as regression coverage (not mislabeled a new production RED/GREEN cycle).

## Tracer 1

Missing CPU contract assertion; no production module existed.

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py::test_text_positions_and_continuation_tracer -q
```

Observed: `1 failed in 0.13s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
```

Observed: `1 passed in 0.04s`.

## Tracer 2

Missing image/grid interface (unexpected grid keyword); text tracer remained green.

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py::test_non_square_image_exact_coordinates_and_delta -q
```

Observed: `1 failed in 0.05s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
```

Observed: `2 passed in 0.04s`.

## Tracer 3

Missing bounded immutable chunk API.

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py::test_bounded_immutable_chunks_cover_boundaries_and_last_token -q
```

Observed: `1 failed in 0.05s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
```

Observed: `3 passed in 0.04s`.

## Tracer 4

Missing versioned request-byte/manifest binding API.

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py::test_versioned_manifest_binds_exact_request_and_positions -q
```

Observed: `1 failed in 0.06s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
```

Observed: `4 passed in 0.05s`.

## Tracer 5

Missing request-scoped lifecycle tracer.

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py::test_lifecycle_physical_cursor_rejects_replay_poison_and_resets -q
```

Observed: `1 failed in 0.06s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
```

Observed: `5 passed in 0.05s`.

## Tracer 6

Missing image/grid/marker admission (23 failures, including unsupported values reaching NumPy instead of admission).

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py::test_image_admission_rejects_malformed_or_unsupported_blocks -q
```

Observed: `23 failed in 0.14s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
```

Observed: `28 passed in 0.06s`.

## Tracer 7

Missing strict flat-ID/identity/cap admission (24 failures; 9 pre-existing unknown-keyword rejections already passed).

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -k "flat_batch or identity_admission or declared_caps or unsupported_input" -q --tb=short
```

Observed: `24 failed, 9 passed, 28 deselected in 0.15s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q --tb=short
```

Observed: `61 passed in 0.07s`.

## Tracer 8

Mutable bytearray/equal-value type forgeries were accepted.

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py::test_mutable_forgery_and_wrong_exact_field_types_fail_closed -q --tb=short
```

Observed: `1 failed in 0.07s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q --tb=short
```

Observed: `62 passed in 0.07s`.

## Tracer 9

Bool/NumPy scalar IDs inside a forged chunk tuple compared equal to integer IDs.

### RED — observed exit 1

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py::test_mutable_forgery_and_wrong_exact_field_types_fail_closed -q --tb=short
```

Observed: `1 failed in 0.06s`.

### GREEN — observed exit 0

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
```

Observed: `62 passed in 0.07s`.

## Final observed verification

- Final import-guarded focused pytest: **63 passed in 0.07s**, no `mlx`, `mlx_lm` or `mlx_vlm` modules loaded.
- Ruff check: **All checks passed!**; format check: **2 files already formatted**. Initial import-order/format failures were fixed only in the two new Python files and tests rerun.
- **10** exact-array/value comparisons against the pinned upstream function body executed with NumPy substituted for `mx`; annotations removed only, original function body retained. This is **not native MLX execution**.
- In-memory function mutations swapping axes or dropping delta were caught by the independent explicit non-square golden. No on-disk production mutations were used. Missing-marker and duplicate/cap rejection tests run in the focused suite.
- Source/config/report hashes and exact full command/output records (including the extracted reference body) are preserved in `results/mlx-vision-contract-cpu.json`.

### FOCUSED-GOLDEN-EXTENSION

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m pytest -p no:cacheprovider tests/test_multimodal_contract.py -q
```

```text
...............................................................          [100%]
63 passed in 0.07s
```

### FINAL-RUFF-CHECK-2

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m ruff check --no-cache multimodal_contract tests/test_multimodal_contract.py
```

```text
All checks passed!
```

### FINAL-RUFF-FORMAT-2

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m ruff format --check --no-cache multimodal_contract tests/test_multimodal_contract.py
```

```text
2 files already formatted
```

## Scope, limitations and non-blocking execution issues

- Only the five authorized new task files were written by this task. Original qualification sources/tests, original existing result artifacts, upstream tree and baseline source/metadata are checked against initial snapshots in the JSON artifact. No original-file edits were made by this subagent.
- **Concurrent filesystem drift was observed:** the workspace root `README.md` changed and additional architecture/admission result files appeared while this task ran. They are outside this task's writes, were not restored or modified, and are enumerated in the artifact. Therefore global filesystem immutability is **not** claimed. The parent/hardware owner must reconcile concurrent changes.
- Original qualification tests were not rerun; no live/runtime qualification is implied by focused passing tests.
- The persistent tool kernel lacked NumPy on the first reference probe (`ModuleNotFoundError`); the probe was rerun successfully using the required existing venv Python, with no installs. An ambiguous patch attempt was rejected atomically before any write, then reapplied with unambiguous context. The workspace is not a git repository; no git diff/commit claim is made.
- One still image only; no pixels, processor, encoder, features, full-attention rotary adapter, physical cache ownership, dual-rank transport, per-layer caches/logits, memory admission, throughput or server. Existing scalar-offset FA remains unsuitable for image positions.
- Padding policy conservatively reserves generation pad/BOS/EOS alias 248044 in prompts; fresh request IDs are a caller obligation. Session reset is terminal; it does not authorize prefix reuse or reset actual caches.
- Approval remains **pending a separate independent review** and any later physical gates owned solely by the parent.

## CPU-CONTRACT-001 repair — new evidence, independent re-review pending

**Not approved.** The original independent review returned FAIL for this finding. This appended section records a later local repair, not a replacement verdict or a rewrite of any historical evidence. The original implementation, parent and FAIL-review JSON receipts remain immutable. Fresh independent re-review follows.

### Root cause and narrow repair

The authoritative `Session` checked several exact immutable types but omitted `config_sha256`. Dataclass equality accepted a mutable zero-dimensional NumPy string array equal to the pinned digest, and the session retained the caller's object. Subsequent mutation could reach JSON serialization rather than admission rejection. The first new regression uses the exact IDs, identity `x`, grid `(1,4,6)` and `np.array(CONFIG_SHA256)` from the review, requiring rejection at admission both before and after payload mutation.

Only `Session.__init__` changed in production: require exact `str` before comparison, require the exact pinned digest, reconstruct and compare a canonical `Request`, then retain only that fresh canonical object. No consume/decode/lifecycle implementation, position arithmetic, limits or GPU integration changed.

### Execution environment and evidence capture

All commands below ran from `/Users/jeremylien/Projects/local-agent-dual-mac` with this explicit environment, the existing venv Python, and no installs:

```text
PYTHONDONTWRITEBYTECODE=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHONPATH=/Users/jeremylien/Projects/local-agent-dual-mac
TMPDIR=/Users/jeremylien/.hermes/cache/scratch
```

The scratch `cpu_contract_001_guarded_pytest.py` runs pytest with `-p no:cacheprovider`, rejects imports of `mlx`, `mlx_lm`, `mlx_vlm`, `torch` and `transformers`, and blocks socket connection calls. Its source and SHA-256, recorder source/hash, full actual commands, stdout, stderr, exits and per-stage owned-file hashes are embedded in the new `results/mlx-vision-contract-fix-verification.json`. The recorder executes each command with the above environment and records the actual child return code; it does not synthesize test results.

### Repair tracer 1 — mutable digest admission, RED then minimal GREEN

Only the exact ndarray regression was added before this RED run; the production module still had SHA-256 `dc33ebc3fed629d28fd9b230c03019edec78836f7ee2aeaf7bb37e01bd9fa2ca`.

```sh
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python /Users/jeremylien/.hermes/cache/scratch/cpu_contract_001_guarded_pytest.py tests/test_multimodal_contract.py::test_config_sha256_mutable_zero_dimensional_array_rejected_at_admission -q --tb=short
```

Observed **exit 1**: `Failed: DID NOT RAISE ValueError`; **1 failed in 0.06s**. This proves the forged mutable payload was accepted by the original `Session`, not merely that a later consume failed.

Only the exact-type and pinned-value guards were then added. The original caller-object retention was deliberately left for the next tracer.

```sh
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python /Users/jeremylien/.hermes/cache/scratch/cpu_contract_001_guarded_pytest.py tests/test_multimodal_contract.py -q --tb=short
```

Observed **exit 0**, **64 passed in 0.07s**; offline socket guard enabled, model runtime imports NONE.

### Repair tracer 2 — canonical retention, RED then minimal GREEN

Only after tracer 1 was GREEN, a new regression required `session.request is not request`, canonical equality and exact bytes/hash, immutable canonical data, exact independent golden positions, contiguous cursor `0→9→10`, decode position 7 at physical offset 10, then terminal cleanup. The caller's immutable bytes/hash must remain unchanged.

```sh
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python /Users/jeremylien/.hermes/cache/scratch/cpu_contract_001_guarded_pytest.py tests/test_multimodal_contract.py::test_session_retains_fresh_canonical_request_without_changing_positions_or_cursor -q --tb=short
```

Observed **exit 1**, assertion `session.request is not request` failed; **1 failed in 0.07s**. The type/pin guards remained in place and no canonical-retention production fix existed yet.

The already reconstructed canonical request was then named and retained, rather than retaining the supplied instance; comparison remained mandatory.

```sh
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python /Users/jeremylien/.hermes/cache/scratch/cpu_contract_001_guarded_pytest.py tests/test_multimodal_contract.py -q --tb=short
```

Observed **exit 0**, **65 passed in 0.07s**; offline socket guard enabled, model runtime imports NONE.

### Post-GREEN coverage and final focused checks

Additional coverage was added after both production tracer fixes: pinned string subclass, wrong-valued string subclass with forged equality, string subclass whose comparison raises, pinned zero-dimensional and singleton arrays, NumPy string scalar, and exact wrong Python string. The subclass comparison trap proves exact-type rejection happens before overloaded comparisons. The exact wrong string test identifies the explicit pinned-value rejection. These seven cases are regression/characterization coverage, **not additional claimed RED→GREEN production cycles**. The pre-existing tests continue to cover valid text requests, positions/chunks, cursor poison, caps and terminal behavior.

```sh
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python /Users/jeremylien/.hermes/cache/scratch/cpu_contract_001_guarded_pytest.py tests/test_multimodal_contract.py -k 'config_sha256 or retains_fresh' -q --tb=short
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python /Users/jeremylien/.hermes/cache/scratch/cpu_contract_001_guarded_pytest.py tests/test_multimodal_contract.py -q --tb=short
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m ruff check --no-cache multimodal_contract tests/test_multimodal_contract.py
/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python -m ruff format --check --no-cache multimodal_contract tests/test_multimodal_contract.py
```

Final observed exits were all **0**:

- Repair selection: **9 passed, 63 deselected in 0.07s**.
- Entire focused suite: **72 passed in 0.08s**; offline socket guard enabled, model runtime imports NONE.
- Ruff: **All checks passed!**
- Format: **3 files already formatted**.

An earlier format check returned **exit 1**, `1 file would be reformatted, 2 files already formatted`, for a newly added long test line. Only that test line was wrapped, and all four final commands above were rerun successfully. The failed check and successful reruns are retained in the new receipt, not discarded.

### Repair scope and remaining gates

Authorized workspace writes: `multimodal_contract/__init__.py`, `tests/test_multimodal_contract.py`, append-only changes here, status/contract updates in `multimodal_contract/README.md`, and only the new `results/mlx-vision-contract-fix-verification.json`. Scratch harnesses/snapshots remain under the provided scratch directory. The new receipt records before/after hashes, all historical workspace JSON hashes and read-only preservation checks. The parent-owned root README, historical JSON, qualification sources/tests, upstream and baseline were not edited. This workspace is not a git repository; no git or commit claim is made. Original qualification tests were not rerun by this repair.

The receipt is local implementation/test evidence, **not self-approval**. Bounds remain **1024 prompt / 256 image / 32 decode / 256 chunk**. No model runtime imports, weights, installs, SSH, networking, GPU work, port/app operations or deployment occurred. No physical-cache, native mRoPE, image encoder, distributed equivalence, autonomy, production readiness or 24x7 approval is established. The parent remains the sole GPU/SSH/deployment owner; fresh independent re-review is pending.
