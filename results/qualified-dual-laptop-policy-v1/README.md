# Fresh reviewed-client laptop-only policy controls

## Verdict: bounded three-case image gate passed

Three fresh configurations were physically exercised in order:

1. Native M4 mini, default residency (`../qualified-native-default-v1/`).
2. M4 mini + M2 Pro RPC, both default (`../qualified-dual-default-v1/`).
3. Same dual configuration, **only the laptop RPC worker** with `GGML_METAL_NO_RESIDENCY=1` (this directory).

The mini supervisor explicitly removes that environment variable in all three configurations. Laptop startup logs explicitly observe default residency enabled in the control and disabled in the candidate, with shared buffers still enabled. This satisfies the isolated-host setup missing from the earlier exploratory both-host test. Neither setting is a system-wide/persistent change.

## Client and input binding

The repaired HTTP client had already passed independent CPU review. It was deployed to an exclusive content-hashed mini snapshot, every written file was read back, and the entire84-test suite passed again on the mini's Python3.13.2. See `../qualification-mini-deployment-controller.json` and `../qualification-client-cpu-approval.json`. The deployment command was initially blocked because an oversized inline command exceeded scanner budget; it performed no deployment. The successful path used explicit inspectable code/payload files instead.

Snapshot tree SHA-256: `51d37c9a7d66682c2d2ba09a7d8d89d5fb32dc5cfde13c06844191df1e28d3aa`.

Each physical run verifies the snapshot, native binary/library hashes, pinned source revision and both actual small model/projector full-file hashes before launching. This is trusted supervisor binding, not cryptographic remote execution attestation.

The same frozen public manifest supplies solid red, solid blue and left-half red/right-half blue PNGs. The third prompt asks only for the left color. No expected answer field is sent in a model request. All nine requests use the actual PNG data, explicit greedy settings and the same server alias/context/template controls.

## Result

- Exactly9/9 semantic answers passed: red, blue, red in each configuration.
- Per-case exact JSON request hashes, output text, stop and complete reported token usage match across all three configurations. Every response is complete; client receipts retain `attestation:false`.
- Effective observed context is2048 for these small controls. Image cap256/actual16-token images remain far below realistic screenshot grounding; no GUI-quality claim follows.
- Logs contain three actual image-encoding events in each mode. Each dual RPC log contains16 actual graph-call headers, not merely device enumeration or per-node trace-line counts.
- The mini pressure samples and both dual laptop pressure samples all remain normal. All server/client workers were reaped and all new API/RPC listeners were explicitly verified closed; see `postcleanup.json` and per-run receipts.
- Complete remote evidence was read back with bounded transfer and hash validation. `comparison.json` contains the mechanically verified counts and exact matches; `remote-readback.json` fingerprints all fetched artifacts.

## Assurance limits

This is only small same-family image-path admission for the isolated resource-policy candidate. It is not full-token/logit/cache numerical equivalence, full27B capacity, quality retention, speedup, screenshot action execution, long-context coding/tool performance, secure persistent API access or unattended/24-hour qualification. Default and candidate runs are sequential, not a latency benchmark. Mini early device properties are not independently logged; its default policy is bound by unchanged supervisor environment/source and actual build identities, while the laptop policy is directly logged.

The independent read-only audit **passed** for this completed bounded gate: `independent-audit.json` (SHA-256 `50fe4bc1c434586b0cbb3b9ce9d12df43776ea2d973aa1ec2fe2d376e13ea61a`). It independently verified340 checks,63 remote artifacts and all9 cases. The parent recounted the check labels/cases and verified every bound source/evidence hash; see `audit-parent-binding.json`. Nonblocking generalized-supervisor hardening and the incorrect inherited alarm comment are recorded for a future version; executed frozen sources remain unchanged.

The subsequent full27B load-only policy trial is a **separate failed result** (`../full-target-load-v3/README.md`): it hit warning pressure during upload and never reached readiness. It cannot inherit success from these nine short answers.
