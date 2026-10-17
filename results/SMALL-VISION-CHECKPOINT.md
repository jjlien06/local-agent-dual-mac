# Small multimodal two-Mac control — physical checkpoint

## Verdict: PARTIAL / bounded image-path smoke test passed

Pinned llama.cpp `ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064` was built for nativearm64 with Metal+RPC on both the M2Pro MacBookPro and M4mini. The API builds disable both UI building and mutable prebuilt UI downloads. The original local build used an upstream UI fallback; it is not the selected API build.

**Actual inference:** Qwen3.5-0.8B Q4_K_M plus its BF16 vision encoder/projector, loaded from the external SSD. This is a small same-architecture control, **not the selected Qwen3.8-27B model**. The two files totaled739,863,648bytes; both exact publisher hashes were read back before inference. File identity is frozen in `qualification-assets.json` at the project root.

## Observed result

The same prompt and exact PNG request bytes were sent to:

1. A native single-M4 Metal server (`small-native-vision-v2`).
2. A server on the mini with local Metal plus M2Pro Metal through private-link TCP RPC (`small-dual-vision-v4`).

| Image | Single M4 answer | Two-Mac answer | Stop / usage |
|---|---|---|---|
| Solid red128×128 PNG | `red` | `red` | `stop`;46prompt/2completion tokens in both |
| Solid blue128×128 PNG | `blue` | `blue` | `stop`;46prompt/2completion tokens in both |

Exact request hashes, answer text, stop reasons and reported usage match. Responses themselves have different timestamps/request metadata; their complete raw bytes and individual hashes are retained. This is **not full-vocabulary/cache numerical equivalence or token-ID equivalence**.

The mini log names Apple M4, loads the actual vision encoder, records image encoding and assigns model/KV/recurrent/compute buffers to RPC0. The M2 log names Apple M2Pro and records actual remote graph computation. The language-model RPC buffer was335.37MiB. Thus the second host was used for inference, not merely enumerated. Vision encoding remained on the mini.

## Failures retained and narrowed

- `small-dual-vision-v1`: initial RPC connection failure, before model requests. Later direct TCP tests reached both the real RPC listener and a Python control listener. The first failure's cause is not proven; do not claim a firewall fix. No firewall setting was changed.
- `small-dual-vision-v2`: remote device enumeration worked, but the test expected the backend/buffer label `RPC0[endpoint]`; the CLI actually names the device `RPC0`.
- `small-dual-vision-v3`: explicit `--device MTL0,RPC0` was parsed before `--rpc` registered RPC0. The exact startup error is preserved.
- `small-dual-vision-v4`: a tight CLI regression first confirmed late `--rpc` fails with `invalid device: RPC0`, then early `--rpc` succeeds; the actual model command was reordered and both image requests passed.

Readiness uses owned listening-PID evidence and an actual RPC device query, not a raw open port as proof of protocol readiness.

## Limits that matter for the final application

- Only two low-resolution color-conditioned image requests per selected mode; no real screenshot grounding, coding, tools or autonomous actions.
- Image cap was256tokens; actual images encoded to16image tokens. Upstream explicitly warns that grounding tasks need at least1024image tokens. A realistic screenshot/context/memory test remains mandatory; do not reuse this tiny-image memory footprint.
- No27B GGUF target loading or6bit model test yet. The previous ParetoQuant MLX/MTP runtime and environment were left unchanged.
- No performance headline or reliable latency estimate from two trivial outputs; first-use and warm samples differ.
- Private RPC is experimental/insecure upstream. It was bound only to the approved bridge address for these bounded tests, never public. The model API bound only to127.0.0.1.
- No persistent service, Hermes-provider switch, power-policy change, or24-hour qualification.

## Cleanup and evidence

The mini model processes were reaped and loopback API listener18380 refused connections afterward. Each local RPC actor was reaped; all test RPC/control ports18552/18554/18555/18556/18558/18560 refused connections afterward. Fresh health checks returned normal memory pressure on both hosts and no previous owned Python inference workers; the mini had no project-owned model binaries remaining.

The successful mini receipts, exact requests/responses and full server logs were fetched through the pinned SSH command and their hashes checked against the live receipt. An initial SCP retrieval attempt could not parse the link-local IPv6 target; no connection settings were changed, and ordinary pinned SSH bounded readback succeeded.

Selected evidence:
- `small-native-vision-v2/ssh-readback/`
- `small-dual-vision-v4/ssh-readback/`
- `small-dual-vision-v4/local-rpc.log`, `local-rpc-receipt.json`
- `rpc-connectivity-diagnostic-v1/`
- Earlier failed version directories remain unchanged.

The reusable HTTP qualification client is implemented; the parent reran its CPU suite (**61 passed**) and lint passed. Its independent security/logic review is separate from the physical proof audit. **The client review failed** (`results/qualification-client-independent-review.json`): malformed chunk separators and a truncated terminal chunk could be marked complete/successful despite those green tests. A focused strict-framing fix is implemented; the parent reran **84 tests**, all passing, with clean lint. **Independent fix re-review passed** (`results/qualification-client-independent-review-v2.json`):84 focused tests and52 independent probe groups, including the exact former failures. The parent verified the saved verdict hash, every reviewed source hash and the probe counts. `results/qualification-client-cpu-approval.json` records approval of the documented CPU transport scope on Python3.13.2 only; no physical model/server/agent approval follows. The failed report and original61-test evidence remain unchanged.

## Independent physical-proof review

`results/small-vision-proof-review.json` preserves the verdict delivered by `deleg_de6fe8ea`: **passed**, no blocking security or logic findings, 42 evidence files fingerprinted, 24 existing hash/argv checks, zero mismatches. The reviewer independently decoded the images, checked matching request bytes/answers/stops/reported usage and verified logs supporting actual M4 vision encoding plus M2 RPC graph work.

This approves only the bounded small-control proof. Historical receipts do not fully attest loaded dylibraries or independently rehash remote model files. The review did not independently recover the comprehensive final all-port/pressure readback; its cleanup assurance is limited to retained per-run receipts and shutdown logs. The API's missing authentication/permissive CORS and experimental RPC must be addressed before persistent deployment. New pre-target build/model manifests are future admission evidence, not retrospective attestation of these historic runs.

**The bounded small-model image-path gate is complete. Overall hands-off27B-agent goal remains incomplete.**
