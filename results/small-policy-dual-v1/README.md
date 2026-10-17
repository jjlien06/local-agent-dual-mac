# No-residency policy — bounded small-image controls

## Verdict: basic image-control passed, full-target policy NOT admitted

Two new bounded runs set `GGML_METAL_NO_RESIDENCY=1` only in their owned processes:

1. Native mini Metal, `../small-policy-native-v1/`.
2. Mini Metal plus laptop Metal RPC, this directory.

Both used the same publisher-hash-verified Qwen3.5-0.8B control + BF16 projector, pinned build/library snapshots, exact original128×128 red/blue images, prompt, greedy sampling and explicit thinking settings. No27B inference was attempted in these runs. The existing HTTP qualification client was not used; its fix re-review is separate.

## Measured result

All four new image requests returned the expected exact text (`red`, `blue`), stop and reported46prompt/2completion/48total/0cached tokens. Per-case image hash, exact request hash, output text, stop and complete reported usage match BOTH archived default-policy configurations. Raw response bytes are retained with individual hashes; timestamp/request metadata differs as expected. See `comparison.json`.

The dual server log records actual mini image encoding and local/RPC model/cache/compute buffers. Laptop RPC logs identify Apple M2 Pro and11 actual `graph_compute` call headers. There are7260 graph-tagged trace lines including per-node detail; those are not7260 graph calls.

Both runs reaped their model server and closed their loopback listener. The RPC controller reaped its owned laptop worker and closed its private listener. `postcleanup.json` retains explicit endpoint/worker/pressure readbacks; both hosts returned normal pressure. No system-wide policy, unrelated application, persistent service or archived default-policy evidence changed.

## Policy assurance and evidence caveats

- Laptop RPC initialization explicitly logs `use residency sets = false` (with native log spacing). On the mini, the selected API server logs omit the early device-property initialization message. Its requested environment is bound by the retained supervisor source and binary/library hashes, but an independently logged M4 property is **not** available. Do not claim full runtime/cryptographic policy attestation.
- The first readback checker incorrectly required that absent M4 log marker, after copying the native artifacts. That failed attempt is not relabeled successful. Original partial `ssh-readback/` remains intact; fresh `ssh-readback-v2/` artifacts were hash-checked independently and the missing observation is recorded explicitly.
- The copied load-controller wrapper contains an inherited `inference_executed:false` field in its local result. That is incorrect if interpreted as an aggregate result for this image test. The authoritative remote receipts and logs demonstrate two image requests; this contradiction is documented rather than silently rewriting the original receipt. This diagnostic wrapper is not a production service supervisor.
- This remains a tiny-image smoke test: cap256, actual16image tokens. It does not meet the1024-token grounding condition or establish useful screenshot/GUI behavior.
- Matching short answers and reported usage do not establish token-ID, full-vocabulary or cache numerical equivalence. No full-target capacity benefit, throughput advantage, latency distribution, long-context, coding, autonomous task completion or24/7 result is claimed.

## Remaining gate

The independent residency-policy review has arrived: `../full-target-load-v2/residency-review-delivered-summary.json`. It supports considering a single isolated experiment, but **does not approve a full-model retry**. Its minimum gate first requires the repaired client's independent approval, then fresh default-policy native/dual controls, followed by a candidate that changes **only the laptop RPC worker** while the mini remains default. These earlier both-host-policy/manual-client controls are exploratory and do not satisfy that stricter isolation sequence. The repaired client has now passed independent CPU re-review on Python3.13.2; see `../qualification-client-cpu-approval.json`. The fresh-default/laptop-only physical isolation gate remains unperformed.

Only after that small-control gate and a fresh complete peak budget may a future full27B load-only candidate use the same weights/source/40-24 placement/8192context/1024image settings, unchanged pressure/deadline/log guards and owned cleanup. Do not treat this small pass as evidence the full model will fit or as permission to ignore a warning.
