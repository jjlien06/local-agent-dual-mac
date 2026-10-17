# Full Qwen3.8-27B multimodal admission — frozen initial gate

## Current status

**Provisioned and hash-verified; three bounded full-model loading trials failed the unchanged normal-pressure gate. No full-model inference requests executed.** The latest isolated laptop-only no-residency trial advanced through239 tensor handler entries, then hit warning pressure before readiness; see `../full-target-load-v3/README.md`. All owned workers/listeners were cleaned up. Parameter retries are paused for a serving-architecture reassessment, not a smaller-model substitution. Header/source budgets remain estimates, not admission.

The external SSD holds the exact two files in `../../full-target-assets.json`: Unsloth Qwen3.8-27B UD-Q4_K_M GGUF and matching BF16 vision projector, immutable revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`. Total: 17,395,586,656 bytes (16.2009025514 GiB). This is a different format and quantization from the archived MLX baseline; it is not a continuation of that baseline's numerical or performance admission.

`provision-readback.json` records the mini's completed full SHA-256 receipt, file sizes, health and actual retained file prefix hash. That prefix exactly matches the independently fetched range used for the header budget. The readback did not repeat the entire full-file hash; the download process performed that hash before publishing its receipt. No engine was running during readback.

## Scope of the initial physical gate

Question: can this exact full model process image inputs across both admitted Metal GPUs with an explicitly sized context/image budget, bounded lifetime, clean shutdown and useful synthetic screenshot answers?

This gate does NOT qualify real GUI actions, repository coding tasks, 24/7 availability, long-horizon autonomy, superiority over other quantizations, or parity with the previous MLX runtime. A useful first gate is not the final product.

## Frozen intended configuration

- Pinned API-only llama.cpp source: `ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064`.
- Native arm64 Release, Metal and RPC enabled, both UI options disabled. Exact current executable/library hashes are in `laptop-build-snapshot.json` and `mini-build-snapshot.json`; these are pre-target snapshots, not retrospective proof of prior execution.
- Mini owns the loopback API and local vision encoder; laptop owns only the temporary RPC worker on the freshly verified private bridge address. Never bind either to all interfaces.
- Register `--rpc` before `--device MTL0,RPC0`.
- Initial context 8192, parallel slots 1, batch and microbatch 256, layer split `1,1`, all supported language layers on the two GPUs, vision device MTL0.
- Set `--fit off`; do not silently reduce context or introduce CPU layer fallback and call the requested configuration admitted.
- Set image minimum AND maximum tokens to 1024. Confirm actual encoded image counts and any positional/grounding constraints in engine logs; flags alone do not prove the effective budget.
- Disable thinking identically for bounded initial answers, greedy temperature, explicit output cap. Do not claim MTP acceleration merely because the GGUF contains one prediction layer.
- Planning budget counts all global tensors and the extra prediction-layer payload conservatively on both hosts, plus the projector on the mini, 8192-token F16 full-attention KV and a 2 GiB unmeasured workspace/state allowance. The resulting approximately 11.50/11.15 GiB host totals are estimates, not measured peaks or actual tensor placement. Inspect physical allocation and OS pressure throughout loading and requests.
- A single-host 16 GiB native full-GGUF control is not required for this first capacity gate: its weight payload alone is about 15.32 GiB before the projector/state/OS. Do not force an unsafe single-host load or transfer the small-model matching-answer claim to the full model.

## Frozen public inputs

`inputs/public-cases.json` SHA-256: `dc4190da0e1d870830cc7c4cc21620f6f500dc5ce8f06fd294c7b488d3e75965`.

There are exactly five cases: red, blue, split-color, dashboard-a and dashboard-b. The two original 1024×768 synthetic screenshots use the SAME prompt but move the target button and change the status badge, preventing a single constant coordinate/status from passing both. Their fonts/layout were visually inspected and all five PNGs passed the client's strict hash/format validation. This inspection was by the setup assistant, not a result from the candidate model.

`inputs/semantic-expectations.json` is independent scorer truth; never include its expected text or target boxes in model requests. Dashboard success requires an actual parsed JSON object with the exact badge text and integer x/y strictly inside the target box. No mouse/keyboard action is executed by this test. Color success is checked separately from the client's transport success. Retain failures and truncations, not only successful answers.

## Prerequisites and evidence

Before launch, refresh both hosts' pressure/swap/free disk and bridge routes, reconcile existing inference workers, require the independent HTTP-client review and the selected small-control proof review, and rebind actual source/model/binary identities. Use a fresh exclusive output directory, finite startup/request/total deadlines, finite logs/response bounds, and ownership-safe cleanup. Preserve all original failed attempts.

Retain exact argv, start/source/binary/model identities, owned PIDs, model/device placement and image-encoding logs, per-host memory observations, exact request/response bytes and hashes, timing/usage/stop fields, semantic scoring, process exits and explicit listener-closure readback. HTTP receipts alone never attest model identity or GPU/RPC execution. Fail closed on missing evidence, pressure problems, deadline overruns, unexpected CPU fallback, incorrect answers or unresolved cleanup.

Subsequent gates remain: useful longer context and growing tool history; structured tool/result continuation; cancellation and overflow; autonomous repository and actual GUI tasks without corrective prompts; restart/link-loss recovery; then a declared 24–48-hour idle/load qualification before persistent supervision/provider setup.
