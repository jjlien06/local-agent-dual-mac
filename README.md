# Two-Mac local development agent

## Goal

Run the strongest defensible coding, terminal and visual computer-use model locally across the existing16GiB M2Pro MacBookPro and16GiB M4mini, eventually available24/7 with authorized task completion without corrective user prompts. Model recommendation and evidence: `../paretoquant-metal/results/local-agent-model-selection-2026-10-16/`.

## Current stage: exact27B MLX vision extension under CPU admission

**Not a deployed service.** The previously measured ParetoQuant text runtime and its environment remain the preserved baseline, not an admitted visual server.

The pinned llama.cpp Metal+RPC small-model image controls passed, but **all three full27B GGUF load attempts failed the unchanged memory-pressure guard before readiness**. Full-model RPC retries are paused. Reducing message staging does not remove resident weights, and the architecture review did not establish enough capacity savings. See `results/full-target-load-v3/architecture-decision.json` and the two source-cited architecture reviews beside it.

The next candidate extends the already-proven exact27B two-host MLX text pipeline with trained vision encoding plus explicit three-axis image positions through attention on both ranks. No pretrained weight or text quantization substitution is planned. Read-only current rehashing verified all three retained MLX shards against publisher hashes and all333 BF16 vision tensors (921,460,192 bytes); see `results/mlx-vision-admission/retained-weight-readback.json`. This establishes file integrity, **not GPU visual compatibility or memory fit**.

The isolated CPU image-position/request contract's first independent review **failed** on `CPU-CONTRACT-001`: a forged mutable NumPy configuration-digest field bypassed canonical admission and could leave the session active after an exception. That failure and the parent's reproduction remain preserved. The narrow repair now validates the exact pinned string and retains a freshly canonicalized immutable request. Parent verification passed156 tests (72 contract tests plus84 existing qualification tests), lint/format and198 historical-JSON hash checks; see `results/mlx-vision-contract-fix-parent-verification.json`. **Fresh independent re-review is pending; the repair is not yet approved.** No encoder, rotary adapter or real cache has been exercised. Native-versus-adapted small-checkpoint numerical/cache gates on each GPU and both ranks, plus a fresh peak-memory budget, remain required before any full-target visual trial. The original text runner and pinned RPC sources remain unchanged.

Source pin: `ggml-org/llama.cpp` commit `ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064`. Build nativearm64 Metal with RPC. Do not expose the experimental RPC listener publicly. No changes to SSH authorization, machine-wide firewall, power settings or Hermes provider until separately admitted.

Workspace is isolated from dirty ParetoQuant experiments. Private connection material stays in the existing approved local profile; do not copy credentials into this project or commit models/build products. Temporary probes belong under `$TMPDIR=/Users/jeremylien/.hermes/cache/scratch`.

## Acceptance gates

1. Pinned source/build identity and Metal/RPC binaries on both hosts.
2. Small-model real-image single-host and distributed generation; authoritative request/output/model/device receipts and bounded cleanup.
3. Full target27B vision, realistic context, tool-result continuation, cache isolation and cancellation/recovery under healthy per-host memory.
4. Real code/terminal/reversible GUI tasks with independent success checks, counting every intervention or failure.
5. Mixed idle/load soak and restart/link-loss recovery before persistent supervision. Never describe this stage as passed from a source audit, file-size fit or unit tests alone.

Current verdict: **PARTIAL**. Native and two-Mac small-model image-path smoke tests passed with exact red/blue answers and matching stop/usage, actual Metal/RPC graph evidence and verified cleanup. This is not the27B target or screenshot/agent readiness. See `results/SMALL-VISION-CHECKPOINT.md`; full target, realistic vision/context and unattended-operation gates remain pending.
