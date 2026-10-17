# Full27B load v3 — isolated laptop policy advances upload, still rejected

## Verdict: NOT ADMITTED; pause parameter retries

The third full-model load-only attempt retained the exact target/projector, source/binary/library identities,40-mini/24-laptop placement, mmap, context8192, image-min/max1024 and fit-off. The only effective resource-policy change fromv2 was disabling residency sets in the laptop RPC worker. The mini explicitly left that flag unset. Normal pressure on BOTH hosts remained mandatory.

This trial followed fresh reviewed-client native/default-dual/laptop-only small image controls:9/9 semantic answers, exact request/output/stop/usage agreement and verified cleanup. Those small controls did not predict the full-model result.

## Observed result

- Laptop startup logs confirm residency disabled and shared buffers enabled.
- Allocation returned and uploading advanced substantially beyondv2. The laptop logged the complete994.63MiB output-head SET_TENSOR handler and239 distinct tensor handler entries, ending at `blk.57.ssm_out.weight`.
- Matching those239 names to the exact retained GGUF tensor directory gives4,970,537,216bytes (4.6291735172GiB) of source tensor payload represented by those handler entries. This is not independent byte-by-byte remote parameter attestation or a complete loaded-model result. See `transfer-progress.json`.
- At55.214seconds after controller start, the laptop pressure sample rose from1 to2 (warning). The resource guard stopped and reaped ONLY its owned RPC worker. The model never reached readiness; no completion/image request ran.
- The mini subsequently aborted with exit-6 on peer loss, then was reaped by its supervisor. As in previous attempts, the resulting RPC assertion is secondary to the controlled pressure stop, not an unexplained initiating RPC crash.
- Disabling eager residency moved/avoided the earlier allocation-stage warning in this one attempt, but **did not solve end-to-end capacity**. No causal speedup, stable peak reduction, model quality or readiness claim is justified.

## Cleanup and evidence

Complete authoritative mini argv/result/started/server-log files were read back through pinned SSH, with exact hash validation. `postcleanup.json` records explicit refusal from all accumulated test listeners, absent owned model/RPC workers and normal pressure on both hosts. Laptop swap remains elevated compared with early-session observations; no unrelated app was stopped and no global system/provider/service setting changed.

## Architecture reassessment

Three guarded full-model attempts have now failed. Do not continue adjusting placement/policy parameters blindly or relax the normal-pressure gate. The archived exact27B MLX text/MTP baseline remains intact and previously validated; the new multimodal GGUF endpoint is not.

Two read-only investigations are comparing:

1. Whether bounded RPC weight transport/loading can materially reduce real transient staging, with exact byte-preservation tests and a small memory-controlled probe before any further full load.
2. Whether the already-proven MLX two-host text runtime can accept the exact model's real visual embeddings/positions/auxiliary states with a properly admitted vision encoder, avoiding this serving backend's loading behavior.

A source-visible full-tensor staging allocation is not proof of a persistent one-GiB vector leak: the RPC receive vector is scoped inside each SET_TENSOR switch case and is destroyed on exit. Allocator/driver accounting and ordinary simultaneous demand must be distinguished with evidence. Neither architectural alternative has been implemented or admitted, and the research must not replace the requested27B model with the small control.

The remaining user goal is still open: real full27B screenshot handling, useful coding/tool/GUI tasks without corrective prompts, cancellation/recovery, and declared24–48hour mixed idle/load qualification before persistent setup.
