# Full27B load v2 — allocation advanced, normal-pressure gate failed

## Verdict: NOT ADMITTED

The source-derived40-mini/24-laptop decoder placement was physically attempted with the same exact27B GGUF, projector, pinned binaries, context8192, image-min/max1024, mmap, fit-off and unchanged normal-pressure requirement. Extra RPC/server logging and read-only memory instrumentation were added. No model completion/image request was executed, and readiness was never reached.

## Observed progress and failure

- RPC debug explicitly records a successful6.26GiB remote buffer allocation. The client reports it returned, followed by remote buffer-base lookup and a20KiB `output_norm.weight` SET_TENSOR. Do not repeat the earlier unconfirmed assertion that remote allocation never returned or that absolutely no tensor upload began.
- The mini logs report the expected storage assignments: CPU_Mapped682.03MiB input embedding, MTL0_Mapped8263.66MiB model buffer, RPC6410.79MiB model buffer. These match the source/header placement budget at the log's rounded precision. The engine's `offloaded66/66` summary is retained verbatim; the checkpoint still has64 main decoder blocks plus an unused prediction block, not66 independently executed decoder layers.
- Around29.58seconds after controller start, the laptop pressure sample rose from1 to2 (**warning**, not the previous trial's4/critical). The unchanged guard intentionally terminated its owned RPC worker. A later sample was1 again; this does not retroactively pass the normal-pressure gate.
- The mini subsequently aborted with exit-6 on peer disconnect. Its pressure samples remained1. As in v1, the peer shutdown is a controlled consequence, not proof of a spontaneous RPC failure.
- The small retained process-footprint/RSS readings are not a measurement of the complete GPU/driver resident working set. Raw VM/footprint/swap evidence is in `local-result.json`. No full-model performance or peak-memory improvement is claimed from these two different load attempts.
- The large output-head upload and complete loaded model, context caches, projector initialization and image encoding were not qualified. A completed allocation is not completed inference.

## Cleanup and integrity

`ssh-readback/` contains complete mini result/argv/started/server-log artifacts read back through the pinned SSH connection; all transferred bytes matched their recorded hashes. `postcleanup.json` retains new explicit checks: all owned test listeners refused connections, owned model/RPC binaries were absent, and both hosts returned normal pressure. No unrelated application was closed, no system policy changed, no model was substituted, and no service remained running.

## Next decision

The fixed64MiB allocation/free probe passed and validates the instrumentation path only; see `../rpc-allocation-64m-v1/README.md`. V2 now narrows the investigation to allocation/residency/initial load behavior with a recorded successful multi-GiB allocation, rather than guessing from missing logs.

An independent CPU-only review is evaluating a documented, source-visible no-residency policy candidate. It is **not enabled or admitted**. Any such behavior change must pass small same-model native/dual image controls before a further full-model trial; keep source/weights/placement/context and the pressure gate unchanged. Do not blindly move more layers, disable mmap simultaneously, apply an already-included upstream fix, or relax the guard to declare success.

The HTTP qualification client remains independently unapproved until its chunk-framing fix is reviewed. Full screenshot/coding/tool/GUI/autonomy and24–48hour qualification remain pending.
