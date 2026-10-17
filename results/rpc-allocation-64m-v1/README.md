# Fixed64MiB RPC allocation accounting probe

## Verdict: instrumentation/control passed, not full-model capacity

The mini's CPU-only client invoked the pinned native GGML RPC buffer API to request exactly67,108,864bytes on laptop Metal device0, held the returned buffer briefly, then freed it. No weights were opened, no tensor bytes were uploaded, no graph or model inference was executed, and no larger allocation was attempted.

## Evidence

- Remote events retained in `ssh-readback/`: requested67,108,864bytes, returned buffer size67,108,864bytes, allocation call returned, and explicit free completed.
- `local-rpc.log` with `GGML_RPC_DEBUG=3` records `ALLOC_BUFFER`, successful64.00MiB remote allocation, and `FREE_BUFFER`.
- All four laptop pressure samples were1; mini final pressure was1.
- `footprint` successfully sampled the owned RPC PID. Its reported physical footprint stayed around42–43million bytes, despite the64MiB buffer size. Returned allocation size, process RSS/footprint and actual GPU/driver residency are therefore not interchangeable. This probe does not prove every buffer page was physically faulted, filled, wired or independently charged to that process.
- The native allocation-call wall time was recorded, but it is not a model latency, upload bandwidth, GPU compute timing or general allocation-performance claim.
- Parent verified retained request/event/result/log hashes, reaped its RPC worker, and confirmed the private listener closed. The remote client returned exit0 with `owned_buffer_freed:true`; no API/model service was started.

## Admission correction

The first attempt was held before launch by an overbroad512MiB immediately-free-page prerequisite applied to the mini. That host only runs a small CPU RPC client in this probe. The revised, archived prerequisite used128MiB immediately-free-page reserve for that client and retained512MiB for the laptop GPU server. Both still required normal pressure and verified bridge routes. See `../full-target-admission/allocation-64m-preflight-v2.json`; the earlier hold remains preserved.

No OS-reported free percentage was converted into a safe allocatable-byte budget; free pages alone omit file-cache reclaimability, while global footprint can include accounting that is not directly subtractable from physical RAM.

## Next bounded question

A source-derived40/24 decoder placement reduces the laptop's raw preferred GPU payload, but this tiny probe does not admit that multi-GiB load. Any full-model trial requires fresh health checks, exact identities, actual placement/allocation logs and unchanged pressure/deadline/log/cleanup gates. This report is not a27B, screenshot, coding, autonomy or24/7 qualification.
