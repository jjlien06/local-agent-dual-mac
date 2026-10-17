# Full-model loading trial v1 — rejected by resource guard

## Verdict: NOT ADMITTED

The first bounded full Qwen3.8-27B UD-Q4_K_M + BF16-projector load was attempted across the M4 mini and M2 Pro laptop. It did **not** reach readiness and no model completion/image request was sent. This is not an inference or autonomy success.

## What actually happened

- Source/build files were rehashed against the retained pinned manifests before launch; complete model download/hash receipt was bound into the admitted configuration. File sizes were checked again, but this trial did not independently repeat the full 17 GB hash operation.
- Both hosts had normal memory pressure at admission; the freshly checked transport route was bridge0 in both directions.
- Context8192, one slot, batch256, layer split1,1, local vision, all requested language layers on Metal/RPC, fit adjustment disabled, image-min/max1024, and no warmup. Requested flags are not evidence that engine initialization or image encoding completed.
- The mini began loading and remained at normal pressure. Its sampled server RSS grew to roughly 9 GiB and later declined; these samples are not peak unified-memory/GPU measurements.
- About34 seconds into the laptop RPC worker's lifetime, the laptop pressure reading changed from1 to4. The supervisor's resource guard terminated **only its owned RPC process**, then reaped it. Pressure4 is a failed capacity/safety gate; it was not ignored or relaxed.
- The mini subsequently aborted with an RPC disconnect/malformed-response message and exit-6. This is consistent with the deliberately terminated peer; it is not evidence that an unexplained spontaneous RPC crash was the initiating failure.
- The sampled laptop RPC process RSS stayed small before the abrupt pressure event. RSS samples are insufficient to identify Metal residency, allocation transients or competing application load. The actual allocation/pressure cause is still under investigation; the nominal header-based planning budget was not physical admission.

## Cleanup

The remote mini supervisor reaped its server and confirmed loopback API18382 closed. Pinned-SSH readback retrieved `result.json`, `argv.json`, `started.json` and the complete server log; hashes matched the remote result. See `ssh-readback/`.

`postcleanup.json` contains new raw readbacks: all seven prior/current local test listener ports explicitly refused connections, the two mini loopback API ports refused connections, no model/RPC binaries remained, and both hosts returned normal pressure. Laptop swap grew during the attempt and remained elevated afterward; normal pressure alone is not a sufficient reason to immediately repeat the same load. No unrelated app was stopped, no power/firewall setting changed, and no persistent service was installed.

## Next diagnostic gate

The read-only source/evidence investigation is complete: `cpu-root-cause-review.json`. Leading hypothesis is upfront multi-GiB Metal buffer allocation/residency on the laptop before weight upload; the exact blocked call and physical footprint remain unmeasured. Source ordering and the absence of later INFO buffer/offload summaries favor that phase, but an abrupt abort/log buffering prevents treating it as proof. The mini's exit-6 remains secondary to the deliberate peer shutdown.

The source-derived ownership calculation was independently reproduced in `../full-target-admission/source-placement-reconciliation.json`: the1,1 split assigns main layers0–32 to the mini and33–63 plus the output head to the laptop (not the intended32/32 split). Raw GPU payloads are7,116,214,144 and8,271,059,072bytes respectively; CPU embedding placement on the mini is intentional. Unused MTP tensors are not loaded. Alternative3,2 and5,3 header budgets match the reviewer, but neither is admitted or a proven fix.

A fixed64MiB allocation/free instrumentation fixture was prepared, with no weights/inference or automatic scaling. **It has not been launched:** its conservative immediately-free-page precondition failed on the mini. The raw counters show substantial file-backed/inactive pages, so this free-page count is not a reliable estimate of total reclaimable headroom and does not establish a new physical-RAM blocker. See `../full-target-admission/allocation-64m-admission-hold.json`. Resolve the accounting method before physical probing rather than forcing this test through or claiming the small allocation succeeded.

Preserve this failed trial and its sources byte-exactly. Do not switch to a smaller model, silently reduce context/CPU-offload, weaken pressure checks, or claim the requested setup is impossible from one guarded attempt. Any revised placement/loading candidate needs a source-grounded reason and a fresh bounded trial.

This load-only diagnostic deliberately did not use the HTTP qualification client; that client's independent review is a separate gate before full visual requests.
