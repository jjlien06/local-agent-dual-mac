# Actual-package CPU attention/position witness

**PARTIAL: CPU witness passed; independent review and physical model admission pending.**

This repeats the earlier synthetic attention witness using actual full imports of installed MLX-LM0.32.0 and pinned MLX-VLM source b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0 in the new .vlm-reference-venv. Unlike v1, no attention class AST is executed and no common SDPA helper is injected: each actual attention class retains its own original package helper.

-126 passing output/live-key/live-value comparisons over42 forward steps at atol1e-6/rtol1e-5.
- F32/F16, synthetic hidden32 weights with24Q/4KV/256-head geometry, one non-square image-position block,256-boundary prefix/final-token handling and two decode steps.
- Eight scalar-text output controls pass; four temporal-only or height/width-swap controls fail the positive parity gate as intended.
- Native KVCache and original LM attention forward retained; only the experimental per-chunk rope callback is replaced. Physical cache length is distinct from compressed image coordinates.
- Full package imports register native Metal kernel factory callables. Owned-process launch wrappers reject any invocation; the recorded custom-kernel launch list is empty. mRoPE uses its explicit eager CPU path. Registering a callable is not a kernel-execution qualification.
- urllib3's import-time IPv6 bind capability probe is disabled only inside the owned offline child. Socket connects/binds/DNS, process creation and signal delivery remain blocked. No machine settings changed.

## Executed

```sh
export TMPDIR=/Users/jeremylien/.hermes/cache/scratch
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
.vlm-reference-venv/bin/python -B results/attention-rope-full-import-cpu-witness-v2/test_binding.py -v
.vlm-reference-venv/bin/python -B results/attention-rope-full-import-cpu-witness-v2/probe.py
```

Both commands exited0. `result.json` retains all metrics, source hashes, registration/invocation counts and limits. The earlier AST witness, initial import guard failure/diagnostic and the5 NumPy/native rejection cases remain preserved, not rewritten. Dependencies were installed from74 individually hash-locked wheels after a disk budget and pip check; the pinned VLM source is imported through its checkout, not substituted by an unverified wheel. The text inference environment and the smaller .vision-venv were not modified.

No actual checkpoint,5120-wide hidden state, quantized projection/KV, BF16, fused Metal execution, GDN, all16 layers, encoder/pixels, distributed inference, MTP or screenshot/task qualification is established here. The new native CPU reference bridge and the attention experiment remain subject to separate review before GPU integration.
