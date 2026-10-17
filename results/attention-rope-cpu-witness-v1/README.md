# Native CPU attention/position hook witness

**PARTIAL: synthetic CPU witness passed; independent review and physical model gates pending.**

The existing MLX-LM Qwen3NextAttention forward is executed unchanged, with only its `.rope` attribute bound to an experimental per-chunk native mRoPE callback. The pinned MLX-VLM Qwen3_5Attention class is the comparison. Both use the original ordinary-KV SDPA helper after verifying its terminal native call matches the VLM helper. Full native KVCache update/state is used; no fake attention outputs or cache values are synthesized.

- Synthetic, unquantized weights with hidden size32 and target head geometry24Q/4KV/256; not the real5120-wide checkpoint.
- F32 and F16 on MLX CPU0.32.4, original tolerance atol1e-6/rtol1e-5.
-126 passing output/live-key/live-value comparisons over42 forward steps, including non-square images, a256-token chunk boundary, final prompt token and two decode steps.
- Eight scalar-text output controls pass. Four temporal-only/height-width-swap negative controls are detected in both outputs and live keys.
- Two test-first binding checks reject incorrect physical offsets and chunk lengths before rotary execution. This small experimental binding is not a production request admission API.
- Original preflight AST-selector failure is preserved separately; ast.walk traversal order is not lexical source order. The corrected check selects the terminal ordinary SDPA call by source line, without modifying native functions.

## Executed commands

From the local-agent-dual-mac workspace, with TMPDIR set to the Hermes scratch directory, PYTHONDONTWRITEBYTECODE=1 and HF_HUB_OFFLINE=1:

```sh
.vision-venv/bin/python -B results/attention-rope-cpu-witness-v1/test_binding.py -v
.vision-venv/bin/python -B results/attention-rope-cpu-witness-v1/probe.py
.vision-venv/bin/python -B results/attention-rope-cpu-witness-v1/probe-with-mutations.py
```

Both numerical probes exited0. Read result.json, result-with-mutations.json and parent-disposition.json for original observations and input/source digests. The NumPy-versus-MLX diagnostic failures under ../native-rotary-reference-v1 remain unchanged; this experiment uses native rotary operations instead of relaxing those tolerances.

## Not qualified

No checkpoint weights, quantized projections/KV, BF16, fused Metal kernels, encoder, real image pixels, actual16-layer installation, GDN state, two-host transport, MTP, logits, task quality or screenshot-grounded actions were tested. Max RSS was approximately172–175MB and is process footprint, not a memory estimate for27B. The callback performs redundant native rotations to preserve the original single-input `.rope` interface; no performance claim or production dispatch decision follows from this witness. All inspected runtime and package sources remain unchanged. Independent review remains required.
