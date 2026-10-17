# Exact-27B MLX vision architecture reassessment

## Decision

**Architecturally feasible to investigate; not currently a visual runtime.** Keep the exact `mlx-community/Qwen3.8-27B-4bit` revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` and the proven two-host language pipeline. The minimum credible extension is a **rank1/mini-only BF16 vision encoder + merged visual embeddings, and explicit multimodal positions through every full-attention layer on both ranks**. Injecting image embeddings alone is incorrect: the installed language attention derives ordinary scalar RoPE from KV offsets. This is a new correctness integration, not a configuration toggle.[12][15][23]

**DeepStack is NOT required for this target.** Its configuration has `deepstack_visual_indexes=[]`; pinned MLX-VLM explicitly rejects nonempty Qwen3.5 DeepStack indexes. Adding Qwen3-VL-style auxiliary residual streams would change the architecture rather than complete it.[4:47–58][12]

**Native MLX-LM pipeline serving is not a drop-in VLM route.** It rejects non-text message fragments, constructs the text-only model, and has no multimodal positional interface. Its default loader remains text-only.[22][25][26] The pinned VLM wrapper/core fail the native pipeline capability contract structurally.[1][2][6]

Recommended order: a bounded CPP staging experiment is the smaller implementation/correctness risk; the MLX extension is the stronger model-byte-preserving fallback if staging cannot establish safe capacity. Do not repeat a full load until one route passes its small control and memory gate. Neither route is implemented or admitted by this review.[27][28]

## Scope and evidence boundary

Read-only local source/metadata inspection and immutable primary-source HTTPS retrieval only. No MLX/VLM imports, GPU work, SSH, installation, model downloads, application actions, commits or original-file edits. Only this new report is published. MLX `0.32.3` and MLX-LM `0.32.0` were read from installed distribution metadata, not imported. MLX-VLM below means the inspected source at **`b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0`**, not an executed or installed compatibility claim.

Performed static checks: 2,180 header names exactly equal the local pinned index; 1,847 language + 333 vision tensors; descriptor dtype × shape equals offset-span bytes; all 27 vision-block layouts match; all 333 vision tensors are BF16; no DeepStack tensor names. Historical download receipt binds this inventory to the target revision and confirms matching headers/full shard hashes. **Current mini payloads were not reread**; these checks prove metadata consistency, not current loaded vision-weight attestation.[13][14]

## 1. Exact image-to-language handoff

1. **Processor.** Pinned `qwen3_5/__init__.py` selects `Qwen3VLProcessor`; the loader uses `AutoProcessor`. The class uses local processor JSON, bicubic RGB resize to multiples of `patch_size*merge_size=32`, rescale/normalize, and temporal duplication for still images. At this target's settings, normalized patches are `[P,1536]`, `image_grid_thw=[1,H/16,W/16]`, with P=H*W/256. Prompt image markers expand to P/4 language tokens before tokenization. Honor the actual local mean/std `[0.5]*3`, minimum65,536 and maximum16,777,216 pixels—not class defaults. Pin/validate actual processor class and outputs: the automatic patch can catch an exception and fall back.[7:279–461,803–876,960–1001][9:977–1006][10:598–659][11][12]
2. **Encoder/merger.** Qwen3.5 reuses the pinned Qwen3-VL vision implementation. Conv3d patches become width1152; interpolated learned 48×48 position embeddings and 2D rotary coordinates follow the same merge-block ordering. Run27 width1152,16-head blocks; the final merger normalizes width1152, groups four patches into4608, then4608→4608→5120. Output is `[P/4,5120]`, plus an empty auxiliary list here.[3][5:63–117,188–232,234–424][12]
3. **Embedding replacement.** `get_input_embeddings` casts pixels to patch-convolution dtype, computes ordinary token embeddings, obtains vision features, checks image-token/feature cardinality, and masked-scatters those rows into the prompt. The returned effective handoff is `inputs_embeds`, `position_ids`, `rope_deltas`; this Qwen3.5 override discards the auxiliary vision return. Inherited `__call__` forwards the bundle to `language_model`. Do not substitute the generic Qwen3-VL DeepStack handoff.[1:58–142][6:215–236][10:75–104]
4. **Positions.** `get_rope_index` constructs three-axis image positions on the merged grid; text resumes at the prior maximum+1. It returns `[3,B,T]` positions and delta=`max_position+1−valid_token_count`. Continuation positions use physical cache offset+delta, not offset alone. Every full-attention layer receives positions; GDN does not consume them. Target head_dim256 × partial_rotary0.25 means64 rotary dimensions, sections `[11,11,10]`, base10,000,000. Pinned rotary uses interleaved axis selection and half-split pairing, not three contiguous frequency bands. Cache length remains the actual token count, never the maximum spatial coordinate.[2:885–1035,1195–1233,1373–1405,1459–1642,1644–1791][8:512–565,1130–1242][12]

Use target token IDs explicitly: image248056, video248057, vision-start248053, vision-end248054, EOS248046/248044; Qwen3.5 class defaults for vision boundaries differ and must not override this config.[4][12]

Initial scope should be one still image, B=1, no padding/video/prefix reuse. Video and padded batching add distinct processor/position/cache behavior; they are not implied by the above path. In the inherited model call, its named `mask` is not explicitly forwarded into embedding preparation; the language core also creates hybrid masks from cache state. Matching argument names alone does not establish padded-batch compatibility.[2:1373–1405][6:215–236]

## 2. What the proven text interfaces lack

`021` and `023` have byte-identical `phased_decode.py` (SHA below). `024` adds hidden-only normalization when `logits=False, return_hidden=True`, but not visual inputs. All accept only token IDs `[1,T]`; rank1 embeds them, runs its prefix, and sends evaluated hidden `[1,T,5120]` over the CPU-stream group. Rank0 receives it before constructing its GPU graph. The native hybrid mask/cache checks are valuable and should stay.[15:62–173][16:62–181]

Both layer-loop variants call `layer(h, mask, state)` without positions. Installed `qwen3_5.DecoderLayer` and its imported `Qwen3NextAttention` expose no position argument. Attention applies scalar `self.rope(...offset=cache.offset)`; the default rope factory returns `nn.RoPE`, ignoring multimodal section information. Although installed outer/core forwards already accept **`input_embeddings`** (not VLM's **`inputs_embeds`**), the custom phased entry does not. Embedding injection alone therefore fails the position contract.[22:208–239,265–337,404–439][23:74–151][30:291–307]

`Command` contains request/sequence/op/tokens only; token chunks are bounded256. `Channel` caps JSON at65,536 bytes. MTP `fresh` chunks raw IDs and feeds shifted token embeddings plus normalized target hidden; reset/snapshot/restore currently manage caches, not multimodal request metadata.[19][20][21:55–169]

### Precise minimum proposed changes (not implemented)

| Boundary | Smallest required contract |
|---|---|
| Admission/loader | Admit exactly333 additional `vision_tower.*` tensors on rank1 only, with fixed names/shapes/BF16/raw hashes; retain all1,847 trusted language tensor hashes and replicated checks. Existing planner deliberately filters vision and existing loader instantiates text-only `Model`, so neither can simply be reused unchanged.[17:51–112][18:194–271] |
| Request preparation | Authoritative expanded token IDs; ordered image grids; pinned processor/config/image hashes; bounded pixel payload and decoded-pixel/token count; `[3,1,T]` positions and `[1,1]` delta. Image bytes/pixels needed only on rank1; rank0 needs positions/delta/cursor, not the encoder or pixels. Reject missing/surplus/ambiguous image markers before embedding. |
| Rank1 forward | Optional premerged embedding chunk `[1,t,5120]` (or image-feature rows+validated replacement indexes), exactly matching the ID slice and current token cursor. Rank1 retains normal ID embedding for text/no-image requests. Compute vision once per request, not once per prefill chunk. |
| Both ranks' language forward | Position slice `[3,1,t]` through phased forward→decoder→every owned FA. Replace/adapt **only** rotary application using pinned multimodal semantics while retaining admitted projections/norms/gating/KV updates; leave GDN/state arithmetic unchanged. A complete VLM-language swap would require re-admitting different operators/cache classes, not just identical weight shapes. |
| Rank handoff | Existing hidden shape/dtype/send/receive ordering can remain. Add a versioned request manifest and chunk offset/position binding with matching ACKs on both ranks. Use bounded binary payloads or bounded position slices; do not base64 image tensors into the existing JSON command frame.[15][19][20] |
| State lifecycle | Make positions/delta/cursor request-scoped; reset on a new image request and release on finish/cancel. Keep KV physical offsets and GDN progression consistent with absolute token slices. Future rollback/replay must restore/derive the cursor and verify positions on both ranks; a cache-only restore must not leave stale positional state. |

No DeepStack tensor transport, per-layer visual injection, new cross-attention cache or extra cross-rank feature stream is needed for this exact configuration. Keep **native width-one generation first; MTP off**. Existing text MTP/rollback admission does not prove image-conditioned target/sidecar positions, shifted visual-token behavior, speculative-width arithmetic or acceptance. That is a separate later gate.[4][12][21]

## 3. Vision invariant inventory available locally

All names below have prefix `vision_tower.` and are BF16; no vision affine scales/biases accompany packed weights. Preserve that checkpoint precision—global4-bit metadata is not permission to newly quantize the encoder.[12][13]

| Suffix | Exact stored shape |
|---|---|
| `patch_embed.proj.weight`, `.bias` | `[1152,2,16,16,3]`, `[1152]` |
| `pos_embed.weight` | `[2304,1152]` |
| `blocks.{0..26}.attn.qkv.weight`, `.bias` | `[3456,1152]`, `[3456]` |
| `blocks.{0..26}.attn.proj.weight`, `.bias` | `[1152,1152]`, `[1152]` |
| `blocks.{0..26}.mlp.linear_fc1.weight`, `.bias` | `[4304,1152]`, `[4304]` |
| `blocks.{0..26}.mlp.linear_fc2.weight`, `.bias` | `[1152,4304]`, `[1152]` |
| `blocks.{0..26}.norm{1,2}.{weight,bias}` | each `[1152]` |
| `merger.norm.{weight,bias}` | each `[1152]` |
| `merger.linear_fc1.weight`, `.bias` | `[4608,4608]`, `[4608]` |
| `merger.linear_fc2.weight`, `.bias` | `[5120,4608]`, `[5120]` |

Total=27×12+9=333 tensors, **921,460,192 bytes =878.773MiB =0.858177GiB**; largest45MiB. Headers already show MLX NDHWC convolution layout. Do not apply another PyTorch transpose or RMSNorm `+1`; pinned sanitizer has guarded paths, and this already-converted layout must remain byte-preserved. There are no DeepStack merger weights.[1:144–175][13]

Inventory is presently under scratch and auto-prunable; this report retains its digest and invariant summary, not all payload hashes. The local index gives names/shard mapping, not shapes/dtypes/payload digests. Before execution the hardware owner must revalidate full mini headers/bytes against trusted checkpoint receipts and bind each vision hash; this review did not manufacture333 unobserved payload hashes.[13][14]

## 4. Memory and copying: plausible, not admission

Computed in stdlib from actual headers, including **1.332102GiB replicated embedding/norm/head per rank**. These are stored-parameter lower bounds, not process/GPU peaks. First-stage rank1 is mini; final rank0 is laptop.[13][15][24]

| Retained layers laptop/mini | Text GiB laptop | Text GiB mini | Mini with full BF16 vision GiB |
|---|---:|---:|---:|
| 24/40 | 6.117634 | 9.307988 | 10.166165 |
| 28/36 | 6.915222 | 8.510400 | 9.368576 |
| 32/32 | 7.712811 | 7.712811 | 8.570988 |

28/36 is a reasonable **planning** starting point if preserving the existing text placement;32/32 is a capacity alternative, not an admitted speed improvement. Add MTP sidecar only if later validated; do not replicate vision. Loading/encoding separately and releasing vision intermediates before language prefill can reduce overlap, but MLX pooled/lazy references must be measured before asserting memory is returned. SSD storage does not supply extra unified RAM or make per-layer weight streaming performant.

**Single square image estimates**, using BF16 vision features; resized dimensions govern the table:

| Side | Raw patches P | Language image tokens | CPU F32 pixels MiB | BF16 pixels MiB | Merged features MiB |
|---|---:|---:|---:|---:|---:|
| 256 | 256 | 64 | 1.5 | 0.75 | 0.625 |
| 512 | 1,024 | 256 | 6 | 3 | 2.5 |
| 1,024 | 4,096 | 1,024 | 24 | 12 | 10 |
| 4,096 (configured max) | 65,536 | 16,384 | 384 | 192 | 160 |

Derived from P×1536 pixel values and (P/4)×5120 features.[7][12][13] At512, individual encoder hidden/QKV/MLP outputs are approximately2.25/6.75/8.41MiB; at1024 they are9/27/33.63MiB. These are not summed peak bounds: Q/K/V, residuals, interpolation gathers, casting, merger buffers and driver workspace can overlap. A materialized BF16 attention-score tensor alone would be32MiB at512,512MiB at1024,128GiB at4096; fused attention avoids that particular allocation only if the physical dispatch actually uses it. **Never admit the processor's16M-pixel default without a separate measured cap.** Start256 or512; later screenshot legibility must be measured rather than assumed.[5][7][12]

Native BF16 full-attention KV is4,096bytes/token/layer: all16 FA layers cost256MiB at4,096 tokens,512MiB at8,192 (plus cache allocation slack). All48 F32 GDN recurrent states cost144MiB, convolution tails2.8125MiB; partitions retain only their own layers. Snapshot/rollback copies add live state, and MTP adds head/cache/workspace. Prompt/image tokens count toward all of these.[12][15][21]

The existing256-token handoff is2.5MiB BF16 per chunk; vision on mini adds **no second cross-rank visual tensor**. It still needs initial screenshot/pixel delivery if capture occurs on laptop. Conversion/transfers are not zero-copy: account for encoded image, CPU float32 patches, BF16 cast, merge/scatter source+destination, retained feature references and CPU-stream handoff buffers. A dense broadcast replacement mask can also materialize T×5120 booleans. Avoid retaining all-token vocab logits:256×248320 values are121.25MiB BF16 or242.5MiB F32; the024 hidden-only path is useful but does not itself add vision.[1][15][16]

Text packed arrays/scales/biases and protected floating tensors stay exactly as admitted; keep BF16 encoder and do not dequantize the entire language model. Preserve768MiB text transfer bound (largest packed embedding/head tensor606.25MiB), one-record loading, hashes and fail-closed shape/dtype checks. Vision's45MiB largest tensor need not introduce a gigabyte transport buffer.[13][17]

## 5. Native pipeline/server alternative: actual compatibility result

Installed `sharded_load` checks `model.model.pipeline`, pipelines the model, selects needed shard filenames, loads again and evaluates parameters; it does not accept a VLM wrapper argument, processor, multimodal kwargs or explicit split. The native mixin supports reverse contiguous splits internally, but this loader calls it without one. File-granularity shard loading is not the current tensor-record streaming memory guarantee.[24][25:671–766]

Default `qwen3_5.Model` is text-only; its sanitizer intentionally drops `vision_tower`/`model.visual`. Its `input_embeddings` support is real but insufficient because no positions reach attention. Pinned VLM `Model` inherits `qwen3_vl.Model`, exposes `layers` rather than native outer `model.pipeline`, and its language core inherits `nn.Module` without `PipelineMixin`. AST inspection verified those declarations, not a physical execution result.[1][2][6][22][23]

Installed server `process_message_content` explicitly raises on non-text content; distributed `_load` uses `sharded_load`, and distributed draft/adapters are rejected. It tokenizes with a tokenizer, not the image processor. Therefore this route currently saves neither the vision/position integration nor memory-policy validation. A wrapper, custom loader, multimodal request protocol, sharded-cache semantics and numerical gates would all be needed; calling that native compatible would be false.[26:121–145,317–340,522–569]

## 6. Ranked risk and smallest next gates

| Candidate | Engineering/correctness risk | Remaining feasibility risk |
|---|---|---|
| **1. Bounded CPP message staging** | Lowest scope: preserve existing model/vision/mRoPE execution and change exact-byte loading transport. Existing offset fields permit bounded writes, but hash-cache semantics, asynchronous queue depth, source-buffer lifetime and completion/fence ownership must also be bounded. | **High capacity uncertainty:** cannot reduce resident final parameters/driver workspace; may merely move the pressure point. |
| **2. Exact MLX pipeline + rank1 vision + FA mRoPE adapter** | Medium/high: processor order, replacement count, positional slicing, all FA layers on both ranks, typed hybrid caches and request lifecycle. No DeepStack lowers scope. | Plausible header budget and proven language-shard loading, but new encoder/workspace residency, image-conditioned numerics, TTFT and useful screenshot detail unmeasured. |
| **3. Native MLX-LM server with VLM wrapper** | High and not presently compatible; new adapter/loader/server work plus loss of existing phased/streamed assumptions. | No demonstrated memory or simplicity advantage here. |

CPP source has a full-payload client serialization copy, server vector resize to message size, and a scoped SET_TENSOR receive vector; it is **not evidence of a persistent vector leak**. Chunking bounds a transient only if in-flight count and lifetime are also bounded. Existing v3 failure cannot be causally attributed to this allocation alone.[27:344–355,828–872,1698–1753,2600–2608][28]

### Proposed CPU-only gates, before any physical run (not executed here)

1. Pure processor fixtures: target local chat/tokenizer/config binding; 256/512 and non-square RGB/RGBA/grayscale inputs; exact patch reorder/temporal duplication/grid and repeated image-token count. Compare pinned reference outputs, not only self-consistency. Reject bad grids, image cardinality, over-budget dimensions/tokens, stale source hashes and double conversion.
2. Position fixtures: ordinary text equals scalar reference; asymmetric non-square image gives distinct t/h/w; multiple interleaved text/image segments; full positions equal sliced chunk positions at255/256/257 and final prompt token. Verify delta and two continuation steps. Poison row/column permutation and dropped axis/delta to ensure tests fail.
3. Pure control/cache tests: replay manifests with missing/duplicate/out-of-order chunks, wrong rank/image/position hash, stale delta, cursor mismatch, aborted request/reset, ragged partitions and JSON/payload caps. Assert full per-layer KV keys/values/offset and both GDN arrays; later snapshot/restore must include positional continuation semantics.
4. Loader invariants: exactly333 BF16 vision descriptors, all27 blocks, unchanged1,847 text names/hashes, correct convolution orientation, no double norm shift, no unintended global requantization. Reject individual missing tensors and malformed storage dtypes. A tiny deterministic quantized fixture can test adapter plumbing; it cannot admit Metal arithmetic.
5. CPP alternative: deterministic random byte tensor, partial offset writes including final ragged chunk, quant-block-safe boundaries, invalid ranges/overflow, exact readback digest; verify bounded queue count/memory and unchanged full-buffer result. Preserve whole-tensor cache identity or explicitly bypass its cache in the bounded control; do not claim chunk hashes preserve old cache semantics automatically.

### Proposed first physical small control, hardware owner only

Use the **already retained** `mlx-community/Qwen3.5-0.8B-4bit` revision `da28692b5f139cb0ec58a356b437486b7dac7462` solely as a same-family control, not target substitution. Its local config includes vision, patch16/temporal2/merge2 and no DeepStack, but smaller encoder dimensions: this cannot prove target-sized numerical dispatch/capacity.[29]

After fresh pressure/headroom admission: one request at a time, no service/MTP, one256px still image. Establish pinned single-host VLM reference on each physical GPU, then the isolated phased adapter, then two-host split with identical image/processor/weights. Compare preprocessing, merged embeddings, full logits, all live recurrent/KV states and continuation positions at unchanged predeclared tolerances; same short token IDs alone are insufficient. Include an asymmetric color/layout control and image swaps while keeping text fixed to prove actual visual dependence. Exercise image→text fresh reset, chunk boundary and cancellation/cleanup. Repeat at512 only after memory measurements pass. Measure TTFT, request wall, per-rank allocator and OS pressure separately.

For CPP staging, first run a fixed small exact-byte write/readback with recorded peak staging/queue limits, then rerun existing0.8B native/dual image controls. Neither path may proceed to27B because a mock test passed. Full target requires re-attested target/vision payloads, current normal pressure on both hosts, retained source snapshots, strict small/full numerical gates, bounded image cap, guarded loading and verified owned cleanup; then task-quality tests on readable real screenshots. MTP and continuous service/recovery qualification come later.

## Known / unknown

**Known:** exact architecture/config and all local header layouts; empty DeepStack; preserved BF16 encoder size; scalar-only current FA positional interface; explicit injection/control/loader changes required; native server's concrete incompatibilities; feasible-looking stored-parameter planning bounds.

**Unknown:** pinned VLM execution/dependency compatibility with this installed environment; physical visual numerical equivalence; current full vision payload hashes; encoder/driver peak residency and safe resolution with ordinary apps open; screenshot OCR/localization/task quality; image-conditioned MTP behavior and performance; whether bounded CPP staging restores full-target capacity. No throughput, full-vision readiness or autonomous-agent capability is claimed.

## Sources and reproducibility

References are line ranges in the fetched raw pinned sources or local files as inspected. Source ledger is kept in memory to honor the read-only scope; no separate cache/ledger was written. Primary-source paths/URLs and all exact digests follow. Local metadata identity hashes appear separately to make rebinding possible.

### Source ledger

- [1] `mlx_vlm/models/qwen3_5/qwen3_5.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_5/qwen3_5.py`
  SHA-256 `7c0e360a35e1f5d3819677cd6f7e4d7349f4f3dd2fad5fc8f1e374bd514e8f1b`
- [2] `mlx_vlm/models/qwen3_5/language.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_5/language.py`
  SHA-256 `6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7`
- [3] `mlx_vlm/models/qwen3_5/vision.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_5/vision.py`
  SHA-256 `a1db23800b564caaedcc79d6a27d76e3186b9315cb68b612d61afbfa69487600`
- [4] `mlx_vlm/models/qwen3_5/config.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_5/config.py`
  SHA-256 `b4db2144b162bbc27c0d843833fdd969413b44260ed1f024866eeadf1832dc0f`
- [5] `mlx_vlm/models/qwen3_vl/vision.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_vl/vision.py`
  SHA-256 `9885d83cf85e037a88f8364817007094a82b5941d49e909d2d9fe11b5e03bf6c`
- [6] `mlx_vlm/models/qwen3_vl/qwen3_vl.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_vl/qwen3_vl.py`
  SHA-256 `12f0393421918eb1cbaa6daa447d93a405762e997f688676688e5c71d987d3ee`
- [7] `mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py`
  SHA-256 `9c6bc521ea6b45c3346967502f2d0b69697371327d7b340c23650e76de91f5d2`
- [8] `mlx_vlm/models/rope_utils.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/rope_utils.py`
  SHA-256 `6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b`
- [9] `mlx_vlm/utils.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/utils.py`
  SHA-256 `2419b2c29adfecb95745914f73bc991a03ec7756d135bdc4ffaafdc0fe51ef72`
- [10] `mlx_vlm/models/base.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/base.py`
  SHA-256 `a7cfa38413bea057946c5d56c9edd34dbcceb33f6613529f5201ca2ad5b41120`
- [11] `mlx_vlm/models/qwen3_5/__init__.py` — `https://raw.githubusercontent.com/Blaizzy/mlx-vlm/b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0/mlx_vlm/models/qwen3_5/__init__.py`
  SHA-256 `d5a1cf335ae0845d0033dcc77dfdd6174f801aa77375cd8d57d69bc0c26f3d2b`
- [12] `/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/config.json`
  SHA-256 `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85`
- [13] `/Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/inventory.json`
  SHA-256 `d0996b2023760e7370b939be8509e3091dfc0c9b8b9357c6fc5c01b14ba81d25`
- [14] `/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-27b-dual-host-v1/preparation/large-download-verified.json`
  SHA-256 `3204653dd524511a889ba3e4175e6cc3e0120f24205c4fd8ae31c6119a06db49`
- [15] `/Users/jeremylien/Projects/paretoquant-metal/spikes/021-qwen38-prefix-rollback/phased_decode.py`
  SHA-256 `b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790`
- [16] `/Users/jeremylien/Projects/paretoquant-metal/spikes/024-qwen38-hidden-prefill/phased_decode.py`
  SHA-256 `97a056ee6db36ae51012c66be2fd305a069d882b928049dab708cd488461a555`
- [17] `/Users/jeremylien/Projects/paretoquant-metal/spikes/021-qwen38-prefix-rollback/stream_weights.py`
  SHA-256 `b3bc1687a12fd91eea7802402bba5fb854dcfa343431e64bb7104c7091c761f9`
- [18] `/Users/jeremylien/Projects/paretoquant-metal/spikes/021-qwen38-prefix-rollback/shard_plan.py`
  SHA-256 `231645bb2e7f90637a3f8ce0a543bbdb1d5d8800960870b1d3cfdaee9942d465`
- [19] `/Users/jeremylien/Projects/paretoquant-metal/spikes/021-qwen38-prefix-rollback/rank_control.py`
  SHA-256 `e522ed335ebd5775132a29044170cedb8d65eb057a316f37ed408d485b1771b5`
- [20] `/Users/jeremylien/Projects/paretoquant-metal/spikes/021-qwen38-prefix-rollback/distributed_control.py`
  SHA-256 `894403782594e8ffc7fb832f336fdd2600925dd0ba330403a701a8a47f695f4e`
- [21] `/Users/jeremylien/Projects/paretoquant-metal/spikes/021-qwen38-prefix-rollback/mlx_backend.py`
  SHA-256 `507c6be2abbd44d80debb9ff5dc7e46b25c349d85e1f0f3202796a6e2c22ed47`
- [22] `/Users/jeremylien/Projects/paretoquant-metal/.venv/lib/python3.13/site-packages/mlx_lm/models/qwen3_5.py`
  SHA-256 `10c59a71d8d739b8cdff38c7e9c7345deb9a9688e821ac37cb774d3086cac42e`
- [23] `/Users/jeremylien/Projects/paretoquant-metal/.venv/lib/python3.13/site-packages/mlx_lm/models/qwen3_next.py`
  SHA-256 `67ca08a4d08cb7be124965e95399735e556f1eaeb21a16bc23cb88d1ac035e11`
- [24] `/Users/jeremylien/Projects/paretoquant-metal/.venv/lib/python3.13/site-packages/mlx_lm/models/pipeline.py`
  SHA-256 `5897f414f550584442594c766fc91ca53d72a8eb1e242ffbc5bababcd7a4ae35`
- [25] `/Users/jeremylien/Projects/paretoquant-metal/.venv/lib/python3.13/site-packages/mlx_lm/utils.py`
  SHA-256 `d442babbfd891bfa61ec9c6feead9024e206a333a357c5e4a07554c4ea138965`
- [26] `/Users/jeremylien/Projects/paretoquant-metal/.venv/lib/python3.13/site-packages/mlx_lm/server.py`
  SHA-256 `57392576de114af80ede723976b9dd156dc234f856b9943b8b44b8b78bc27f79`
- [27] `/Users/jeremylien/Projects/local-agent-dual-mac/upstream/llama.cpp/ggml/src/ggml-rpc/ggml-rpc.cpp`
  SHA-256 `583cf1ee149c5b94c215eca07a293564d1b755427b57b1a7846f0328e724eacd`
- [28] `/Users/jeremylien/Projects/local-agent-dual-mac/results/full-target-load-v3/README.md`
  SHA-256 `34884673e483127956ef86bc8cf0565c726103ed2230d3a9eefbc26d8fd27b3d`
- [29] `/Users/jeremylien/.hermes/cache/scratch/paretoquant-qwen38-preflight/small-control/config.json`
  SHA-256 `ba7770da23eae5ebd6827571f086e331956b33f4442a9e876fb4aa10969a6772`
- [30] `/Users/jeremylien/Projects/paretoquant-metal/.venv/lib/python3.13/site-packages/mlx_lm/models/rope_utils.py`
  SHA-256 `06466d57dfe1ccc1e61ba595c0a8f2dd3a14bcb5c6a1d06c67591dc03128606b`

### Additional exact metadata identities

- `/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/model.safetensors.index.json` — SHA-256 `13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8`
- `/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/processor_config.json` — SHA-256 `222e060084f093738ca7af4c89ef09d844e415876021325fdf7d5c6821b119d4`
- `/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/preprocessor_config.json` — SHA-256 `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516`
- `/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/tokenizer_config.json` — SHA-256 `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b`

- `023-qwen38-workload-bench/phased_decode.py` SHA-256 equals021: `b54cd54f56f51480716e77a76bb947290719d8647d6326bceae87ccf50b9b790`.
