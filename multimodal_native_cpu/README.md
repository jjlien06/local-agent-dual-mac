# Source-bound native MLX CPU rotary reference

`rotate_reference_cpu(q_numpy, k_numpy, position_ids_numpy, config=multimodal_rotary.TARGET)`
returns a tuple of **NumPy Q and K arrays produced by the actual pinned upstream
MLX eager CPU implementation**. This is an offline reference, not a model adapter.
It neither calls `rotate_numpy` nor implements replacement rotary arithmetic.

## Admission and execution

Each call, before importing MLX or executing upstream code:

1. Checks the exact immutable `TargetRotaryConfig` class and every metadata field
   against the pinned target (including exact Python field/tuple-element types).
2. Requires exact native NumPy `ndarray` inputs: Q `[1,24,T,256]`, K `[1,4,T,256]`,
   matching float16/float32 dtypes and token count, `1 <= T <= 256`. Q/K must be
   finite. Positions must be exact native int32 `[3,1,T]`, with values `0..1055`.
3. Reads and SHA-256 verifies the actual metadata-only target config and both
   complete upstream source files; verifies the upstream checkout HEAD.
4. Requires installed versions `mlx==0.32.4`, `mlx-metal==0.32.4`, `numpy==2.5.3`.
   Nothing is installed, downloaded, or loaded from a model checkpoint.

It then imports `mlx.core` lazily and checks the installed CPU APIs. It uses the
real `mx.cpu`; it never fabricates a device or falls back to GPU. Before upstream
module execution, class construction, array construction, and evaluation, it
establishes the CPU default. It executes the **entire verified `rope_utils.py`
bytes**, sets the private module's `_HAS_METAL = False` before constructing the
rotary object, and executes only the unmodified `Qwen3_5RotaryEmbedding` class and
`apply_multimodal_rotary_pos_emb` function AST nodes from verified `language.py`.
The rest of the language/model module is not executed or imported.

Constructor arguments come from the actual verified config (`text_config` and
`rope_parameters`): rotary dimension 64 of head dimension 256, theta 10000000,
sections `[11,11,10]`, maximum positions 262144, interleaved half-split pairing.
`rotary.apply_rotary(...)` supplies the returned outputs. Private module namespaces
are fresh on every call; neither upstream source nor shared upstream modules are
modified. Source bytes are verified once per call and those exact bytes are
compiled, avoiding a second unverified source read.

The original default device is restored in `finally`, including constructor,
evaluation, and output-conversion failures. Allocator/cache limits are **not
changed**. NumPy input views are copied in C order before transport: MLX rejects
negative DLPack strides, and `ascontiguousarray` alone can retain a negative stride
on a singleton axis. Inputs remain unmodified; trailing 192 dimensions remain
byte-exact, including signed zeros and subnormals. The bridge does not promise
finite outputs for arbitrary finite values large enough to overflow native FP16.
Use this bounded reference in isolation, not concurrently with a live model or
other code changing MLX default device/stream state.

## Source identity

- Upstream checkout: `../upstream/mlx-vlm`, HEAD
  `b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0`.
- `mlx_vlm/models/rope_utils.py` SHA-256:
  `6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b`.
- `mlx_vlm/models/qwen3_5/language.py` SHA-256:
  `6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7`.
- Target: `mlx-community/Qwen3.8-27B-4bit`, revision
  `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.
- Actual metadata-only config:
  `/Users/jeremylien/Projects/paretoquant-metal/models/qwen3.8-27b-4bit-tokenizer-only/config.json`.
  SHA-256: `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85`.

Paths are intentionally bound to this workspace and its existing metadata-only
config. Missing or changed metadata/source/runtime is a rejection, not permission
to retrieve another copy or weaken the binding.

## Reproduce (workspace root, existing isolated environment only)

```sh
PYTHONDONTWRITEBYTECODE=1 .vision-venv/bin/python -B -m unittest discover -s tests -p test_multimodal_native_cpu.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .vision-venv/bin/python -B tests/test_multimodal_native_cpu.py --probe
```

The unittest suite uses no pytest or extra dependencies. The probe compares the
bridge with separately loaded, independently invoked native upstream class and
language-wrapper calls. It uses explicit nonsquare 4x6, 6x4, and 16x32 positions,
first decode positions, text positions 0/17/1023/1055, Q/K, and float16/float32:
28 comparison rows at unchanged `atol=1e-6`, `rtol=1e-5`, plus 24 role-specific
strict-tolerance negative controls for t-only and h/w-swapped positions. Separate
unittests cover T=1/256, read-only/negative-stride arrays, tail bits, lazy imports,
source/config/runtime/API rejection, and failure restoration. Native evaluations
are observed under CPU defaults; `mx.fast.metal_kernel`, allocator setters,
network sockets, and `rotate_numpy` are forbidden during the bounded probe.

Durable commands, RED/GREEN outputs (including unsuccessful fix attempts),
source/artifact hashes, native comparison rows, actual counts, imports, device
observations, and limits are recorded in
`../results/native-rotary-bridge-implementation-v1.json`.

## Scope and non-admission

A passing native-reference result is **not NumPy/native parity or Metal parity**.
The existing `results/native-rotary-reference-v1/result.json` remains a numerical
rejection: five original float32 NumPy/native rows failed its original tolerance.
This bridge does not edit that failure, adjust tolerances, or invalidate the
separate NumPy-only AST review. No fused Metal kernel, GPU timing, BF16 path,
attention, cache adapter, vision encoder, checkpoint/model loading, logits,
full-model capacity, screenshot approval, service, or integration is qualified.
