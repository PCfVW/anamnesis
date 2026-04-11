# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`dequantize_gguf_blocks_to_bf16`** — new streaming public API that
  emits one block's worth of `BF16` bytes per call into a caller-supplied
  `FnMut(&[u8]) -> Result<()>` sink closure (64 B per call for legacy
  quants, 512 B for K-quants). Peak heap is O(one scratch + one block
  output) regardless of tensor size — around 1.5 KB — enabling
  dequantisation of 70 B-parameter models on modest-RAM machines by
  streaming directly to disk. The existing `dequantize_gguf_to_bf16`
  `Vec`-returning variant is now a thin convenience wrapper that sinks
  into `Vec::with_capacity`, so both entry points share the same
  validation and the same scalar kernels.
- **`GGUF` block-quant dequantisation to `BF16`** (`dequantize_gguf_to_bf16`,
  feature-gated behind `gguf`) — scalar reference kernels for all 12 block
  types covered by the parser: legacy `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
  `Q8_1` (32-element blocks) and K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`,
  `Q6_K`, `Q8_K` (256-element super-blocks). Formulas ported verbatim from
  `ggml-quants.c`'s `dequantize_row_*` reference functions, with
  block-at-a-time loop fission (packed-bit unpacking into an `[f32; QK]`
  stack scratch buffer, then a branch-free `f32 × scale → BF16` pass via
  the shared `f32_bits_to_bf16_bits` helper). `IQ*`/`TQ*`/`MXFP4` return
  `AnamnesisError::Unsupported` (deferred to a later phase). No new crate
  dependencies — reuses `half` and the existing `gguf` feature. Phase 4
  step 2 toward v0.4.0; bit-for-bit cross-validation against `llama.cpp`
  is Phase 4 step 4.

### Changed

- **GGUF dequant kernels refactored to close over a `FnMut` sink**
  instead of each returning an owned `Vec<u8>`. Per-type pass-1 unpacking
  is now a small closure fed to a generic `run_legacy_kernel` /
  `run_super_kernel` outer-loop helper that handles `chunks_exact(TS)`
  iteration, scratch-buffer management, and the shared pass-2 `BF16`
  writer. `Vec::with_capacity` + `extend_from_slice` replaces the old
  `vec![0u8; n_elements * 2]`, avoiding the zero-init memset (~10–15 %
  of dequant wall time on `Q8_0`/`Q4_0` saved on platforms without lazy
  zero pages). Per-block `.get_mut(range).ok_or_else(...)?` bounds
  checks are replaced with `chunks_exact_mut`-style iteration on the
  inner kernel runners — ~4 M branches removed per 1 M-block tensor.
- **Infallible byte readers**: `read_f16_le(&[u8], usize) -> Result<f32>`
  is replaced with `read_f16_bytes([u8; 2]) -> f32` (and analogously for
  `read_f32_bytes`), eliminating dead `Result` shuffling on every
  hot-loop call. Callers slice fixed-length arrays out of their
  already-validated block slices.
- **Output size overflow guard**: `dequantize_gguf_to_bf16` now checks
  `n_elements.checked_mul(2)` in its shared validation helper, turning
  what would have been a silent `Vec` allocation truncation on 32-bit
  targets with > 2 GiB of `BF16` output into a clean `AnamnesisError::Parse`.
- **`GGUF` file parser** (`parse_gguf`, feature-gated behind `gguf`) — lean
  in-house parser for `GGUF` v2 and v3 files. Reads header, metadata
  key-value pairs (all 13 value types including nested `ARRAY`), and tensor
  info table. Resolves absolute tensor-data offsets from the tensor-info
  table's relative offsets plus the effective `general.alignment` (default
  32 bytes). `ParsedGguf::tensors` returns zero-copy `Cow::Borrowed` slices
  into the memory-mapped file for every dtype with a known `type_size`
  (`F32`, `F16`, `BF16`, `F64`, `I8`–`I64`, `Q4_0`–`Q8_1`, `Q2_K`–`Q8_K`).
  `IQ*`/`TQ*`/`MXFP4` tensors are listed in `tensor_info()` with
  `byte_len = None` and will be sized when dequantisation lands. No new
  third-party crate — reuses the `memmap2` dependency already pulled in by
  the `pth` feature. First commit of Phase 4 toward v0.4.0.
- **`GgufMetadataArray`** — new `#[non_exhaustive]` enum holding natively
  typed arrays (`Vec<u8>`, `Vec<f32>`, `Vec<String>`, …). Replaces the old
  `GgufMetadataValue::Array(Vec<GgufMetadataValue>)` storage to eliminate
  the ~8× enum-discriminant bloat on homogeneous numeric metadata arrays.
- **`ParsedGguf::tensors` returns `impl Iterator<Item = GgufTensor<'_>>`**
  instead of `Result<Vec<GgufTensor<'_>>>` — zero heap allocation per call.
  `GgufTensor::{name, shape}` now borrow from the parsed handle as
  `&'a str` / `&'a [usize]` rather than cloning owned `String`/`Vec`.
  Worst-case allocation dropped from ~130 MB per call on a 1 M-tensor file
  to 0 MB. Callers needing random access should use `.collect::<Vec<_>>()`.
- **`GgufMetadataValue::Array` now holds `Box<GgufMetadataArray>`** (was
  `Vec<GgufMetadataValue>`). `GgufMetadataValue::as_array` now returns
  `Option<&GgufMetadataArray>`. As a side effect, `GgufMetadataValue`
  shrinks from 32 bytes to 24 bytes (25% reduction) across every metadata
  value, not just arrays, because the max-sized variant is now `String`.

### Security / Performance

- **Cap trust-the-header pre-allocation** at `PREALLOC_SOFT_CAP = 256`
  entries for every `Vec::with_capacity` / `HashMap::with_capacity` call
  keyed on a file-declared count (metadata kv count, tensor count,
  per-array length). Previously a ~40-byte adversarial header claiming
  1 M of each could force ~175 MB of eager heap allocation before a single
  entry was read (empirically measured: 114 MB `HashMap` + 61 MB
  `Vec<RawTensorInfo>`); the cap drops this to ~34 KB (5 000× reduction).
  An adversarial `ARRAY` header claiming 16 M `f32` elements forced
  ~488 MB of eager allocation; combined with the typed-array fix, this is
  now capped at ~8 KB (60 000× reduction). Legitimate files grow the
  containers geometrically and are unaffected.
- **`Cursor::read_string` validates UTF-8 on the borrowed mmap slice
  before copying** — an adversarial 16 MiB non-UTF-8 string now costs
  zero heap allocation on the rejection path (was: a full 16 MiB
  `to_vec()` followed by `String::from_utf8`).
- **`ParsedGguf::inspect` dedups distinct dtypes via a `[bool; 32]`
  bitmap** keyed on a dense `GgufType` discriminant, replacing
  `Vec::contains` in the per-tensor loop. Drops the dtype-dedup hot path
  from O(n × d) to O(n) — ~10 ms → ~1 ms on a 1 M-tensor inspect call.
  First-occurrence order of `GgufInspectInfo::dtypes` is preserved.
- **`parse_gguf` builds `tensor_infos` in a single pass** instead of
  first materialising a throwaway `Vec<RawTensorInfo>` and then iterating
  it. The relative tensor-data offset is stored in `data_offset` during
  the read pass and patched to the absolute offset in a short sweep once
  `data_section_start` is known. Peak tensor-info heap on a 1 M-tensor
  file drops by ~60 MB; `RawTensorInfo` and `read_raw_tensor_info` are
  deleted.

### Fixed

- **`parse_gguf` accepted tensors whose relative offset was not a
  multiple of `general.alignment`** (Phase 4 audit I1). The GGUF spec
  mandates that every tensor's offset field is a multiple of the file's
  declared alignment, but the patch sweep only checked the upper bound
  of each tensor's byte range. A malformed file encoding
  `relative_offset = 1` for every tensor would parse successfully and
  hand out unaligned byte slices through `ParsedGguf::tensors`, which
  downstream SIMD dequant kernels would then reinterpret as `f32`/`f16`
  words — unaligned access is undefined behaviour in the `unsafe`
  intrinsics planned for Phase 9. `parse_gguf` now rejects such files
  with `AnamnesisError::Parse` naming the offending tensor.

## [0.3.2] - 2026-04-05

### Fixed

- **`copy_to_contiguous` silent data corruption on mismatched shape/strides** (NI1) —
  added ndim guard rejecting `shape.len() != strides.len()`. Previously, `.zip()`
  silently truncated to the shorter iterator, producing corrupted output. Defence-in-depth
  check also added in `parse_rebuild_args`
- **`copy_to_contiguous` inner loop used `.get()` despite `// INDEX:` annotation** (NI2) —
  switched to direct indexing `storage[range]`, matching CONVENTIONS.md and the
  pre-validation that proves bounds safety. Eliminates dead `.ok_or_else()` branches
- **NPZ `extract_descr` mixed-quote bug** — quote character detection now reads the
  first quote in the value portion, not the entire header tail. Fixes silent
  mis-extraction for mixed-quote headers like `{'descr': "<f4", ...}`
- **`parse_pth` stale return-type doc** (D1) — claimed "Returns `Vec<PthTensor>`" but
  actually returns `Result<ParsedPth>`. Updated to reference `ParsedPth` and
  `ParsedPth::tensors()`
- **`inspect_npz` overflow saturation undocumented** (D2) — added note explaining
  `byte_len` saturates to `usize::MAX` on shape overflow (best-effort metadata),
  unlike `parse_npz` which returns `Err`
- **Misused `// EXPLICIT:` in `build_entry_index`** (D3) — was a char-boundary
  assertion, not a no-op arm or stateful loop. Downgraded to plain comment
- **Per-line `// BORROW:` annotations** — replaced single block-level annotation in
  `execute()` with per-call annotations on all 12 `.to_owned()`/`.to_vec()` sites.
  Added missing annotations in `build_entry_index`
- **`// VECTORIZED:` on `copy_to_contiguous`** — added `scalar fallback` annotation
  documenting why the inner loop cannot auto-vectorize (cross-iteration coords state)
- **`# Memory` section on `ParsedPth::tensors`** — documents zero-copy vs owned
  allocation paths
- **`const fn` on `ParsedPth::len`/`is_empty`** — `Vec::len()`/`is_empty()` are
  `const fn` since Rust 1.39
- **`# Errors` sections** on `parse_rebuild_args`, `build_entry_index`, and
  `copy_to_contiguous` — consistency with other private fallible functions
- **`lib.rs` architecture doc** (D6) — added `pth_to_safetensors()` to bullet list
- **`NpzDtype` `Display` doc** (D7) — documented as canonical uppercase string used
  in inspection output and cross-validation tests
- **`parse/mod.rs` stale docstring** — updated from "wraps `npyz`" to reflect own parser
- **`byteswap_inplace` missing `// VECTORIZED:` annotation** — added per CONVENTIONS.md
- **NPZ annotations** — `// EXPLICIT:` for `=` native-endian prefix, `// EXPLICIT:`
  for `extract_fortran_order` default
- **`bench_npz_adhoc.rs` hardcoded path** — replaced with `dirs::home_dir()` fallback

### Added

- **48 new unit tests** covering code review findings G1–G36, NI1–NI2, NN1–NN4:
  pickle VM opcodes (FRAME, NONE, NEWTRUE/NEWFALSE, BININT, BININT2, BINUNICODE,
  SHORT_BINSTRING, BINSTRING, SHORT_BINBYTES, BINBYTES, EMPTY_LIST, EMPTY_TUPLE,
  TUPLE1, TUPLE3, SETITEMS, APPEND, APPENDS, STACK_GLOBAL, REDUCE, NEWOBJ, BUILD,
  BINPERSID, LONG_BINPUT/LONG_BINGET, MEMOIZE), `long1_to_i64` 8-byte boundary,
  `MEMOIZE` overflow at `u32::MAX`, `MAX_PICKLE_NESTING` enforcement (both
  `unwrap_to_rebuild` and `extract_dict_pairs`), `copy_to_contiguous` (transposed,
  zero-element, overflow, zero-stride broadcast, ndim mismatch, storage boundary),
  missing/compressed `data.pkl` ZIP entries, zero-length ZIP entry,
  NPZ Fortran-order end-to-end rejection, empty NPZ archive (parse + inspect),
  native-endian `=` prefix, big-endian through `parse_npz`, `inspect_npz` overflow

## [0.3.1] - 2026-04-02

### Added

- **PyTorch `.pth` parsing** (`src/parse/pth.rs`) — minimal pickle
  interpreter (~36 opcodes) that parses `PyTorch` ≥ 1.6 `state_dict` ZIP
  archives with a safe, explicit `GLOBAL` allowlist (rejects non-`torch.*`
  callables — equivalent to `weights_only=True` but stricter). Zero-copy
  I/O via `memmap2` with `Cow::Borrowed` tensor data sliced directly from
  the mmap. Handles shared storage, non-contiguous strides, big-endian
  byte order, and both newer (`archive/`) and older (`{model_name}/`)
  `PyTorch` ZIP prefix conventions. Feature-gated behind `pth`. Supports
  `F16`, `BF16`, `F32`, `F64`, `I8`–`I64`, `U8`, `Bool` storage types.
  **11–31× faster** than `torch.load()` on torchvision models (resnet18,
  resnet50, ViT-B/16)
- **`.pth` → safetensors conversion** (`src/remember/pth.rs`) — lossless
  format conversion preserving original dtypes (no dequantization). The
  conversion pipeline writes directly from mmap slices to the output file
  — zero intermediate data copies. Byte-exact roundtrip verified against
  `PyTorch` reference on all 3 test models
- **`.pth` cross-validation** against `PyTorch` on 3 real
  [AlgZoo](https://github.com/alignment-research-center/alg-zoo) models
  (MIT-0 license): `2nd_argmax_2_2` RNN (10 params), `longest_cycle_2_3`
  Transformer (50 params), `one_layer_16_hidden` RNN blog example (432
  params). Byte-exact match on all tensors against `PyTorch` reference
- **CLI `.pth` support** — `amn parse`, `amn inspect`, and `amn remember`
  now accept `.pth`, `.pt`, and `.bin` files when built with
  `--features pth`. Format detection by extension with ZIP magic fallback
  for `.bin` files. `amn remember model.pth` converts to safetensors;
  `amn parse model.pth` shows per-tensor details (name, dtype, shape,
  size)
- **`ParsedPth`** container — owns the mmap, provides zero-copy
  `tensors()`, `inspect()` → `PthInspectInfo`, `tensor_info()` →
  `PthTensorInfo` (metadata only, no data access), and
  `to_safetensors()` convenience method
- **`PthInspectInfo`** — summary struct (tensor count, total bytes,
  dtypes, byte order) with `Display` impl
- **`PthTensorInfo`** — lightweight per-tensor metadata (name, shape,
  dtype, `byte_len`) for display paths that don't need tensor data
- **`PthDtype::to_safetensors_dtype()`** — direct single-hop conversion
  to `safetensors::Dtype`, bypassing the intermediate anamnesis `Dtype`
- **`inspect_npz()`** (`src/parse/npz.rs`) — header-only `NPZ` inspection
  that reads only `NPY` headers (~128 bytes per array), no tensor data.
  Returns `NpzInspectInfo` + `NpzTensorInfo` (name, shape, dtype,
  `byte_len`). For a 300 MB file, uses kilobytes instead of 300 MB
- **CLI `.npz` support** — `amn parse` and `amn inspect` now accept
  `.npz` files when built with `--features npz`, using the header-only
  `inspect_npz` path. `amn remember` for `.npz` returns a clear
  unsupported error (tensors are already full-precision)

### Changed

- Extracted `byteswap_inplace` from `src/parse/npz.rs` to shared
  `src/parse/utils.rs` module (`pub(crate)`) so that multiple format
  parsers (`NPZ`, `.pth`) can reuse it without duplication
- Widened `From<ZipError>` impl in `error.rs` from `npz`-only to
  `any(npz, pth)` feature gate
- Changed `unsafe_code` lint from `forbid` to `deny` to allow
  feature-gated `memmap2` usage in the `pth` module (with `// SAFETY:`
  annotation)

### Fixed

- **`has_zip_magic`** now reads only 4 bytes via `read_exact` instead of
  loading the entire file into heap (prevented 7 GB allocation on large
  `.bin` files)
- **`build_entry_index`** now returns `AnamnesisError::Parse` for corrupt
  ZIP entries whose data range exceeds the file size, instead of silently
  skipping them
- **`extract_dict_pairs`** unreachable `Reduced{OrderedDict}` branch now
  returns `Err` instead of `Ok(&[])`, preventing silent data loss
- **MEMOIZE** opcode uses `checked_add(1)` instead of plain `+= 1`,
  preventing silent `u32` wraparound on adversarial pickles
- **`build_entry_index`** `u64`→`usize` casts replaced with `TryFrom`,
  consistent with codebase conventions (no truncation on 32-bit)
- **`inspect()`** element count uses `saturating_mul` instead of
  `checked_mul().unwrap_or(0)`, avoiding silently wrong `total_bytes`
- **`--to`** argument for `.pth` files is now validated: accepts
  `safetensors` or `bf16`, errors on unsupported values
- **`copy_to_contiguous`** uses the two-level bounds pattern from
  `CONVENTIONS.md` — pre-validates max source offset once before the
  loop, removing 6 per-element `checked_*` calls

### New dependencies

- `memmap2` v0.9 (optional, `pth` feature only) — memory-mapped file I/O

## [0.3.0] - 2026-03-24

### Added

- **NPZ/NPY parsing** (`src/parse/npz.rs`) — `parse_npz(path)` reads `NumPy`
  `.npz` archives into framework-agnostic `NpzTensor` structs. Custom `NPY`
  header parser with bulk `read_exact` data extraction — zero per-element
  deserialization for LE data on LE machines. Feature-gated behind `npz`.
  Supports `F16`/`F32`/`F64`, all integer types, `Bool`, and `BF16` (JAX `V2`
  void dtype). **3,586 MB/s** on 302 MB Gemma Scope file (1.3× raw I/O
  overhead), **17.7× faster** than `npyz`-backed parser
- **NPZ cross-validation** against Gemma Scope 2B SAE weights (`params.npz`,
  5 `F32` arrays). Byte-exact match against `NumPy` reference on all arrays

### Fixed

- **BnB4 output shape recovery** — `NF4`/`FP4` dequantized weights now have their
  original 2D shape (e.g., `[2048, 8192]`) instead of flat `[total_elements]`.
  Shape is recovered from the `quant_state.bitsandbytes__nf4`/`__fp4` companion
  tensor's `JSON` blob, which is stored inside the safetensors file itself (no
  `config.json` needed). Falls back to 1D if the companion is absent
- **`extract_descr` mixed-quote header bug** — quote character detection now reads
  the first character of the value, not the entire header tail. Fixes silent
  mis-extraction for mixed-quote headers (e.g., `{'descr': "<f4", ...}`)
- Added mandatory `// VECTORIZED:`, `// EXPLICIT:` annotations on
  `byteswap_inplace`, `extract_fortran_order`, native-endian `=` treatment,
  and unused minor version byte
- Removed hardcoded absolute path from `bench_npz_adhoc.rs` — now resolves
  from `USERPROFILE`/`HOME` environment variables
- Added missing `[0.2.0]` changelog section for Phase 2 (GPTQ, AWQ, BnB)

## [0.2.0] - 2026-03-24

### Added

- **GPTQ dequantization** (`src/remember/gptq.rs`) — INT4 and INT8 with
  group-wise scale + zero-point, activation-order via `g_idx`. Feature-gated
  behind `gptq`. Bit-exact against PyTorch on 4 real models from 2 quantizers
  (AutoGPTQ, GPTQModel), **6.5–12.2× faster** than CPU PyTorch (AVX2)
- **GPTQ parsing layer** — `TensorRole::ZeroPoint` / `GroupIndex`,
  `QuantScheme::Gptq`, `GptqConfig` (bits + group_size inference from metadata
  or tensor shapes), `find_gptq_companions()`, `gptq` feature gate
- **GPTQ cross-validation** against PyTorch on 4 models: Falcon3-1B INT4/INT8
  (AutoGPTQ), Llama-3.2-1B INT4 (AutoGPTQ), Llama-3.2-1B INT8 (GPTQModel)
- **GPTQ inspect/CLI** — zero-point, group-index counts in `inspect` and
  `parse` output; format-aware size label (FP8/GPTQ/unquantized)
- **AWQ dequantization** (`src/remember/awq.rs`) — INT4 (and INT8 path,
  unit-tested) with per-group scales, no +1 zero-point offset. Feature-gated
  behind `awq`. Bit-exact against PyTorch on 2 real models (AutoAWQ GEMM),
  **4.7–5.7× faster** than CPU PyTorch (AVX2). Loop fission applied from the
  start; full AVX2 `vsubps`/`vmulps` ymm confirmed
- **AWQ parsing layer** — `QuantScheme::Awq`, `AwqConfig`, `AwqCompanions`,
  shape-based detection distinguishing AWQ (packed along cols) from GPTQ
  (packed along rows), `awq` feature gate
- **AWQ cross-validation** against PyTorch on 2 models: Llama-3.2-1B and
  Falcon3-1B (both AutoAWQ GEMM, 4-bit). Note: no real 8-bit AWQ models
  exist in the standard AutoAWQ `.qweight` format — all "8-bit AWQ" models
  on HuggingFace are either dequantized F16, `compressed-tensors` (vLLM), or
  mislabeled 4-bit
- **BitsAndBytes dequantization** (`src/remember/bnb.rs`) — NF4, FP4 (both
  4-bit lookup-table with per-block absmax), double-quant NF4/FP4 (nested
  absmax), and INT8 (`LLM.int8()` with per-row absmax). Feature-gated behind
  `bnb`. Bit-exact against PyTorch on 4 real models, **18–54× faster** for
  NF4/FP4 (AVX2), **1.2× faster** for INT8 (near memory bandwidth limit).
  Loop fission for NF4/FP4; single-pass AVX2 for INT8 (`vpmovsxbd` →
  `vcvtdq2ps` → `vmulps`)
- **BnB parsing layer** — `QuantScheme::Bnb4` / `BnbInt8`,
  `TensorRole::QuantMap` / `NestedScale`, `BnbConfig` (block_size,
  double_quant), `Bnb4Companions`, detection by `.weight.quant_map` (NF4/FP4)
  and `.SCB` (INT8) naming patterns, `bnb` feature gate
- **BnB cross-validation** against PyTorch on 4 models: Llama-3.2-1B NF4,
  Llama-3.2-1B NF4 double-quant, Llama-3.2-1B FP4, Llama-3.2-1B INT8
- **BnB model.rs integration** — `Bnb4` and `BnbInt8` arms in
  `remember_bf16_inner` with companion lookup, double-quant detection, and
  shape handling (flat output for NF4/FP4, preserved 2D for INT8)
- `FromStr` impl for `TargetDtype` — centralizes string-to-enum parsing so new
  variants cannot be silently missed in the CLI
- `ParsedModel::remember_with_progress()` — dequantize with a per-tensor
  callback, enabling progress reporting in CLI contexts
- `indicatif` progress bar during `remember`/`dequantize` when built with the
  `indicatif` feature (`amn remember` shows `[====================] 2.1s`)
- CONVENTIONS.md: two-level bounds checking pattern (reconciles `// INDEX:`
  safety with SIMD rule #2) and loop fission for mixed-domain pipelines

### Fixed

- Added unit tests for `dequantize_per_channel_fp8_to_bf16` covering F32, BF16,
  and F16 scale dtypes, single-row, NaN handling, and validation errors
- Added fine-grained dequantization tests for all three scale dtypes (F32, BF16,
  F16) and multi-block F32 scale path
- Added CLI integration tests (`tests/cli.rs`) — 9 tests covering `parse`,
  `inspect`/`info`, `remember`/`dequantize`, error handling, and `--version`
- Documented single-scheme assumption in `detect_scheme` (all quantized tensors
  in a file use the same scheme; early-return on first scale companion found)
- CLI `parse` subcommand now displays the actual scale dtype (BF16, F16, F32)
  instead of always printing "F32"
- `inspect` Display now shows the actual scale dtype instead of hardcoded "F32"
- `dequantize_per_tensor_fp8_to_bf16` now uses `checked_mul` for output size,
  consistent with the other two dequantize functions (**breaking**: returns
  `Result<Vec<u8>>` instead of `Vec<u8>`)
- Fine-grained dequantization now validates that the scale grid is rectangular
  (rejects `scale_elements % scale_rows != 0` instead of silently truncating)
- `serialize_to_file` I/O errors now surface as `AnamnesisError::Io` instead of
  being misclassified as `AnamnesisError::Parse`
- `derive_output_path` now matches `TargetDtype` exhaustively instead of using a
  wildcard that would silently produce broken paths for future variants
- Simplified `shape_to_rows_cols` 2D arm: direct indexing with `// INDEX:`
  annotation instead of redundant `Option` unwrapping
- **`classify_tensor` AWQ-only builds** — `.qweight`/`.qzeros`/`.scales` were
  gated on the `gptq` feature only; AWQ-only builds silently misclassified all
  quantized tensors as passthrough. Now gated on `any(gptq, awq)` with `.g_idx`
  remaining `gptq`-only
- **`detect_scheme` silent fallthrough** — `return` statements inside the
  GPTQ/AWQ detection block were feature-gated, causing misdetection when only
  one scheme was enabled. Detection is now unconditional; feature-disabled
  errors are handled downstream in `model.rs`
- `derive_output_path` now strips GPTQ, AWQ, and BitsAndBytes suffixes
  (e.g., `-GPTQ-Int4`, `-awq`, `-bnb-4bit`) in addition to FP8 suffixes
- `read_scale_f32` now uses `checked_add` for all byte offset computations,
  consistent with `read_u32_le` in the same codebase
- Extracted duplicated `read_u32_le` and `read_scale_f32` from `gptq.rs` and
  `awq.rs` into shared `remember/quant_utils.rs` module
- Replaced dead-code `checked_mul(1)` in `bnb.rs` with direct parity check
- GPTQ/AWQ outer loop offsets (`i * out_features * 2`) now use `checked_mul`
  for consistency with the codebase's zero-panic discipline
- `parse_g_idx` offset (`i * 4`) now uses `checked_mul`
- Updated GPTQ docstring memory estimate from "~1 MB" to "up to ~8 MB per
  weight tensor" to reflect fine-grained group configurations

## [0.1.0] - 2026-03-24

### Added

- **Safetensors parsing** (`src/parse/safetensors.rs`) — header parsing, tensor
  metadata extraction, dtype classification, tensor role classification (quantized,
  scale, passthrough), quantization scheme detection by scale tensor shape
- **Three FP8 dequantization schemes** (`src/remember/fp8.rs`):
  - **Fine-grained** — 128×128 block scale factors (`dequantize_fp8_to_bf16`)
  - **Per-tensor** — single scalar scale (`dequantize_per_tensor_fp8_to_bf16`)
  - **Per-channel** — one scale per output row (`dequantize_per_channel_fp8_to_bf16`)
- **Three scale dtypes** — `F32`, `BF16`, and `F16` scale tensors all supported
- **Branchless E4M3 → BF16 pipeline** — const subnormal lookup table, bitwise NaN
  select, round-to-nearest-even; auto-vectorized to SSE2 (default) and AVX2
  (`target-cpu=native`), verified with `cargo-show-asm`
- **Inspect module** (`src/inspect.rs`) — format, tensor counts, current/dequantized
  size estimates, Lethe distance
- **Parse-first public API** (`src/model.rs`) — `parse(path)` → `ParsedModel` →
  `.inspect()` / `.remember(path, target)`
- **CLI binary** (`src/bin/main.rs`) — subcommands: `parse`, `inspect`/`info`,
  `remember`/`dequantize`. Installed as both `anamnesis` and `amn`. Feature-gated
  behind `cli`
- **Cross-validation against PyTorch** — 7 fixtures from real models, bit-exact
  match (0 ULP on 65,536 elements each), 2.7–9.7× faster than PyTorch (AVX2)
- **Validated against 7 real models** from 5 quantization tools (LG AI, Qwen,
  Mistral, RedHat, NVIDIA)
