# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **PyTorch `.pth` pickle VM + tensor extraction** (`src/parse/pth.rs`) —
  minimal pickle interpreter (~36 opcodes) that parses `PyTorch` ≥ 1.6
  state_dict ZIP archives. Explicit `GLOBAL` allowlist rejects
  non-`torch.*` callables (security boundary equivalent to
  `weights_only=True`). Handles shared storage, non-contiguous strides,
  and big-endian byte order. Dynamic ZIP prefix discovery supports both
  newer (`archive/`) and older (`{model_name}/`) `PyTorch` formats.
  Feature-gated behind `pth`. Supports `F16`, `BF16`, `F32`, `F64`,
  `I8`–`I64`, `U8`, `Bool` storage types
- **`.pth` cross-validation** against `PyTorch` on 3 real
  [AlgZoo](https://github.com/alignment-research-center/alg-zoo) models
  (MIT-0 license): `2nd_argmax_2_2` RNN (10 params), `longest_cycle_2_3`
  Transformer (50 params), `one_layer_16_hidden` RNN blog example (432
  params). Byte-exact match on all tensors against `PyTorch` reference

### Changed

- Extracted `byteswap_inplace` from `src/parse/npz.rs` to shared
  `src/parse/utils.rs` module (`pub(crate)`) so that multiple format
  parsers (`NPZ`, `.pth`) can reuse it without duplication
- Widened `From<ZipError>` impl in `error.rs` from `npz`-only to
  `any(npz, pth)` feature gate

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
