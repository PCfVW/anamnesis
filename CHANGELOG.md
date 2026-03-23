# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.1.0] — FP8 Dequantization

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
