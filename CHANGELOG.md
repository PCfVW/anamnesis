# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Crate scaffold** — `Cargo.toml` with metadata, dual license (MIT OR Apache-2.0),
  feature gates (`cli`, `npz`, `indicatif`), two `[[bin]]` targets (`anamnesis` and
  `amn`), `#![forbid(unsafe_code)]`, `#![deny(warnings)]`
- **CI/CD** — `.github/workflows/ci.yml` with MSRV (1.88) + stable matrix,
  `.github/workflows/publish.yml` with tag-triggered `cargo publish`
- **Safetensors parsing foundation** (`src/parse/safetensors.rs`) — header parsing,
  tensor metadata extraction, dtype classification (`F8_E4M3`, `BF16`, `F32`, etc.),
  tensor role classification (quantized, scale, passthrough), quantization scheme
  detection (fine-grained FP8, per-tensor FP8, unquantized), scale factor lookup
- **Inspect module** (`src/inspect.rs`) — `InspectInfo` struct with format, tensor
  counts, current/dequantized size estimates, Lethe distance; `Display` impl matching
  the flagship CLI output
- **Fine-grained FP8 dequantization** (`src/remember/fp8.rs`) — branchless E4M3 → f32
  → BF16 pipeline with const subnormal lookup table, bitwise select for NaN/subnormal,
  round-to-nearest-even f32 → BF16, block iteration with `chunks_exact(128)` and
  hoisted scale factors; `dequantize_fp8_to_bf16()` public API; exhaustive
  cross-validation against the `float8` crate
- **Per-tensor FP8 dequantization** (`src/remember/fp8.rs`) — single scale factor
  per tensor; `dequantize_per_tensor_fp8_to_bf16()` public API; reuses the same
  branchless E4M3 → BF16 pipeline
- **Parse-first public API** (`src/model.rs`) — `parse(path)` reads a `.safetensors`
  file and returns a `ParsedModel` holding header metadata + byte data.
  `ParsedModel::inspect()` returns format info. `ParsedModel::remember(path, target)`
  dequantizes all quantized tensors and writes a standard `.safetensors` file.
  `TargetDtype` enum (`BF16`). Round-trip tested with synthetic FP8 safetensors files.
- **CLI binary** (`src/bin/main.rs`) — `clap`-based CLI with subcommands: `parse`
  (tensor summary), `inspect` / `info` (format, sizes, Lethe distance), `remember` /
  `dequantize` (FP8 → BF16 conversion with auto-derived output path). Installed as
  both `anamnesis` and `amn`. Feature-gated behind `cli`.
- **`format_bytes`** made public for reuse by CLI and downstream consumers
- **README.md** with badges (CI, crates.io, docs.rs, MSRV), motto, dev warning
- **SPDX license identifiers** on all `.rs` files
