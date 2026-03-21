# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
