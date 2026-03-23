# Tests

## Integration Tests

| File | Feature gate | What it tests |
|------|-------------|---------------|
| `cli.rs` | `cli` | CLI binary (`anamnesis`/`amn`): subcommand routing, argument parsing, output format, error messages, `--version` |
| `cross_validation.rs` | — | FP8 dequantization: bit-exact comparison against PyTorch on 7 real models (3 FP8 schemes × 3 scale dtypes) |
| `cross_validation_gptq.rs` | `gptq` | GPTQ dequantization: bit-exact comparison against PyTorch on 4 real models (2 quantizers × 2 bit widths) |
| `cross_validation_awq.rs` | `awq` | AWQ dequantization: bit-exact comparison against PyTorch on 2 real models (AutoAWQ GEMM, 4-bit) |
| `cross_validation_bnb.rs` | `bnb` | BitsAndBytes dequantization: bit-exact comparison against PyTorch on 4 real models (NF4, FP4, double-quant NF4, INT8) |

## Fixtures

Each `fixtures/<scheme>_reference/` directory contains:

- `generate_<scheme>.py` — Python script that extracts a small slice from a real quantized model, dequantizes with PyTorch/bitsandbytes, and writes a binary fixture file
- `*.bin` — Binary fixture files containing packed input tensors + expected BF16 output

The fixture format is scheme-specific (documented in each generator script). All fixtures are committed to the repo so that `cargo test` works without downloading models or running Python.

| Directory | Models | Slice size | Generator |
|-----------|--------|-----------|-----------|
| `fp8_reference/` | 7 models (EXAONE, Qwen3, Llama, Ministral) | 256x256 | `generate_fp8.py` |
| `gptq_reference/` | 4 models (Falcon3, Llama-3.2) | 256x256 | `generate_gptq.py` |
| `awq_reference/` | 2 models (Llama-3.2, Falcon3) | 256x256 | `generate_awq.py` |
| `bnb_reference/` | 4 models (Llama-3.2 NF4/FP4/INT8) | 4096 elements (NF4/FP4), 256x256 (INT8) | `generate_bnb.py` |

## Regenerating Fixtures

Fixtures only need regeneration if the dequantization formula changes. To regenerate:

1. Download test models via `hf-fm` (see generator scripts for model IDs)
2. Run the Python generator: `python tests/fixtures/<scheme>_reference/generate_<scheme>.py`
3. Commit the updated `.bin` files

## Running Tests

```bash
# All tests (requires all features)
cargo test --all-features

# Specific scheme
cargo test --features gptq --test cross_validation_gptq

# Release mode with AVX2 (for timing comparison)
RUSTFLAGS="-C target-cpu=native" cargo test --release --all-features -- --nocapture
```
