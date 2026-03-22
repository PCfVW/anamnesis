# anamnesis

[![CI](https://github.com/PCfVW/anamnesis/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/anamnesis/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/anamnesis.svg)](https://crates.io/crates/anamnesis)
[![docs.rs](https://docs.rs/anamnesis/badge.svg)](https://docs.rs/anamnesis)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue.svg)](https://www.rust-lang.org)
[![license](https://img.shields.io/crates/l/anamnesis.svg)](https://github.com/PCfVW/anamnesis#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

**ἀνάμνησις** — *Parse any format, recover any precision.*

> ⚠️ **This crate is under active development.** See [ROADMAP.md](ROADMAP.md) for the plan and [CHANGELOG.md](CHANGELOG.md) for current progress.

## Tested Models

Cross-validated against PyTorch on 7 real FP8 models from 5 quantization tools. Bit-exact output (0 ULP difference). Auto-vectorized: SSE2 on any x86-64, AVX2 with `target-cpu=native`.

| Model | Quantizer | Scheme | Scales | vs PyTorch (AVX2) |
|---|---|---|---|---|
| EXAONE-4.0-1.2B-FP8 | LG AI | Fine-grained | BF16 | 6.0x faster |
| Qwen3-1.7B-FP8 | Qwen | Fine-grained | BF16 | 3.9x faster |
| Qwen3-4B-Instruct-2507-FP8 | Qwen | Fine-grained | F16 | 3.0x faster |
| Ministral-3-3B-Instruct-2512 | Mistral | Per-tensor | BF16 | 9.7x faster |
| Llama-3.2-1B-Instruct-FP8 | RedHat | Per-tensor | BF16 | 3.9x faster |
| Llama-3.2-1B-Instruct-FP8-dynamic | RedHat | Per-channel | BF16 | 2.7x faster |
| Llama-3.1-8B-Instruct-FP8 | NVIDIA | Per-tensor | F32 | 6.3x faster |

### GPTQ Dequantization

Cross-validated against PyTorch on 4 real GPTQ models from 2 quantizers (AutoGPTQ, GPTQModel). Bit-exact output (0 ULP difference). Loop fission for full AVX2 vectorization.

| Model | Quantizer | Bits | vs PyTorch (AVX2) |
|---|---|---|---|
| Falcon3-1B-Instruct-GPTQ-Int4 | AutoGPTQ | 4 | 6.5x faster |
| Llama-3.2-1B-Instruct-GPTQ | AutoGPTQ | 4 | 12.2x faster |
| Falcon3-1B-Instruct-GPTQ-Int8 | AutoGPTQ | 8 | 7.0x faster |
| Llama-3.2-1B-gptqmodel-8bit | GPTQModel | 8 | 7.9x faster |

### AWQ Dequantization

Cross-validated against PyTorch on 2 real AWQ models (AutoAWQ GEMM). Bit-exact output (0 ULP difference). Loop fission for full AVX2 vectorization.

| Model | Quantizer | Bits | vs PyTorch (AVX2) |
|---|---|---|---|
| llama-3.2-1b-instruct-awq | AutoAWQ | 4 | 5.7x faster |
| Falcon3-1B-Instruct-AWQ | AutoAWQ | 4 | 4.7x faster |

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code) (dev) and [Augment Code](https://www.augmentcode.com/) (review)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset designed to improve AI coding accuracy.