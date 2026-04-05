# anamnesis

[![CI](https://github.com/PCfVW/anamnesis/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/anamnesis/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/anamnesis.svg)](https://crates.io/crates/anamnesis)
[![docs.rs](https://docs.rs/anamnesis/badge.svg)](https://docs.rs/anamnesis)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue.svg)](https://www.rust-lang.org)
[![license](https://img.shields.io/crates/l/anamnesis.svg)](https://github.com/PCfVW/anamnesis#license)
[![unsafe: deny](https://img.shields.io/badge/unsafe-deny_(mmap_only)-blue.svg)](https://github.com/rust-secure-code/safety-dance/)

**ἀνάμνησις** — *Parse any format, recover any precision.*

> ⚠️ **This crate is under active development.** See [ROADMAP.md](ROADMAP.md) for the plan and [CHANGELOG.md](CHANGELOG.md) for current progress.

## Table of Contents

- [Install](#install)
- [CLI Commands](#cli-commands)
- [Tested Models](#tested-models)
  - [FP8 Dequantization](#fp8-dequantization)
  - [GPTQ Dequantization](#gptq-dequantization)
  - [AWQ Dequantization](#awq-dequantization)
  - [BitsAndBytes Dequantization](#bitsandbytes-dequantization)
- [NPZ/NPY Parsing](#npznpy-parsing)
- [PyTorch `.pth` Parsing](#pytorch-pth-parsing)
- [Used by](#used-by)
- [License](#license)
- [Development](#development)

## Install

```sh
cargo install anamnesis --features cli,pth
```

Installs both `anamnesis` and `amn` (short alias). Feature flags: `gptq`, `awq`, `bnb`, `npz`, `pth`, `indicatif` (progress bars).

## CLI Commands

| Command | |
|---------|---|
| `amn parse <file>` | Parse and summarize a model file (`.safetensors`, `.pth`, `.npz`) |
| `amn inspect <file>` | Show format, tensor counts, size estimates, and byte order |
| `amn remember <file>` | Dequantize to BF16 (safetensors) or convert `.pth` → `.safetensors` |

Aliases: `amn info` = `amn inspect`, `amn dequantize` = `amn remember`.

Format detection is automatic: `.safetensors` files go through the dequantization pipeline, `.pth`/`.pt` files go through the pickle parser, `.npz` files go through the header-only NPZ inspector. `.bin` files are probed for ZIP magic to distinguish PyTorch from safetensors.

```
$ amn parse model.pth
Parsed model.pth (PyTorch state_dict)
  Tensors:    3
  Total size: 1.7 KB
  Dtypes:     F32
  Byte order: little-endian

  rnn.weight_ih_l0               F32 [16, 1]         64 B
  rnn.weight_hh_l0               F32 [16, 16]        1.0 KB
  linear.weight                  F32 [10, 16]        640 B

$ amn inspect weights.npz
Format:      NPZ archive
Tensors:     5
Total size:  160 B
Dtypes:      F32

$ amn remember model.pth
Converting model.pth → model.safetensors
  3 tensors, 1.7 KB
  Done.
```

## Tested Models

### FP8 Dequantization

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

### BitsAndBytes Dequantization

Cross-validated against PyTorch on 4 real BitsAndBytes models (NF4, FP4, double-quant, INT8). Bit-exact output (0 ULP difference). Loop fission for AVX2 on NF4/FP4; single-pass AVX2 on INT8 (`vpmovsxbd` → `vcvtdq2ps` → `vmulps`).

| Model | Format | Elements | vs PyTorch (AVX2) |
|---|---|---|---|
| Llama-3.2-1B-Instruct-bnb-nf4 | NF4 | 4,096 | 21.8x faster |
| Llama-3.2-1B-BNB-FP4 | FP4 | 4,096 | 18.0x faster |
| Llama-3.2-1B-Instruct-bnb-nf4-double-quant | NF4 double-quant | 4,096 | 54.0x faster |
| Llama-3.2-1B-BNB-INT8 | INT8 | 65,536 | 1.2x faster |

> **Note:** INT8 speedup is modest because the operation is trivially simple (`i8→f32→multiply`). Both PyTorch and anamnesis are near memory bandwidth limits at ~0.7–0.8 ns/element. The AVX2 hot loop is fully vectorized — the 1.2× reflects the inherent ceiling, not a missed optimization.

### NPZ/NPY Parsing

Feature-gated behind `npz`. Custom NPY header parser with bulk `read_exact` — zero per-element deserialization for little-endian data on little-endian machines. Cross-validated byte-exact against NumPy on Gemma Scope 2B SAE weights.

| Metric | Value |
|---|---|
| Throughput (302 MB Gemma Scope, F32) | **3,586 MB/s** |
| Overhead vs raw I/O | 1.3x |
| vs `npyz` crate | **17.7x faster** |
| Supported dtypes | F16, BF16, F32, F64, Bool, U8–U64, I8–I64 |

BF16 support via JAX `V2` void-dtype convention. Big-endian NPY files handled with in-place byte-swap.

### PyTorch `.pth` Parsing

Feature-gated behind `pth`. Minimal pickle VM (~36 opcodes) with security allowlist. Memory-mapped I/O with zero-copy tensor access (`Cow::Borrowed` from mmap). Cross-validated byte-exact against PyTorch `torch.load()` on 3 [AlgZoo](https://github.com/alignment-research-center/alg-zoo) models (MIT-0 license).

| Model | Size | Tensors | vs `torch.load` |
|---|---|---|---|
| torchvision ResNet-18 | 45 MB | 102 | **11.2x faster** |
| torchvision ResNet-50 | 98 MB | 267 | **12.7x faster** |
| torchvision ViT-B/16 | 330 MB | 152 | **30.8x faster** |

Lossless `.pth` → `.safetensors` conversion preserving original dtypes (F16, BF16, F32, F64, I8–I64, U8, Bool). The conversion pipeline writes directly from mmap slices to the output file — zero intermediate data copies.

Handles both newer (`archive/` prefix) and older (`{model_name}/` prefix) PyTorch ZIP conventions. Legacy (pre-1.6) raw-pickle files are rejected with a clear error.

## Used by

- [candle-mi](https://github.com/PCfVW/candle-mi) — Mechanistic interpretability toolkit for language models

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code) (dev) and [Augment Code](https://www.augmentcode.com/) (review)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset designed to improve AI coding accuracy.