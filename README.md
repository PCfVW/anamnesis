# anamnesis

[![CI](https://github.com/PCfVW/anamnesis/actions/workflows/ci.yml/badge.svg)](https://github.com/PCfVW/anamnesis/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/anamnesis.svg)](https://crates.io/crates/anamnesis)
[![docs.rs](https://docs.rs/anamnesis/badge.svg)](https://docs.rs/anamnesis)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue.svg)](https://www.rust-lang.org)

**ἀνάμνησις** — *Parse any format, recover any precision.*

> ⚠️ **This crate is under active development.** See [ROADMAP.md](ROADMAP.md) for the plan and [CHANGELOG.md](CHANGELOG.md) for current progress.

## Tested Models

Validated against 7 real FP8 models from 5 different quantization tools, covering both fine-grained (128x128 block scales) and per-tensor quantization schemes.

| Model | Quantizer | Scheme | Scales |
|---|---|---|---|
| EXAONE-4.0-1.2B-FP8 | LG AI | Fine-grained | BF16 |
| Qwen3-1.7B-FP8 | Qwen | Fine-grained | F32 |
| Qwen3-4B-Instruct-2507-FP8 | Qwen | Fine-grained | F32 |
| Ministral-3-3B-Instruct-2512 | Mistral | Per-tensor | BF16 |
| Llama-3.2-1B-Instruct-FP8 | RedHat | Per-tensor | F32 |
| Llama-3.2-1B-Instruct-FP8-dynamic | RedHat | Per-tensor | F32 |
| Llama-3.1-8B-Instruct-FP8 | NVIDIA | Per-tensor | F32 |
