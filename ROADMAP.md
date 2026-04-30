# Roadmap: anamnesis — Tensor Format Transformation for Rust

> *Parse any format, recover any precision.*

**Date:** March 20, 2026 (updated April 30, 2026)
**Status:** Phases 1–4.5 complete (v0.4.2 published). FP8/GPTQ/AWQ/BnB dequantization + NPZ parsing + PyTorch `.pth` parsing + GGUF parsing & dequantization — all 22 of 22 production block-quant kernels (12 from Phase 4 + 10 IQ/TQ/MXFP4 from Phase 4.5), bit-exact against the `gguf` Python reference (mirrors `llama.cpp`'s `ggml-quants.c`). **Next:** Phase 4.7 (remote-only NPZ inspection via reader-generic API, v0.4.3) → Phase 5 (Lethe — BnB encode + round-trip validation harness, v0.5.0).
**Context:** The Rust ML ecosystem (candle, burn, tch) cannot load quantized models (FP8, GPTQ, AWQ) or NumPy weight archives (NPZ/NPY for SAEs). The only workaround is a Python script. anamnesis fills this gap: a framework-agnostic, pure-Rust crate that parses tensor formats and recovers precision when needed. Used by hf-fetch-model (download + transform pipeline) and candle-mi (MI framework).

---

## Table of Contents

- [1. Landscape](#1-landscape)
  - [1.1 The Problem](#11-the-problem)
  - [1.2 Existing Solutions](#12-existing-solutions)
- [2. Architecture](#2-architecture)
  - [2.1 Parsing as Foundation](#21-parsing-as-foundation)
  - [2.2 Module Structure](#22-module-structure)
  - [2.3 Ecosystem Fit](#23-ecosystem-fit)
- [3. Phased Development Plan](#3-phased-development-plan)
  - [3.0 Git Workflow](#30-git-workflow)
  - [Phase 1: FP8 Dequantization](#phase-1-fp8-dequantization)
  - [Phase 2: Additional Quantization Schemes](#phase-2-additional-quantization-schemes)
  - [Phase 3: NPZ/NPY Parsing](#phase-3-npznpy-parsing)
  - [Phase 3.5: PyTorch `.pth` Parsing](#phase-35-pytorch-pth-parsing)
  - [Phase 4: GGUF Parsing & Dequantization](#phase-4-gguf-parsing--dequantization)
  - [Phase 4 patch: API polish + dogfooding (v0.4.1)](#phase-4-patch-api-polish--dogfooding-v041)
  - [Phase 4.5: GGUF Completeness](#phase-45-gguf-completeness)
  - [Phase 4.7: Remote-only NPZ inspection (Reader-generic API)](#phase-47-remote-only-npz-inspection-reader-generic-api)
  - [Phase 5: Quantization (Lethe)](#phase-5-quantization-lethe)
  - [Phase 6: Format Conversion Matrix](#phase-6-format-conversion-matrix)
  - [Phase 6.5: Benchmarking & Performance Validation](#phase-65-benchmarking--performance-validation)
  - [Phase 7: Python Bindings (PyO3)](#phase-7-python-bindings-pyo3)
  - [Phase 7.5: Lethe Encode Completion](#phase-75-lethe-encode-completion)
  - [Phase 8: Emerging Quantization Formats](#phase-8-emerging-quantization-formats)
  - [Phase 9: CPU SIMD Pass](#phase-9-cpu-simd-pass)
  - [Phase 10: Streaming Output](#phase-10-streaming-output)
  - [Future Directions](#future-directions)
- [4. Key Design Decisions](#4-key-design-decisions)
- [5. Relationship to Other Projects](#5-relationship-to-other-projects)

---

## 1. Landscape

### 1.1 The Problem

Quantized models are the norm. Mistral ships Ministral 3B as FP8 safetensors. Community quantizers (TheBloke, NeuralMagic, etc.) distribute Llama and other models in GPTQ and AWQ. SAE weights (Gemma Scope) ship as NPZ archives. None of these load in any Rust ML framework:

```rust
let vb = VarBuilder::from_mmaped_safetensors(&[path], DType::BF16, &device)?;
// → Error: unsupported safetensor dtype F8_E4M3
```

This blocks candle, candle-mi, burn, and tch. The entire Rust ML ecosystem stops at the file format boundary.

### 1.2 Existing Solutions

*Surveyed March 2026. No framework-agnostic tensor format transformation library exists in any language — not in Rust, not in Python, not in C++. Dequantization is universally implemented ad-hoc inside inference engines.*

#### Rust ecosystem

| Solution | What it does | Why it's not enough |
|---|---|---|
| [`float8`](https://crates.io/crates/float8) v0.7.0 | FP8 primitive types (E4M3, E5M2) with `to_f32()` / `from_f32()` conversions | Type-level only. No tensors, no safetensors, no scale factors, no block-wise dequant. A useful **building block** — anamnesis should depend on it for FP8 ↔ f32 element conversion. |
| [candle](https://github.com/huggingface/candle) v0.9.2+ | Added `F8E4M3` DType. [DeepSeek V3 PR #2745](https://github.com/huggingface/candle/pull/2745) implements block-wise FP8 dequant with `scale_inv` | Dequant logic is **model-specific** (DeepSeek V3 only), embedded in `candle-transformers`. Not a reusable library. No E5M2. Still fails on generic FP8 safetensors: `unsupported safetensor dtype F8_E4M3`. |
| [mistralrs-quant](https://lib.rs/crates/mistralrs-quant) | GPTQ, AWQ, HQQ, BnB, FP8 via `QuantMethod` trait with `dequantize_w()` | **CUDA-only** for GPTQ/AWQ. Tightly coupled to mistral.rs inference (742K SLoC, depends on candle-core, tokio, rayon). Not reusable as a standalone library. |
| [pmetal-gguf](https://docs.rs/pmetal-gguf) | Standalone GGUF dequantization with SIMD (`dequant::dequantize()`) | **GGUF-specific.** Does not handle safetensors FP8, GPTQ, AWQ, or NPZ. |
| [`gguf-rs-lib`](https://crates.io/crates/gguf-rs-lib) v0.2.5 | Type-safe GGUF reader/writer. Pure safe Rust. | Parsing only — no dequantization. Potential Phase 4 dependency (TBD). |
| [`llama-gguf`](https://crates.io/crates/llama-gguf) v0.14.0 | Pure Rust llama.cpp reimplementation with GPU dequant kernels (CUDA/Metal/DX12/Vulkan). | Full inference engine, not a standalone library. K-quant dequant math is a useful reference. |
| [`npyz`](https://crates.io/crates/npyz) v0.8.4 | Mature NPY/NPZ parser (4.6M downloads). f16 support via `half` feature. Full read/write. | **No bf16.** Initial Phase 3 approach wrapped `npyz`, but benchmarking revealed 23× overhead vs raw I/O. Replaced with anamnesis's own fast parser (17.7× faster). |
| [`ndarray-npy`](https://crates.io/crates/ndarray-npy) v0.10.0 | Most popular NPY/NPZ crate (7.2M downloads). Actively maintained. | **No f16/bf16.** Tightly coupled to ndarray. Less suitable than `npyz` for framework-agnostic use. |
| [safetensors](https://crates.io/crates/safetensors) | Reads/writes safetensors files | No quantization awareness. Pure format I/O. |
| [burn](https://github.com/tracel-ai/burn) v0.14.0+ | INT8 quantization (beta) | No FP8, no GPTQ, no AWQ. Own internal format only. |
| tch-rs | Defines FP8 variants (`Float8e4m3fn`, `Float8e5m2`) | Wraps PyTorch C++ API. Requires full libtorch installation. No Rust-native dequant. |
| candle-mi's NPZ parser | Loads Gemma Scope SAE weights from `.npz` | Tightly coupled to candle. Not reusable outside candle-mi. |

#### Python ecosystem

| Solution | What it does | Why it's not enough |
|---|---|---|
| [compressed-tensors](https://github.com/vllm-project/compressed-tensors) (vLLM) | Safetensors extension for sparse/quantized storage. Closest conceptual competitor to anamnesis. | Python-only, vLLM-coupled, decompression-to-bf16 still immature ([users requesting basic examples](https://huggingface.co/moonshotai/Kimi-K2-Thinking/discussions/2) as of late 2025). |
| [optimum-quanto](https://github.com/huggingface/optimum-quanto) (HuggingFace) | In-memory quantize/dequantize on PyTorch tensors | Being merged into `optimum`. In-memory only — no file-to-file conversion. PyTorch-coupled. |
| [TorchAO](https://github.com/pytorch/ao) | INT4, INT8, FP8 quantization within PyTorch | PyTorch tensors only, not a file format tool. |
| transformers `dequantize=True` | Loads fine-grained FP8 with dequantization during model load | Embedded in model loading pipeline. Not a standalone operation. |
| AutoGPTQ → [GPTQModel](https://pypi.org/project/GPTQModel/) | GPTQ quantization/dequantization | Dequant happens in fused CUDA kernels during inference. No standalone API. AutoGPTQ archived April 2025. |
| AutoAWQ, bitsandbytes | AWQ and NF4/INT8 quantization | Same pattern: dequant is implicit during inference, not exposed as a file conversion. |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) (vLLM) | "Model-free PTQ" on safetensors (FP8) | **Quantization direction only.** No dequantization. |

#### C/C++ ecosystem

| Solution | What it does | Why it's not enough |
|---|---|---|
| llama.cpp | Converts safetensors → GGUF (Python scripts); dequantizes GGUF during inference (C++) | GGUF-specific. Not a library. Conversion scripts are Python. |
| NVIDIA TensorRT | `IQuantizeLayer` / `IDequantizeLayer` in graph IR | Operates within TensorRT's graph representation, not on raw files. NVIDIA-coupled. |
| NVIDIA TransformerEngine | FP8 compute library (matmul kernels) | Compute library, not a format conversion tool. |

#### The gap

No tool in any language provides **file → transform → file** as a first-class, framework-agnostic operation for quantized tensor formats. Dequantization is always buried inside inference engines or model loading pipelines. anamnesis is the first library designed for this purpose.

---

## 2. Architecture

### 2.1 Parsing as Foundation

Every operation in anamnesis begins with parsing. You cannot remember what you have not first parsed. You cannot quantize what you have not first parsed. You cannot inspect what you have not first parsed.

Parsing is the act of making contact with the weights: decoding their structure, validating their format, understanding what is present and what Lethe took. Sometimes parsing alone is enough (NPZ — nothing was forgotten). Other times, parsing reveals that precision recovery is needed (FP8 — Lethe took something), and remembering follows.

### 2.2 Module Structure

```
anamnesis (library crate)
│
├── parse/              ← decode + validate any tensor format
│   ├── safetensors        safetensors (including quantized metadata)
│   ├── npz                NPZ/NPY archives (feature-gated)
│   ├── pth                PyTorch .pth state_dict (feature-gated, Phase 3.5)
│   └── gguf               GGUF (feature-gated, Phase 4)
│
├── remember/           ← built on parse: precision recovery (dequantize)
│   ├── fp8                fine-grained, per-channel, per-tensor FP8 (E4M3, E5M2)
│   ├── gptq               GPTQ dequantization
│   ├── awq                AWQ dequantization
│   ├── bnb                BitsAndBytes (NF4, INT8) dequantization
│   ├── pth                PyTorch .pth → safetensors conversion (Phase 3.5)
│   └── gguf               GGUF K-quant dequantization (Phase 4)
│
├── lethe/              ← built on parse: precision reduction (quantize)
│   ├── bnb                BnB encode (NF4, FP4, INT8) (Phase 5)
│   ├── round_trip         Bit-exact decode↔encode validation harness (Phase 5)
│   ├── fp8                FP8 encode (E4M3, E5M2; per-tensor / per-channel / fine-grained) (Phase 7.5)
│   ├── gguf_legacy        GGUF legacy block encode (Q4_0–Q8_1) (Phase 7.5)
│   ├── gguf_kquants       GGUF K-quant encode (Q2_K–Q8_K) (Phase 7.5)
│   ├── gguf_iq            GGUF IQ-quant encode (IQ1/IQ2/IQ3/IQ4) (Phase 7.5)
│   ├── gguf_tq            GGUF TQ encode (TQ1_0, TQ2_0) (Phase 7.5)
│   └── mxfp4              MXFP4 encode (Phase 7.5)
│
├── inspect             ← built on parse: report without transforming
│
└── bin/
    └── main.rs         ← CLI binary (feature-gated behind "cli")
                           installed as both `anamnesis` and `amn`
                           subcommands: parse, inspect/info, remember/dequantize, forget/quantize
```

### 2.3 Ecosystem Fit

```
safetensors / npz / pth / gguf  ← file formats
    ↓
anamnesis (library)     ← parse, remember, lethe
anamnesis / amn (CLI)   ← amn parse, amn inspect, amn remember, amn forget
    ↓
hf-fetch-model          ← download + transform pipeline
                           (hf-fm --dequantize bf16, download_and_parse_npz)
candle / candle-mi      ← load + run + interpret
burn / tch / ...        ← any Rust ML consumer
```

hf-fetch-model depends on anamnesis for two things:
1. `--dequantize bf16` calls `anamnesis::parse()` + `.remember()` after download.
2. `download_and_parse_npz()` calls `download()` + `anamnesis::parse_npz()`.

---

## 3. Phased Development Plan

### 3.0 Git Workflow

Single `main` branch. Each task ends with a commit. Phases end with a push and a version tag. Same workflow as candle-mi and hf-fetch-model.

Commit style: imperative mood, lowercase, no trailing period. Examples:
- `add safetensors header parsing`
- `implement fine-grained FP8 dequantization`
- `validate FP8 dequantization against EXAONE-4.0-1.2B`

### Phase 1: FP8 Dequantization

**Goal:** The flagship use case — parse an FP8 safetensors file, dequantize to BF16, write a standard safetensors file loadable by any Rust ML framework. Proves the parse → remember architecture end-to-end.

- [x] Scaffold crate:
  - `git init` + GitHub repo creation + remote setup
  - `cargo init --lib`
  - `Cargo.toml` with metadata (name, version `0.1.0`, description, license MIT OR Apache-2.0, keywords, categories), feature gates (`cli = ["dep:clap"]`, `npz = ["dep:npyz"]`), two `[[bin]]` targets (`anamnesis` and `amn`, both pointing to `src/bin/main.rs`, `required-features = ["cli"]`) following the hf-fetch-model pattern
  - `src/lib.rs` with `#![forbid(unsafe_code)]`, `#![deny(warnings)]`
  - `CLAUDE.md` — "Before writing or modifying any Rust code, read and follow CONVENTIONS.md." (same as hf-fetch-model)
  - `LICENSE-MIT` + `LICENSE-APACHE` — dual license files
  - `.gitignore` — standard Rust (`/target`, etc.)
  - `.github/workflows/ci.yml` + `.github/workflows/publish.yml` — reused from hf-fetch-model's pattern with `actions/checkout@v5`: format check, clippy (`--all-targets` + `--all-features`), tests, doc check on publish, tag-triggered `cargo publish`
  — **commit**
- [x] Safetensors parsing foundation (`src/parse/safetensors.rs`) — read header JSON, extract tensor names, shapes, dtypes, byte offsets. Identify quantized tensors (FP8) vs passthrough (BF16/F32 norms, embeddings). Detect fine-grained FP8 metadata (scale factor tensors, block structure). This is the "make contact" layer — **commit**
- [x] Inspect (`src/inspect.rs`) — built on parse. Report format, tensor count, quantized vs passthrough breakdown, size estimate (current and dequantized). Matches the flagship CLI output in `amn-flagship-v2.md` — **commit**
- [x] Fine-grained FP8 dequantization (`src/remember/fp8.rs`) — E4M3 with 128×128 block scale factors. Scalar implementation written for auto-vectorization: contiguous `&[u8]` → `&mut [u16]` slices, no branches in the hot path (bitwise select for NaN/subnormal), `chunks_exact(128)` matching block size, scale factor hoisted before inner loop. Verify with `cargo-show-asm` that the compiler emits SIMD instructions; annotate with `// VECTORIZED:`. Built on the parsed representation from `parse/safetensors` — **commit**
- [x] Per-tensor FP8 dequantization — single scale factor per tensor (simpler case). Same module, same SIMD-friendly loop structure, different scale broadcast — **commit**
- [x] Parse-first public API (`src/lib.rs`) — `parse(path)` returns a `ParsedModel` struct holding header metadata + byte data. `ParsedModel::inspect()` returns format info. `ParsedModel::remember(output_path, TargetDtype)` dequantizes and writes a standard safetensors file. No file is re-read after the initial parse. The Rust API in `amn-flagship-v2.md` should work — **commit**
- [x] CLI binary (`src/bin/main.rs`) — thin `clap` wrapper over the library API. Subcommands: `parse`, `inspect` (alias `info`), `remember` (alias `dequantize`). Each subcommand calls `anamnesis::parse()` then the appropriate method. Progress output via `indicatif` (optional, behind `indicatif` feature). Same binary serves both `anamnesis` and `amn` names — **commit**
- [x] Download FP8 test models via `hf-fetch-model` (v0.8.1, with `list-files`). 7 models from 5 quantization tools, covering 3 FP8 schemes discovered during validation:

  | Model | Size | FP8 scheme | Scale dtype | Quantizer |
  |---|---|---|---|---|
  | `LGAI-EXAONE/EXAONE-4.0-1.2B-FP8` | 1.39 GB | **Fine-grained** | BF16 | LG AI |
  | `Qwen/Qwen3-1.7B-FP8` | 2.47 GB | **Fine-grained** | BF16 | Qwen |
  | `Qwen/Qwen3-4B-Instruct-2507-FP8` | 4.83 GB | **Fine-grained** | **F16** | Qwen |
  | `mistralai/Ministral-3-3B-Instruct-2512` | 4.35 GB | **Per-tensor** (scalar) | BF16 | Mistral |
  | `RedHatAI/Llama-3.2-1B-Instruct-FP8` | 1.88 GB | **Per-tensor** | BF16 | RedHat |
  | `RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic` | 1.89 GB | **Per-channel** `[N,1]` | BF16 | RedHat |
  | `nvidia/Llama-3.1-8B-Instruct-FP8` (shard 1) | 4.65 GB | **Per-tensor** (scalar) | F32 | NVIDIA |

  Discoveries during validation:
  - **Per-channel FP8** — a third scheme (one scale per row, shape `[N,1]`), used by RedHat/vLLM dynamic quantization. New `dequantize_per_channel_fp8_to_bf16()` and `PerChannelFp8` scheme variant.
  - **F16 scales** exist in the wild (Qwen3-4B-Instruct-2507-FP8, Qwen/Alibaba), undocumented in the ecosystem. All three scale dtypes (F32, BF16, F16) now supported.
  - **Scheme detection by scale shape**, not name suffix — both fine-grained and per-tensor use `_scale_inv`, distinguished by scale tensor dimensionality.
  — **commits** (multiple: BF16 scale fix, scheme detection fix, F16 + per-channel support)

- [x] Validation against FP8 test models — cross-validated against PyTorch (`torch.float8_e4m3fn` → `torch.bfloat16`) on 256×256 slices from all 7 models. **Bit-exact match** (0 ULP difference on 65,536 elements per fixture). Fixtures generated by `tests/fixtures/fp8_reference/generate.py`, Rust tests in `tests/cross_validation.rs`. Auto-vectorization verified with `cargo-show-asm`: SSE2 (default), AVX2 (`target-cpu=native`). **2.7–9.7× faster than PyTorch** (AVX2). — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.1.0 — FP8 dequantization works. — **PUSH + tag `v0.1.0`**

**Dependencies:** `safetensors`, `half`, `float8` (FP8 ↔ f32 element conversion). CLI: `clap` 4 with `derive` (behind `cli` feature), `indicatif` (behind `indicatif` feature, optional progress bars).

**Note on cached test assets:** The local HuggingFace cache (`~/.cache/huggingface/hub/`) contains 21 models, none with FP8/GPTQ/AWQ quantization. The one asset relevant to anamnesis is **google/gemma-scope-2b-pt-res** which contains `params.npz` (302 MB, 5 F32 arrays: `W_dec`, `W_enc`, `b_dec`, `b_enc`, `threshold`) — this will be used for Phase 3 NPZ validation.

**Note on candle-mi auto-config:** Validating EXAONE-4.0 and Qwen3 after dequantization requires extending candle-mi's auto-config with `exaone4` and `qwen3` `model_type` support. These are both text-only decoder transformers close to existing supported families:
- `exaone4` — LLaMA-like with alternating sliding window / full attention layers ("LLLG" pattern). Similar to Gemma 2's alternating window scheme.
- `qwen3` — extends `qwen2` with QK LayerNorm (additional `q_norm` / `k_norm` tensors per layer) and thinking mode support. The core architecture is otherwise identical to `qwen2`.

### Phase 2: Additional Quantization Schemes

**Goal:** Extend `remember/` to cover the major quantization formats. Each scheme adds a new submodule under `remember/` and extends the parsing layer to detect its metadata.

- [x] GPTQ dequantization (`src/remember/gptq.rs`) — INT4/INT8 with group-wise scale + zero-point. Parse GPTQ metadata from safetensors, reconstruct full-precision weights. Bit-exact against PyTorch on 4 real models (2 quantizers × 2 bit widths), 6.5–12.2× faster than CPU PyTorch (AVX2). Loop fission for full AVX2 vectorization — **commit**
- [x] AWQ dequantization (`src/remember/awq.rs`) — activation-aware INT4 with per-group scales, no +1 zero-point offset. Packs along out_features (columns), unlike GPTQ (rows). Bit-exact against PyTorch on 2 real models (AutoAWQ GEMM), 4.7–5.7× faster than CPU PyTorch (AVX2). Loop fission applied from the start. Note: 8-bit AWQ path is unit-tested but no real 8-bit AWQ models exist in the standard AutoAWQ `.qweight` format — all "8-bit AWQ" on HuggingFace use `compressed-tensors` (vLLM) or are mislabeled 4-bit — **commit**
- [x] BitsAndBytes dequantization (`src/remember/bnb.rs`) — NF4, FP4, double-quant NF4/FP4, and INT8 (LLM.int8()). 4-bit uses 16-entry lookup table + per-block absmax; INT8 uses per-row absmax / 127.0. Bit-exact against PyTorch on 4 real models, 18–54× faster for NF4/FP4 (AVX2), 1.2× for INT8 (near bandwidth limit). Loop fission for NF4/FP4; single-pass AVX2 for INT8 (`vpmovsxbd` → `vcvtdq2ps` → `vmulps`). Format discovery via `hf-fm inspect` (remote header probing) — **commit**
- [x] Feature gates — each scheme behind its own feature flag (`gptq`, `awq`, `bnb`). Default features: `fp8` only — **commit**
- [x] Validation against real models for each scheme — **commits as needed** — **PUSH**

**Deliverable:** `anamnesis` v0.2.0 — all major quantization schemes supported. — **PUSH + tag `v0.2.0`**

**New dependencies:** None expected (pure bit manipulation), but feature-gated if any arise.

### Pre-Phase 3: BnB4 output shape recovery

**Goal:** BnB4 weights are stored flattened (`[N, 1]`). The current dequantization emits a 1D `[total_elements]` output shape (see `model.rs`). The original 2D shape (`[out_features, in_features]`) must be recovered from `config.json` for downstream consumers that expect shaped weight tensors.

- [x] Recover original weight shape for `BnB4` tensors from `quant_state.bitsandbytes__nf4`/`__fp4` companion tensor (`JSON` blob with `"shape"` field). No `config.json` needed — the shape is stored inside the safetensors file itself — **commit**

### Phase 3: NPZ/NPY Parsing

**Goal:** Extend the `parse/` layer to NumPy archives. This is the case where parsing alone suffices — nothing was forgotten, the weights just need extracting from a foreign container. Migrates candle-mi's tightly-coupled NPZ parser into anamnesis as a reusable, framework-agnostic module.

**Own the fast path.** Initial implementation wrapped `npyz` (v0.8.4), but benchmarking revealed 23× overhead vs raw I/O due to per-element deserialization. Replaced with a custom `NPY` header parser (~80 lines) and bulk `read_exact` — for LE data on LE machines (>99% of ML files), the raw bytes ARE the correct output. Uses `zip` v2 directly for `ZIP` extraction. Result: 84 ms for 302 MB (1.3× raw I/O overhead), **17.7× faster** than `npyz`, **4.9× faster** than candle-mi's custom parser.

- [x] NPZ integration module (`src/parse/npz.rs`) — custom `NPY` header parser with bulk `read_exact` data extraction. `parse_npz(path) -> HashMap<String, NpzTensor>`. Zero per-element deserialization for LE data on LE machines. BF16 interpretation layer (JAX `V2` void dtype). **3,586 MB/s** on 302 MB file (1.3× raw I/O overhead, 17.7× faster than `npyz`) — **commit**
- [x] `NpzTensor` type — framework-agnostic struct: `name: String`, `shape: Vec<usize>`, `dtype: NpzDtype`, `data: Vec<u8>`. The `NpzDtype` enum includes `BF16` which `NumPy` cannot represent natively — **commit**
- [x] Feature-gated behind `npz` feature (adds `zip` v2 with `deflate`) — **commit**
- [x] Validation against Gemma Scope SAE weights — already cached locally at `~/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/` (`params.npz`, 302 MB, 5 F32 arrays: `W_dec` [16384×2304], `W_enc` [2304×16384], `b_dec` [2304], `b_enc` [16384], `threshold` [16384]). Parse, verify shapes and values match Python reference — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.3.0 — NPZ parsing works. candle-mi can migrate its NPZ dependency from internal to `anamnesis`. — **PUSH + tag `v0.3.0`**

**New dependencies:** `zip` v2 with `deflate` feature (direct dependency, no `npyz`). Feature-gated behind anamnesis's `npz` feature.

### Phase 3.5: PyTorch `.pth` Parsing

**Goal:** Add PyTorch `.pth` (state_dict) parsing and lossless conversion to safetensors. This enables loading pre-trained weights from repositories that only ship `.pth` files (e.g., AlgZoo's 400+ tiny models on GCS, plus thousands of older HuggingFace models). Unlike Phases 1–2, no dequantization is needed — `.pth` tensors are already full-precision (F32/F64/BF16/F16).

**Approach:** Implement a minimal pickle VM (~36 opcodes, not the full protocol) that interprets the `data.pkl` stream inside the ZIP archive, extracts tensor metadata (shape, dtype, storage reference), and reads raw bytes from `data/{index}` entries. Security boundary: explicit GLOBAL allowlist rejects non-`torch.*` callables. Lossless format conversion to safetensors via `safetensors::tensor::serialize_to_file`. Feature-gated behind `pth`.

**Detailed plan:** See `PLAN-PTH-v0.3.1.md`.

- [x] Extract `byteswap_inplace` from `npz.rs` to shared `src/parse/utils.rs` — **commit**
- [x] Pickle VM + tensor extraction (`src/parse/pth.rs`) — minimal stack machine, GLOBAL allowlist, storage cache for shared storage, stride handling — **commit**
- [x] Test fixtures — 3 AlgZoo `.pth` models (2–3.5 KB each) from GCS + Python-generated JSON reference manifests — **commit**
- [x] Cross-validation tests (`tests/cross_validation_pth.rs`) — byte-exact comparison against `PyTorch` on 3 AlgZoo models (RNN + Transformer, newer and older ZIP formats) — **commit**
- [x] Safetensors conversion (`src/remember/pth.rs`) — lossless format writer, byte-exact roundtrip verified — **commit**
- [x] `ParsedPth` + `PthInspectInfo` — inspect/to_safetensors methods, Display impl — **commit**
- [x] CLI integration (`src/bin/main.rs`) — format dispatch by extension + ZIP magic, `parse`/`inspect`/`remember` for `.pth` — **commit**
- [x] Module wiring — `mod.rs`, `lib.rs`, `Cargo.toml` feature gate, `error.rs` cfg update — done incrementally across prior commits

**Deliverable:** `anamnesis` v0.3.1 — PyTorch `.pth` state_dict parsing + safetensors conversion. — **PUSH + tag `v0.3.1`**

**New dependencies:** None (reuses `zip` v2 from `npz`). Feature-gated behind `pth`.

### Phase 4: GGUF Parsing & Dequantization

**Status:** Complete (`v0.4.0` published). Parser + all 12 block-quant dequantisation kernels + streaming API + cross-validation against `llama.cpp` reference (10/12 kernels bit-exact, 6.3–31.3× faster than Python reference).

**Goal:** Add support for GGUF, the dominant format for local inference (~166,000 models on HuggingFace as of March 2026). GGUF is to llama.cpp/Ollama/LM Studio what safetensors is to HuggingFace Transformers. No standalone, framework-agnostic Rust crate exists for GGUF dequantization — candle-core has GGUF support but it is tightly coupled to candle's tensor types and inference pipeline.

**Approach:** Parse GGUF metadata and tensor layout. Dequantize K-quant types (`Q2_K`–`Q8_K`) and legacy types (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`) to BF16 using the same loop-fission + auto-vectorization strategy as GPTQ/AWQ/BnB. Feature-gated behind `gguf`.

**Key projects studied:**
- **ggml-org/ggml** (`src/ggml-common.h`, `src/ggml-quants.c`) — canonical block struct layouts with `_Static_assert` byte counts and scalar `dequantize_row_*` reference implementations. **Used as the ground truth for every kernel in this phase** (ported verbatim with CONVENTIONS annotations applied).
- **candle-core `quantized`** (`k_quants.rs`) — secondary math reference for K-quants; not depended on.
- **gguf-rs-lib** — considered as a parsing dependency but rejected in favour of a lean in-house parser (matches the `.npy` and `.pth` pattern, keeps all error paths inside `AnamnesisError`).
- **llama-gguf (Lexmata)** — pure Rust reimplementation of llama.cpp with GPU dequant kernels; not used.

- [x] **GGUF file parser** (`src/parse/gguf.rs`, commit `2acaf1a`) — lean in-house parser for `GGUF` v2 and v3. Memory-mapped via `memmap2` (reused from `pth`), returns `Cow::Borrowed` tensor views with zero per-tensor heap allocation. Reads header, metadata KV pairs (all 13 value types including nested `ARRAY`), and the tensor info table; resolves absolute tensor-data offsets from the `general.alignment` metadata key (default 32 B). Adversarial-input DoS guards on tensor count (1 M), KV count (1 M), string length (16 MiB), array length (16 M), array nesting depth (4), tensor dimensions (8), and element product (1 T). Public types: `GgufType`, `GgufMetadataValue`, `GgufMetadataArray`, `GgufTensor`, `GgufTensorInfo`, `GgufInspectInfo`, `ParsedGguf`. **21 unit tests.** **commit**
- [x] **K-quant dequantization** (`src/remember/gguf.rs`, commit `cc4ecf8`) — scalar reference kernels for `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` (256-element super-blocks). Each block is processed with two-pass loop fission: pass 1 unpacks the packed-bit storage into an `[f32; 256]` stack scratch buffer (L1-resident), pass 2 walks the scratch and emits `BF16` bytes via the shared `f32_bits_to_bf16_bits` helper from `remember::fp8`. `get_scale_min_k4` (Q4_K/Q5_K's 6-bit packed-scale extractor) and `q3_k_unpack_scales` (Q3_K's `kmask1`/`kmask2` permute) are ported verbatim as private `#[inline]` helpers with their own unit tests — the two most error-prone bit-packing pieces are verifiable in isolation. **commit**
- [x] **Legacy quant types** (`src/remember/gguf.rs`, commit `cc4ecf8`) — scalar kernels for `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` (32-element blocks). `Q8_1` is a trivial variant of `Q8_0` (same `d * qs[j]` formula; the auxiliary `d × Σ qs` field is ignored for reconstruction). Landed together with the K-quants in a single commit — the loop-fission template is identical; only the per-type pass-1 unpack differs. **commit**
- [x] **Streaming dequant API** (`dequantize_gguf_blocks_to_bf16`, commit `b6610fe`) — `FnMut(&[u8]) -> Result<()>` sink closure receives one block's worth of `BF16` bytes at a time (64 B legacy, 512 B K-quant). Peak heap is O(one scratch + one block buffer) ≈ 1.5 KB, independent of tensor size — enabling dequantisation of 70 B-parameter models on modest-RAM machines by streaming directly to disk. The `Vec`-returning `dequantize_gguf_to_bf16` is now a thin wrapper around the streaming variant; both entry points share the same validation and the same scalar kernels via a generic `run_legacy_kernel` / `run_super_kernel` outer-loop helper.
- [x] **Algorithmic audit fixes + consistency pass** (commits `b6610fe`, `3156104`) — `Vec::with_capacity` + `extend_from_slice` instead of `vec![0u8; n]` (no zero-init memset); `chunks_exact` outer loop eliminates per-block bounds checks; infallible `read_f16_bytes([u8; 2]) -> f32` / `read_f32_bytes([u8; 4]) -> f32` replace the old `Result`-returning readers; `n_elements.checked_mul(2)` overflow guard catches 32-bit targets with > 2 GiB of `BF16` output; `read_scales12` helper deduplicates the 12-byte packed-scale load across Q3_K/Q4_K/Q5_K. **30 unit tests** after the refactor.
- [x] **Feature-gated behind `gguf` feature** — added in commit `2acaf1a` as part of the parser commit (`gguf = ["dep:memmap2"]` in `Cargo.toml`). Reuses `memmap2` (already a `pth` dep) and `half` (already mandatory). No new third-party crate in any Phase 4 commit.
- [x] **Cross-validation against `llama.cpp` reference dequantization** (commit `73f8e74`) — 10 of 12 production kernels bit-exact (0 ULP) against the `gguf` Python package's `dequantize` function (the official `ggml-org` reference mirroring `ggml-quants.c`'s `dequantize_row_*`). Legacy quants: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`. K-quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`. Fixtures from bartowski SmolLM2-135M-Instruct and TheBloke TinyLlama-1.1B-Chat (65 536-element slices). `Q8_1` and `Q8_K` are not shipped by any real model (internal `llama.cpp` activation quant types) and are already covered by unit tests. **6.3–31.3× faster** than the Python reference (AVX2). **10 integration tests.** — **PUSH + tag `v0.4.0`**

**Deferred follow-ups:** every out-of-scope item is rehomed to **Phase 4.5: GGUF Completeness** (`IQ*`/`TQ*`/`MXFP4` kernels — targeting `v0.4.2`) and **Phase 9: CPU SIMD Pass** (cross-format AVX2/NEON pass-2 SIMD). User-facing polish (GGUF CLI subcommands, `ParsedGguf::dequantize_tensor`) is pulled into v0.4.1 alongside the in-memory safetensors API from the candle-mi dogfooding report. Big-endian GGUF v3 support has no committed target — the parser detects byte-swapped magic and returns a clear `Unsupported` error, which is enough until a real big-endian model ships.

**Deliverable:** `anamnesis` v0.4.0 — GGUF parsing + dequantization works end-to-end with bit-exact `llama.cpp`-validated output. anamnesis becomes the only Rust crate that can parse *both* safetensors-based (GPTQ/AWQ/BnB/FP8) and GGUF-based quantized models and dequantize everything to BF16. — **PUSH + tag `v0.4.0`**

**New dependencies:** None. The `gguf` feature pulls in `memmap2` (already used by `pth`) and relies on `half` (already mandatory). No third-party GGUF parser in the dependency tree.

### Phase 4 patch: API polish + dogfooding (v0.4.1)

**Goal:** Quick backward-compatible release bundling two GGUF polish items pulled forward from Phase 4.5 and one in-memory safetensors API addition requested by candle-mi dogfooding. Unblocks candle-mi stoicheia module (agnostic `.pth` loading) and gives GGUF users CLI + convenience-method access without waiting for the full `IQ*`/`TQ*`/`MXFP4` kernel work.

**Dogfooding report:** [`docs/dogfooding-feedbacks/in-memory-safetensors-for-candle-mi.md`](docs/dogfooding-feedbacks/in-memory-safetensors-for-candle-mi.md) — candle-mi's stoicheia module needs an in-memory path from `.pth` bytes to safetensors bytes (`VarBuilder::from_buffered_safetensors`). The existing `pth_to_safetensors` writes to disk; the new `pth_to_safetensors_bytes` returns `Vec<u8>`.

- [x] **`pth_to_safetensors_bytes` + `ParsedPth::to_safetensors_bytes`** — in-memory `.pth` → safetensors conversion returning `Vec<u8>` instead of writing to disk. Enables downstream crates (candle-mi) to load `.pth` files without a temp file round-trip. Re-export from `lib.rs`. **5 new tests** (2 unit + 3 integration roundtrip-via-bytes). ~50 lines — **commit**
- [x] **`ParsedGguf::dequantize_tensor` convenience method** — thin wrapper that infers `n_elements` from `shape.iter().try_fold(checked_mul)` and slices the mmap at `data_offset..data_offset + byte_len` before delegating to `dequantize_gguf_to_bf16`. ~40 lines — **commit**
- [x] **GGUF CLI subcommands** — extend `src/bin/main.rs` format detection to recognise the `"GGUF"` magic and add `amn parse model.gguf`, `amn inspect model.gguf`, and `amn remember model.gguf --to bf16 -o out.safetensors`. Dequantizes quantized tensors to `BF16`, passes through non-quantized tensors with their original dtype, reverses GGUF MSB-first shape to safetensors row-major. ~130 lines — **commit** — **PUSH + tag `v0.4.1`**

**Deliverable:** `anamnesis` v0.4.1 — in-memory safetensors API for `.pth`, GGUF CLI subcommands, and `dequantize_tensor` convenience method. ~100 lines total, fully backward compatible. — **PUSH + tag `v0.4.1`**

### Phase 4.5: GGUF Completeness

**Goal:** Close the last remaining coverage gap in GGUF dequantisation so that anamnesis can handle every GGUF block type shipping on HuggingFace today. `v0.4.0` covers the 12 block types that back most mainstream models (`Q4_0`–`Q8_1`, `Q2_K`–`Q8_K`), but a growing fraction of 2025–2026 GGUF uploads use the newer `IQ*` family — `IQ4_XS` in particular has become a common "small but accurate" quant for many Llama 3, Qwen 2.5, and Mistral Nemo variants — plus a handful of models ship `TQ*` or `MXFP4`. Without those kernels, anamnesis rejects real files with `AnamnesisError::Unsupported`.

**Approach:** Port the scalar `dequantize_row_*` reference functions for the remaining block types from `ggml-quants.c` using the same loop-fission template as Phase 4. Most of the work is verbatim translation of the reference kernels plus verbatim copying of the lattice / codebook constant tables that the `IQ*` family uses (`iq2xxs_grid: [u64; 256]`, `iq3s_grid`, `iq1s_grid`, and siblings). Each variant has a distinct block struct layout documented in `ggml-common.h`; the test playbook is unchanged from Phase 4 — hand-constructed single-block fixtures per type, plus bit-exact cross-validation against `llama.cpp` reference output.

- [x] **`IQ4_NL` and `IQ4_XS` dequant kernels** (`src/remember/gguf.rs`, commits `486fa85` + `89abaa6`) — non-linear 4-bit quants (`IQ4_NL` 32-element at 18 B/block, `IQ4_XS` 256-element super-block at 136 B/block) sharing a single 16-entry `kvalues_iq4nl` codebook constant. Ported verbatim from `ggml-quants.c` using the Phase 4 two-pass loop-fission template. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/SmolLM2-135M-Instruct-GGUF` (6 new unit tests + 2 new cross-validation tests). First `IQ*` kernels in the crate; `IQ4_XS` is the most widely used member of the `IQ*` family on HuggingFace.
- [x] **`IQ2_XXS`, `IQ2_XS`, `IQ2_S` dequant kernels** (`src/remember/gguf.rs`, commits `cc83de8` + `6385c73`) — three 2-bit super-quants (66 / 74 / 82 B/block) sharing a lattice-codebook design: `IQ2XXS_GRID: [u64; 256]`, `IQ2XS_GRID: [u64; 512]`, `IQ2S_GRID: [u64; 1024]`, plus the `ksigns_iq2xs` / `kmask_iq2xs` sign tables (~14 KB total), all ported verbatim from `ggml-common.h` into a private `iq_grids` submodule at the bottom of the file. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`IQ2_XXS` + `IQ2_XS`) and `bartowski/Qwen2.5-0.5B-Instruct-GGUF` IQ2_M mix (`IQ2_S`). Discovered along the way: `Mistral-7B-v0.3-IQ2_S.gguf` is misleadingly named — it ships `IQ2_XS` + `IQ3_S`, no `IQ2_S`. 7 new unit tests + 3 new cross-validation tests.
- [x] **`IQ3_XXS` and `IQ3_S` dequant kernels** (`src/remember/gguf.rs`, commits `acdcfcd` + `62b4aa3`) — two 3-bit super-quants (98 / 110 B/block) sharing two new codebook grids: `IQ3XXS_GRID: [u32; 256]` (1 KB) and `IQ3S_GRID: [u32; 512]` (2 KB). Both reuse the Phase 4.5 step 2 `write_signed_grid` helper — the combined-grid/signs packing format is shared across every sign-masked `IQ*` kernel. `IQ3_S` introduces the unusual odd-integer scale formula `d × (1 + 2·nibble)` and pairs low/high nibbles of `scales[outer]` across two consecutive sub-blocks. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` — a single 2.64 GB download (`Mistral-7B-Instruct-v0.3-IQ3_XXS.gguf`) covers both fixtures because that file ships 96 `IQ3_XXS` tensors + 33 `IQ3_S` tensors. 6 new unit tests + 2 new cross-validation tests (3.32× / 4.37× faster than Python reference, release-mode best-of-5).
- [x] **`IQ1_S` and `IQ1_M` dequant kernels** (`src/remember/gguf.rs`, commits `16bd880` + `ecaca11` + `891e4b6`) — two 1-bit super-quants, smallest footprint in the IQ family. `IQ1_S` (50 B/block, top-level `d: f16`, 11-bit grid index, ±`IQ1S_DELTA = 0.125` additive bias); `IQ1_M` (56 B/block, **no top-level `d`** — super-block scale reconstructed from a scattered 16-bit pattern across `scales[8]` reinterpreted as `f16`). Both share the 2048-entry `IQ1S_GRID: [u64; 2048]` codebook of signed `i8` 8-element vectors (~16 KB, the largest single grid in the crate). Inner-loop math is `dl × (grid[j] + delta)` — additive bias instead of IQ2/IQ3's multiplicative ±1 sign — needing the new `write_delta_grid` helper. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (separate `IQ1_S.gguf` and `IQ1_M.gguf` files; no shared-file trick available like step 3). 6 new unit tests + 2 new cross-validation tests (13.77× / 7.88× faster than Python reference, release-mode best-of-5 — fastest IQ kernels in the crate).
- [x] **`TQ1_0` and `TQ2_0` dequant kernels** (`src/remember/gguf.rs`, commits `72f8e3c` + `111de18` + `53dd3ab`) — two ternary super-quants (54 / 66 B/block) decoding to `{-d, 0, +d}`. `TQ1_0` uses base-3 packing (5 ternaries per `qs` byte, 4 per `qh` byte) decoded via the `pow3 = [1, 3, 9, 27, 81, 243]` multiplication trick. `TQ2_0` is plain 2-bit packing. New `decode_pow3_ternary` helper alongside the existing `write_signed_grid` / `write_delta_grid` family. **First step using synthetic fixtures** — only ~15 BitNet-derivative GGUFs ship TQ types on HuggingFace, but Python `gguf.quants.quantize()` implements both, so a deterministic synthetic random tensor (seed=42, scale=0.1) is the practical fixture source. Bit-exact (0 ULP) against the `gguf` Python reference. 5 new unit tests + 2 new cross-validation tests (35.59× / 26.31× faster than Python reference, release-mode best-of-5 — fastest GGUF kernels in the crate, beating Q5_0's 31.3× record).
- [x] **`MXFP4` dequant kernel** (`src/remember/gguf.rs`, commits `b87d877` + `acfe8a6` + `89e596d`) — 32-element microscaling FP4 (OCP MX standard, added to `ggml` in 2024), 17 B/block: `e: u8` (E8M0 byte exponent) + `qs[16]` (4-bit packed). Distinct from the `IQ*` / `TQ*` family because it uses a standardised IEEE-like sub-block exponent (`E8M0`) rather than a learned codebook, decoded by a new `e8m0_to_fp32_half` helper that matches `llama.cpp`'s `ggml_e8m0_to_fp32_half` (no NaN guard for `e == 0xFF`, deviating from raw OCP MX spec). 16-entry signed `i8` codebook (`K_VALUES_MXFP4`, storing 2× the OCP E2M1 magnitudes) with the doubling cancelled by the half-scale exponent. Same low/high split-nibble layout as `Q4_0` / `IQ4_NL` via the existing `run_legacy_kernel` runner. Bit-exact (0 ULP) against the `gguf` Python reference on a deterministic synthetic fixture (Python `gguf.quants.quantize()` supports MXFP4 too — same synthetic-fixture path as TQ1_0/TQ2_0 from step 5). 4 new unit tests + 1 new cross-validation test (30.14× faster than Python reference, release-mode best-of-5).
- [x] **Cross-validation extension** — landed incrementally with steps 1–6: every `IQ*` / `TQ*` / `MXFP4` kernel ships its own cross-validation test against the `gguf` Python reference (mirrors `ggml-quants.c`). Bit-exactness contract holds at 0 ULP for all 22 of 22 production kernels — **PUSH + tag `v0.4.2`**.

**Deliverable:** `anamnesis` v0.4.2 — anamnesis now dequantises every GGUF block type shipping on HuggingFace. `IQ*`/`TQ*`/`MXFP4` are the last meaningful coverage gap; after this release anamnesis is a drop-in replacement for `llama.cpp`'s CPU dequant path. — **PUSH + tag `v0.4.2`**

**New dependencies:** None. Reuses the `gguf` feature gate, `memmap2`, and `half` from Phase 4.

**Explicitly out of scope:**

- **Big-endian GGUF v3 support** — the parser detects byte-swapped magic and returns `Unsupported`. A reader rewrite can reuse `parse::utils::byteswap_inplace` if demand ever materialises, but no mainstream model is distributed big-endian today. No committed target.
- **CPU SIMD optimisation of pass 2** — cross-cutting optimisation that would benefit FP8, GPTQ, AWQ, and BnB equally, not just GGUF. See [Phase 9: CPU SIMD Pass](#phase-9-cpu-simd-pass).
- **Remote-only NPZ inspection (HTTP-range probe)** — rehomed to its own scheduled milestone, [Phase 4.7](#phase-47-remote-only-npz-inspection-reader-generic-api), shipping in `v0.4.3` ahead of Phase 5.

### Phase 4.7: Remote-only NPZ inspection (Reader-generic API)

**Goal:** Eliminate the bandwidth cliff candle-mi v0.1.10 dogfooding (GemmaScope `open()` flow, 2026-04-30) hit when a 288 MiB layer-0 NPZ download was wasted just to read 16 bytes of `W_enc` shape metadata. Add a reader-generic `inspect_npz_from_reader<R: Read + Seek>` so callers can plug in any positional source — including an HTTP-range-backed adapter — without anamnesis itself taking on a network or TLS dependency. Cuts candle-mi's GemmaScope `open()` cold-start from ~30 s on a 100 Mbps link to <1 s. Call site reference: `candle-mi/src/clt/mod.rs::open_gemmascope` (TODO comment marks the spot).

**Approach:** Lift the file-opening boilerplate out of the existing `inspect_npz(path)` and re-express it as a thin wrapper over a new `inspect_npz_from_reader<R: Read + Seek>(reader: R) -> Result<NpzInspectInfo>`. ZIP places its central directory at the end of the file and `zip::ZipArchive::new` already accepts any `R: Read + Seek`; the existing per-entry `parse_npy_header` flow translates verbatim — only the I/O substrate changes. Anamnesis stays HTTP-free; remote inspection is composed at the call site by feeding an HTTP-range-backed `Read + Seek` adapter (which `hf-fm` already implements for safetensors and can extend to NPZ). Architectural fit: this is the same "parse from reader" doctrine the rest of the crate already follows (`parse_npy_header(&mut impl Read)`, the safetensors header parser, the GGUF mmap reader) — the public API surface just hadn't been pushed up to the inspect layer yet.

- [ ] **`inspect_npz_from_reader<R: Read + Seek>` public function** (`src/parse/npz.rs`) — accepts any `Read + Seek` source, returns `NpzInspectInfo` exactly as today. Refactor existing `inspect_npz(path)` into a 2-line wrapper that opens the file and delegates. No change to `NpzInspectInfo` / `NpzTensorInfo` / `NpzDtype` (already public from v0.3.0). ~30 LOC of refactor + the new public entry-point. Re-export from `lib.rs` alongside existing `inspect_npz` — **commit**
- [ ] **In-memory cursor coverage test** (`src/parse/npz.rs`, `tests/`) — verify `inspect_npz_from_reader(Cursor::new(&npz_bytes))` matches `inspect_npz(path)` byte-for-byte on the existing Gemma Scope SAE fixture. Locks the contract that the path-based and reader-based APIs return identical `NpzInspectInfo` regardless of substrate — **commit**
- [ ] **Range-read access pattern documentation** (`src/parse/npz.rs` rustdoc) — document the seek pattern an HTTP-range adapter should be able to satisfy efficiently: end-of-file scan for EOCD (~64 KiB), one read for the central directory (~few KiB for typical ML NPZ), one read per entry for the local file header + NPY header (~512 B each). Spells out for downstream adapter authors what shape of caching layer makes the implementation efficient (e.g., a single CD prefetch on first seek). No code, just rustdoc — **commit** — **PUSH + tag `v0.4.3`**

**Deliverable:** `anamnesis` v0.4.3 — reader-generic NPZ inspection. candle-mi (and any other downstream consumer) can now inspect a remote NPZ archive without materialising the data segment by composing `inspect_npz_from_reader` with an HTTP-range-backed `Read + Seek` adapter. Backward compatible: the existing `inspect_npz(path)` entry-point is unchanged (now implemented as a 2-line wrapper). No new dependencies, no feature gate — the whole change is a public-API extension on the already-shipping `npz` feature. — **PUSH + tag `v0.4.3`**

**New dependencies:** None. Reuses the existing `npz` feature gate, the existing `zip` v2 dep, and the existing `NpzInspectInfo` types.

**Why reader-generic, not built-in HTTP:** A `npz-remote` feature with an embedded HTTP/TLS transport (e.g., `ureq` + rustls) was considered and rejected. Reasons: (a) anamnesis's dep tree is consciously lean — adding TLS would expand the audit surface and re-trigger the full publish dry-run gauntlet for a single download-orchestration concern; (b) HTTP belongs in `hf-fm`, which already range-reads safetensors and is the natural home for a `Read + Seek` HTTP adapter; (c) the reader-generic API has the additional payoff of in-memory `Cursor<&[u8]>` inspection, mirroring the in-memory path the v0.4.1 candle-mi dogfooding round opened on the `.pth` side. The HTTP convenience layer can land later in a dedicated crate or as an opt-in feature if multiple downstream users request it.

**Explicitly out of scope:**

- **Built-in HTTP transport** — see "Why reader-generic, not built-in HTTP" above. The HTTP-range-backed `Read + Seek` adapter lives in `hf-fm` (or any other downstream caller), not in anamnesis.
- **Remote `parse_npz` (data-fetching variant)** — only `inspect_npz_from_reader` (metadata-only) is in scope. A reader-generic `parse_npz_from_reader` is a natural follow-up if a concrete downstream need arises, but the current dogfooding signal is exclusively about inspection (16 bytes of shape metadata), not data extraction.

### Phase 5: Quantization (Lethe)

**Goal:** The opposite direction — take full-precision weights and quantize them. Built on parse (read the source) + lethe (compress). With Phase 4 complete, this enables **cross-format conversion** via the dequantize-then-requantize path (e.g., GPTQ safetensors → BF16 → GGUF Q4_K_M). No Rust tool can do this today.

**Strategy:** Phase 5 ships a deliberately narrow first cut and **claims its own minor version (`v0.5.0`)** because Lethe is the architectural inverse of Anamnesis (Phases 1–4.5) and deserves a minor-version namespace of its own — even when the initial coverage is one quant family rather than the full encode-side matrix. The minimum viable end-to-end pipeline ships **BnB encode** as step 1 (this Phase), then proceeds through Phase 6 (Format Conversion Matrix), Phase 6.5 (Benchmarking), and Phase 7 (Python Bindings) to validate the complete `parse → remember → forget → convert → bind` chain on a single quant family. Once the architecture is proven end-to-end, the remaining encode kernel families (FP8, GGUF legacy block, GGUF K-quants, etc.) land in **Phase 7.5+** as a focused encode-completion milestone.

- [ ] **Step 1 — `BnB` encode (`NF4` + `FP4` + `INT8`) + round-trip validation harness** (`src/lethe/bnb.rs` + `src/lethe/round_trip.rs`) — three encode kernels mirroring the Phase 2 BnB decode kernels (shipped at v0.2.0, refactored at v0.4.0): `encode_nf4` and `encode_fp4` use the same fixed compile-time codebooks (no search — just nearest-codebook-entry lookup over the 16-entry table); `encode_bnb_int8` is per-row absmax-derived scale + `round(x / scale)`. The round-trip validation harness asserts bit-exactness for `encode(decode(q, scale)) == q` across the entire valid index range — the codebook itself is the oracle, no external Python reference needed. Plus cross-validation against PyTorch `bitsandbytes` on the existing 4 BnB fixtures: encode → load in PyTorch → assert equal. The harness pattern (generic `assert_bit_exact_decode_encode` helper) is the foundation for every subsequent encode kernel; subsequent steps extend it rather than reinvent it. ~1100 LOC, ~4 effort units — **commit** — **PUSH + tag `v0.5.0`**

**Deliverable:** `anamnesis` v0.5.0 — **Lethe lands.** Phase 5 ships BnB encode (NF4 / FP4 / INT8) + the round-trip validation harness that every subsequent encode kernel will reuse. Lethe gets its own minor version even though the initial coverage is one quant family — the architectural commitment (`src/lethe/`, the round-trip oracle, the encode-side public API surface) is what `v0.5.0` represents. The remaining encode kernel families (FP8, GGUF legacy block, GGUF K-quants, IQ4_NL/XS, MXFP4, TQ) move to **Phase 7.5: Lethe Encode Completion**, landing after the full chain has been validated through Python bindings in Phase 7. — **PUSH + tag `v0.5.0`**

### Phase 6: Format Conversion Matrix

**Goal:** Wire the conversion pipeline — `parse → remember → forget → write` end-to-end via a single CLI primitive. With Phase 5 complete (BnB encode), v0.6.0 ships every decode-side conversion (any input format → safetensors BF16) plus the BnB encode path (any input → BnB safetensors); the remaining encode-side rows of the matrix (GGUF / FP8 / IQ / TQ / MXFP4) light up at v0.7.5 once Phase 7.5 lands the missing encode kernels, accessible through the same `convert()` API and the same `amn convert` subcommand.

**Key conversions unlocked (cumulative across Phase 6 and Phase 7.5):**

| From | To | Use case | Available |
|------|----|----------|-----------|
| AWQ safetensors | safetensors BF16 | Remove quantization for fine-tuning | v0.6.0 |
| GGUF | safetensors BF16 | Load llama.cpp models in candle/burn | v0.6.0 |
| NPZ | safetensors | Migrate JAX weights to HuggingFace ecosystem | v0.6.0 |
| PyTorch .pth | safetensors | Migrate legacy PyTorch weights | v0.3.1 (shipped) |
| safetensors BF16 | safetensors BnB-NF4 | Compact BnB-quantized model from BF16 source | v0.6.0 (Phase 5 forget) |
| GPTQ safetensors | GGUF Q4_K_M | Deploy HuggingFace models in llama.cpp/Ollama | **v0.7.5** (Phase 7.5) |
| safetensors BF16 | GGUF Q4_K_M | Quantize for local inference | **v0.7.5** (Phase 7.5) |
| safetensors BF16 | GGUF IQ4_XS | Compact "small but accurate" GGUF for local inference | **v0.7.5** (Phase 7.5) |

- [ ] GGUF output writer (header + tensor info + F32 / BF16 passthrough emit) — full quantized-block emit (K-quant / IQ / TQ / MXFP4) unlocks once Phase 7.5 encoders land and plug into the same writer scaffold — **commit**
- [ ] `amn convert` CLI subcommand — `amn convert model.gguf --to safetensors -o model.safetensors` (and `--to bnb-nf4` from v0.6.0); `--to gguf-q4km` and other quantized targets light up at v0.7.5 via the same dispatch — **commit**
- [ ] Cross-format round-trip validation — convert through every v0.6.0-available pair, then back, measure distortion — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.6.0 — the convert primitive plus every decode-side conversion (any input → safetensors BF16) and the BnB encode path (any input → BnB safetensors). No Rust or Python tool offers this convert primitive today. The remaining encode-side rows complete in Phase 7.5. — **PUSH + tag `v0.6.0`**

### Phase 6.5: Benchmarking & Performance Validation

**Goal:** Establish reproducible performance baselines before the Python bindings expose anamnesis to a much larger audience. Benchmark every dequantization kernel (throughput) and validate memory efficiency claims (peak heap). All benchmarking infrastructure is dev-only — zero impact on the published crate.

**Approach:** Use [criterion](https://github.com/bheisler/criterion.rs) for runtime throughput benchmarks (statistical rigor, regression detection, baselines). Use [dhat-rs](https://github.com/nnethercote/dhat-rs) for peak-heap assertions in integration tests. Both are `[dev-dependencies]` only — bench code in `benches/`, memory tests in `tests/`.

**Runtime benchmarks (`benches/`):**

- [ ] Dequantization throughput — one benchmark per kernel family (FP8, GPTQ 4-bit, AWQ 4-bit, BnB NF4, BnB INT8, GGUF Q4_K), reporting elements/sec on synthetic tensors sized like real model layers (e.g., 4096x11008) — **commit**
- [ ] Parsing throughput — safetensors, NPZ, PTH, GGUF header parsing, reporting MB/s against raw `fs::read` baseline — **commit**

**Peak-memory validation (`tests/`):**

- [ ] Peak-heap assertions for GPTQ/AWQ dequantization — verify lazy precomputation keeps peak heap within `output_size + O(out_features)`, not `output_size + O(num_groups × out_features)` — **commit**
- [ ] Peak-heap assertions for BnB double-quant — verify no intermediate byte-serialization allocation — **commit** — **PUSH**

**New dev-dependencies:** `criterion` (runtime), `dhat` (peak heap). Not compiled into the published crate.

**Deliverable:** Baselines recorded, CI can detect regressions, performance claims in README are backed by reproducible numbers. No version bump — this is infrastructure, not a user-facing release.

### Phase 7: Python Bindings (PyO3)

**Goal:** Expose anamnesis to the Python ecosystem via `pip install anamnesis`. Phases 1–6 give anamnesis the fastest dequantization (2.7–54× vs PyTorch), the fastest NPZ parser (17.7× vs npyz), PyTorch `.pth` parsing, GGUF support, BnB encoding, and the `convert()` pipeline scaffold — all in pure Rust. Python bindings multiply the audience by ~100×, replacing ad-hoc dequantization scripts across the ML community. By shipping after the format conversion scaffold, `pip install anamnesis` exposes `convert()` from day one; the remaining encode-side targets (GGUF / FP8 / IQ / TQ / MXFP4) light up at v0.7.5 / Phase 7.5 through the same Python API once those kernels land.

**Approach:** Use [PyO3](https://pyo3.rs/) + [maturin](https://github.com/PyO3/maturin) to build a native Python extension. The Python API should mirror the Rust library API closely: `parse()`, `inspect()`, `remember()`, `forget()`, `convert()`, `parse_npz()`. Returns NumPy arrays (via `numpy` interop) or raw bytes. Ships as a wheel on PyPI.

- [ ] PyO3 module scaffold (`src/python.rs`) — feature-gated behind `python` — **commit**
- [ ] `parse_npz()` binding — returns `dict[str, NpzTensor]` with NumPy-compatible arrays — **commit**
- [ ] Safetensors `parse()` + `inspect()` bindings — **commit**
- [ ] `remember()` / `forget()` / `convert()` bindings — dequantize, quantize, and convert from Python — **commit**
- [ ] `maturin` build config, PyPI packaging, CI workflow — **commit**
- [ ] Python test suite — validate against PyTorch reference on the same fixtures — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.7.0 — `pip install anamnesis` works, with the full conversion matrix exposed. — **PUSH + tag `v0.7.0`**

**New dependencies:** `pyo3`, `numpy` (PyO3 interop). Feature-gated behind `python`.

### Phase 7.5: Lethe Encode Completion

**Goal:** Close the encode-side coverage gap left by Phase 5's narrow first cut. `v0.5.0` ships **BnB** encode (`NF4` / `FP4` / `INT8`) plus the round-trip validation harness; this phase extends that harness to every other codebook-style kernel family already supported on the decode side: **FP8** (Phase 1), **GGUF legacy block** + **K-quants** (Phase 4), and **GGUF IQ-quants** + **TQ** + **MXFP4** (Phase 4.5). After this release, anamnesis can encode into every block-quant format it can decode — closing the loop that the Phase 6 conversion matrix opened against the BnB-only encode side, and giving Phase 7's Python bindings a complete encode surface.

**Approach:** Mirror the Phase 4 / Phase 4.5 playbook on the encode side. Every kernel here is a codebook-LUT inversion (find the nearest entry in a fixed table) plus a per-block scale derivation (per-row absmax, per-block max, or sub-block stats). The encode loop body is structurally identical across families; only the codebook constants and the per-block scale formula differ — exactly the same pattern that justified collapsing six decode-kernel families into Phase 4.5.

Each kernel reuses the generic `assert_bit_exact_decode_encode` round-trip harness from Phase 5 (the codebook itself is the oracle for codebook-LUT kernels — no external Python reference needed for the round-trip). Each kernel ALSO ships a cross-validation test against the same Python reference used on the decode side: `gguf.quants.quantize()` for the GGUF families (legacy block, K-quants, IQ, TQ, MXFP4), and `torch.float8_e4m3fn` / `torch.float8_e5m2` casts for FP8. Fixtures are the existing `bartowski/...` GGUF slices and the seven FP8 model slices from Phase 1 — no new downloads.

**Why now (not earlier):** sequencing this *after* Phase 7 means the full `parse → remember → forget → convert → bind` chain has already been validated end-to-end through Python on a single quant family (BnB). Architectural mistakes in `lethe/` would have been caught by then. Phase 7.5 is then a focused encode-completion sweep against a stable architecture rather than co-evolving with one.

- [ ] **Step 1 — `FP8` encode (`E4M3` + `E5M2`, per-tensor / per-channel / fine-grained block schemes)** (`src/lethe/fp8.rs`) — three encode flows mirroring the v0.1.0 decode kernels: per-tensor (single scale), per-channel (one scale per row, `[N,1]`), and fine-grained 128×128 block scales. For each scheme, `scale = absmax / fp8_max` at the appropriate granularity, divide and round-to-nearest-even via `float8::F8E4M3::from_f32` (and `F8E5M2::from_f32` for the E5M2 variant). The round-trip harness asserts `decode(encode(x)) ≈ x` to within FP8 representation error (FP8 is lossy by construction); the *re-encode* path on the existing 7 FP8 fixtures DOES require bit-exactness — `encode(decode(q)) == q` once the scales stabilise. Cross-validation against `torch.float8_e4m3fn` cast on 256×256 slices from the existing `Llama-3.2-1B-Instruct-FP8`, `Qwen3-1.7B-FP8`, and `EXAONE-4.0-1.2B-FP8` fixtures. ~600 LOC, ~3 effort units — **commit**

- [ ] **Step 2 — GGUF legacy block encode (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`)** (`src/lethe/gguf_legacy.rs`) — six encode kernels for the 32-element legacy block family. Per-block absmax → `d` (block scale); divide and round each element to the nearest signed integer in the block-type's bit width; pack low/high nibbles for the 4-bit and 5-bit variants exactly inverting the decode-side unpack. `Q*_1` variants additionally derive `m` (block min) for asymmetric quantisation, mirroring the `(scale, min)` pair the decode side reads. Reuses the legacy-block runner that Phase 4 introduced (`run_legacy_kernel`) on the encode side. Cross-validation against `gguf.quants.quantize()` on the same `bartowski/SmolLM2-135M-Instruct-GGUF` slices used for decode-side validation. ~800 LOC, ~4 effort units — **commit**

- [ ] **Step 3 — GGUF K-quants encode (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`)** (`src/lethe/gguf_kquants.rs`) — six encode kernels for the 256-element super-block family. Per-block absmax → top-level `d`; per-sub-block `(scale, min)` derivation matching the decode-side `get_scale_min_k4` 6-bit packed-scale layout. `Q3_K` requires the inverse of the `kmask1`/`kmask2` permute (`q3_k_pack_scales` mirroring the existing `q3_k_unpack_scales`); a private `#[inline]` helper with its own unit tests, same pattern as the decode side. Reuses every constant table already ported from `ggml-common.h`. Cross-validation against `gguf.quants.quantize()` on the existing `bartowski/TinyLlama-1.1B-Chat` and `SmolLM2-135M-Instruct` slices. ~1200 LOC, ~5 effort units — **commit**

- [ ] **Step 4 — IQ4 + IQ2 family encode (`IQ4_NL`, `IQ4_XS`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`)** (`src/lethe/gguf_iq.rs`) — five codebook-LUT encode kernels. `IQ4_NL` / `IQ4_XS` walk the 16-entry `kvalues_iq4nl` codebook finding the nearest signed value per element, then pack into the existing low/high-nibble layout. `IQ2_XXS` / `IQ2_XS` / `IQ2_S` walk the 256 / 512 / 1024-entry signed-vector grids (`IQ2XXS_GRID`, `IQ2XS_GRID`, `IQ2S_GRID`) finding the nearest 8-element signed-vector entry per source chunk, then pack the index plus the matching `ksigns_iq2xs` sign mask into the block's `qs` / `qh` / `scales` fields. Lattice search is the new primitive; a single `nearest_signed_vector_index` helper covers all three IQ2 variants by varying grid size. Cross-validation against `gguf.quants.quantize()` on the existing `bartowski/Mistral-7B-Instruct-v0.3-GGUF` slices. ~1500 LOC, ~7 effort units — **commit**

- [ ] **Step 5 — IQ3 + IQ1 family encode (`IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M`)** (`src/lethe/gguf_iq.rs`) — four codebook-LUT encode kernels with two new wrinkles. `IQ3_XXS` / `IQ3_S` reuse the lattice-search helper from step 4 against the 256-entry `IQ3XXS_GRID` and 512-entry `IQ3S_GRID`; `IQ3_S` additionally inverts the unusual odd-integer scale formula `d × (1 + 2·nibble)` and the low/high nibble pairing across two consecutive sub-blocks. `IQ1_S` / `IQ1_M` use the 2048-entry `IQ1S_GRID` plus the `±IQ1S_DELTA = 0.125` *additive* bias — encode subtracts the bias before nearest-grid lookup, instead of IQ2/IQ3's multiplicative ±1 sign. `IQ1_M`'s scattered-`f16` super-scale layout (no top-level `d`) requires the inverse pack of the four-byte `scales[8]`-as-`f16` reconstruction. Cross-validation on the existing `bartowski/Mistral-7B-Instruct-v0.3-GGUF` slices (`IQ3_XXS` and `IQ3_S` ride the same shared file as decode-side step 3; `IQ1_S` / `IQ1_M` use their dedicated fixtures). ~1300 LOC, ~6 effort units — **commit**

- [ ] **Step 6 — TQ + MXFP4 encode (`TQ1_0`, `TQ2_0`, `MXFP4`)** (`src/lethe/gguf_tq.rs` + `src/lethe/mxfp4.rs`) — three encode kernels sharing the synthetic-fixture validation pattern the decode side established for the same kernels. **TQ:** per-block absmax → `d`; map each element to `{-1, 0, +1}` via threshold `d/2`; for `TQ1_0` re-pack five ternaries per byte using base-3 multiplication (inverse of the `pow3` decode trick); for `TQ2_0` re-pack four ternaries per byte using plain 2-bit packing. **MXFP4:** per-block absmax → `e: u8` (E8M0 byte exponent) via a new `f32_to_e8m0_half` helper (inverse of the existing `e8m0_to_fp32_half`); divide each element by the half-scale, find nearest entry in the 16-entry signed `K_VALUES_MXFP4` codebook, pack low/high nibbles. Reuses the `run_legacy_kernel` outer-loop runner. Synthetic-fixture cross-validation against `gguf.quants.quantize()` (Python supports both, mirrors Phase 4.5 step 5/6). ~600 LOC, ~3 effort units — **commit**

- [ ] **Step 7 — `forget()` dispatch + `forget` CLI subcommand + cumulative cross-validation** (`src/lib.rs` + `src/bin/main.rs`) — extend the `TargetKernel` enum and `ParsedModel::forget()` dispatch (introduced in Phase 5 for BnB) to recognise every kernel landed in steps 1–6, plus the `amn forget model.safetensors --to <kernel> -o out.gguf` CLI subcommand (alias `quantize`). The cross-validation suite is the cumulative roll-up of every per-step cross-validation test — bit-exactness contract holds at 0 ULP for the codebook-LUT kernels (steps 2–6) and within FP8 representation error for the FP8 path (step 1). Mirrors the v0.4.2 closeout step from Phase 4.5. ~300 LOC, ~2 effort units — **commit** — **PUSH + tag `v0.7.5`**

**Deliverable:** `anamnesis` v0.7.5 — Lethe encode completion. Every block-quant format anamnesis can decode out of can now also be encoded into. Combined with Phase 6's any-to-any conversion CLI and Phase 7's Python bindings, `pip install anamnesis` ships a complete read/write/convert matrix for the GGUF + FP8 + BnB universe — the first time any tool, Rust or Python, has covered all three families end-to-end. — **PUSH + tag `v0.7.5`**

**New dependencies:** None. Reuses the `lethe/` namespace (introduced in Phase 5), the `gguf` and `bnb` feature gates, `half`, and `float8`. Codebook constants from Phase 4.5 are reused verbatim — encode-side helpers live alongside them in the existing `iq_grids` submodule.

**Explicitly out of scope:**

- **GPTQ encoding** — requires a calibration dataset and OBQ/Hessian-based weight optimisation. Calibration-aware quantisation is a fundamentally different architecture (gradient computation, per-layer quant search) and belongs in its own dedicated phase, not in the codebook-LUT family of Phase 7.5.
- **AWQ encoding** — same reason as GPTQ; needs activation statistics from a calibration loop.
- **Big-endian GGUF v3 emit** — mirrors the Phase 4.5 deferral on the decode side. No committed target.

### Phase 8: Emerging Quantization Formats

**Goal:** Extend coverage to newer quantization formats that currently have zero Rust implementations. Prioritized by ecosystem adoption (HuggingFace model count as of March 2026).

**Landscape (March 2026):**

| Format | HF Models | Ecosystem | Rust Status |
|--------|-----------|-----------|-------------|
| EXL2 | ~4,775 | ExLlamaV2-only (GPU, hobbyist/local) | None |
| HQQ | ~190 | Transformers, vLLM (calibration-free) | mistral.rs only (inference-coupled) |
| AQLM | ~120 | Transformers, vLLM (academic, sub-2-bit) | None |

**Note:** EXL2's successor (EXL3) is emerging. HQQ and AQLM have strong technical merits but very low adoption. These formats are monitored; implementation is deferred until adoption grows or a concrete downstream need arises.

- [ ] EXL2 dequantization — mixed-bitrate (2–8 bpw), ExLlamaV2 binary format — **commit**
- [ ] HQQ dequantization — half-quadratic quantization, 1–8 bit, calibration-free — **commit**
- [ ] AQLM dequantization — additive quantization, sub-2-bit with vector codebooks — **commit**
- [ ] Cross-validation for each format — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.8.0 — emerging format coverage. — **PUSH + tag `v0.8.0`**

### Phase 9: CPU SIMD Pass

**Goal:** Replace the scalar pass-2 `f32 → BF16` loop in every dequantiser with an explicit AVX2 / NEON vectorised version, giving a 4–8× speedup on the hot path without touching the correctness-verified scalar kernels. This is the first time anamnesis commits to explicit SIMD intrinsics — up to v0.8.0 the project has relied entirely on auto-vectorisation through the loop-fission + `chunks_exact_mut(2)` pattern.

**Why this is a single phase and not per-format work:** Phases 1–4.5 all share the same `f32_bits_to_bf16_bits` pattern (defined once at [src/remember/fp8.rs:101](src/remember/fp8.rs#L101) and reused verbatim by `gptq.rs`, `awq.rs`, `bnb.rs`, and `gguf.rs`). A single `f32x8_to_bf16x8` SIMD helper retrofits every existing dequantiser at once, with one `unsafe` module, one `// SAFETY:` contract, one benchmark harness, and one cross-format round of cross-validation. Splitting the work per format would duplicate the intrinsic surface area across five modules and violate `CONVENTIONS.md`'s "unsafe lives in a single, dedicated module" rule.

The Phase 5 / 7.5 encode kernels write quantized bytes (not BF16), so they don't share this exact pass — but their absmax-then-round-and-pack inner loops are similarly auto-vectorisation friendly and could pick up SIMD intrinsics in a Phase 9 follow-up if encode-side benchmarks (Phase 6.5 + Phase 7.5 cross-validation results) demand it.

**Why now (not earlier):** Auto-vectorisation handles the pass-2 loop reasonably well already — the branch-free `chunks_exact_mut(2)` pattern was chosen specifically to give LLVM a clean target. Phase 9 is triggered when benchmark pressure from real users (enabled by the Phase 6 format conversion CLI and the Phase 7 Python bindings) surfaces kernels where the scalar fallback is a bottleneck. A `TODO(phase4-followup)` comment already sits on [`write_scratch_to_bf16`](src/remember/gguf.rs) flagging the intent.

**Approach:** Add a `simd` feature flag. Under that flag, introduce `src/remember/simd.rs` — a single dedicated module containing `#[target_feature(enable = "avx2")] unsafe fn f32x8_to_bf16x8(...)` and its NEON equivalent. The `write_scratch_to_bf16` helper (currently in `src/remember/gguf.rs` but logically shared across all dequantisers) moves to this module and becomes a runtime dispatcher: `is_x86_feature_detected!("avx2")` → AVX2 path, `std::arch::is_aarch64_feature_detected!("neon")` → NEON path, else fall through to the existing scalar loop. The scalar path stays the canonical correctness reference; the SIMD paths are tested bit-exactly against it via golden-vector cross-checks.

Per [`CONVENTIONS.md`](CONVENTIONS.md)'s accepted-unsafe table, all of the following must hold:

1. `unsafe` lives in a **single, dedicated module** (`src/remember/simd.rs`) — never scattered.
2. Every `unsafe` block carries a `// SAFETY:` comment documenting the invariants (lane count, alignment, target-feature preconditions).
3. The module is gated behind `#[cfg(feature = "simd")]` — users who don't enable the feature get `#![forbid(unsafe_code)]` unchanged.
4. The safe scalar fallback exists and is tested identically to the SIMD path.

- [ ] **`simd` feature flag + `remember::simd` module scaffold** — new `Cargo.toml` entry, `#[cfg(feature = "simd")] pub mod simd;` in `remember/mod.rs`, `cfg_attr(feature = "simd", allow(unsafe_code))` in `lib.rs`, and a new row in the `CONVENTIONS.md` accepted-unsafe table — **commit**
- [ ] **`f32x8_to_bf16x8` AVX2 intrinsic** — 8-wide round-to-nearest-even `f32 → BF16` lane-for-lane equivalent to the scalar `f32_bits_to_bf16_bits`. Bit-exact against the scalar reference (golden-vector test) — **commit**
- [ ] **`f32x4_to_bf16x4` NEON intrinsic** — ARM64 equivalent for Apple Silicon and AWS Graviton. Same bit-exactness requirement as the AVX2 path — **commit**
- [ ] **Runtime dispatch in `write_scratch_to_bf16`** — move the shared helper out of `src/remember/gguf.rs` into `src/remember/simd.rs` (or a new `remember::bf16_writer` shim). Retrofit FP8, GPTQ, AWQ, BnB, and GGUF pass-2 loops to call the shared dispatcher. Delete the per-kernel scalar copies — single source of truth — **commit**
- [ ] **Criterion bench harness** — benchmarks comparing scalar vs SIMD vs PyTorch CPU on the existing FP8/GPTQ/AWQ/BnB/GGUF cross-validation fixtures. Target: ≥ 4× speedup on `BF16`-writing pass 2 for tensors ≥ 1 MB. Publish the numbers in the crate README — **commit**
- [ ] **`copy_to_contiguous` flat-index pass-2 conversion** — replace the cross-iteration `coords[]` carry chain in [`src/parse/pth.rs::copy_to_contiguous`](src/parse/pth.rs) with the stateless `c_i = (flat_idx / stride_i) % shape_i` formula plus specialised fast paths for `ndim ∈ {1, 2, 3}`. Verify with `cargo-show-asm` that the loop emits SIMD strided gathers on AVX2/NEON. Path is rare (<0.1% of `state_dict` files) so this is a polish item, not a hot-path priority. Bit-exact unit tests already cover the function — **commit**
- [ ] **AWQ/GPTQ pass-2 zip-chain audit** — verify whether the four-way `chunks_exact_mut(2).zip(unpacked).zip(zeros).zip(scales)` pattern in [`src/remember/awq.rs`](src/remember/awq.rs) and [`src/remember/gptq.rs`](src/remember/gptq.rs) actually pessimises auto-vectorisation under stable LLVM. If `cargo-show-asm` shows scalar fallback, refactor to an indexed `for i in 0..len { … }` loop over pre-validated equal-length slices (per [`CONVENTIONS.md`](CONVENTIONS.md) "Reconciling bounds checking with vectorization"). Bit-exactness against PyTorch must hold at 0 ULP after refactor — **commit**
- [ ] **Cross-format correctness sweep** — re-run every Phase 1–4.5 cross-validation test with `--features simd` and assert bit-identical output against the scalar path. No format may regress — **commit** — **PUSH + tag `v0.9.0`**

**Deliverable:** `anamnesis` v0.9.0 — `BF16`-writing pass 2 runs 4–8× faster on AVX2 and NEON CPUs, scalar fallback preserved for `forbid(unsafe_code)` users, with cross-kernel consistency (FP8 / GPTQ / AWQ / BnB / GGUF all benefit from the same single-source-of-truth SIMD helper). — **PUSH + tag `v0.9.0`**

**New dependencies:** None at runtime — uses `std::arch` intrinsics directly. Optional dev-dependency on `criterion` for the bench harness.

### Phase 10: Streaming Output

**Goal:** Drop peak heap from `O(total_dequantised_size)` to `O(largest_tensor_BF16)` for whole-model dequantisation paths (`ParsedModel::remember` and the `amn remember model.gguf -o out.safetensors` CLI). Unblocks 70 B+ model conversion on commodity hardware (≤ 32 GB) where the current eager-buffering approach OOMs.

**Background:** The dequantisation kernels already provide streaming entry points — [`dequantize_gguf_blocks_to_bf16`](src/remember/gguf.rs) is `O(one block)` per call, and the `FP8`/`GPTQ`/`AWQ`/`BnB` kernels emit one tensor at a time. The orchestrators do not stream: [`ParsedModel::remember_bf16_inner`](src/model.rs) accumulates every dequantised tensor's `Vec<u8>` into a `dequantized_data: Vec<(name, bytes, shape)>` and then hands all `TensorView`s to `safetensors::serialize_to_file` simultaneously. The same pattern lives in [`src/bin/main.rs::run_remember_gguf`](src/bin/main.rs). The `safetensors` 0.4 crate's writer **already streams tensor bodies** (one `BufWriter::write_all` per tensor inside `serialize_to_file`) — the eager buffering is ours, not the crate's.

**Approach:** Implement a custom `View` impl whose `data()` lazily dequantises a single tensor on demand, paired with an iterator that yields `(name, LazyView)` derived from the parsed model. The crate's `prepare()` step only needs `data_len()` / `dtype()` / `shape()` for the header — all deterministic and cheap. Hand the iterator to `safetensors::serialize_to_file`; the writer pulls bytes per tensor and drops them after each `write_all`. Peak heap drops to `O(largest_tensor_BF16)` — for a 70 B model's biggest layer (~500 MB BF16) this fits on commodity hardware. True per-block O(1) streaming is **out of scope** for Phase 10 — it would require a custom safetensors writer (the `View` trait returns a single `Cow<[u8]>`, not a stream) and the per-tensor variant is sufficient for every model size in current circulation.

- [ ] **`LazyDequantView<'a>` type** in `src/remember/streaming.rs` — implements `safetensors::tensor::View`, captures a borrow into the parsed model + the per-tensor metadata needed to dispatch into the right kernel, and returns `Cow::Owned(dequantised_bytes)` from `data()` on demand — **commit**
- [ ] **`ParsedModel::remember_streaming`** — yields lazy views to `safetensors::serialize_to_file`. Rewrite `remember()` to delegate to `remember_streaming()` so existing call sites pick up the win automatically — **commit**
- [ ] **GGUF CLI streaming path** — refactor `run_remember_gguf` in `src/bin/main.rs` to drop the `tensor_data: Vec<(...)>` accumulator in favour of a lazy iterator. Same `LazyDequantView` infrastructure — **commit**
- [ ] **Peak-heap regression tests via `dhat-rs`** — assert peak heap ≤ `2 × largest_tensor_BF16 + small_constant` on a representative 7 B fixture (FP8 + GGUF). Reuses Phase 6.5 dhat-rs infrastructure — **commit**
- [ ] **Cross-format correctness sweep** — re-run every Phase 1–4.5 cross-validation test against the streaming output path; output bytes must be byte-identical to the eager path on the same input — **commit** — **PUSH + tag `v0.10.0`**

**Deliverable:** `anamnesis` v0.10.0 — peak heap drops from `O(model_size)` to `O(largest_tensor)` for whole-model dequantisation. 70 B+ models become convertible on commodity hardware. All existing public APIs preserved (`remember()` becomes a thin wrapper). — **PUSH + tag `v0.10.0`**

**New dependencies:** None at runtime. Reuses the Phase 6.5 `dhat-rs` dev-dependency for peak-heap assertions.

**Out of scope:**

- **True per-block O(1) streaming** — requires a custom safetensors writer (the `View::data()` method returns `Cow<[u8]>`, not a stream). Per-tensor lazy dequant covers every plausible model size; per-block remains a Future Directions item if real-world peak-heap measurements after Phase 10 ever show insufficient headroom.
- **`WASM32` linear-memory adaptation** — WASM32's 4 GB memory cap is independent of `usize` typing; supporting it requires WASM64 or a fundamentally different I/O model. Listed under Future Directions.

### Future Directions

The following are potential extensions beyond Phase 10, listed for context. They will be promoted to full phases when a concrete need arises.

- **Model surgery** — extract specific layers, merge LoRA adapters with base weights at the file level, split/shard for distributed loading
- **Quantization quality analysis** — per-layer distortion reports, optimal mixed-precision selection, diff two quantized versions of the same model
- **WASM target** — compile anamnesis to WASM for browser-based model inspection (drop a safetensors file, see tensor layout and quantization scheme instantly)
- **GPU-accelerated dequantization** — compute shader (wgpu/Vulkan) kernels for dequantization hot loops, for cases where the Phase 9 CPU SIMD pass is still insufficient (e.g., bulk 70 B-parameter conversions on a workstation with an idle GPU)

---

## 4. Key Design Decisions

### Framework-agnostic output

anamnesis outputs standard safetensors files (for dequantization) and raw byte arrays with shape/dtype metadata (for NPZ). It never depends on candle, burn, or tch. Any framework can consume its output.

### Parse-first architecture

Every code path begins with parsing. The `parse/` module is the foundation; `remember/`, `lethe/`, and `inspect` are built on top. This is not incidental — it reflects the design principle that you cannot remember what you have not first parsed.

### Feature gates per format

Each quantization scheme and container format is behind its own feature flag. Users pay only for what they need. Default: `fp8` only.

### No GPU dependency

All operations are pure CPU. FP8 dequantization is bit manipulation, not matrix multiplication. This keeps the crate lightweight and universally deployable.

### SIMD strategy

Conversion loops process billions of elements and are embarrassingly parallel
(each element is independent). The strategy is layered:

1. **Phase 1: SIMD-friendly scalar code.** Write loops that follow the
   CONVENTIONS.md SIMD-friendly rules (contiguous slices, no branches,
   `chunks_exact`, hoisted invariants, separate input/output buffers).
   Verify auto-vectorization with `cargo-show-asm`. This requires **no
   `unsafe`**, no nightly, no extra dependencies, and builds on stable Rust.
   The compiler typically emits AVX2 on x86-64 (Intel + AMD) and NEON on ARM.

2. **If benchmarks demand more: hand-written intrinsics** behind a `simd`
   feature gate. `#[target_feature(enable = "avx2")]` + runtime
   `is_x86_feature_detected!` dispatch, with the scalar path as fallback.
   One dedicated module (`src/remember/simd.rs`), tested against the scalar
   reference. Requires `unsafe` (see CONVENTIONS.md `// SAFETY:` rules).

3. **Not pursued: `std::simd` (portable SIMD).** Requires nightly Rust, which
   would virally infect downstream crates (candle-mi, hf-fetch-model). If
   `std::simd` stabilizes, it becomes the preferred path and replaces both
   the scalar and intrinsic implementations.

Coverage: AVX2 covers all Intel + AMD CPUs since ~2015. NEON covers Apple
Silicon and ARM servers. The scalar fallback covers everything else.

### Error type

A single `AnamnesisError` enum with variants per failure category:
- `Parse { reason }` — format decoding failures (including `NPZ`/`NPY` and `ZIP` errors)
- `Unsupported { format, detail }` — recognized but unimplemented format
- `Io(std::io::Error)` — file system errors

---

## 5. Relationship to Other Projects

### hf-fetch-model

Download crate. Depends on anamnesis for `--dequantize` and `download_and_parse_npz()`. anamnesis provides the format intelligence; hf-fetch-model provides the network I/O. See hf-fetch-model v0.8.0 roadmap.

### candle-mi

MI framework. anamnesis intersects with candle-mi in two ways:

1. **NPZ migration.** candle-mi contained a tightly-coupled NPZ parser for Gemma Scope SAE weights. Phase 3 of anamnesis replaces it. Migration design: `candle-mi/design/migrate-npz-to-anamnesis.md`. `anamnesis` v0.3.0+ provides `parse_npz()` with a `From<AnamnesisError>` bridge to `MIError`. Target: candle-mi v0.1.5.

2. **Auto-config extensions.** The FP8 test models chosen for anamnesis Phase 1 validation introduce two new architectures that candle-mi should support:
   - **`exaone4`** (`LGAI-EXAONE/EXAONE-4.0-1.2B-FP8`) — LLaMA-like with alternating sliding window / full attention ("LLLG" pattern). Close to Gemma 2's alternating scheme. Requires a new `parse_exaone4()` in `config.rs`.
   - **`qwen3`** (`Qwen/Qwen3-1.7B-FP8`) — extends `qwen2` with QK LayerNorm (`q_norm` / `k_norm` per layer). Requires a new `parse_qwen3()` in `config.rs` and handling the extra norm tensors in the forward pass.

   These extensions benefit candle-mi independently of anamnesis — both model families are mainstream and worth supporting regardless.

### candle

HuggingFace's Rust ML framework. anamnesis outputs standard safetensors that candle can load directly. No dependency in either direction — anamnesis reads the safetensors format, not candle's tensor types.
