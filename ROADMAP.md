# Roadmap: anamnesis — Tensor Format Transformation for Rust

> *Parse any format, recover any precision.*

**Date:** March 20, 2026 (updated April 11, 2026)
**Status:** Phases 1–3.5 complete (v0.3.2 published). FP8/GPTQ/AWQ/BnB dequantization + NPZ parsing + PyTorch `.pth` parsing. **Phase 4 in progress:** parser + 12 block-quant dequant kernels (both legacy `Q4_0`–`Q8_1` and K-quants `Q2_K`–`Q8_K`) committed, streaming + Vec APIs shipped. Next: cross-validation against `llama.cpp` reference output, then `v0.4.0` tag.
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
  - [Phase 5: Quantization (Lethe)](#phase-5-quantization-lethe)
  - [Phase 6: Python Bindings (PyO3)](#phase-6-python-bindings-pyo3)
  - [Phase 7: Format Conversion Matrix](#phase-7-format-conversion-matrix)
  - [Phase 8: Emerging Quantization Formats](#phase-8-emerging-quantization-formats)
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

**Status:** Parser + all 12 block-quant dequantisation kernels + streaming API committed (`2acaf1a`, `cc4ecf8`, `b6610fe`, `3156104`). Cross-validation against `llama.cpp` + `v0.4.0` tag still pending.

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
- [ ] **Cross-validation against `llama.cpp` reference dequantization** on real GGUF models — pick a small model (e.g., `TinyLlama-1.1B-Q4_K_M`), run `llama.cpp`'s `quantize --inspect` or a purpose-built C harness to dump golden dequant output per tensor, then assert bit-exactness against anamnesis output across all 12 block types. Required to graduate from "scalar reference ported from ggml" to "proven equivalent to the ground truth". — **commit** — **PUSH + tag `v0.4.0`**

**Known follow-ups (deferred out of v0.4.0 scope):**

- **`IQ*` / `TQ*` / `MXFP4` dequant kernels** — recognised by the parser with `byte_len = None`, rejected by the dispatcher with `Unsupported`. Each family has its own block struct layout in `ggml-common.h` that would need porting. Not blocking v0.4.0 because no mainstream model relies on them yet.
- **Big-endian GGUF v3 support** — the parser detects byte-swapped magic and returns a clear `Unsupported` error; a reader rewrite can reuse `parse::utils::byteswap_inplace`.
- **AVX2 `f32x8 → bf16x8` pass-2 SIMD** — would give a ~4-8× speedup on pass 2 (~40-60% of total dequant time) but requires a `simd` feature, a new `remember::simd` module, an entry in the CONVENTIONS accepted-unsafe table, and runtime `is_x86_feature_detected!` dispatch. Flagged as a `TODO(phase4-followup)` comment on `write_scratch_to_bf16`.
- **GGUF CLI subcommands** — `amn parse model.gguf`, `amn inspect model.gguf`, `amn remember model.gguf --to bf16`. The library API is ready; CLI wiring in `src/bin/main.rs` is a small follow-up.
- **`ParsedGguf::dequantize_tensor(&self, info: &GgufTensorInfo)` convenience method** — users can call `dequantize_gguf_to_bf16(&tensor.data, tensor.dtype, tensor.shape.iter().product())` directly; a wrapper method can land if real callers would benefit.

**Deliverable:** `anamnesis` v0.4.0 — GGUF parsing + dequantization works end-to-end with bit-exact `llama.cpp`-validated output. anamnesis becomes the only Rust crate that can parse *both* safetensors-based (GPTQ/AWQ/BnB/FP8) and GGUF-based quantized models and dequantize everything to BF16. — **PUSH + tag `v0.4.0`**

**New dependencies:** None. The `gguf` feature pulls in `memmap2` (already used by `pth`) and relies on `half` (already mandatory). No third-party GGUF parser in the dependency tree.

### Phase 5: Quantization (Lethe)

**Goal:** The opposite direction — take full-precision weights and quantize them. Built on parse (read the source) + lethe (compress). With Phase 4 complete, this enables **cross-format conversion** via the dequantize-then-requantize path (e.g., GPTQ safetensors → BF16 → GGUF Q4_K_M). No Rust tool can do this today.

- [ ] FP8 quantization (`src/lethe/fp8.rs`) — BF16/F32 → FP8 E4M3 with fine-grained block scale factors — **commit**
- [ ] INT8/INT4 quantization — per-channel and group-wise schemes — **commit**
- [ ] `ParsedModel::forget()` public API + `forget` CLI subcommand (alias `quantize`) — **commit**
- [ ] Round-trip validation — quantize then dequantize, measure lethe distance — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.5.0 — full quantize + dequantize cycle. — **PUSH + tag `v0.5.0`**

### Phase 6: Python Bindings (PyO3)

**Goal:** Expose anamnesis to the Python ecosystem via `pip install anamnesis`. Phases 1–5 give anamnesis the fastest dequantization (2.7–54× vs PyTorch), the fastest NPZ parser (17.7× vs npyz), PyTorch `.pth` parsing, GGUF support, and quantization — all in pure Rust. Python bindings multiply the audience by ~100×, replacing ad-hoc dequantization scripts across the ML community.

**Approach:** Use [PyO3](https://pyo3.rs/) + [maturin](https://github.com/PyO3/maturin) to build a native Python extension. The Python API should mirror the Rust library API closely: `parse()`, `inspect()`, `remember()`, `forget()`, `parse_npz()`. Returns NumPy arrays (via `numpy` interop) or raw bytes. Ships as a wheel on PyPI.

- [ ] PyO3 module scaffold (`src/python.rs`) — feature-gated behind `python` — **commit**
- [ ] `parse_npz()` binding — returns `dict[str, NpzTensor]` with NumPy-compatible arrays — **commit**
- [ ] Safetensors `parse()` + `inspect()` bindings — **commit**
- [ ] `remember()` / `forget()` bindings — dequantize and quantize from Python — **commit**
- [ ] `maturin` build config, PyPI packaging, CI workflow — **commit**
- [ ] Python test suite — validate against PyTorch reference on the same fixtures — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.6.0 — `pip install anamnesis` works. — **PUSH + tag `v0.6.0`**

**New dependencies:** `pyo3`, `numpy` (PyO3 interop). Feature-gated behind `python`.

### Phase 7: Format Conversion Matrix

**Goal:** Wire the full pipeline — any supported input format to any supported output format in a single command. With Phases 1–6 complete, the core is `parse → remember → forget → write`. Phase 7 adds the CLI and Python API for end-to-end conversion, plus output writers for each target format.

**Key conversions unlocked:**

| From | To | Use case |
|------|----|----------|
| GPTQ safetensors | GGUF Q4_K_M | Deploy HuggingFace models in llama.cpp/Ollama |
| AWQ safetensors | safetensors BF16 | Remove quantization for fine-tuning |
| GGUF | safetensors BF16 | Load llama.cpp models in candle/burn |
| safetensors BF16 | GGUF Q4_K_M | Quantize for local inference |
| NPZ | safetensors | Migrate JAX weights to HuggingFace ecosystem |
| PyTorch .pth | safetensors | Migrate legacy PyTorch weights (already supported since v0.3.1) |

- [ ] GGUF output writer — write GGUF files with K-quant types from BF16 input — **commit**
- [ ] `amn convert` CLI subcommand — `amn convert model.safetensors --to gguf-q4km -o model.gguf` — **commit**
- [ ] Python `convert()` API — **commit**
- [ ] Cross-format round-trip validation — convert, then convert back, measure distortion — **commit** — **PUSH**

**Deliverable:** `anamnesis` v0.7.0 — any-to-any format conversion. No Rust or Python tool does this today. — **PUSH + tag `v0.7.0`**

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

### Future Directions

The following are potential extensions beyond Phase 8, listed for context. They will be promoted to full phases when a concrete need arises.

- **Streaming / memory-mapped processing** — mmap-based safetensors/GGUF parsing for 70B+ models (100+ GB) that exceed available RAM
- **Model surgery** — extract specific layers, merge LoRA adapters with base weights at the file level, split/shard for distributed loading
- **Quantization quality analysis** — per-layer distortion reports, optimal mixed-precision selection, diff two quantized versions of the same model
- **WASM target** — compile anamnesis to WASM for browser-based model inspection (drop a safetensors file, see tensor layout and quantization scheme instantly)
- **GPU-accelerated dequantization** — compute shader (wgpu/Vulkan) kernels for dequantization hot loops, for cases where CPU auto-vectorization is insufficient

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
