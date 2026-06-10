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
- [Parsing untrusted input](#parsing-untrusted-input)
- [Tested Models](#tested-models)
  - [FP8 Dequantization](#fp8-dequantization)
  - [GPTQ Dequantization](#gptq-dequantization)
  - [AWQ Dequantization](#awq-dequantization)
  - [BitsAndBytes Dequantization](#bitsandbytes-dequantization)
  - [BitsAndBytes Quantization (Lethe — Phase 5)](#bitsandbytes-quantization-lethe--phase-5)
  - [Format Conversion Pipeline (Phase 6, v0.6.0)](#format-conversion-pipeline-phase-6-v060)
  - [GGUF Block-Quant Dequantization](#gguf-block-quant-dequantization)
- [Safetensors Header Inspection](#safetensors-header-inspection)
- [NPZ/NPY Parsing](#npznpy-parsing)
- [GGUF Inspection](#gguf-inspection)
- [PyTorch `.pth` Parsing](#pytorch-pth-parsing)
- [PyTorch `.pth` Inspection](#pytorch-pth-inspection)
- [Parser robustness (Phases 6.6–6.8, v0.6.1–v0.6.3)](#parser-robustness-phases-6668-v061v063)
- [Performance validation (Phase 6.5)](#performance-validation-phase-65)
- [Used by](#used-by)
- [License](#license)
- [Development](#development)

## Install

```sh
cargo install anamnesis --features cli,pth,gguf
```

Installs both `anamnesis` and `amn` (short alias). Feature flags: `gptq`, `awq`, `bnb`, `npz`, `pth`, `gguf`, `indicatif` (progress bars).

## CLI Commands

| Command | |
|---------|---|
| `amn parse <file>` | Parse and summarize a model file (`.safetensors`, `.pth`, `.npz`, `.gguf`) |
| `amn inspect <file>` | Show format, tensor counts, size estimates, and byte order |
| `amn remember <file>` | Dequantize to BF16 (safetensors) or convert `.pth`/`.gguf` → `.safetensors` |
| `amn convert <file> --to <target>` | Convert any input format to `safetensors` / `gguf` / `bnb-nf4` through a single dispatch (Phase 6, v0.6.0) |

Aliases: `amn info` = `amn inspect`, `amn dequantize` = `amn remember`.

Format detection is automatic: `.safetensors` files go through the dequantization pipeline, `.pth`/`.pt` files go through the pickle parser, `.npz` files go through the header-only NPZ inspector, `.gguf` files go through the GGUF parser. `.bin` files are probed for ZIP/GGUF magic to distinguish PyTorch, GGUF, and safetensors. `amn convert` reuses the same detector and dispatches to the appropriate (input × target) pair; combinations not in the v0.6.0 matrix (e.g. `pth → bnb-nf4`) return a clear `Unsupported` error rather than silently falling through.

Build with `cargo install anamnesis --features cli,pth,gguf,ollama` to also enable the **`ollama:` URL scheme** — every subcommand resolves `ollama:<model>:<tag>` to the local Ollama cache's GGUF blob:

```
$ amn inspect ollama:llama3.2:1b
Format:      GGUF v3
Arch:        llama
Tensors:     147
Total size:  1.22 GB
Dtypes:      F32, Q8_0
Alignment:   32 bytes
```

The resolver reads the Ollama manifest at `~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>` and returns the blob path under `~/.ollama/models/blobs/sha256-<hash>` — pure path arithmetic plus a single JSON read, no `ollama` CLI shell-out, no Go interop. Honours the `OLLAMA_MODELS` environment variable for non-default cache locations.

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

## Parsing untrusted input

A `.safetensors`, `.npz`, `.pth`, or `.gguf` file is attacker-controllable: it
can *declare* arbitrary tensor counts, shapes, string lengths, or (for
compressed `.npz`) expansion ratios. anamnesis is built **parse-first** so a
multi-tenant or edge host can reject a hostile or simply too-large file against
limits **it** chooses, before committing memory. The recommended recipe is
**inspect → check policy → parse under limits**:

```rust
use anamnesis::{inspect_npz_from_reader, parse_npz_with_limits, ParseLimits};

// 1. Inspect cheaply — header-only, bounded; never materialises tensor data.
let info = inspect_npz_from_reader(&mut reader)?;

// 2. Check the declared totals against YOUR environment's policy, and reject
//    early — no parse, no allocation.
if info.total_bytes > my_ram_budget || info.tensors.len() > my_tensor_cap {
    return Err(/* 400 Too Large */);
}

// 3. Parse under a `ParseLimits` matched to this worker — the authoritative
//    gate, enforced fail-fast *before* each allocation. (`path` is the same
//    file; over HTTP-range you inspect remotely, then fetch and parse.)
let limits = ParseLimits::default()
    .with_max_total_bytes(my_ram_budget)
    .with_max_item_count(my_tensor_cap)
    .with_max_decompression_ratio(1000); // reject a DEFLATE entry inflating >1000×
let tensors = parse_npz_with_limits(path, &limits)?;
```

Every parser has a `parse_*_with_limits` sibling (`parse_with_limits` for
safetensors, `parse_gguf_with_limits`, `parse_npz_with_limits`,
`parse_pth_with_limits`). The four `ParseLimits` axes:

| Axis | Builder | Bounds |
|---|---|---|
| Single allocation | `with_max_single_alloc` | the largest single header-declared buffer a parser allocates |
| Cumulative heap | `with_max_total_bytes` | the running sum of those allocations across the whole file (the many-small-items blow-up) |
| Item count | `with_max_item_count` | declared tensors / arrays / KV entries / archive members |
| Decompression ratio | `with_max_decompression_ratio` | a compressed entry's uncompressed:compressed ratio (`DEFLATE` `.npz` zip-bomb cap) |

`ParseLimits::default()` is **unbounded** on every axis, so the limit-free
`parse_*` functions and existing call sites are byte-for-byte unchanged — limits
are opt-in. Limits are **tighten-only**: where a built-in per-format cap already
exists (the single-allocation caps and the `GGUF` counts) the effective limit is
`min(cap, your limit)` — a caller can only make the parser *stricter*, never
weaker than that floor; the cumulative-heap and decompression-ratio axes have no
built-in cap and are pure caller-supplied bounds. The inspection totals
(`total_bytes`, `tensor_count` / `tensors.len()`) are a cheap *lower-bound*
pre-filter — they omit per-entry header overhead — so they are for the early
reject; `parse_*_with_limits` is the authoritative enforcement.

See also [Parser robustness](#parser-robustness-phases-6668-v061v063) for the
underlying per-format caps.

## Tested Models

> **Validation provenance (v0.6.4):** "Cross-validated" below means the expected output comes from the **canonical library's own code** — `bitsandbytes` (`dequantize_4bit` / `int8_vectorwise_dequant`), AutoAWQ (`unpack_awq` + `reverse_awq_order`), GPTQModel (`dequantize_weight` + its v1→v2 zero-point conversion), PyTorch's native fp8 cast, and `gguf-py`'s `dequantize` — never a hand-rolled reimplementation of the formula. v0.6.4 adopted this rule after dogfooding in candle-mi exposed a circularly-validated nibble-order bug (full post-mortem: [`docs/dogfooding-feedbacks/bnb-nibble-order-and-circular-fixture-validation.md`](docs/dogfooding-feedbacks/bnb-nibble-order-and-circular-fixture-validation.md)) and fixed three real defects (BnB nibble order, BnB double-quant `nested_offset`, AWQ GEMM interleave). The "vs PyTorch" speed columns are historical CPU-PyTorch timings (pre-v0.6.4 baselines); they document throughput, not correctness.

### FP8 Dequantization

Cross-validated against PyTorch's native fp8→f32 cast on 7 real FP8 models from 5 quantization tools. Bit-exact output (0 ULP difference). Auto-vectorized: SSE2 on any x86-64, AVX2 with `target-cpu=native`.

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

Cross-validated against GPTQModel's own dequant pipeline (`TorchLinear.dequantize_weight` + its loader-side v1→v2 zero-point conversion) on 4 real GPTQ models from 2 quantizers (AutoGPTQ, GPTQModel). Bit-exact output (0 ULP difference). Loop fission for full AVX2 vectorization.

| Model | Quantizer | Bits | vs PyTorch (AVX2) |
|---|---|---|---|
| Falcon3-1B-Instruct-GPTQ-Int4 | AutoGPTQ | 4 | 6.5x faster |
| Llama-3.2-1B-Instruct-GPTQ | AutoGPTQ | 4 | 12.2x faster |
| Falcon3-1B-Instruct-GPTQ-Int8 | AutoGPTQ | 8 | 7.0x faster |
| Llama-3.2-1B-gptqmodel-8bit | GPTQModel | 8 | 7.9x faster |

### AWQ Dequantization

Cross-validated against AutoAWQ's own unpack + reorder (`packing_utils.unpack_awq` + `reverse_awq_order`, including the GEMM nibble interleave `[0, 2, 4, 6, 1, 3, 5, 7]`) on 2 real AWQ models. Bit-exact output (0 ULP difference). 4-bit only — AutoAWQ's GEMM format defines no other width, so anamnesis rejects the rest rather than guess an interleave. Loop fission for full AVX2 vectorization.

| Model | Quantizer | Bits | vs PyTorch (AVX2) |
|---|---|---|---|
| llama-3.2-1b-instruct-awq | AutoAWQ | 4 | 5.7x faster |
| Falcon3-1B-Instruct-AWQ | AutoAWQ | 4 | 4.7x faster |

### BitsAndBytes Dequantization

Cross-validated against real `bitsandbytes` (`functional.dequantize_4bit` / `int8_vectorwise_dequant`) on 7 real BitsAndBytes models across 4 architecture families (NF4, FP4, double-quant incl. the `nested_offset` absmax recovery, INT8). Bit-exact output (0 ULP difference). Loop fission for AVX2 on NF4/FP4; single-pass AVX2 on INT8 (`vpmovsxbd` → `vcvtdq2ps` → `vmulps`).

| Model | Format | Elements | vs PyTorch (AVX2) |
|---|---|---|---|
| Llama-3.2-1B-Instruct-bnb-nf4 | NF4 | 4,096 | 21.8x faster |
| Llama-3.2-1B-BNB-FP4 | FP4 | 4,096 | 18.0x faster |
| Llama-3.2-1B-Instruct-bnb-nf4-double-quant | NF4 double-quant | 4,096 | 54.0x faster |
| Llama-3.2-1B-BNB-INT8 | INT8 | 65,536 | 1.2x faster |

> **Note:** INT8 speedup is modest because the operation is trivially simple (`i8→f32→multiply`). Both PyTorch and anamnesis are near memory bandwidth limits at ~0.7–0.8 ns/element. The AVX2 hot loop is fully vectorized — the 1.2× reflects the inherent ceiling, not a missed optimization.

### BitsAndBytes Quantization (Lethe — Phase 5)

The inverse direction. Phase 5 ships the `lethe` namespace alongside `remember`: `encode_bnb4` / `encode_bnb4_double_quant` / `encode_bnb_int8` plus the bit-exact `round_trip` validation harness. Cross-validated against real `bitsandbytes` on **7 fixtures across 4 architecture families** (Llama 3.2 / Qwen3 / Qwen2.5 / Phi-3.5): every fixture round-trips **byte-exact** (0 byte diffs) against the original `bitsandbytes`-quantised on-disk bytes — including the high-nibble-first packing order and the double-quant `nested_offset` (both fixed in v0.6.4).

| Fixture | Format | Elements | Byte-exact round-trip | vs PyTorch quantize (CPU) |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct-bnb-nf4 | NF4 plain | 4,096 | ✓ 0 / 2048 | 0.22× (slower) |
| Llama-3.2-1B-BNB-FP4 | FP4 plain | 4,096 | ✓ 0 / 2048 | 0.24× (slower) |
| Llama-3.2-1B-Instruct-bnb-nf4-double-quant | NF4 double-quant | 4,096 | ✓ 0 / 2048 | 0.22× (slower) |
| Llama-3.2-1B-BNB-INT8 | INT8 | 65,536 | ✓ 0 / 65536 | 0.03× (32× slower) |
| ema1234/qwen_mcqa_bnb_fp4 | FP4 plain (Qwen3) | 4,096 | ✓ 0 / 2048 | 0.20× (slower) |
| unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit | NF4 double-quant (Qwen2.5) | 4,096 | ✓ 0 / 2048 | 0.18× (slower) |
| unsloth/Phi-3.5-mini-instruct-bnb-4bit | NF4 double-quant (Phi-3.5) | 4,096 | ✓ 0 / 2048 | 0.18× (slower) |

> **Sign-of-zero preservation finding (FP4):** The on-disk `bitsandbytes` Python `FP4` `quant_map` stores `+0.0` at *both* index 0 *and* index 8 — collapsing the `±0` pair. A naive `decode → encode` round-trip would be mathematically impossible under that codebook. Phase 5 introduces a narrow, principled tweak in `dequantize_bnb4_to_bf16`: when a codebook entry is exactly `+0.0` AND the nibble has its high bit set (`nibble & 0x8 != 0`), the emitted `BF16` is `-0.0`. This recovers the sign information `bitsandbytes`' Python decode discards. Arithmetically invisible (both are IEEE 754 zero), affects `0.2 %` of `FP4` elements, no-op for `NF4`. The encoder mirrors the rule with `apply_sign_magnitude_encode_correction`. Confirmed to generalise: the Qwen3 FP4 fixture shows the same `+0.0` / `+0.0` codebook collapse and round-trips byte-exact under the rule.

> **Ecosystem finding (NF4 double-quant):** `hf-fm inspect` HTTP-range probes during cross-architecture candidate selection revealed that **every** non-Llama BnB-NF4 model checked uses double-quant — bitsandbytes' default. Plain NF4 is effectively a Llama-fixture-only phenomenon. Without `encode_bnb4_double_quant` (Step 1c), anamnesis would only encode a tiny corner of real-world BnB-4bit models. Promoted from deferred polish to required Step 1c gate on `v0.5.0`.

> **On the "slower than PyTorch" column:** The encode kernels are 4–6× slower than PyTorch's broadcast-vectorised quantize on `BnB4`, 32× slower on `INT8`. This is expected — PyTorch encode uses a single broadcast tensor op (`(blocks.unsqueeze(-1) - codebook).abs().argmin(dim=-1)`) that vectorises across the whole tensor; the Rust encode loop is currently scalar per element. **Phase 9 (CPU SIMD pass) is the natural target** — this table makes the gap visible. The same loop-fission + `target-cpu=native` infrastructure that gave the decode path its 18–54× wins is the candidate retrofit on the encode side.

### Format Conversion Pipeline (Phase 6, v0.6.0)

`amn convert <input> --to <target>` routes any v0.6.0-available format pair through a single dispatch. Targets at v0.6.0: `safetensors` (alias `bf16`), `gguf` (unquantised passthrough), `bnb-nf4`. Quantised GGUF targets (`gguf-q4km`, …) land in Phase 7.5 through the same dispatch.

End-to-end runtime, **release build, `target-cpu=native`, single 4096×4096 tensor (32 MiB BF16 or 64 MiB F32) on CPU**, vs the closest Python equivalents (best-of-5 median, PyTorch 2.10.0 with the CPU device, NumPy 2.4, safetensors-py 0.7, gguf-py 0.18, bitsandbytes 0.49.1's CPU kernel):

| Conversion | anamnesis | Python ecosystem default | PyTorch-CPU equivalent |
|---|---:|---:|---:|
| `npz → safetensors` | **11.2 ms** | 75.7 ms (numpy + safetensors-py) — **6.75× faster** | 92.5 ms (torch.from_numpy + safetensors.torch) — **8.24× faster** |
| `pth → safetensors` | **5.7 ms** | — | 29.6 ms (torch.load + safetensors.torch) — **5.18× faster** |
| `safetensors-BF16 → GGUF` | **13.6 ms** | 15.1 ms (gguf-py) — **1.11× faster** | 29.6 ms (safetensors.torch.load + gguf-py) — **2.17× faster** |
| `safetensors-BF16 → BnB-NF4` | **141 ms** | 376.8 ms (bitsandbytes CPU) — **2.67× faster** | — *(bitsandbytes is the PyTorch-native path)* |

> **What "PyTorch-CPU equivalent" means here:** the two non-PyTorch baselines (NPZ via NumPy, GGUF via gguf-py) get a second row that routes the data through `torch.from_numpy` / `safetensors.torch.load_file` before the writer step — the way Python practitioners typically pipeline these formats when they already work in PyTorch tensors. The PyTorch row is consistently 1.2–1.6× slower than the ecosystem default because the extra `torch.from_numpy` / `tensor.numpy()` hop is non-zero work, not because PyTorch is inefficient. We report both so the table is honest about Python's choice space, not just the fastest available Python path.

> **Methodology:** Each row's number is the median of 5 release-mode runs at the same 4096×4096 shape both sides agree on, captured in the `tests/fixtures/convert_reference/*.timing.json` sidecars and re-validated by `t14_perf_vs_python_size_matched` in [`tests/cross_validation_convert.rs`](tests/cross_validation_convert.rs). Run `cargo test --release --all-features --test cross_validation_convert -- --ignored --nocapture t14` to reproduce on your own machine, and re-generate the sidecars via `tests/fixtures/convert_reference/generate_convert_timings.py`. The 13 byte-exact round-trip tests in the same file (default, not `--ignored`) exercise every conversion pair at small synthetic fixtures so the correctness suite stays fast.

> **Limitations:** The single-tensor `O(input size)` shape of the perf measurement applies only to the convert primitive itself, not to whole-model conversion. Multi-shard models retain every dequantised tensor in heap until `safetensors::serialize_to_file` returns — same constraint as `ParsedModel::remember`, see the GGUF block-quant note below. Phase 10 (streaming output) is the planned remedy.

### GGUF Block-Quant Dequantization

Cross-validated against the `gguf` Python package (`ggml-org` reference, mirrors `ggml-quants.c`) on **22 block-quant kernels** from 4 real models (bartowski SmolLM2-135M-Instruct, TheBloke TinyLlama-1.1B-Chat, bartowski Mistral-7B-Instruct-v0.3, bartowski Qwen2.5-0.5B-Instruct) plus 3 synthetic fixtures (`TQ1_0` / `TQ2_0` / `MXFP4` — only ~15 BitNet-derivative GGUFs ship the `TQ*` types on HuggingFace, and mainstream `MXFP4` only ships inside the 11 GB `gpt-oss-20b` upload, so a deterministic random tensor is the practical fixture source). Bit-exact output (0 ULP difference). **All 22 of 22 GGUF block types now supported** — Phase 4.5 closed in step 6 (MXFP4). Feature-gated behind `gguf`.

| Kernel | Model | vs `gguf` Python (AVX2) |
|---|---|---|
| Q4_0 | SmolLM2-135M | 6.9x faster |
| Q4_1 | SmolLM2-135M | 6.3x faster |
| Q5_0 | TinyLlama-1.1B | 31.3x faster |
| Q5_1 | SmolLM2-135M | 11.4x faster |
| Q8_0 | SmolLM2-135M | 6.3x faster |
| IQ4_NL | SmolLM2-135M | 12.2x faster |
| Q2_K | TinyLlama-1.1B | 6.7x faster |
| Q3_K | SmolLM2-135M | 10.9x faster |
| Q4_K | SmolLM2-135M | 8.1x faster |
| Q5_K | SmolLM2-135M | 11.6x faster |
| Q6_K | SmolLM2-135M | 26.6x faster |
| IQ4_XS | SmolLM2-135M | 12.6x faster |
| IQ2_XXS | Mistral-7B-v0.3 | 3.45x faster |
| IQ2_XS | Mistral-7B-v0.3 | 2.84x faster |
| IQ2_S | Qwen2.5-0.5B | 4.10x faster |
| IQ3_XXS | Mistral-7B-v0.3 | 3.32x faster |
| IQ3_S | Mistral-7B-v0.3 | 4.37x faster |
| IQ1_S | Mistral-7B-v0.3 | 15.00x faster |
| IQ1_M | Mistral-7B-v0.3 | 7.85x faster |
| TQ1_0 | synthetic | 35.59x faster |
| TQ2_0 | synthetic | 26.31x faster |
| MXFP4 | synthetic | 30.14x faster |

> **Note:** `Q8_1` and `Q8_K` are internal `llama.cpp` activation quant types, not shipped as model weights — they are covered by unit tests only. Speedup measured on 65,536 elements (release build, `target-cpu=native`, best-of-5 per kernel). The `IQ2_*` and `IQ3_*` kernels land in the 2.8×–4.4× range rather than the 6×–31× range of the pure-arithmetic `Q*` kernels because their pass 1 involves a codebook LUT gather and a per-element sign branch — neither of which the auto-vectoriser can eliminate. The `IQ1_*` kernels are notably faster (7.9×–15.0×) because their inner loop replaces the per-element sign branch with a single scalar `±delta` per 8-element group, and the codebook gather is a plain `[u64; 2048]` table lookup. The ternary `TQ*` kernels are the **fastest in the crate** (26×–36×) — no codebook lookup at all, just bit shifts (`TQ2_0`) or a base-3 multiplication trick (`TQ1_0`) decoding directly to `{-d, 0, +d}`. `MXFP4` lands at 30× — structurally identical to `IQ4_NL` (12.2×) but with a tighter 17 B/block layout (1 B `E8M0` exponent vs 2 B `f16`) and a smaller codebook (16 entries × 4-bit nibble lookup) that the auto-vectoriser handles cleanly. Phase 9 (CPU SIMD pass) will further address the IQ2/IQ3 case with hand-written AVX2 intrinsics.

> **Limitations (peak heap):** Whole-model dequantisation via `ParsedModel::remember` or `amn remember model.gguf -o out.safetensors` retains every dequantised tensor in heap memory simultaneously until the underlying `safetensors::serialize_to_file` call returns. Peak heap is `O(total_BF16_output_size)` ≈ `2 × n_parameters` bytes — comfortable for **≤7 B** models on a 32 GB system, **tight at 13 B**, **OOMs at 70 B+**. The single-tensor kernel `dequantize_gguf_blocks_to_bf16` is already streaming (O(one block)); the orchestrator-level streaming output path is planned for Phase 10 — see [ROADMAP.md](ROADMAP.md). Phase 9 (SIMD) and Phase 10 (streaming) are independent; this perf table will be unaffected by Phase 10 because the per-tensor kernel timings stay the same.

### Safetensors Header Inspection

Header-only safetensors parsing ships in three forms. The path-based `parse(path)` memory-maps the file and returns a `ParsedModel` (header + mmap-backed buffer) ready for `inspect()` or `remember()`; the slice-based `parse_safetensors_header(&[u8])` operates on a buffer that already contains the prefix and JSON; and `parse_safetensors_header_from_reader<R: Read>(reader)` accepts any `Read` substrate (in-memory `Cursor`, HTTP-range-backed adapter, custom transport) and reads only the 8-byte length prefix plus the JSON header. Total transfer ≈ header size (~1 MiB on a multi-GB shard) instead of the full file.

`Read` — not `Read + Seek` — is sufficient because the safetensors layout is purely prefix-then-JSON: two contiguous reads in order, never seek-back. This keeps the simplest possible HTTP-range adapter (one connection, two range fetches) viable. Anamnesis itself takes on no network or TLS dependency — downstream crates plug in their own adapter when remote inspection is needed. See the rustdoc on `parse_safetensors_header_from_reader` for the access pattern.

### NPZ/NPY Parsing

Feature-gated behind `npz`. Custom NPY header parser with bulk `read_exact` — zero per-element deserialization for little-endian data on little-endian machines. Cross-validated byte-exact against NumPy on Gemma Scope 2B SAE weights.

| Metric | Value |
|---|---|
| Throughput (302 MB Gemma Scope, F32) | **3,586 MB/s** |
| Overhead vs raw I/O | 1.3x |
| vs `npyz` crate | **17.7x faster** |
| Supported dtypes | F16, BF16, F32, F64, Bool, U8–U64, I8–I64 |

BF16 support via JAX `V2` void-dtype convention. Big-endian NPY files handled with in-place byte-swap.

Header-only inspection ships in two forms: `inspect_npz(path)` for files on disk and `inspect_npz_from_reader<R: Read + Seek>(reader)` for any other substrate (in-memory `Cursor`, HTTP-range-backed adapter, custom transport). Anamnesis itself takes on no network or TLS dependency — downstream crates plug in their own `Read + Seek` adapter when remote inspection is needed. See the rustdoc on `inspect_npz_from_reader` for the access pattern an HTTP-range adapter must satisfy.

### GGUF Inspection

Feature-gated behind `gguf`. The path-based `parse_gguf(path)` memory-maps the file and returns a `ParsedGguf` with zero-copy `Cow::Borrowed` tensor views into the mapping; the reader-generic `inspect_gguf_from_reader<R: Read + Seek>(reader)` accepts any positional substrate (in-memory `Cursor`, HTTP-range-backed adapter, custom transport) and returns just the `GgufInspectInfo` summary (version, architecture, tensor count, total size, dtypes, alignment) without materialising the data segment.

`Read + Seek` — not just `Read` — is required because `GGUF`'s parser computes the absolute tensor-data offset by combining the relative offsets in the tensor-info table with the post-tensor-info `data_section_start` anchor, then validates each offset against the captured stream length. The simplest correct refactor preserves this positional access pattern via `Seek`. A pure-`Read` reformulation would require restructuring the parser into a strict forward pass and is out of scope. A 2 GiB quantised `GGUF` is inspectable in two or three small range requests covering a few MiB of front-loaded metadata — no weight data downloaded. Anamnesis itself takes on no network or TLS dependency; downstream crates plug in their own adapter when remote inspection is needed. See the rustdoc on `inspect_gguf_from_reader` for the access pattern.

### PyTorch `.pth` Parsing

Feature-gated behind `pth`. Minimal pickle VM (~36 opcodes) with security allowlist. Memory-mapped I/O with zero-copy tensor access (`Cow::Borrowed` from mmap). Cross-validated byte-exact against PyTorch `torch.load()` on 3 [AlgZoo](https://github.com/alignment-research-center/alg-zoo) models (MIT-0 license).

| Model | Size | Tensors | vs `torch.load` |
|---|---|---|---|
| torchvision ResNet-18 | 45 MB | 102 | **11.2x faster** |
| torchvision ResNet-50 | 98 MB | 267 | **12.7x faster** |
| torchvision ViT-B/16 | 330 MB | 152 | **30.8x faster** |

Lossless `.pth` → `.safetensors` conversion preserving original dtypes (F16, BF16, F32, F64, I8–I64, U8, Bool). The conversion pipeline writes directly from mmap slices to the output file — zero intermediate data copies.

Handles both newer (`archive/` prefix) and older (`{model_name}/` prefix) PyTorch ZIP conventions. Legacy (pre-1.6) raw-pickle files are rejected with a clear error.

### PyTorch `.pth` Inspection

Feature-gated behind `pth`. The path-based `parse_pth(path)` memory-maps the file and returns a `ParsedPth` with zero-copy `tensors()` views into the mapping; the reader-generic `inspect_pth_from_reader<R: Read + Seek>(reader)` accepts any positional substrate (in-memory `Cursor`, HTTP-range-backed adapter, custom transport) and returns just the `PthInspectInfo` summary (`tensor_count`, `total_bytes`, `dtypes`, `big_endian`) without materialising any of the tensor-data files inside the archive (`data/0`, `data/1`, …).

Only the ZIP central directory and the `data.pkl` entry — typically <100 KiB even on torchvision-class 300 MB models — are fetched. A 300 MB torchvision `.pth` is inspectable through an HTTP-range adapter in well under 100 KiB of network transfer, instead of 300 MB.

Measured across the **full 6 960-file [AlgZoo](https://github.com/alignment-research-center/alg-zoo) corpus** (the `algzoo_weights/` set imported for `candle-mi` v0.1.9's `stoicheia` module; best-of-5 release-mode median per file, `target-cpu=native`, PyTorch 2.10.0+cu130):

| Substrate                                    | Median per file | vs `torch.load` |
|---|---:|---:|
| `parse_pth(path).inspect()` (mmap)           | 124.0 µs |  **4.07x faster** |
| `inspect_pth_from_reader(File)` (reader)     | 168.7 µs |  **2.99x faster** |
| `torch.load(weights_only=True)` (PyTorch)    | 504.3 µs |        baseline   |

PyTorch has no separate inspect-only primitive — `torch.load(weights_only=True)` is the closest comparable; it fully materialises every tensor before the caller can iterate the `state_dict` for summary stats, so the speedup is a **lower bound** that grows by orders of magnitude on larger models (the reader path stays bounded by `data.pkl` size while `torch.load` scales linearly in total tensor-data size). Per-family breakdown and the full method are in [`docs/perf-experiments.md`](docs/perf-experiments.md) Experiment 6.

`Read + Seek` (not just `Read`) is required because the ZIP format keeps its central directory at the end of the file, then seeks back to each local-file header to read entry payloads. `zip::ZipArchive::new` already requires `Read + Seek` for that reason, and `inspect_pth_from_reader` inherits the constraint verbatim. The pickle interpreter itself runs over an owned `Vec<u8>` (the materialised `data.pkl`) — same security allowlist as the path-based `parse_pth`, shared by construction so the two entry points cannot diverge. Anamnesis itself takes on no network or TLS dependency; downstream crates plug in their own adapter when remote inspection is needed. See the rustdoc on `inspect_pth_from_reader` for the full access pattern.

### Parser robustness (Phases 6.6–6.8, v0.6.1–v0.6.3)

Every parser bounds its header-derived allocations against the [`candle #3533`](https://github.com/huggingface/candle/issues/3533) unguarded-allocation DoS class (CWE-770 / CWE-1284 / CWE-400): a length / count / dimension field read from a file header is validated against an explicit cap *before* it reaches `vec!` / `Vec::with_capacity` / the pickle VM. GGUF and safetensors were already capped; v0.6.1 closes the NPZ `header_len` window (NPY v2/v3 decode it as a `u32`, reachable to 4 GiB) and the PyTorch `.pth` mmap-path `data.pkl` size gate (previously bounded only by file size), and adds NPZ array-byte and per-opcode pickle payload caps as defence in depth. Malformed or malicious inputs fail fast with a clear `AnamnesisError::Parse`; legitimate files are unaffected, and there is no API change. This matters because remote header inspection over an `HTTP`-range adapter parses bytes from arbitrary, untrusted repositories.

**v0.6.2** follows up with a full pre-Phase-7 security audit (all four parsers plus the dequant/encode layers) and adds a *container-size cross-check* — a third defence beyond constant caps and checked arithmetic. NPZ rejects a declared array size larger than the ZIP entry's own uncompressed `size()` before allocating, and the GGUF reader validates each variable-length read against the bytes physically remaining before allocating, so a tiny file can no longer drive an eager allocation up to a type cap. A shared checked shape-product helper removes a debug-panic / release-wrap on adversarial shapes. Backing the audit is a coverage-guided [`cargo fuzz`](fuzz/README.md) harness — one libFuzzer target per parser, dev-only and excluded from the published crate, focused on the pickle VM (the deepest state machine) — run with zero crashes across the parser surface.

**v0.6.3** makes those fixed, server-scale caps *caller-configurable* (Phase 6.8). A new `ParseLimits` budget — `max_single_alloc`, `max_total_bytes` (the cumulative-heap aggregate), `max_item_count`, and `max_decompression_ratio` (the `DEFLATE` zip-bomb cap) — threads through every parser via `parse_*_with_limits`, enforced fail-fast and tighten-only, so a memory-constrained edge board or per-slot MLaaS worker can tighten the parser to *its* environment; `ParseLimits::default()` is unbounded, so default behaviour is byte-for-byte unchanged. The recommended **inspect → check policy → parse-under-limits** recipe is documented under [Parsing untrusted input](#parsing-untrusted-input), and the `cargo fuzz` harness gains targets that co-explore malformed files against tightened limits, exercising the enforcement branches with zero crashes. v0.6.3 also drops the unused `zopfli` DEFLATE *compressor* from the dependency tree (`zip` set to inflate-only).

### Performance validation (Phase 6.5)

Three dev-only validation tracks ship alongside v0.6.0 — none of them affect the published crate (excluded from the tarball by Cargo's defaults). Each is a regression detector that catches drift before it reaches a release:

- **[Criterion runtime benchmarks](benches/README.md)** — `benches/dequant.rs` covers 7 synthetic-layer-sized kernel groups (FP8, GPTQ, AWQ, BnB NF4, BnB INT8, GGUF Q4_K, plus FP8 fine-grained) at `4096 × 11008`, reporting **657 Melem/s → 2.01 Gelem/s** on the dev machine. `benches/parsing.rs` covers header-only parses for all four formats vs an `fs::read` baseline. Plus a real-world group on the Ollama-cached `llama3.2:1b` `Q8_0` slice (**3.92 Gelem/s**). Run with `cargo bench --features gptq,awq,bnb,gguf,npz,pth`.
- **[`dhat-rs` peak-heap assertions](tests/peak_heap_README.md)** — three `#[ignore]`d test binaries that wrap the global allocator and assert observed peak heap stays within the documented ceiling. **Every kernel's observed scratch matches the documented `# Memory` claim to the byte**: GPTQ/AWQ at `3 × out_features × 4`, BnB-DQ at `num_blocks × 4 + block_size × 4`. Run with `cargo test --release --features gptq --test peak_heap_gptq -- --ignored --nocapture` (and similarly for `awq` / `bnb_dq`).
- **[Ollama cross-validation](tests/cross_validation_ollama.rs)** — bit-exact `Q8_0` dequant against the `gguf-py` reference on a slice extracted from the local Ollama cache's `llama3.2:1b` blob. Same kernel anamnesis already validates against bartowski / `TheBloke` quantisations, now on the dominant local-LLM distribution channel too. Result: **0 ULP mismatches**.

## Used by

- [candle-mi](https://github.com/PCfVW/candle-mi) — Mechanistic interpretability toolkit for language models

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT License](LICENSE-MIT) at your option.

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code) (dev) and [Augment Code](https://www.augmentcode.com/) (review)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset designed to improve AI coding accuracy.