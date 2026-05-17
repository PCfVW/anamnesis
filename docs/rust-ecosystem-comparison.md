# Rust Ecosystem Comparison

**Date surveyed:** 2026-05-17
**anamnesis version at survey:** v0.4.6 + Phase 5 step 1a/1b/1c (commits `a5c452d` / `24cba42` / `ab4e735`, pending `v0.5.0` tag)
**Scope:** Rust crates and binaries only. Python tooling (`bitsandbytes`, `autogptq`, `autoawq`, `GPTQModel`, `llama.cpp quantize`, `safetensors` Python bindings) is out of scope by design — this matrix exists to help downstream Rust consumers (`candle`, `mistral.rs`, `burn`, `candle-mi`, plus any new HF-quantised-model loader) evaluate whether `anamnesis` covers what they need vs the alternatives in their own ecosystem.

---

## What anamnesis does (the comparison reference)

`anamnesis` is a Rust library AND CLI binary for **parse-first tensor format work**: parse any HuggingFace-ecosystem tensor archive, inspect metadata without materialising weights, dequantise to `BF16` for downstream consumption, and (Phase 5+) encode `BF16` back into compact quantised formats. Framework-agnostic by design — produces raw bytes, not bound to a specific tensor library's types. The 14 capability areas used as comparison axes:

| # | Header | Capability |
|---|---|---|
| 1 | **STParse** | `.safetensors` header parsing — path-based (mmap) **and** reader-generic over any `Read + Seek` substrate (HTTP-range friendly), including dequant-companion-tensor awareness (`.qzeros`, `.absmax`, `.quant_map`, `.SCB`, `.g_idx`, …) |
| 2 | **NPZParse** | NumPy `.npz` archive parsing — bulk-read NPY parser with `F16`/`BF16`/`F32`/`F64`/integer support, 3,586 MB/s on real fixtures (17.7× `npyz`) |
| 3 | **PTHParse** | PyTorch `.pth` parsing — minimal pickle VM (~36 opcodes) with security allowlist, zero-copy `Cow::Borrowed` mmap, 11–31× faster than `torch.load()` on torchvision models |
| 4 | **GGUFParse** | `.gguf` parsing — header + metadata + tensor-info, reader-generic inspection variant |
| 5 | **Inspect** | header-only metadata across **all four** formats (zero further I/O after header read; reader-generic for `.safetensors`, `.npz`, `.gguf`, `.pth`) |
| 6 | **DQ-FP8** | `FP8 E4M3` dequant: fine-grained 128×128 block-scale, per-channel `[N,1]`, per-tensor; 2.7–9.7× vs PyTorch CPU |
| 7 | **DQ-GPTQ** | `GPTQ` `INT4`/`INT8` dequant: group-wise scales, `g_idx`, zero-points; 6.5–12.2× vs PyTorch CPU |
| 8 | **DQ-AWQ** | `AWQ` `INT4` dequant: per-group activation-aware; 4.7–5.7× vs PyTorch CPU |
| 9 | **DQ-BnB** | `BitsAndBytes` `NF4`/`FP4`/`INT8` dequant including double-quant; 18–54× vs PyTorch CPU for 4-bit |
| 10 | **DQ-GGUF** | `GGUF` dequant: legacy block (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`/`Q8_1`), K-quants (`Q2_K`–`Q8_K`), `IQ` family (`IQ1_S`/`IQ1_M`/`IQ2_*`/`IQ3_*`/`IQ4_NL`/`IQ4_XS`), `TQ1_0`/`TQ2_0`, `MXFP4` |
| 11 | **EncBnB** | `BitsAndBytes` **encode** — `NF4`/`FP4`/`INT8` plain **and** double-quant, plus sign-of-zero preservation rule; shipped in Phase 5 commits `a5c452d` / `24cba42` / `ab4e735` |
| 12 | **Convert** | lossless format conversion — `.pth` → `.safetensors` (one-shot CLI subcommand) |
| 13 | **Lib** | embeddable Rust library API |
| 14 | **CLI** | CLI binary (`anamnesis` + alias `amn`) |

Quality properties shared across every kernel (not table columns — would otherwise mark all-✓ for anamnesis trivially): **bit-exact** (0 ULP) validation against PyTorch reference on real model fixtures; `#![deny(unsafe_code)]` at the crate root with a documented, tightly-scoped opt-in for `memmap2::Mmap::map` only.

---

## Per-tool detail

### A. Direct competitors (multi-family Rust dequantisation libraries)

**None known.** No other Rust crate exposes dequantisation across multiple families (`FP8` + `GPTQ` + `AWQ` + `BnB` + `GGUF`) as a standalone primitive. Dequant in the current Rust ecosystem lives **inside** inference frameworks (see category C), tightly coupled to those frameworks' tensor types and loading paths. `anamnesis` is the first crate to make dequant a framework-agnostic library that produces raw `BF16` bytes any downstream tensor system can consume. Phase 5 extends this to the encode side: **no other Rust crate ships `BnB` encode kernels at all** — quantisation in Rust today means "call `bitsandbytes` from Python via PyO3" or "use `mistral.rs quantize` and accept its custom UQFF format".

### B. Tensor format parsers / inspectors (single-format or narrow scope)

**safetensors** — [github.com/safetensors/safetensors](https://github.com/safetensors/safetensors) | [crates.io/crates/safetensors](https://crates.io/crates/safetensors)
The canonical format crate (~3.7k stars, library-only, no CLI). Header read/write, lazy mmap-backed views, supports the format spec end-to-end. **Does not** dequantise, doesn't expose dtype histograms, doesn't know about quant companion tensors (`.qzeros` / `.absmax` / `.quant_map`); those are anamnesis-side concepts.

**safetensors_explorer** — [github.com/EricLBuehler/safetensors_explorer](https://github.com/EricLBuehler/safetensors_explorer) | [crates.io/crates/safetensors_explorer](https://crates.io/crates/safetensors_explorer)
Interactive TUI inspector for `.safetensors` AND `.gguf`. Tree view, fuzzy search, sharded model index detection. CLI-only (no library API), inspect-only (no dequant, no encode, no format conversion). The closest analog to `anamnesis inspect` but on metadata only.

**safetensors-cli** — [github.com/gzsombor/safetensors-cli](https://github.com/gzsombor/safetensors-cli) | [crates.io/crates/safetensors-cli](https://crates.io/crates/safetensors-cli)
Tiny (~57 LOC), 0 stars, effectively unmaintained. Local-file `.safetensors` header dump only.

**gguf-rs / gguf-cli** — [github.com/ThreatFlux/gguf](https://github.com/ThreatFlux/gguf) | [crates.io/crates/gguf-rs](https://crates.io/crates/gguf-rs)
GGUF parsing library + `gguf-cli` binary (`info`/`tensors`/`metadata`/`validate`). Zero-copy parsing, optional mmap. GGUF only — no `.safetensors`/`.npz`/`.pth` support, no dequantisation.

**inspector-gguf** — [docs.rs/inspector-gguf](https://docs.rs/inspector-gguf)
GUI (egui drag-and-drop) + CLI + library. Exports analysis as CSV/YAML/Markdown/HTML/PDF. Tokenizer + chat-template surface. GGUF only, inspect only, heavyweight footprint.

**npyz** — [crates.io/crates/npyz](https://crates.io/crates/npyz) | [docs.rs/npyz](https://docs.rs/npyz)
`.npy` / `.npz` parser. Per-element deserialisation (slower than anamnesis's bulk `read_exact` path on LE data; the anamnesis README documents a 17.7× speed delta on a 302 MB fixture). No dequantisation, no `.safetensors` / `.pth` / `.gguf` support.

**ndarray-npy** — [crates.io/crates/ndarray-npy](https://crates.io/crates/ndarray-npy)
`.npy` ↔ `ndarray::Array` adapter. Useful when the downstream type is `ndarray`; not framework-agnostic.

**serde-pickle** — [crates.io/crates/serde-pickle](https://crates.io/crates/serde-pickle)
General Python pickle decoder via the serde ecosystem. Pure VM with no `.pth`-specific opcode allowlist, no zero-copy tensor extraction, no security-allowlist on `GLOBAL` references — usable as a primitive, but **substantial gap** between "pickle parser" and "safe `.pth` loader" that anamnesis closes.

### C. Inference frameworks (parse + dequant + inference bundled)

**candle** — [github.com/huggingface/candle](https://github.com/huggingface/candle)
Hugging Face's Rust ML framework (~20k stars). Parses `.safetensors` and `.gguf` natively; dequant happens **inside** the tensor-loading code path (`candle-quantized`, `quantized-var-builder`), tightly coupled to `candle::Tensor`. GGUF dequant is the most complete family on the candle side; no `FP8`, no first-class `GPTQ`/`AWQ`/`BnB` loaders (those exist in downstream candle-* crates like `candle-transformers` but as model-specific glue, not standalone dequant primitives). `.pth` loading via the `tch` bridge or hand-rolled paths; no equivalent of anamnesis's pickle-VM-with-allowlist. No encode side.

**mistral.rs** — [github.com/EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
LLM inference engine (~7k stars). Internally handles `BnB`/`GPTQ`/`AWQ`/`GGUF` for its own model loading; ships a `mistralrs quantize` subcommand that emits its own **UQFF** format (a candle-friendly bundle), not `BnB` or `GGUF`. Dequant is not exposed as a reusable library primitive; the quantisation/dequantisation kernels are entangled with the inference graph. Closest "all-format" coverage on the inference side but the wrong shape if you want raw `BF16` bytes for a different framework.

**burn** — [github.com/tracel-ai/burn](https://github.com/tracel-ai/burn)
Multi-backend training+inference framework. Weight loading is in-code via `PyTorch`/`Safetensors`/`ONNX` import; no standalone CLI for parse/inspect/dequant. Does **not** cover the `GPTQ`/`AWQ`/`BnB`/`GGUF` quantised families as first-class loaders. Out of scope for the comparison axes here.

**tract** — ONNX-only inference. Out of scope (no `.safetensors`/`.pth`/`.gguf`/quant-family overlap).

**llm** — superseded by `mistral.rs` / `candle`; no active development.

**apr-cli** — [docs.rs/apr-cli](https://docs.rs/crate/apr-cli/latest) | part of [paiml/aprender](https://github.com/paiml/aprender)
v0.33.0 (2026-05-14), Rust monorepo (~70–80 workspace crates). CLI binary only (no library API). Subcommands: `inspect` / `list` / `tui` / `diff` / `chat` / `quantize` / `serve` / `qa`. Reads APR (native), GGUF, and SafeTensors. Documentation explicitly says it "specialises in quantization rather than dequantization". The most relevant overlap: `apr convert model.safetensors --quantize q4_k -o model.gguf` — a one-shot SafeTensors → GGUF Q4_K conversion. This is the **closest existing Rust analog** to anamnesis's Phase 5 / Phase 6 / Phase 7.5 encode + convert flow, but with a different output format (GGUF Q4_K/Q4_0 vs anamnesis's BnB Phase 5 today, BnB+FP8+GGUF families at Phase 7.5). Positioning is different: apr-cli is heavyweight inference machinery (`chat`/`serve`/`tui` as the primary value); anamnesis is a thin framework-agnostic library that produces raw bytes. Also ships `apr serve plan --gpu` for the VRAM-fit verdict (the same niche `hf-fm inspect --check-gpu` plans to fill).

### D. PyTorch `.pth` ecosystem

**tch** — [github.com/LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs)
Rust bindings for `libtorch`. Loads `.pth` natively because it ships the full PyTorch C++ runtime. ~250 MB native dependency, opaque tensor type (`tch::Tensor`), wrong shape if you want a thin parse-and-convert tool. No quant-family dequant exposed.

**pickle** / **serde-pickle** — see category B.

### E. Encode (quantisation) Rust tools

**`mistral.rs quantize`** — emits UQFF (mistral.rs's custom packaging of candle tensors). Not a general-purpose quantisation primitive; tightly bound to mistral.rs's inference path.

**No other Rust crate** ships dequant-family encode kernels (`BnB`/`GPTQ`/`AWQ`/`GGUF`). Quantisation in Rust today is "run Python and import the bytes back". `anamnesis` Phase 5 ships the first standalone `BnB` encode kernel set; Phase 7.5 extends to `FP8`/`GGUF` (legacy/K-quants/`IQ`/`TQ`/`MXFP4`).

---

## Comparison matrix

Cells: ✓ = present · ◐ = partial / framework-coupled / narrow · — = absent.

| Tool | STParse | NPZParse | PTHParse | GGUFParse | Inspect | DQ-FP8 | DQ-GPTQ | DQ-AWQ | DQ-BnB | DQ-GGUF | EncBnB | Convert | Lib | CLI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **anamnesis v0.5.0** *(reference)* | ✓ path + reader | ✓ | ✓ pickle-VM | ✓ | ✓ all 4 fmts | ✓ E4M3 3 modes | ✓ INT4/INT8 + g_idx | ✓ INT4 | ✓ NF4/FP4/INT8 + DQ | ✓ block/K/IQ/TQ/MXFP4 | ✓ NF4/FP4/INT8 + DQ | ✓ .pth→.safetensors | ✓ | ✓ `anamnesis`/`amn` |
| safetensors 0.7.x | ✓ lib only | — | — | — | — no CLI | — | — | — | — | — | — | — | ✓ | — |
| safetensors_explorer 0.2.0 | ◐ inspect TUI | — | — | ◐ inspect TUI | ✓ ST + GGUF metadata | — | — | — | — | — | — | — | — bin only | ✓ TUI |
| safetensors-cli 0.1.0 | ◐ minimal | — | — | — | ◐ minimal local | — | — | — | — | — | — | — | — | ✓ |
| gguf-rs 0.2.5 | — | — | — | ✓ | ◐ GGUF only | — | — | — | — | — | — | — | ✓ | ✓ `gguf-cli` |
| inspector-gguf 0.3.1 | — | — | — | ✓ | ◐ GGUF only, GUI+CLI | — | — | — | — | — | — | — | ✓ | ✓ |
| npyz | — | ✓ slower 1× | — | — | ◐ NPY metadata | — | — | — | — | — | — | — | ✓ | — |
| ndarray-npy | — | ◐ `ndarray`-bound | — | — | — | — | — | — | — | — | — | — | ✓ | — |
| serde-pickle | — | — | ◐ generic pickle, no `.pth` opcode allowlist or zero-copy tensor extraction | — | — | — | — | — | — | — | — | — | ✓ | — |
| candle 0.x | ✓ via `candle-core` | — | ◐ via `tch` bridge | ✓ via `candle-quantized` | ◐ in-code only | — | ◐ model-specific glue | ◐ model-specific glue | ◐ model-specific glue | ◐ Q\*\_\* + K-quants | — | — | ✓ framework | — no central CLI |
| mistral.rs 0.8.x | ✓ for inference | — | — | ✓ for inference | ◐ via `doctor` | — | ◐ for inference, tensor-coupled | ◐ for inference, tensor-coupled | ◐ for inference, tensor-coupled | ◐ for inference, tensor-coupled | ◐ UQFF only, not BnB | ◐ UQFF target | ✓ | ✓ `mistralrs` |
| burn | ◐ via importer | — | ◐ via importer | — | — | — | — | — | — | — | — | — | ✓ framework | — |
| apr-cli 0.33.0 | ✓ read for inference | — | — | ✓ read + write | ◐ APR-centric | — | — | — | — | — | — bound to GGUF Q4\_K | ◐ ST→GGUF Q4\_K (lossy) | — bin only | ✓ |
| tch | — | — | ✓ via `libtorch` | — | — | — | — | — | — | — | — | — | ✓ | — |

---

## Where anamnesis stands (preliminary read — happy to expand)

- **Multi-format parser**: anamnesis is the only Rust crate covering all four of `.safetensors` / `.npz` / `.pth` / `.gguf` in one library with a unified `inspect` story.
- **Multi-family dequant**: anamnesis is the only Rust crate exposing dequantisation across **all five** of `FP8` / `GPTQ` / `AWQ` / `BnB` / `GGUF` as standalone, framework-agnostic primitives that produce raw `BF16` bytes. Every other tool in the matrix either covers one family (gguf-rs) or covers many families but tightly coupled to a specific tensor library's loader path (candle, mistral.rs).
- **Encode side (Phase 5)**: anamnesis is **the first** Rust crate to ship `BnB` encode kernels at all. The closest adjacent is `mistral.rs quantize` which targets its own UQFF format, not the `BnB` on-disk layout.
- **`.pth` loading without `libtorch`**: anamnesis's pickle-VM-with-security-allowlist is the only pure-Rust safe `.pth` loader; the alternatives are (a) `tch` (pulls in ~250 MB of native `libtorch`) or (b) `serde-pickle` (general pickle, no `.pth`-specific opcode allowlist, no zero-copy tensor extraction, no security model for `GLOBAL` references).
- **CLI + Lib parity**: anamnesis ships both, mirroring `hf-fm`'s pattern. `safetensors_explorer` / `safetensors-cli` / `gguf-cli` are bin-only; `safetensors` / `tch` / `burn` are lib-only; `candle` has no central CLI for parse/inspect work.

### Closest competitor footprint

The closest competitor coverage requires a **union of four or five tools**: `safetensors` (parse only) + `safetensors_explorer` (inspect TUI, ST+GGUF) + `gguf-rs` (GGUF parse + CLI inspect) + `mistral.rs` (inference-coupled BnB/GPTQ/AWQ/GGUF dequant + UQFF encode) + optionally `apr-cli` (SafeTensors → GGUF Q4_K quantise + convert). Even that union still does not cover:

- Cross-format **lossless** conversion (anamnesis: `.pth` → `.safetensors` is one CLI call; apr-cli's `convert` is always lossy via quantize)
- Standalone dequant primitives (mistral.rs dequant is bound to its inference graph; apr-cli explicitly doesn't expose dequant)
- `BnB` encode in `BnB`'s on-disk layout (mistral.rs encodes to UQFF, apr-cli to GGUF Q4_K — neither produces BnB)
- `.pth` loading without a `libtorch` dependency
- Reader-generic header parsing (HTTP-range remote inspect over any `Read + Seek` substrate)
- Library API for embedders (`apr-cli` / `safetensors_explorer` / `gguf-cli` are bin-only; embedders cannot reuse the orchestration)

### Beyond Rust (context — not in the matrix by design)

**ollama** ([github.com/ollama/ollama](https://github.com/ollama/ollama)) — v0.24.0, 172 k stars, 5,389 commits. The dominant local-LLM CLI in the broader ecosystem, written in Go (67 %) + C (27 %, mostly llama.cpp glue) + TS (3 %). Most relevant for our scope: ships a `convert/` directory in Go with `reader_safetensors.go` and `reader_torch.go` plus architecture-specific converters (Llama / Gemma / Qwen / Mistral / …) — independent re-implementations of the same parse work `anamnesis` does in Rust, targeting the same GGUF output that `apr-cli` (and `anamnesis` Phase 7.5) targets. The pipeline is `HF SafeTensors → ollama's Go converter → GGUF intermediate → llama.cpp quantize → final GGUF`. **No dequantisation primitives exposed** — entirely bundled with the inference path, same pattern as `candle` / `mistral.rs`. **No library API for Rust embedders** (Python + JS clients only).

The parse + quantise pipeline is being solved *somewhere* in every active local-LLM ecosystem:

| Ecosystem | Tool | Quantise output |
|---|---|---|
| Python | `bitsandbytes` + `autogptq` + `autoawq` + `llama.cpp quantize` | BnB, GPTQ, AWQ, GGUF (each via separate tool) |
| Go | `ollama convert` (+ llama.cpp) | GGUF only |
| Rust | `apr-cli` quantize | GGUF Q4_K / Q4_0 only |
| Rust | `mistral.rs quantize` | UQFF only |
| Rust | `anamnesis` (Phase 5+) | BnB plain + DQ now; FP8 + GGUF families at Phase 7.5 |

What's distinctive about `anamnesis` isn't *that* it exists — it's the **library-first + framework-agnostic + multi-family-dequant** shape. None of the cross-language analogs ship dequant as a reusable primitive consumable from outside their own inference path.

---

## Sources

- [safetensors GitHub](https://github.com/safetensors/safetensors) / [crates.io](https://crates.io/crates/safetensors)
- [safetensors_explorer GitHub](https://github.com/EricLBuehler/safetensors_explorer) / [crates.io](https://crates.io/crates/safetensors_explorer)
- [safetensors-cli GitHub](https://github.com/gzsombor/safetensors-cli) / [crates.io](https://crates.io/crates/safetensors-cli)
- [gguf-rs GitHub](https://github.com/ThreatFlux/gguf) / [crates.io](https://crates.io/crates/gguf-rs)
- [inspector-gguf docs.rs](https://docs.rs/inspector-gguf)
- [npyz docs.rs](https://docs.rs/npyz) / [crates.io](https://crates.io/crates/npyz)
- [ndarray-npy crates.io](https://crates.io/crates/ndarray-npy)
- [serde-pickle crates.io](https://crates.io/crates/serde-pickle)
- [candle GitHub](https://github.com/huggingface/candle) / [candle-quantized](https://docs.rs/candle-core/latest/candle_core/quantized/index.html)
- [mistral.rs GitHub](https://github.com/EricLBuehler/mistral.rs)
- [apr-cli docs.rs](https://docs.rs/crate/apr-cli/latest) / [aprender monorepo](https://github.com/paiml/aprender)
- [burn GitHub](https://github.com/tracel-ai/burn)
- [tch-rs GitHub](https://github.com/LaurentMazare/tch-rs)
- [ollama GitHub](https://github.com/ollama/ollama) (Go, ecosystem context only)

---

## Next steps (not done — flagging for your call)

This first cut leans on my own knowledge of the ecosystem for version numbers and capability claims. To bring it up to `hf-fm`'s level of precision (specific release dates, star counts, recent activity verification, version-pin accuracy), the table cells should be cross-checked via `WebFetch` against each crate's `crates.io` / GitHub page. Worth doing before any wider circulation; happy to do that pass if you want the doc shippable as-is. Other open questions you might want to direct:

- **NoUnsafe** and **BitExact** as table columns? Currently relegated to the "quality properties" paragraph since they're properties rather than capabilities, but a "rigour" column would be a real differentiator vs e.g. unsafe-heavy framework loaders.
- Per-tool detail length: kept brief (~1–3 sentences each) for this first cut; happy to expand each tool's section to `hf-fm`'s level (4–8 lines each with versions / star counts / reverse-dep notes) once we've agreed on the tool set.
- **Adjacent CLI tools** I haven't surveyed but that might be worth a row: `apr-cli` (has a `quantize` subcommand for its own APR format), `llmserve` (cache discovery), `ollama` (out of scope — Go, not Rust).
- Should the doc also include a `Phase 7.5` forward-looking column for the deferred `FP8`/`GGUF`/`IQ`/`TQ`/`MXFP4` encode kernels, analogous to the `VRAMFit` planned-feature column in `hf-fm`'s doc?
