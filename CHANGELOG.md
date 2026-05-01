# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`TensorEntry::num_elements` overflow saturation** — replaced the
  unguarded `shape.iter().product()` with `try_fold(checked_mul)`
  saturating to `usize::MAX`, matching the contract `inspect_npz`
  already documents. On a malformed or adversarial header that
  declares a shape whose element count overflows `usize` (e.g.,
  `[u32::MAX, 2]` on a 32-bit target), the previous implementation
  would silently wrap to a small value, after which downstream
  validation could accept a tiny data slice as if it described the
  full tensor. Public-API contract is unchanged: still
  `pub fn num_elements(&self) -> usize`. **3 new unit tests** cover
  the saturation, the exact path on normal shapes, and the empty-shape
  scalar case (product is 1, not 0). Identified by the v0.4.x
  algorithmic-weakness audit (finding #12 of 12).
- **GGUF CLI `remember` `n_elements` overflow guard** — the
  `tensor.shape.iter().product()` in `src/bin/main.rs` now uses
  `try_fold(checked_mul)` returning `AnamnesisError::Parse` on
  overflow, naming the offending tensor and shape. Previously a
  malformed GGUF tensor entry could silently wrap `n_elements` to a
  small value and dequantize a fraction of the data with no error.
  No public-API surface (CLI binary only).

### Added

- **`inspect_npz_from_reader<R: Read + Seek>`** — reader-generic `NPZ`
  inspection. Accepts any `Read + Seek` substrate (in-memory `Cursor`,
  HTTP-range-backed adapter, custom transport) and returns the same
  `NpzInspectInfo` as the existing path-based `inspect_npz`. The legacy
  `inspect_npz(path)` entry point is now a two-line wrapper that opens a
  `std::fs::File` and delegates here — fully backward-compatible. Unblocks
  remote `NPZ` inspection without materialising the data segment: a
  downstream HTTP-range adapter (e.g., `hf-fm`'s safetensors range-reader
  extended to `NPZ`) can satisfy this function in ~7 small range requests
  totalling well under 100 KiB on a typical Gemma Scope `params.npz` —
  cutting candle-mi's GemmaScope `open()` cold-start from ~30 s on a
  100 Mbps link to <1 s. Anamnesis itself takes on no network or TLS
  dependency. Phase 4.7 (v0.4.3); see [`ROADMAP.md`](ROADMAP.md).
- **3 new `NPZ` unit tests** covering the reader-generic path:
  `inspect_from_reader_matches_path` (substrate-equivalence on a
  multi-array in-memory archive), `inspect_from_reader_empty_archive`,
  and `inspect_from_reader_rejects_fortran_order` — confirming the
  refactor preserves every guard. Plus a new `cross_validation_npz`
  integration test (`inspect_path_and_reader_agree_on_gemma_scope_fixture`)
  asserting field-for-field parity between `inspect_npz` and
  `inspect_npz_from_reader` on the real Gemma Scope SAE fixture.

### Changed

- **`ROADMAP.md`** — inserted Phase 4.7 (Remote-only NPZ inspection,
  Reader-generic API, v0.4.3) between Phase 4.5 and Phase 5. Updated
  status header `Next:` pointer (Phase 4.7 → Phase 5), added the new
  section to the TOC, and rehomed the "Remote-only NPZ inspection
  (HTTP-range probe)" out-of-scope bullet from Phase 4.5 to a one-line
  pointer at its now-scheduled milestone.
- **`ROADMAP.md`** — added Phase 7.5 (Lethe Encode Completion, v0.7.5) and applied a consistency pass: refreshed header status (Phases 1–4.5 / v0.4.2 / next Phase 5), added Phase 7.5 + Phase 10 to the TOC, populated the `lethe/` module box, scoped Phase 6 / Phase 7 claims to actual v0.6.0 / v0.7.0 reality (BnB-only encode at v0.6.0, full encode matrix at v0.7.5), corrected the "v0.4.0 BnB decode kernels" reference in Phase 5 step 1, removed the v0.4.2 "awaiting user review" note now that the tag has shipped, and added a Phase 9 follow-up note covering Phase 5/7.5 encode-side pass-2 loops.

## [0.4.2] - 2026-04-25

### Added

- **`GGUF` `MXFP4` dequant kernel** — the last GGUF block-quant type, closing Phase 4.5. 32-element block, 17 B/block: `e: u8` (E8M0 byte exponent) + `qs[16]` (4-bit packed nibbles). Decodes via a 16-entry signed `i8` codebook (`K_VALUES_MXFP4`, storing 2× the OCP E2M1 magnitudes) and a 1-byte E8M0 exponent decoded by the new `e8m0_to_fp32_half` helper — the doubled codebook plus the half-scale exponent cancel out so the dequantised value matches the raw OCP MX spec. Same low/high split-nibble layout as `Q4_0` / `IQ4_NL`. Bit-exact (0 ULP) against the `gguf` Python reference on a deterministic synthetic fixture (Python `gguf.quants.quantize()` supports MXFP4 too — same synthetic-fixture path as TQ1_0/TQ2_0 from step 5; mainstream MXFP4 GGUFs only ship inside the 11 GB `gpt-oss-20b` upload, too large to justify when synthetic is bit-exact). After this step **anamnesis dequantises every GGUF block type shipping on `HuggingFace` today** — 22 of 22 production kernels, no remaining coverage gap. Sixth and final step of Phase 4.5.
- **`GGUF` `TQ1_0` + `TQ2_0` dequant kernels** — two ternary super-quants invented for BitNet-style 1.58-bit models. `TQ1_0` (~1.6875 bpw, 54 B/block) uses base-3 packing (5 ternaries per byte for `qs`, 4 for `qh`) decoded via the `pow3 = [1, 3, 9, 27, 81, 243]` multiplication trick: after `byte * pow3[n]` (wrapping `u8`), the n-th digit lives in the top bits and is recovered by `(q * 3) >> 8`. `TQ2_0` (~2.0625 bpw, 66 B/block) uses plain 2-bit packing — 4 ternaries per byte. Both produce values in `{-d, 0, +d}` (true ternary). New shared helper `decode_pow3_ternary` alongside the existing `write_signed_grid` / `write_delta_grid` family. Bit-exact (0 ULP) against the `gguf` Python reference on a deterministic synthetic fixture (no model download needed — only ~15 BitNet-derivative uploads exist on HuggingFace, so synthetic input via Python `gguf.quants.quantize()` is the practical path). Fifth step of Phase 4.5.
- **`GGUF` `IQ1_S` + `IQ1_M` dequant kernels** — two 1-bit super-quants, the smallest IQ-family members. `IQ1_S` (~1.56 bpw, 50 B/block, 11-bit grid index via `qs` + `qh`, per-sub-block 3-bit scale + ±`IQ1S_DELTA = 0.125` additive bias selected by the top bit of `qh`); `IQ1_M` (~1.75 bpw, 56 B/block, **no top-level `d` field** — the super-block float scale is reconstructed from a scattered 16-bit pattern across `scales[8]` reinterpreted as `f16` via `half::f16::from_bits`). Both share the 2048-entry `IQ1S_GRID: [u64; 2048]` codebook of signed `i8` 8-element vectors (~16 KB). Inner-loop math is `dl × (grid[j] + delta)` rather than the multiplicative-sign pattern of IQ2/IQ3 — needs the new `write_delta_grid` helper sitting next to `write_signed_grid`. Bit-exact (0 ULP) against `gguf` Python on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`...-IQ1_S.gguf` and `...-IQ1_M.gguf` — the IQ1 variants don't share files like IQ3 did, so two downloads were needed). Fourth step of Phase 4.5 (GGUF completeness).
- **`GGUF` `IQ3_XXS` + `IQ3_S` dequant kernels** — two 3-bit super-quants. `IQ3_XXS` (~3.06 bpw, 98 B/block, grid-only `scales_and_signs` per sub-block like `IQ2_XXS`); `IQ3_S` (~3.44 bpw, 110 B/block, 9-bit grid index via `qs` + `qh`, inline sign bytes like `IQ2_S`, unusual odd-integer scale formula `d × (1 + 2·nibble)`). Codebook grids (`IQ3XXS_GRID: [u32; 256]`, `IQ3S_GRID: [u32; 512]`, ~3 KB total) ported verbatim from `ggml-common.h`. Both reuse the Phase 4.5 step 2 `write_signed_grid` helper unchanged — the combined-grid/signs packing format is shared across all sign-masked `IQ*` kernels. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`IQ3_XXS` from the `IQ3_XXS.gguf` variant, `IQ3_S` from the already-local `IQ2_S.gguf` which happens to ship 37 `IQ3_S` tensors). Third step of Phase 4.5 (GGUF completeness).
- **`GGUF` `IQ2_XXS` + `IQ2_XS` + `IQ2_S` dequant kernels** — three 2-bit super-quants from the `IQ*` family, each using a per-sub-block lattice codebook of packed 8-element `u8` vectors with a 7- or 8-bit sign mask flipping individual element signs. `IQ2_XXS` (66 B/block, 8-bit grid index), `IQ2_XS` (74 B/block, 9-bit grid index), and `IQ2_S` (82 B/block, 10-bit grid index with the high 2 bits from a separate `qh` array + inline sign bytes). Codebook grids (`IQ2XXS_GRID: [u64; 256]`, `IQ2XS_GRID: [u64; 512]`, `IQ2S_GRID: [u64; 1024]`) and the `ksigns_iq2xs` / `kmask_iq2xs` sign tables are ported verbatim from `ggml-common.h` into a private `iq_grids` submodule. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`IQ2_XXS` + `IQ2_XS`) and `bartowski/Qwen2.5-0.5B-Instruct-GGUF` IQ2_M mix (`IQ2_S`). Second step of Phase 4.5 (GGUF completeness).
- **`GGUF` `IQ4_NL` + `IQ4_XS` dequant kernels** — two more GGUF block types dequantised to `BF16`: `IQ4_NL` (32-element non-linear 4-bit, 18-byte blocks) and `IQ4_XS` (256-element non-linear 4-bit super-block, 136-byte blocks — the most widely used member of the `IQ*` family on HuggingFace). Both share the 16-entry `kvalues_iq4nl` codebook. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/SmolLM2-135M-Instruct-GGUF`. First step of Phase 4.5 (GGUF completeness).
- **`[package.metadata.docs.rs]`** — docs.rs now builds with `features = ["npz", "pth", "gguf", "awq", "gptq", "bnb"]`, exposing all feature-gated public API items on docs.rs.
- **`docs/formats/gemmascope.md`** — one-page reference for loading GemmaScope (Gemma 2 JumpReLU SAEs). Documents the two-repo split (metadata in `mntss/gemma-scope-transcoders`, weights in `google/gemma-scope-2b-pt-transcoders`), NPZ tensor layout (`W_enc` transpose, `threshold` for JumpReLU, no `W_skip`), and links to the canonical `circuit-tracer` Python loader. No new parser needed — loads via existing NPZ support.

### Changed

- **Peak-heap documentation** — clarified the `# Memory` rustdoc on [`ParsedModel::remember`](src/model.rs) and [`dequantize_gguf_to_bf16`](src/remember/gguf.rs) to accurately describe the orchestrator-level eager buffering: every dequantised tensor's `Vec<u8>` is retained simultaneously until `safetensors::serialize_to_file` returns. Earlier wording suggested individual frees during the loop, which is incorrect. Added matching **Limitations (peak heap)** note to the README's GGUF section with model-size guidance (≤7 B comfortable, 13 B tight, 70 B+ OOMs on 32 GB systems). The streaming-output milestone is now a concrete planned phase ([ROADMAP.md](ROADMAP.md) Phase 10) rather than a Future Directions bullet — the dequantisation kernels already provide streaming entry points; only the orchestrator-level wiring is missing. No code change to dequantisation behaviour.
- **ROADMAP follow-ups from external audit** — added two Phase 9 (CPU SIMD pass) bullets capturing audit-identified opportunities to verify and, if needed, refactor the [`copy_to_contiguous`](src/parse/pth.rs) coordinate-carry loop and the [`AWQ`](src/remember/awq.rs)/[`GPTQ`](src/remember/gptq.rs) four-way `chunks_exact_mut(2).zip(...)` chains. Both are deferred from v0.4.2 — they require `cargo-show-asm` evidence before any refactor, which is exactly Phase 9's scope.
- **GPTQ/AWQ lazy precomputation** — scale and zero-point arrays are now computed per-group on demand instead of precomputing the full `num_groups × out_features` grid upfront. Reduces intermediate memory from `O(num_groups × out_features)` to `O(out_features)` with no throughput regression.
- **BnB double-quant refactor** — extracted shared `dequantize_bnb4_core` accepting `&[f32]` absmax directly. The double-quant path no longer serializes `Vec<f32>` to `Vec<u8>` and back; eliminates one allocation and one copy loop.
- **GPTQ g_idx pre-validation** — `g_idx` entries are now validated against `num_groups` in a single pass before the hot loop, failing fast on corrupted files instead of mid-dequantization.
- **CLI stale-binary guard** — `binary_path()` in CLI integration tests now checks that the binary version matches `Cargo.toml` and panics with a diagnostic message if stale. Pre-commit checks updated to include `cargo build --features cli`.
- **`docs/formats/gemmascope.md`** — added "Where to find files on HuggingFace" section documenting the `google/gemma-scope-{size}-{tune}-{site}` slug convention (sizes 2b/9b/27b/2-270m/2-1b, pt vs it, hook sites res/att/mlp/transcoders) and notable community ports (mwhanna, EleutherAI, weijie210).

### Fixed

- **CLI feature-gate fallback defect** — `detect_format` in [src/bin/main.rs](src/bin/main.rs) now returns a `Result<Format>` and emits an `AnamnesisError::Unsupported` carrying a feature-flag hint when the input matches a format whose Cargo feature is not enabled in this build. Previously a `.pth` / `.npz` / `.gguf` file (or a `.bin` file with the corresponding magic) silently fell through to the safetensors parser when its feature was disabled, producing cryptic downstream errors like `HeaderTooLarge` instead of a useful "rebuild with `--features cli,<flag>`" message. This matters in practice because `cli = ["dep:clap"]` does **not** transitively activate `pth` / `npz` / `gguf` — `cargo install anamnesis --features cli` ships a safetensors-only CLI. The library API (`anamnesis::parse_pth` etc.) was already returning `Unsupported` properly; this fix brings the CLI's UX in line.
- **Rust 1.95 clippy lints** — fixed `clippy::unnecessary_trailing_comma` in `src/inspect.rs` and `clippy::map_unwrap_or` (with the `is_ok_and` suggestion) in `src/bin/main.rs`. Both lints landed/strengthened in Rust 1.95.0 (2026-04-14) and would have broken CI's `stable` matrix job under `#![deny(warnings)]`. No user-visible behavior change.

## [0.4.1] - 2026-04-13

### Added

- **`pth_to_safetensors_bytes`** — in-memory `.pth` → safetensors conversion
  returning `Vec<u8>` instead of writing to disk. Enables downstream crates
  (candle-mi) to load `.pth` files without a temp file round-trip via
  `VarBuilder::from_buffered_safetensors`.
- **`ParsedPth::to_safetensors_bytes`** — convenience method combining
  `tensors()` + `pth_to_safetensors_bytes()`.
- **`ParsedGguf::dequantize_tensor`** — convenience method that slices the
  mmap, infers element count from shape, and delegates to
  `dequantize_gguf_to_bf16`. Saves consumers the three-line boilerplate on
  every tensor iteration.
- **GGUF CLI subcommands** — `amn parse model.gguf`, `amn inspect model.gguf`,
  and `amn remember model.gguf --to bf16 -o out.safetensors` now work when
  built with `--features gguf`. Format detection by `.gguf` extension with
  `GGUF` magic fallback for `.bin` and unknown extensions. Quantized tensors
  are dequantized to `BF16`; non-quantized tensors (`F32`, `F16`, `BF16`,
  integer types) are passed through with their original dtype.

## [0.4.0] - 2026-04-12

### Added

- **`dequantize_gguf_blocks_to_bf16`** — new streaming public API that
  emits one block's worth of `BF16` bytes per call into a caller-supplied
  `FnMut(&[u8]) -> Result<()>` sink closure (64 B per call for legacy
  quants, 512 B for K-quants). Peak heap is O(one scratch + one block
  output) regardless of tensor size — around 1.5 KB — enabling
  dequantisation of 70 B-parameter models on modest-RAM machines by
  streaming directly to disk. The existing `dequantize_gguf_to_bf16`
  `Vec`-returning variant is now a thin convenience wrapper that sinks
  into `Vec::with_capacity`, so both entry points share the same
  validation and the same scalar kernels.
- **`GGUF` block-quant dequantisation to `BF16`** (`dequantize_gguf_to_bf16`,
  feature-gated behind `gguf`) — scalar reference kernels for all 12 block
  types covered by the parser: legacy `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
  `Q8_1` (32-element blocks) and K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`,
  `Q6_K`, `Q8_K` (256-element super-blocks). Formulas ported verbatim from
  `ggml-quants.c`'s `dequantize_row_*` reference functions, with
  block-at-a-time loop fission (packed-bit unpacking into an `[f32; QK]`
  stack scratch buffer, then a branch-free `f32 × scale → BF16` pass via
  the shared `f32_bits_to_bf16_bits` helper). `IQ*`/`TQ*`/`MXFP4` return
  `AnamnesisError::Unsupported` (deferred to a later phase). No new crate
  dependencies — reuses `half` and the existing `gguf` feature. Phase 4
  step 2 toward v0.4.0; bit-for-bit cross-validation against `llama.cpp`
  is Phase 4 step 4.

### Changed

- **GGUF dequant kernels refactored to close over a `FnMut` sink**
  instead of each returning an owned `Vec<u8>`. Per-type pass-1 unpacking
  is now a small closure fed to a generic `run_legacy_kernel` /
  `run_super_kernel` outer-loop helper that handles `chunks_exact(TS)`
  iteration, scratch-buffer management, and the shared pass-2 `BF16`
  writer. `Vec::with_capacity` + `extend_from_slice` replaces the old
  `vec![0u8; n_elements * 2]`, avoiding the zero-init memset (~10–15 %
  of dequant wall time on `Q8_0`/`Q4_0` saved on platforms without lazy
  zero pages). Per-block `.get_mut(range).ok_or_else(...)?` bounds
  checks are replaced with `chunks_exact_mut`-style iteration on the
  inner kernel runners — ~4 M branches removed per 1 M-block tensor.
- **Infallible byte readers**: `read_f16_le(&[u8], usize) -> Result<f32>`
  is replaced with `read_f16_bytes([u8; 2]) -> f32` (and analogously for
  `read_f32_bytes`), eliminating dead `Result` shuffling on every
  hot-loop call. Callers slice fixed-length arrays out of their
  already-validated block slices.
- **Output size overflow guard**: `dequantize_gguf_to_bf16` now checks
  `n_elements.checked_mul(2)` in its shared validation helper, turning
  what would have been a silent `Vec` allocation truncation on 32-bit
  targets with > 2 GiB of `BF16` output into a clean `AnamnesisError::Parse`.
- **`GGUF` file parser** (`parse_gguf`, feature-gated behind `gguf`) — lean
  in-house parser for `GGUF` v2 and v3 files. Reads header, metadata
  key-value pairs (all 13 value types including nested `ARRAY`), and tensor
  info table. Resolves absolute tensor-data offsets from the tensor-info
  table's relative offsets plus the effective `general.alignment` (default
  32 bytes). `ParsedGguf::tensors` returns zero-copy `Cow::Borrowed` slices
  into the memory-mapped file for every dtype with a known `type_size`
  (`F32`, `F16`, `BF16`, `F64`, `I8`–`I64`, `Q4_0`–`Q8_1`, `Q2_K`–`Q8_K`).
  `IQ*`/`TQ*`/`MXFP4` tensors are listed in `tensor_info()` with
  `byte_len = None` and will be sized when dequantisation lands. No new
  third-party crate — reuses the `memmap2` dependency already pulled in by
  the `pth` feature. First commit of Phase 4 toward v0.4.0.
- **`GgufMetadataArray`** — new `#[non_exhaustive]` enum holding natively
  typed arrays (`Vec<u8>`, `Vec<f32>`, `Vec<String>`, …). Replaces the old
  `GgufMetadataValue::Array(Vec<GgufMetadataValue>)` storage to eliminate
  the ~8× enum-discriminant bloat on homogeneous numeric metadata arrays.
- **`ParsedGguf::tensors` returns `impl Iterator<Item = GgufTensor<'_>>`**
  instead of `Result<Vec<GgufTensor<'_>>>` — zero heap allocation per call.
  `GgufTensor::{name, shape}` now borrow from the parsed handle as
  `&'a str` / `&'a [usize]` rather than cloning owned `String`/`Vec`.
  Worst-case allocation dropped from ~130 MB per call on a 1 M-tensor file
  to 0 MB. Callers needing random access should use `.collect::<Vec<_>>()`.
- **`GgufMetadataValue::Array` now holds `Box<GgufMetadataArray>`** (was
  `Vec<GgufMetadataValue>`). `GgufMetadataValue::as_array` now returns
  `Option<&GgufMetadataArray>`. As a side effect, `GgufMetadataValue`
  shrinks from 32 bytes to 24 bytes (25% reduction) across every metadata
  value, not just arrays, because the max-sized variant is now `String`.

### Security / Performance

- **Cap trust-the-header pre-allocation** at `PREALLOC_SOFT_CAP = 256`
  entries for every `Vec::with_capacity` / `HashMap::with_capacity` call
  keyed on a file-declared count (metadata kv count, tensor count,
  per-array length). Previously a ~40-byte adversarial header claiming
  1 M of each could force ~175 MB of eager heap allocation before a single
  entry was read (empirically measured: 114 MB `HashMap` + 61 MB
  `Vec<RawTensorInfo>`); the cap drops this to ~34 KB (5 000× reduction).
  An adversarial `ARRAY` header claiming 16 M `f32` elements forced
  ~488 MB of eager allocation; combined with the typed-array fix, this is
  now capped at ~8 KB (60 000× reduction). Legitimate files grow the
  containers geometrically and are unaffected.
- **`Cursor::read_string` validates UTF-8 on the borrowed mmap slice
  before copying** — an adversarial 16 MiB non-UTF-8 string now costs
  zero heap allocation on the rejection path (was: a full 16 MiB
  `to_vec()` followed by `String::from_utf8`).
- **`ParsedGguf::inspect` dedups distinct dtypes via a `[bool; 32]`
  bitmap** keyed on a dense `GgufType` discriminant, replacing
  `Vec::contains` in the per-tensor loop. Drops the dtype-dedup hot path
  from O(n × d) to O(n) — ~10 ms → ~1 ms on a 1 M-tensor inspect call.
  First-occurrence order of `GgufInspectInfo::dtypes` is preserved.
- **`parse_gguf` builds `tensor_infos` in a single pass** instead of
  first materialising a throwaway `Vec<RawTensorInfo>` and then iterating
  it. The relative tensor-data offset is stored in `data_offset` during
  the read pass and patched to the absolute offset in a short sweep once
  `data_section_start` is known. Peak tensor-info heap on a 1 M-tensor
  file drops by ~60 MB; `RawTensorInfo` and `read_raw_tensor_info` are
  deleted.

### Fixed

- **GGUF dequantization cross-validated against `llama.cpp` reference**
  (Phase 4 step 4). 10 of 12 production kernels bit-exact (0 ULP) against
  the `gguf` Python package's `dequantize` function (the official
  `ggml-org` reference mirroring `ggml-quants.c`). Legacy quants: `Q4_0`,
  `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`. K-quants: `Q2_K`, `Q3_K`, `Q4_K`,
  `Q5_K`, `Q6_K`. Fixtures from 3 real models: bartowski SmolLM2-135M-
  Instruct and TheBloke TinyLlama-1.1B-Chat. `Q8_1` and `Q8_K` are not
  shipped by any real model (internal `llama.cpp` activation quant types)
  and are already covered by unit tests.

- **`parse_gguf` accepted tensors whose relative offset was not a
  multiple of `general.alignment`** (Phase 4 audit I1). The GGUF spec
  mandates that every tensor's offset field is a multiple of the file's
  declared alignment, but the patch sweep only checked the upper bound
  of each tensor's byte range. A malformed file encoding
  `relative_offset = 1` for every tensor would parse successfully and
  hand out unaligned byte slices through `ParsedGguf::tensors`, which
  downstream SIMD dequant kernels would then reinterpret as `f32`/`f16`
  words — unaligned access is undefined behaviour in the `unsafe`
  intrinsics planned for Phase 9. `parse_gguf` now rejects such files
  with `AnamnesisError::Parse` naming the offending tensor.

## [0.3.2] - 2026-04-05

### Fixed

- **`copy_to_contiguous` silent data corruption on mismatched shape/strides** (NI1) —
  added ndim guard rejecting `shape.len() != strides.len()`. Previously, `.zip()`
  silently truncated to the shorter iterator, producing corrupted output. Defence-in-depth
  check also added in `parse_rebuild_args`
- **`copy_to_contiguous` inner loop used `.get()` despite `// INDEX:` annotation** (NI2) —
  switched to direct indexing `storage[range]`, matching CONVENTIONS.md and the
  pre-validation that proves bounds safety. Eliminates dead `.ok_or_else()` branches
- **NPZ `extract_descr` mixed-quote bug** — quote character detection now reads the
  first quote in the value portion, not the entire header tail. Fixes silent
  mis-extraction for mixed-quote headers like `{'descr': "<f4", ...}`
- **`parse_pth` stale return-type doc** (D1) — claimed "Returns `Vec<PthTensor>`" but
  actually returns `Result<ParsedPth>`. Updated to reference `ParsedPth` and
  `ParsedPth::tensors()`
- **`inspect_npz` overflow saturation undocumented** (D2) — added note explaining
  `byte_len` saturates to `usize::MAX` on shape overflow (best-effort metadata),
  unlike `parse_npz` which returns `Err`
- **Misused `// EXPLICIT:` in `build_entry_index`** (D3) — was a char-boundary
  assertion, not a no-op arm or stateful loop. Downgraded to plain comment
- **Per-line `// BORROW:` annotations** — replaced single block-level annotation in
  `execute()` with per-call annotations on all 12 `.to_owned()`/`.to_vec()` sites.
  Added missing annotations in `build_entry_index`
- **`// VECTORIZED:` on `copy_to_contiguous`** — added `scalar fallback` annotation
  documenting why the inner loop cannot auto-vectorize (cross-iteration coords state)
- **`# Memory` section on `ParsedPth::tensors`** — documents zero-copy vs owned
  allocation paths
- **`const fn` on `ParsedPth::len`/`is_empty`** — `Vec::len()`/`is_empty()` are
  `const fn` since Rust 1.39
- **`# Errors` sections** on `parse_rebuild_args`, `build_entry_index`, and
  `copy_to_contiguous` — consistency with other private fallible functions
- **`lib.rs` architecture doc** (D6) — added `pth_to_safetensors()` to bullet list
- **`NpzDtype` `Display` doc** (D7) — documented as canonical uppercase string used
  in inspection output and cross-validation tests
- **`parse/mod.rs` stale docstring** — updated from "wraps `npyz`" to reflect own parser
- **`byteswap_inplace` missing `// VECTORIZED:` annotation** — added per CONVENTIONS.md
- **NPZ annotations** — `// EXPLICIT:` for `=` native-endian prefix, `// EXPLICIT:`
  for `extract_fortran_order` default
- **`bench_npz_adhoc.rs` hardcoded path** — replaced with `dirs::home_dir()` fallback

### Added

- **48 new unit tests** covering code review findings G1–G36, NI1–NI2, NN1–NN4:
  pickle VM opcodes (FRAME, NONE, NEWTRUE/NEWFALSE, BININT, BININT2, BINUNICODE,
  SHORT_BINSTRING, BINSTRING, SHORT_BINBYTES, BINBYTES, EMPTY_LIST, EMPTY_TUPLE,
  TUPLE1, TUPLE3, SETITEMS, APPEND, APPENDS, STACK_GLOBAL, REDUCE, NEWOBJ, BUILD,
  BINPERSID, LONG_BINPUT/LONG_BINGET, MEMOIZE), `long1_to_i64` 8-byte boundary,
  `MEMOIZE` overflow at `u32::MAX`, `MAX_PICKLE_NESTING` enforcement (both
  `unwrap_to_rebuild` and `extract_dict_pairs`), `copy_to_contiguous` (transposed,
  zero-element, overflow, zero-stride broadcast, ndim mismatch, storage boundary),
  missing/compressed `data.pkl` ZIP entries, zero-length ZIP entry,
  NPZ Fortran-order end-to-end rejection, empty NPZ archive (parse + inspect),
  native-endian `=` prefix, big-endian through `parse_npz`, `inspect_npz` overflow

## [0.3.1] - 2026-04-02

### Added

- **PyTorch `.pth` parsing** (`src/parse/pth.rs`) — minimal pickle
  interpreter (~36 opcodes) that parses `PyTorch` ≥ 1.6 `state_dict` ZIP
  archives with a safe, explicit `GLOBAL` allowlist (rejects non-`torch.*`
  callables — equivalent to `weights_only=True` but stricter). Zero-copy
  I/O via `memmap2` with `Cow::Borrowed` tensor data sliced directly from
  the mmap. Handles shared storage, non-contiguous strides, big-endian
  byte order, and both newer (`archive/`) and older (`{model_name}/`)
  `PyTorch` ZIP prefix conventions. Feature-gated behind `pth`. Supports
  `F16`, `BF16`, `F32`, `F64`, `I8`–`I64`, `U8`, `Bool` storage types.
  **11–31× faster** than `torch.load()` on torchvision models (resnet18,
  resnet50, ViT-B/16)
- **`.pth` → safetensors conversion** (`src/remember/pth.rs`) — lossless
  format conversion preserving original dtypes (no dequantization). The
  conversion pipeline writes directly from mmap slices to the output file
  — zero intermediate data copies. Byte-exact roundtrip verified against
  `PyTorch` reference on all 3 test models
- **`.pth` cross-validation** against `PyTorch` on 3 real
  [AlgZoo](https://github.com/alignment-research-center/alg-zoo) models
  (MIT-0 license): `2nd_argmax_2_2` RNN (10 params), `longest_cycle_2_3`
  Transformer (50 params), `one_layer_16_hidden` RNN blog example (432
  params). Byte-exact match on all tensors against `PyTorch` reference
- **CLI `.pth` support** — `amn parse`, `amn inspect`, and `amn remember`
  now accept `.pth`, `.pt`, and `.bin` files when built with
  `--features pth`. Format detection by extension with ZIP magic fallback
  for `.bin` files. `amn remember model.pth` converts to safetensors;
  `amn parse model.pth` shows per-tensor details (name, dtype, shape,
  size)
- **`ParsedPth`** container — owns the mmap, provides zero-copy
  `tensors()`, `inspect()` → `PthInspectInfo`, `tensor_info()` →
  `PthTensorInfo` (metadata only, no data access), and
  `to_safetensors()` convenience method
- **`PthInspectInfo`** — summary struct (tensor count, total bytes,
  dtypes, byte order) with `Display` impl
- **`PthTensorInfo`** — lightweight per-tensor metadata (name, shape,
  dtype, `byte_len`) for display paths that don't need tensor data
- **`PthDtype::to_safetensors_dtype()`** — direct single-hop conversion
  to `safetensors::Dtype`, bypassing the intermediate anamnesis `Dtype`
- **`inspect_npz()`** (`src/parse/npz.rs`) — header-only `NPZ` inspection
  that reads only `NPY` headers (~128 bytes per array), no tensor data.
  Returns `NpzInspectInfo` + `NpzTensorInfo` (name, shape, dtype,
  `byte_len`). For a 300 MB file, uses kilobytes instead of 300 MB
- **CLI `.npz` support** — `amn parse` and `amn inspect` now accept
  `.npz` files when built with `--features npz`, using the header-only
  `inspect_npz` path. `amn remember` for `.npz` returns a clear
  unsupported error (tensors are already full-precision)

### Changed

- Extracted `byteswap_inplace` from `src/parse/npz.rs` to shared
  `src/parse/utils.rs` module (`pub(crate)`) so that multiple format
  parsers (`NPZ`, `.pth`) can reuse it without duplication
- Widened `From<ZipError>` impl in `error.rs` from `npz`-only to
  `any(npz, pth)` feature gate
- Changed `unsafe_code` lint from `forbid` to `deny` to allow
  feature-gated `memmap2` usage in the `pth` module (with `// SAFETY:`
  annotation)

### Fixed

- **`has_zip_magic`** now reads only 4 bytes via `read_exact` instead of
  loading the entire file into heap (prevented 7 GB allocation on large
  `.bin` files)
- **`build_entry_index`** now returns `AnamnesisError::Parse` for corrupt
  ZIP entries whose data range exceeds the file size, instead of silently
  skipping them
- **`extract_dict_pairs`** unreachable `Reduced{OrderedDict}` branch now
  returns `Err` instead of `Ok(&[])`, preventing silent data loss
- **MEMOIZE** opcode uses `checked_add(1)` instead of plain `+= 1`,
  preventing silent `u32` wraparound on adversarial pickles
- **`build_entry_index`** `u64`→`usize` casts replaced with `TryFrom`,
  consistent with codebase conventions (no truncation on 32-bit)
- **`inspect()`** element count uses `saturating_mul` instead of
  `checked_mul().unwrap_or(0)`, avoiding silently wrong `total_bytes`
- **`--to`** argument for `.pth` files is now validated: accepts
  `safetensors` or `bf16`, errors on unsupported values
- **`copy_to_contiguous`** uses the two-level bounds pattern from
  `CONVENTIONS.md` — pre-validates max source offset once before the
  loop, removing 6 per-element `checked_*` calls

### New dependencies

- `memmap2` v0.9 (optional, `pth` feature only) — memory-mapped file I/O

## [0.3.0] - 2026-03-24

### Added

- **NPZ/NPY parsing** (`src/parse/npz.rs`) — `parse_npz(path)` reads `NumPy`
  `.npz` archives into framework-agnostic `NpzTensor` structs. Custom `NPY`
  header parser with bulk `read_exact` data extraction — zero per-element
  deserialization for LE data on LE machines. Feature-gated behind `npz`.
  Supports `F16`/`F32`/`F64`, all integer types, `Bool`, and `BF16` (JAX `V2`
  void dtype). **3,586 MB/s** on 302 MB Gemma Scope file (1.3× raw I/O
  overhead), **17.7× faster** than `npyz`-backed parser
- **NPZ cross-validation** against Gemma Scope 2B SAE weights (`params.npz`,
  5 `F32` arrays). Byte-exact match against `NumPy` reference on all arrays

### Fixed

- **BnB4 output shape recovery** — `NF4`/`FP4` dequantized weights now have their
  original 2D shape (e.g., `[2048, 8192]`) instead of flat `[total_elements]`.
  Shape is recovered from the `quant_state.bitsandbytes__nf4`/`__fp4` companion
  tensor's `JSON` blob, which is stored inside the safetensors file itself (no
  `config.json` needed). Falls back to 1D if the companion is absent
- **`extract_descr` mixed-quote header bug** — quote character detection now reads
  the first character of the value, not the entire header tail. Fixes silent
  mis-extraction for mixed-quote headers (e.g., `{'descr': "<f4", ...}`)
- Added mandatory `// VECTORIZED:`, `// EXPLICIT:` annotations on
  `byteswap_inplace`, `extract_fortran_order`, native-endian `=` treatment,
  and unused minor version byte
- Removed hardcoded absolute path from `bench_npz_adhoc.rs` — now resolves
  from `USERPROFILE`/`HOME` environment variables
- Added missing `[0.2.0]` changelog section for Phase 2 (GPTQ, AWQ, BnB)

## [0.2.0] - 2026-03-24

### Added

- **GPTQ dequantization** (`src/remember/gptq.rs`) — INT4 and INT8 with
  group-wise scale + zero-point, activation-order via `g_idx`. Feature-gated
  behind `gptq`. Bit-exact against PyTorch on 4 real models from 2 quantizers
  (AutoGPTQ, GPTQModel), **6.5–12.2× faster** than CPU PyTorch (AVX2)
- **GPTQ parsing layer** — `TensorRole::ZeroPoint` / `GroupIndex`,
  `QuantScheme::Gptq`, `GptqConfig` (bits + group_size inference from metadata
  or tensor shapes), `find_gptq_companions()`, `gptq` feature gate
- **GPTQ cross-validation** against PyTorch on 4 models: Falcon3-1B INT4/INT8
  (AutoGPTQ), Llama-3.2-1B INT4 (AutoGPTQ), Llama-3.2-1B INT8 (GPTQModel)
- **GPTQ inspect/CLI** — zero-point, group-index counts in `inspect` and
  `parse` output; format-aware size label (FP8/GPTQ/unquantized)
- **AWQ dequantization** (`src/remember/awq.rs`) — INT4 (and INT8 path,
  unit-tested) with per-group scales, no +1 zero-point offset. Feature-gated
  behind `awq`. Bit-exact against PyTorch on 2 real models (AutoAWQ GEMM),
  **4.7–5.7× faster** than CPU PyTorch (AVX2). Loop fission applied from the
  start; full AVX2 `vsubps`/`vmulps` ymm confirmed
- **AWQ parsing layer** — `QuantScheme::Awq`, `AwqConfig`, `AwqCompanions`,
  shape-based detection distinguishing AWQ (packed along cols) from GPTQ
  (packed along rows), `awq` feature gate
- **AWQ cross-validation** against PyTorch on 2 models: Llama-3.2-1B and
  Falcon3-1B (both AutoAWQ GEMM, 4-bit). Note: no real 8-bit AWQ models
  exist in the standard AutoAWQ `.qweight` format — all "8-bit AWQ" models
  on HuggingFace are either dequantized F16, `compressed-tensors` (vLLM), or
  mislabeled 4-bit
- **BitsAndBytes dequantization** (`src/remember/bnb.rs`) — NF4, FP4 (both
  4-bit lookup-table with per-block absmax), double-quant NF4/FP4 (nested
  absmax), and INT8 (`LLM.int8()` with per-row absmax). Feature-gated behind
  `bnb`. Bit-exact against PyTorch on 4 real models, **18–54× faster** for
  NF4/FP4 (AVX2), **1.2× faster** for INT8 (near memory bandwidth limit).
  Loop fission for NF4/FP4; single-pass AVX2 for INT8 (`vpmovsxbd` →
  `vcvtdq2ps` → `vmulps`)
- **BnB parsing layer** — `QuantScheme::Bnb4` / `BnbInt8`,
  `TensorRole::QuantMap` / `NestedScale`, `BnbConfig` (block_size,
  double_quant), `Bnb4Companions`, detection by `.weight.quant_map` (NF4/FP4)
  and `.SCB` (INT8) naming patterns, `bnb` feature gate
- **BnB cross-validation** against PyTorch on 4 models: Llama-3.2-1B NF4,
  Llama-3.2-1B NF4 double-quant, Llama-3.2-1B FP4, Llama-3.2-1B INT8
- **BnB model.rs integration** — `Bnb4` and `BnbInt8` arms in
  `remember_bf16_inner` with companion lookup, double-quant detection, and
  shape handling (flat output for NF4/FP4, preserved 2D for INT8)
- `FromStr` impl for `TargetDtype` — centralizes string-to-enum parsing so new
  variants cannot be silently missed in the CLI
- `ParsedModel::remember_with_progress()` — dequantize with a per-tensor
  callback, enabling progress reporting in CLI contexts
- `indicatif` progress bar during `remember`/`dequantize` when built with the
  `indicatif` feature (`amn remember` shows `[====================] 2.1s`)
- CONVENTIONS.md: two-level bounds checking pattern (reconciles `// INDEX:`
  safety with SIMD rule #2) and loop fission for mixed-domain pipelines

### Fixed

- Added unit tests for `dequantize_per_channel_fp8_to_bf16` covering F32, BF16,
  and F16 scale dtypes, single-row, NaN handling, and validation errors
- Added fine-grained dequantization tests for all three scale dtypes (F32, BF16,
  F16) and multi-block F32 scale path
- Added CLI integration tests (`tests/cli.rs`) — 9 tests covering `parse`,
  `inspect`/`info`, `remember`/`dequantize`, error handling, and `--version`
- Documented single-scheme assumption in `detect_scheme` (all quantized tensors
  in a file use the same scheme; early-return on first scale companion found)
- CLI `parse` subcommand now displays the actual scale dtype (BF16, F16, F32)
  instead of always printing "F32"
- `inspect` Display now shows the actual scale dtype instead of hardcoded "F32"
- `dequantize_per_tensor_fp8_to_bf16` now uses `checked_mul` for output size,
  consistent with the other two dequantize functions (**breaking**: returns
  `Result<Vec<u8>>` instead of `Vec<u8>`)
- Fine-grained dequantization now validates that the scale grid is rectangular
  (rejects `scale_elements % scale_rows != 0` instead of silently truncating)
- `serialize_to_file` I/O errors now surface as `AnamnesisError::Io` instead of
  being misclassified as `AnamnesisError::Parse`
- `derive_output_path` now matches `TargetDtype` exhaustively instead of using a
  wildcard that would silently produce broken paths for future variants
- Simplified `shape_to_rows_cols` 2D arm: direct indexing with `// INDEX:`
  annotation instead of redundant `Option` unwrapping
- **`classify_tensor` AWQ-only builds** — `.qweight`/`.qzeros`/`.scales` were
  gated on the `gptq` feature only; AWQ-only builds silently misclassified all
  quantized tensors as passthrough. Now gated on `any(gptq, awq)` with `.g_idx`
  remaining `gptq`-only
- **`detect_scheme` silent fallthrough** — `return` statements inside the
  GPTQ/AWQ detection block were feature-gated, causing misdetection when only
  one scheme was enabled. Detection is now unconditional; feature-disabled
  errors are handled downstream in `model.rs`
- `derive_output_path` now strips GPTQ, AWQ, and BitsAndBytes suffixes
  (e.g., `-GPTQ-Int4`, `-awq`, `-bnb-4bit`) in addition to FP8 suffixes
- `read_scale_f32` now uses `checked_add` for all byte offset computations,
  consistent with `read_u32_le` in the same codebase
- Extracted duplicated `read_u32_le` and `read_scale_f32` from `gptq.rs` and
  `awq.rs` into shared `remember/quant_utils.rs` module
- Replaced dead-code `checked_mul(1)` in `bnb.rs` with direct parity check
- GPTQ/AWQ outer loop offsets (`i * out_features * 2`) now use `checked_mul`
  for consistency with the codebase's zero-panic discipline
- `parse_g_idx` offset (`i * 4`) now uses `checked_mul`
- Updated GPTQ docstring memory estimate from "~1 MB" to "up to ~8 MB per
  weight tensor" to reflect fine-grained group configurations

## [0.1.0] - 2026-03-24

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
