# Performance Experiments — Tested and Rejected

This file is the **case-study log of perf hypotheses that were tested and either
rejected, partially confirmed, or contradicted by measurement**. It exists so
future audits and reviews don't re-propose the same ideas without first reading
what already happened.

The binding rule for any new perf-claim commit lives in [`CLAUDE.md`'s
Performance Changes section](../CLAUDE.md). This file is the historical record
backing it.

## Why this file exists

In late April 2026 a multi-finding "algorithmic-weakness audit" was run against
the crate. Several findings were framed in absolute-sounding terms ("saves
~30 % on Gemma Scope", "saves ~10 % on every dequant kernel") but turned out to
be wrong in direction or much smaller than claimed once measured on real
fixtures and real hardware. After the second consecutive revert
([commit `5f2632b`](../README.md), then a never-committed FP8 refactor),
the project adopted the rule: **measure on a real fixture before committing any
perf-claim change**. This file catalogs what's been tested.

## Experiment index

| # | Experiment | Verdict | Commit / status |
|---|---|---|---|
| 1 | NPZ `read_array_data` memset elimination | **Regressed −33 %** | Committed as `67d6db0`, reverted in `5f2632b` |
| 2 | FP8 per-tensor chunked extend | **Regressed −23 %** | Never committed (this session, branch is clean) |
| 3 | v0.4.0 GGUF refactor re-validation | **Split: Q4_0 wins ~8 %, Q8_0 loses ~6 %** | Re-measurement only — current code unchanged |
| 4 | `parse()`: `fs::read` → `memmap2::Mmap` | **~3000× faster on 11 GiB safetensors** | Shipped |

---

## Experiment 1 — NPZ `read_array_data` memset elimination

**Audit finding:** "`vec![0u8; data_bytes]` zero-inits the buffer immediately
before `read_exact` overwrites every byte — pure dead work. Switching to
`Vec::with_capacity(data_bytes)` + `reader.take(data_bytes).read_to_end(...)`
should save ~30 % of the parse time on Gemma Scope `params.npz` (302 MB)."

**Method:** [`tests/bench_npz_adhoc.rs`](../tests/bench_npz_adhoc.rs),
best-of-5 release-mode median, target-cpu=native, warmed FS cache. Compared
the two versions by `git checkout`-ing only `src/parse/npz.rs` between runs.

**Result:**

| Variant | Median | Range (min/max) |
|---|---|---|
| pre-#4 (`vec![0u8;n]` + `read_exact`) | **82.9 ms** | 82.2–83.2 (σ≈0.4) |
| post-#4 (`Vec::with_capacity` + `take().read_to_end`) | **110.8 ms** | 104.3–131.8 (σ≈11) |

A **+33 %** regression, opposite direction from the audit's prediction.

**Why the prediction was wrong:**

- A SIMD-optimised memset on a fresh allocation runs at ~25 GB/s on modern x86,
  so `vec![0u8; 302_000_000]` costs ~10 ms — not the ~25 ms the audit implied.
- `read_to_end` reads in ~8 KiB chunks via `read_buf`; for 302 MB that's ~37 000
  `read` syscalls vs the **single** `read_exact` syscall the old code issued.
  Even with `Vec::with_capacity` pre-allocating exactly the right size (so no
  reallocations), the iteration overhead dominates and swamps the memset
  saving.

**Disposition:** reverted in [`5f2632b`](../CHANGELOG.md). The full pre/post
numbers and analysis are preserved in that commit's message.

**Re-attempting this requires:** a safe-Rust replacement that beats
`read_exact` over a pre-allocated buffer. The only mechanism that would work is
`unsafe { buf.set_len(n) }` + `read_exact`, which requires amending
[`CONVENTIONS.md`](../CONVENTIONS.md)'s accepted-`unsafe` table. Not justified
for a single read site that saves ~10 ms.

---

## Experiment 2 — FP8 per-tensor chunked extend

**Audit finding:** "`vec![0u8; out_byte_len]` in
`dequantize_per_tensor_fp8_to_bf16` is dead work; the v0.4.0 GGUF refactor
saved ~10–15 % on `Q8_0`/`Q4_0` with the same change." Predicted ~10 % win on
FP8 per-tensor.

**Method:** [`tests/bench_dequant_adhoc.rs`](../tests/bench_dequant_adhoc.rs)
`bench_fp8_per_tensor`. 4096 × 11008 = 45 M FP8 elements, ~90 MB BF16 output
(typical Llama-class FFN layer). Best-of-5 release-mode median.

**Replacement design:** Chunked extend with a 2048-element stack scratch
buffer ([CONVENTIONS.md](../CONVENTIONS.md) SIMD-friendly loop rules
preserved: `chunks_exact` outer loop, vectorisable inner zip into
`[u8; 4096]`, single `extend_from_slice` per chunk).

**Result:**

| Variant | Median | Range (min/max) |
|---|---|---|
| BEFORE (`vec![0u8;n]` + zip) | **39.63 ms** | 39.41–42.89 (σ≈1.0) |
| AFTER (`Vec::with_capacity` + chunked extend) | **48.63 ms** | 48.59–48.79 (σ≈0.07) |

A **+23 %** regression, opposite direction from the audit's prediction. The
post-refactor σ is ~14× tighter, suggesting the regression is a stable cost
attribution, not measurement noise.

**Why the prediction was wrong:**

1. **The memset cost the audit assumed wasn't actually paid.** `vec![0u8; n]`
   on Windows allocates via `HeapAlloc` → `VirtualAlloc` with `MEM_COMMIT`. The
   kernel returns *demand-zero pages* — virtual addresses that map to a magic
   zero page lazily, then get individually zero-filled on first write. So the
   "memset" we thought we were eliminating wasn't a separable cost; it was a
   constant per-page tax that any allocation pays. (Linux and macOS also use
   demand-zero pages.)
2. **The chunked structure adds a doubled memory pass.** In the original, each
   element does `read 1 input byte → arithmetic → write 2 output bytes` in one
   tight zip the compiler interleaves. In the chunked refactor, each element
   does `read 1 input byte → arithmetic → write 2 bytes to scratch (L1) →
   memcpy 4096 bytes from scratch to output`. Even though scratch lives in L1,
   the additional memcpy is a measurable secondary cost.

**Disposition:** never committed. `src/remember/fp8.rs` remains on the
pre-refactor pattern.

**Re-attempting this requires:** evidence that one of the lazy-zero-page
absorbing assumptions doesn't hold (e.g., a target where the memset actually
runs eagerly), AND a refactor that doesn't add a second memory pass.

---

## Experiment 3 — v0.4.0 GGUF refactor re-validation

**Background:** The v0.4.0 CHANGELOG ([2026-04-12](../CHANGELOG.md))
described the `Vec::with_capacity` + `extend_from_slice` GGUF dequant
refactor as **"~10–15 % of dequant wall time on `Q8_0`/`Q4_0` saved on
platforms without lazy zero pages"**. The "platforms without lazy zero pages"
caveat is doing a lot of work — Windows, Linux, and macOS all *have* lazy zero
pages. After Experiment 2's null result, this claim looked suspect and was
re-measured.

**Method:** [`tests/bench_dequant_adhoc.rs`](../tests/bench_dequant_adhoc.rs)
`bench_gguf_size_sweep`. Same kernel logic driven two ways via the public
streaming API `dequantize_gguf_blocks_to_bf16`:
- **NEW** — current `dequantize_gguf_to_bf16` (`Vec::with_capacity` +
  per-block `extend_from_slice`).
- **OLD** — bench-local replay of the pre-refactor pattern: pre-allocate
  `vec![0u8; out_byte_len]`, drive the streaming API with a sink that tracks
  an offset and writes via indexed `copy_from_slice`.

Sweep across four output sizes: 2 MB (L3-resident) → 16 MB (L3 boundary) →
90 MB (Llama-class FFN) → 200 MB (deeply DRAM-bound). Best-of-5 release-mode
median per cell.

**Result (NEW vs OLD median delta, negative = NEW faster):**

| Output BF16 size | Q8_0 | Q4_0 |
|---|---|---|
| 2 MB | **+3.0 %** slower | **−6.9 %** faster |
| 16 MB | **+9.5 %** slower | **−8.2 %** faster |
| 90 MB | **+6.3 %** slower | **−7.2 %** faster |
| 200 MB | **+3.9 %** slower | **−9.1 %** faster |
| **Average** | **+5.7 %** slower | **−7.9 %** faster |

The sign is stable across all 4 sizes for each kernel — the result is a
structural property of the kernels, not size-dependent measurement noise.

**Verdict:** the v0.4.0 CHANGELOG claim was **partially wrong**:

- **`Q4_0`** is a real win, but the magnitude was overstated (~8 % measured vs
  10–15 % claimed).
- **`Q8_0`** is a real **regression** (~6 % slower than the pre-refactor
  pattern). The CHANGELOG asserted a uniform improvement; reality is a
  net wash across the two kernels.

**Why `Q8_0` and `Q4_0` disagree** (best understanding):

- Both kernels emit BF16 through the same `dispatch_streaming` → sink-closure
  pipeline. The OLD vs NEW difference is just
  `out[offset..].copy_from_slice(block_out); offset += len;` versus
  `out.extend_from_slice(block_out)` — almost identical machine code.
- **`Q8_0`** is bandwidth-bound (`d × i8 → BF16`, no bit unpacking). The
  output-write bandwidth is the bottleneck. Anything that adds even small
  overhead per block (e.g., extra Vec metadata bookkeeping) shows up.
- **`Q4_0`** does packed-nibble unpacking (`(q & 0xF) - 8`,
  `(q >> 4) - 8`), so the kernel has more CPU work per output byte. The
  per-block overhead is amortised across that work, and `Vec::extend_from_slice`'s
  internal length update apparently has slightly less overhead than the
  manual `offset += ...` pattern for this kernel.

**Disposition:** **current code unchanged.** Two reasons:

1. The deltas roughly cancel: a 5.7 % regression on `Q8_0` and a 7.9 % win on
   `Q4_0`. Splitting the dispatch by kernel (Q4_0 keeps the new pattern, Q8_0
   reverts) would add complexity for a 1–2 ms saving on a 22 ms kernel.
2. The bench file is now the audit-trail. The next person tempted to "fix
   `Q8_0`" can read this entry and Experiment 2 first.

**The principle:** "save the memset" is **not a reliable rationale** in this
codebase. Three of four perf-claim experiments based on it have measured null
or regression. Future audit findings using this framing should be treated as
hypotheses to disprove with measurement, not as actionable.

---

## Experiment 4 — `parse()`: `fs::read` → `memmap2::Mmap`

**Audit finding:** "[`src/model.rs:90`](../src/model.rs) calls `std::fs::read(path)`, materialising the entire safetensors file into a `Vec<u8>` before the header is even parsed. On a 70 GiB shard this peaks at 70 GiB even when the caller only intends to `inspect()`. Switching to `memmap2::Mmap::map(&file)` would let the kernel page bytes in lazily — `parse()` + `inspect()` would then only fault in the header (~1 MiB), and full `remember()` paths gain OOM-resilience because file-backed pages can be dropped by the kernel under memory pressure (whereas `Vec<u8>` pages cannot, they need swap)."

**Method:** [`tests/bench_parse_adhoc.rs`](../tests/bench_parse_adhoc.rs)
`bench_parse_safetensors_large`. Fixture: a locally-cached 11 560 MiB
single-file safetensors model (`bigcode/starcoder2-3b/model.safetensors`).
Best-of-5 release-mode median, 2-iteration warmup to populate the OS file
cache. Compared `parse()` alone and `parse()` + `inspect()`.

**Result:**

| | BEFORE (`fs::read` + `Vec<u8>`) | AFTER (`memmap2::Mmap::map`) | Delta |
|---|---|---|---|
| `parse()` median | **2881.93 ms** (range 2787.82–2887.74, σ ≈ 40 ms) | **0.89 ms** (range 0.86–0.91, σ ≈ 0.02 ms) | **~3236× speedup** |
| `parse()` + `inspect()` median | 2715.84 ms | 0.94 ms | ~2889× speedup |
| `inspect()` overhead | (within noise) | 0.05 ms | ✓ as expected |

The "before" parse() rate is ~4 GiB/s — consistent with `memcpy` from
the warm OS file cache to a fresh `Vec<u8>`. The "after" rate is
file-size-independent: `mmap` setup + parsing the ~1 MiB header.

**Why the prediction was right (and the magnitude):**

`std::fs::read` is `open + read_exact(n) + close` where `n` is the file
size. The dominant cost on a warm cache is the `memcpy` from the FS
cache to the freshly-allocated `Vec<u8>` — ~4 GiB/s on this hardware,
linear in file size.

`memmap2::Mmap::map` is `open + mmap + close` where `mmap` is a
kernel call that establishes virtual address translations without
copying anything — constant time, file-size-independent. Subsequent
reads through the mapping fault in pages on demand. For
`parse_safetensors_header`, only the first ~1 MiB is touched, so for
the inspect-only path the resident-set growth is bounded by header
size, not file size.

The ~3000× speedup is the ratio of (file size / `memcpy` bandwidth)
to (constant `mmap` setup + header parse). It scales with file size:
on a 70 GiB shard the speedup would be larger still.

**Disposition:** **Shipped**. Commit hash recorded in this entry's
index row when the commit lands. All 320 unit tests + every
cross-validation suite (FP8, GPTQ, AWQ, BnB, GGUF, NPZ, PTH) still
pass — the refactor is semantically equivalent because the public
API surface (`ParsedModel::inspect`, `ParsedModel::remember`,
`tensor_data`) all consume the buffer through `&[u8]` slices, and
`memmap2::Mmap` derefs to `[u8]` so callers see no change.

**Trade-offs accepted:**

- `memmap2` becomes a mandatory dependency (was optional, gated behind
  `pth`/`gguf`). This adds ~one small crate to the dependency tree of
  every build, including the safetensors-only minimal build. Justified
  by the always-on speedup.
- Concurrent file modification by another process is now undefined
  behaviour — the same assumption every other tensor parser in this
  crate (`parse_pth`, `parse_gguf`) and the upstream `safetensors`
  crate's mmap path already rely on. Documented in the `// SAFETY:`
  comment and the [CONVENTIONS.md](../CONVENTIONS.md) accepted-`unsafe`
  table.

**Re-attempting this requires:** N/A — this is the success case. If
the change ever needs to be reverted, the `bench_parse_adhoc` harness
is in place to detect a regression.

## How to add an entry

When you ship (or attempt to ship) a perf-claim change, add a row to the index
table and a section below. The minimum content is:

- **Audit finding** (one paragraph) — what was claimed and why.
- **Method** — bench file, fixture, hardware/OS, harness type (best-of-N
  median, etc.).
- **Result** — a table of before/after numbers with σ or range.
- **Why the prediction was right or wrong** — root-cause analysis,
  preferably citing measured behaviour rather than asymptotic argument.
- **Disposition** — committed (with hash), reverted (with hash), or never
  committed.
- **Re-attempting this requires** — what new evidence would make the
  experiment worth retrying.

Keep entries even when the experiment *succeeds*: a successful experiment with
documented before/after numbers is the strongest possible defense against
future regressions.
