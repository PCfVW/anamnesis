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
| 5 | `inspect_gguf_from_reader`: internal `BufReader<R>` | **~52× faster on `File` substrate, mmap parity** | Shipped |
| 6 | `inspect_pth_from_reader` vs `parse_pth(path).inspect()` vs `torch.load` | **Reader ≤1.6× mmap on KiB fixtures; both 2.4–3.6× faster than `torch.load`** | Shipped |

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

## Experiment 5 — `inspect_gguf_from_reader`: internal `BufReader<R>` (Tier 1)

**Audit finding:** The Phase 4.9 substrate-equivalence test surfaced
that `inspect_gguf_from_reader(File::open(path)?)` was 30–100× slower
than `parse_gguf(path).inspect()` on the same file (e.g., 213 ms vs.
3.0 ms on a 2.7 GiB Mistral-7B-IQ3_XXS). Diagnosis: the parser issues
many small `read_exact` calls (4–8 B per typed primitive, variable per
`gguf_string_t`), and on a `File` substrate every one is a syscall.
Hypothesis: wrapping the user's reader in a `std::io::BufReader<R>`
(64 KiB buffer) inside `inspect_gguf_from_reader` collapses those into
one underlying read per buffer-fill, with no API change and no
correctness risk (the only `Seek` calls happen at `GgufReader::new`
*before* any reads, so the buffer is empty when seek is issued — no
invalidation cost).

**Method:** [`tests/bench_gguf_inspect_adhoc.rs`](../tests/bench_gguf_inspect_adhoc.rs)
(`bench_gguf_inspect_paths`), best-of-5 release-mode median per file
with min/max range, target-cpu=native (`$env:RUSTFLAGS = "-C target-cpu=native"`),
1 warm-up iteration before timing. Compared baseline (no `BufReader`)
vs. post-Tier-1 (`BufReader::with_capacity(64 * 1024, reader)`) by
running the bench, applying the patch, running again. 17 real `GGUF`
files from `tests/fixtures/gguf_reference/models/` spanning 4
architectures × 11 distinct dtypes × 84 MiB to 2.7 GiB:

- `bartowski/SmolLM2-135M-Instruct` (8 quants: `Q2_K`, `Q3_K_M`, `Q4_0`,
  `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ4_XS`)
- `bartowski/Mistral-7B-Instruct-v0.3` (5 quants: `IQ1_S`, `IQ1_M`,
  `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`)
- `bartowski/Qwen2.5-{0.5,1.5}B-Instruct-IQ2_M`
- `TheBloke/TinyLlama-1.1B-chat-v1.0` (`Q2_K`, `Q5_0`)

**Result:**

| Aggregate | Baseline reader/mmap ratio | Post-Tier-1 reader/mmap ratio |
|---|---|---|
| Min | 46.6× slower | 0.9× (slightly **faster** than mmap) |
| Median | 51.7× slower | 1.0× (parity) |
| Mean | 56.6× slower | 1.0× (parity) |
| Max | 71.4× slower | 1.0× (parity) |

Per-file reader medians (μs), best-of-5:

| File | Baseline | Tier 1 | Reader speedup |
|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.3-IQ1_M | 209,452 | 2,845 | **73.6×** |
| Mistral-7B-Instruct-v0.3-IQ1_S | 213,157 | 2,856 | **74.6×** |
| Mistral-7B-Instruct-v0.3-IQ2_XS | 214,694 | 2,826 | **76.0×** |
| Mistral-7B-Instruct-v0.3-IQ2_XXS | 214,768 | 2,881 | **74.5×** |
| Mistral-7B-Instruct-v0.3-IQ3_XXS | 213,215 | 2,829 | **75.4×** |
| Qwen2.5-0.5B-Instruct-IQ2_M | 1,228,412 | 25,712 | **47.8×** |
| Qwen2.5-1.5B-Instruct-IQ2_M | 1,229,113 | 25,424 | **48.3×** |
| SmolLM2-135M-Instruct-IQ4_XS | 399,473 | 7,538 | **53.0×** |
| SmolLM2-135M-Instruct-Q2_K | 397,048 | 8,338 | **47.6×** |
| SmolLM2-135M-Instruct-Q3_K_M | 400,154 | 7,753 | **51.6×** |
| SmolLM2-135M-Instruct-Q4_0 | 400,510 | 8,054 | **49.7×** |
| SmolLM2-135M-Instruct-Q4_K_M | 399,283 | 7,578 | **52.7×** |
| SmolLM2-135M-Instruct-Q5_K_M | 397,638 | 7,558 | **52.6×** |
| SmolLM2-135M-Instruct-Q6_K | 398,046 | 7,641 | **52.1×** |
| SmolLM2-135M-Instruct-Q8_0 | 430,908 | 7,560 | **57.0×** |
| TinyLlama-1.1B-chat-v1.0.Q2_K | 440,615 | 6,961 | **63.3×** |
| TinyLlama-1.1B-chat-v1.0.Q5_0 | 437,530 | 7,132 | **61.4×** |

The `parse_gguf(path).inspect()` (mmap-backed) numbers are unchanged
across the two runs — Tier 1 only touches the reader-generic entry
point, by design. Median mmap times: ~3.0 ms for Mistral-7B,
~26.4 ms for Qwen2.5, ~8.0 ms for SmolLM2, ~7.9 ms for TinyLlama.

**Why the prediction was right and the headline result was bigger
than expected:** On a `File` substrate with cold-then-warm fs cache,
the post-Tier-1 reader path occasionally measures *faster than mmap*
(0.9× ratio). The likely explanation: BufReader does one syscall per
64 KiB of metadata, while mmap incurs one minor page fault per 4 KiB
page touched (the front matter is ~few MiB on these fixtures, so
dozens of syscalls vs. a few hundred page faults). Both backends
ultimately read the same OS-cached pages — but BufReader's larger
batch granularity wins on this access pattern.

The 30–100× baseline ratio was an underestimate of the syscall cost
on Windows; the 47–76× per-file speedups are the empirical answer.
The Qwen2.5 fixtures are the slowest in absolute terms (1.23 s
baseline) because their `tokenizer.ggml.tokens` arrays are larger
than the SmolLM2/TinyLlama equivalents (Qwen has a 152K-entry
vocabulary vs. SmolLM2's 49K), giving more per-element reads to
amortise.

**Disposition:** **Shipped.** All 28 GGUF parser unit tests + the
real-fixture substrate-equivalence test (17/17) still pass — every
field of `GgufInspectInfo` is identical pre- and post-Tier-1 because
the bytes read are identical, only the syscall granularity changed.
The `# Performance` rustdoc on `inspect_gguf_from_reader` was updated
to reflect the new numbers and to remove the now-stale "use mmap for
local files" guidance.

**Trade-offs accepted:**

- **+~64 KiB heap per call** for the BufReader's internal buffer.
  Negligible vs. the parsed metadata `HashMap` (often hundreds of KiB
  to a few MiB for the tokenizer arrays).
- **Caller can no longer pass a non-buffered `Read + Seek` and rely
  on its own buffering decisions** — but the type signature is
  unchanged (`R: Read + Seek` in, `Result<GgufInspectInfo>` out), so
  this is a strictly internal optimisation. Callers that want to
  control buffering can pass any `Read + Seek`; the internal
  `BufReader` will wrap it (mostly redundantly for an in-memory
  `Cursor`, but the per-call memcpy cost is dwarfed by the parsing
  work).

**Tier 2 not pursued:** The original analysis identified a "bulk-read
typed arrays in `read_typed_array`" optimisation (collapse the
per-element `Vec::push` loop into one `read_into` + `chunks_exact`
convert) as a Tier 2 follow-up. With Tier 1 closing the gap to mmap
parity, Tier 2's added complexity (security guard for "fail-before-
allocate" on adversarial array-length headers) is no longer
justified. The geometric-growth `Vec::push` pattern stays.

**Re-attempting this requires:** N/A — this is the success case.
[`tests/bench_gguf_inspect_adhoc.rs`](../tests/bench_gguf_inspect_adhoc.rs)
is in place to detect any regression. If a future change to
`GgufReader` reintroduces per-element reads on top of `BufReader`
(e.g., dropping `read_into` for some other pattern), the bench will
catch it.

## Experiment 6 — `inspect_pth_from_reader` reader vs mmap vs `torch.load`

**Question:** Does Phase 4.10's reader-generic
[`inspect_pth_from_reader`](../src/parse/pth.rs) match the mmap-backed
`parse_pth(path).inspect()` in throughput, and how does each compare to the
closest Python equivalent (`torch.load(weights_only=True)` + iterate the
returned `state_dict` to compute the same summary fields)? Phase 4.10's
PTH parser does not adopt the GGUF Tier-1 `BufReader` win because the I/O
pattern is structurally different (bulk reads through `zip`, not many small
`read_exact` calls) — does the measurement confirm that the parity claim
holds without buffering?

**Method:**
[`tests/bench_pth_inspect_adhoc.rs`](../tests/bench_pth_inspect_adhoc.rs)
(`bench_pth_inspect_paths`, `#[ignore]`-gated) plus the Python script at
[`tests/fixtures/pth_reference/bench_python_inspect.py`](../tests/fixtures/pth_reference/bench_python_inspect.py).
Best-of-5 release-mode median per file, `target-cpu=native`, warmed FS
cache, one warm-up iteration before timing. Three AlgZoo fixtures (the
only `.pth` fixtures checked into the repo): `algzoo_rnn_small.pth`
(2.0 KiB), `algzoo_transformer_small.pth` (3.5 KiB), `algzoo_rnn_blog.pth`
(3.3 KiB). PyTorch 2.10.0+cu130 on the Python side; same machine, same
files.

**Result — Rust mmap vs Rust reader (this commit):**

| Fixture | mmap median | reader median | reader / mmap |
|---|---:|---:|---:|
| `algzoo_rnn_small.pth` (2.0 KiB) | 134.4 µs | 220.1 µs | 1.64× |
| `algzoo_transformer_small.pth` (3.5 KiB) | 154.0 µs | 236.1 µs | 1.53× |
| `algzoo_rnn_blog.pth` (3.3 KiB) | 133.1 µs | 151.5 µs | 1.14× |
| **median across fixtures** | — | — | **1.53×** |

**Result — Rust paths vs Python `torch.load`:**

| Fixture | torch.load median | mmap speedup | reader speedup |
|---|---:|---:|---:|
| `algzoo_rnn_small.pth` (2.0 KiB) | 532.7 µs | 4.0× | 2.4× |
| `algzoo_transformer_small.pth` (3.5 KiB) | 858.7 µs | 5.6× | 3.6× |
| `algzoo_rnn_blog.pth` (3.3 KiB) | 530.6 µs | 4.0× | 3.5× |
| **median across fixtures** | — | **4.0×** | **3.5×** |

**Why the reader path is slower than mmap on these fixtures:** the AlgZoo
files are 2–4 KiB — at this scale, *fixed* costs dominate over per-byte
work. The reader path pays for:

1. `seek(End(0))` to capture total length (one syscall).
2. The 4-byte magic probe + rewind (one read, one seek — kept so the
   *"legacy pre-1.6 raw pickle"* diagnostic remains distinct from the
   generic *"not a valid ZIP"* diagnostic).
3. `zip::ZipArchive::new` doing its own EOCD scan (several reads near the
   end of the file, with `zip`'s internal buffering).
4. One `Vec::with_capacity(pkl_size)` + `read_to_end` for `data.pkl`.

The mmap path skips (1) and (3)'s syscall costs because the file is already
in the page cache; it pays one minor page fault for the 4 KiB containing
the EOCD and central directory. On a 2 KiB fixture the entire archive fits
in one page, so the mmap path is essentially free past the initial mmap
syscall.

On larger files the relative overhead collapses. Linear extrapolation from
the Phase 4.9 GGUF benchmark (where reader and mmap reached parity ~3 ms on
multi-GB models) plus the per-byte work being dominated by the pickle VM
(which is identical across substrates and takes O(`pkl_size`) ≈ O(tens of
microseconds for tens of KiB)) gives reader/mmap parity at a few hundred
KiB of `data.pkl` — which is the realistic range for torchvision-class
models (~50 KiB of `data.pkl` on ResNet-50 / ViT-B/16).

**Why both Rust paths are faster than `torch.load`:** PyTorch has no
separate inspect-only primitive — `torch.load(weights_only=True)`
materialises every tensor as a `torch.Tensor` on CPU before the caller
can iterate the `state_dict` for summary stats. Even on a 2 KiB fixture
that involves tens of `torch.Tensor` constructions plus the surrounding
Python overhead. The Rust paths skip *all* tensor materialisation — only
the pickle metadata is interpreted.

The 3.5× median speedup on tiny fixtures is a **lower bound** for the
reader path: scaling to a torchvision-class 300 MB `.pth`, `torch.load`'s
time grows linearly with the total `data/N` size while
`inspect_pth_from_reader`'s time stays bounded by `data.pkl` size (tens
of KiB). On 300 MB models the reader path would beat `torch.load` by
multiple orders of magnitude, mirroring the 11.2–30.8× full-parse
speedups already documented in the project README for the mmap path.

**Disposition:** **Shipped.** The reader-generic path lands without an
internal `BufReader` because:

- The parity gap on KiB fixtures (1.14–1.64× of mmap) is small in absolute
  terms (~50–90 µs) and would not benefit meaningfully from buffering —
  the `zip` crate already does its own buffering on the central-directory
  scan, and our two payload reads are bulk `read_to_end` calls.
- Adding `BufReader<R>` would introduce one extra memcpy per buffer-fill
  on every substrate (including `Cursor`), with no syscall reduction (the
  fixed-cost path is dominated by `seek + EOCD scan + central-directory
  parse`, not per-element reads).
- The 3.5× absolute speedup vs `torch.load` is already comfortable; the
  remaining ~80 µs gap to the mmap path is a fixed cost of the ZIP-archive
  abstraction, not a syscall pattern that buffering would amortise.

The Phase 4.9 GGUF rationale for adding `BufReader<R>` (collapsing many
4–8 B `read_exact` calls on a `File` substrate into one syscall per buffer
fill) does not apply here.

**Re-attempting this requires:** evidence that on a torchvision-class
real `.pth` (≥45 MB, ≥100 tensors, ≥50 KiB `data.pkl`) the reader path is
more than ~1.5× slower than the mmap path — at which point the parity
claim in the rustdoc would need to be revised and `BufReader<R>`
reconsidered. The current bench harness
([`bench_pth_inspect_paths`](../tests/bench_pth_inspect_adhoc.rs)) accepts
arbitrary additional fixtures dropped into
`tests/fixtures/pth_reference/`.

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
