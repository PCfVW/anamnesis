# anamnesis Coding Conventions (Grit + Grit-AMN Extensions)

This document describes the [Amphigraphic coding](https://github.com/PCfVW/Amphigraphic-Strict) conventions used in anamnesis. It is a superset of
the [Grit — Strict Rust for AI-Assisted Development](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit).

## Annotation Patterns

Every annotation below is mandatory when the corresponding situation applies.

### `// TRAIT_OBJECT: <reason>`
Required on every `Box<dyn Trait>` or `&dyn Trait` usage.
> Example: `// TRAIT_OBJECT: heterogeneous format parsers require dynamic dispatch`

### `// EXHAUSTIVE: <reason>`
Required on `#[allow(clippy::exhaustive_enums)]` or
`#[allow(clippy::wildcard_enum_match_arm)]` when a wildcard is used on a
foreign `#[non_exhaustive]` enum that we cannot match exhaustively.
> Example: `// EXHAUSTIVE: internal dispatch enum; crate owns and matches all variants`
> Example: `// EXHAUSTIVE: SafeTensorError is a foreign type that may gain variants`

### `// EXPLICIT: <reason>`
Required when a match arm is intentionally a no-op, or when an imperative
loop is used instead of an iterator chain for a stateful computation.
> Example: `// EXPLICIT: FP8 block accumulation is stateful; .map() would hide the scale update`

### `// BORROW: <what is converted>`
Required on explicit `.as_str()`, `.as_bytes()`, `.to_owned()` conversions (Grit Rule 2).
> Example: `// BORROW: explicit .as_str() instead of Deref coercion`

### `// SAFETY: <invariants>`
Required on every `unsafe` block or function (inline comment, not a doc comment).

anamnesis is `#![forbid(unsafe_code)]` **by default**. The one anticipated
exception is explicit SIMD intrinsics (e.g., `#[target_feature(enable = "avx2")]`),
which require `unsafe` by language rules even though the operations are safe
in practice. If `unsafe` becomes necessary, follow the candle-mi pattern:

| Feature | Accepted `unsafe` scope |
|---------|------------------------|
| `simd`  | `#[target_feature]` functions for explicit SIMD in conversion loops |

Each accepted use must satisfy all of:
1. The `unsafe` block is in a **single, dedicated module** (e.g., `src/remember/simd.rs`)
   — never scattered across the codebase.
2. Every `unsafe` block carries a `// SAFETY:` comment documenting the invariants.
3. The module is gated behind `#[cfg(feature = "...")]` — users who don't
   enable the feature get `forbid(unsafe_code)` with zero exceptions.
4. A safe scalar fallback exists and is tested identically.

### `// INDEX: <reason>`
Required on every direct slice index (`slice[i]`, `slice[a..b]`) that cannot
be replaced by an iterator. Direct indexing panics on out-of-bounds; prefer
`.get(i)` with `?` or explicit error handling. Use direct indexing only when
the bound is provably valid and an iterator idiom would be significantly less
readable. See also "Reconciling bounds checking with vectorization" below for
how this rule interacts with the SIMD rules in hot loops.
> Example: `// INDEX: offset is bounded by header.data_offsets checked above`
> Example: `// INDEX: chunks_exact(4) guarantees exactly 4 bytes per chunk`

### `// CAST: <from> → <to>, <reason>`
Required on every `as` cast between numeric types. Prefer `From`/`Into` for
lossless conversions and `TryFrom`/`TryInto` with `?` for fallible ones.
Use `as` only when truncation or wrapping is the deliberate intent, or when
the bit-level reinterpretation is the whole point (as in dequantization).
> Example: `// CAST: u8 → f32, FP8 E4M3 mantissa bits promoted for arithmetic`
> Example: `// CAST: usize → u64, byte offset for safetensors; file size fits in u64`

### `// BITWISE: <operation>`
Required on every raw bit manipulation that implements part of a quantization
or dequantization algorithm. Document what the bits represent and what the
operation achieves — the mapping between storage format and numerical value
must be traceable through the annotations.
> Example: `// BITWISE: extract 3-bit mantissa from E4M3 byte (bits [2:0])`
> Example: `// BITWISE: combine sign, exponent, mantissa into IEEE 754 half-precision`
> Example: `// BITWISE: extract unsigned 4-bit value from packed I32 at bit position shift`

### `// VECTORIZED: <verification>`
Required on every bulk conversion loop that is expected to auto-vectorize.
Document how vectorization was verified and on which target.
> Example: `// VECTORIZED: confirmed AVX2 vmulps + vpermb in cargo-show-asm, x86-64, opt-level=3`
> Example: `// VECTORIZED: confirmed AVX2 vsubps+vmulps (target-cpu=native) via --emit=asm`
> Example: `// VECTORIZED: scalar fallback — loop body contains branch that defeats auto-vectorization`

---

## SIMD-Friendly Loop Rules

anamnesis processes billions of elements (one per model parameter). Performance
depends on the compiler auto-vectorizing the inner conversion loops. The
following rules ensure that hot loops remain vectorizable.

### Write loops that the compiler can vectorize

The compiler auto-vectorizes when it can prove: no aliasing, no branches,
no cross-iteration dependencies, fixed stride, and known trip count. Write
loops that satisfy these conditions:

1. **Process contiguous slices, not iterators over structs.**
   Work on flat slices (e.g., `&[u8]` for packed data, `&[f32]` for
   precomputed values, `&mut [u8]` for `BF16` output), not abstracted types.

   > ✅ `for i in 0..block.len() { out[i] = convert(block[i], scale); }`
   > ✅ `block.iter().zip(out.iter_mut()).for_each(|(&b, o)| *o = convert(b, scale));`
   > ❌ `for tensor in tensors { for elem in tensor.elements() { ... } }` ← indirect, defeats vectorization

2. **No branches in the hot path.** Conditional logic (NaN handling, subnormal
   detection) must be expressed as bitwise select or arithmetic, not `if`/`match`.
   This includes bounds checks — see "Reconciling bounds checking with
   vectorization" below.

   > ✅ `let is_nan_mask = ((exp == 0xF) & (mant != 0)) as u16;` then blend with bitwise ops
   > ❌ `if exp == 0xF && mant != 0 { return BF16_NAN; }` ← branch per element kills SIMD

3. **Hoist invariants out of the loop.** The scale factor is constant for an
   entire block (128 elements in fine-grained FP8) or an entire row (GPTQ
   group). Compute it once before the loop, not inside.

   > ✅ `let scale = load_scale(block_idx);  for i in 0..128 { ... scale ... }`
   > ❌ `for i in 0..128 { let scale = load_scale(block_idx); ... }` ← redundant load, may confuse optimizer

4. **Use exact chunk sizes.** `chunks_exact()` tells the compiler the trip
   count is a multiple of the chunk size, enabling full-width SIMD with no
   remainder loop.

   > ✅ `for chunk in data.chunks_exact(128) { ... }`
   > ❌ `for chunk in data.chunks(128) { ... }` ← last chunk may be short, generates scalar remainder

5. **Avoid `as` casts that widen through intermediate types.** A chain like
   `u8 → u32 → f32 → f32 (multiply) → u16` may force the compiler to emit
   widening/narrowing shuffles that break vectorization. Prefer bitwise
   manipulation that stays in the target width. When the pipeline inevitably
   crosses domains (e.g., GPTQ: byte extraction → float arithmetic → integer
   `BF16` rounding), use loop fission — see below.

6. **Separate input and output slices.** The compiler cannot vectorize if it
   suspects the output aliases the input. Use distinct `&[u8]` input and
   `&mut [u8]` output buffers — never in-place transformation on a single
   `&mut [u8]`.

### Reconciling bounds checking with vectorization

The `// INDEX:` rule (above) says to prefer `.get()` with `?` for bounds
checking. In hot loops, `.get()` generates a branch on every element — the
exact pattern that rule 2 ("no branches in the hot path") forbids.

Satisfy both rules with a **two-level pattern**:

1. **Before the loop:** validate bounds **once** using `.get(range)` with `?`
   on the enclosing slice. This produces a pre-validated sub-slice whose
   length is known at the start of the loop.

2. **Inside the loop:** iterate the pre-validated slice with `chunks_exact`,
   `.iter().zip()`, or direct indexing annotated with `// INDEX:` referencing
   the pre-validation. No `.get()`, no `?`, no branches.

> ✅ Pre-validate, then iterate branch-free:
> ```rust
> let row = data.get(start..start + cols).ok_or_else(|| err())?;  // bounds checked once
> let out = output.get_mut(o_start..o_start + cols * 2).ok_or_else(|| err())?;
> // VECTORIZED: inner loop is branch-free, pre-validated slices
> for (&byte, out_pair) in row.iter().zip(out.chunks_exact_mut(2)) {
>     out_pair.copy_from_slice(&convert(byte, scale).to_le_bytes());
> }
> ```
>
> ❌ Per-element `.get()` inside the loop — defeats vectorization:
> ```rust
> for j in 0..cols {
>     let byte = data.get(start + j).ok_or_else(|| err())?;   // branch per element
>     let scale = scales.get(j).ok_or_else(|| err())?;         // branch per element
>     // compiler cannot vectorize — too many conditional paths
> }
> ```

This pattern appears throughout anamnesis: the FP8 dequantization loops
pre-slice weight and output rows before the inner `chunks_exact` iteration.
Every new dequantization module (GPTQ, AWQ, BnB) must follow the same
two-level structure.

### Loop fission for mixed-domain pipelines

When a single loop mixes data domains (byte extraction, float arithmetic,
integer bit-manipulation for `BF16` rounding), the compiler often cannot
vectorize because it sees cross-domain dependencies in the loop body.

**Split the loop into separate passes**, one per domain. Each pass has a
uniform data flow that the compiler can vectorize independently. Use a
scratch buffer (typically one row, fits in L1 cache) to communicate between
passes.

> ✅ Two passes — byte extraction then float arithmetic:
> ```rust
> // Pass 1: unpack bytes → f32 scratch buffer (partially vectorizes)
> for (chunk, dst) in raw.chunks_exact(4).zip(scratch.iter_mut()) {
>     let packed = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
>     *dst = ((packed >> shift) & mask) as f32;
> }
> // Pass 2: pure f32 arithmetic → BF16 output (fully vectorizes to AVX2)
> for ((&val, &zero, &scale), out) in scratch.iter()
>     .zip(zeros.iter()).zip(scales.iter()).zip(output.chunks_exact_mut(2)) {
>     let bf16 = f32_bits_to_bf16_bits(((val - zero) * scale).to_bits());
>     out.copy_from_slice(&bf16.to_le_bytes());
> }
> ```
>
> ❌ Single combined loop — mixes byte, float, and integer domains:
> ```rust
> for (chunk, out) in raw.chunks_exact(4).zip(output.chunks_exact_mut(2)) {
>     let packed = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
>     let qw = ((packed >> shift) & mask) as f32;       // byte → int → float
>     let val = (qw - zero) * scale;                     // float
>     let bf16 = f32_bits_to_bf16_bits(val.to_bits());   // float → int → truncate
>     out.copy_from_slice(&bf16.to_le_bytes());           // compiler gives up
> }
> ```

The cost of the extra pass is negligible: the scratch buffer is one row
(typically 1–32 KB, fits in L1 cache). The benefit is full-width SIMD on
the arithmetic pass. In GPTQ dequantization, loop fission turned scalar
`vsubss`/`vmulss` into AVX2 `vsubps`/`vmulps` (8-wide), improving the
function from 1.5–2.8× to **6.5–12.2× faster** than CPU PyTorch.

**When to expect the need for fission:** if the input format packs multiple
logical values per storage word (GPTQ: 8 INT4 per `I32`, AWQ: similar), the
unpacking step will mix byte/integer domains with the float arithmetic.
Plan two passes from the start. Conversely, if the input-to-output mapping
is 1:1 with a small branchless inline kernel (FP8: one byte → one `BF16`),
the compiler typically fuses the pipeline without fission.

### Verify vectorization

After writing a hot loop, **verify** that the compiler actually vectorized it.
Do not assume — auto-vectorization is fragile and can silently regress.

- Use `cargo-show-asm` or `RUSTFLAGS="-C target-cpu=native --emit=asm" cargo
  build --release` to inspect the generated assembly.
- Look for packed SIMD instructions: `vmulps`, `vsubps`, `vpsrld`, `vpaddd`,
  `vpackusdw` (AVX2) or `fmul.4s`, `ushll` (NEON). Scalar variants
  (`vmulss`, `vsubss`) indicate the loop did not vectorize.
- If the loop did not vectorize, add a `// VECTORIZED: scalar fallback — <reason>`
  annotation explaining why and what would need to change.
- If it did vectorize, add `// VECTORIZED: confirmed <ISA> <key instruction>
  in cargo-show-asm, <target>, opt-level=3`.
- Re-verify after any change to the loop body or its dependencies.

### When auto-vectorization is not enough

If a verified scalar loop is too slow and the compiler refuses to vectorize
despite following the rules above, escalate in this order:

1. **`#[target_feature(enable = "avx2")]` + `is_x86_feature_detected!`** —
   explicit opt-in to wider SIMD on x86-64 while keeping a portable fallback.
   Requires `unsafe` (see `// SAFETY:` rules) and a feature gate.
2. **`pulp` or `portable-simd`** — stable-Rust portable SIMD abstractions.
   Adds a dependency but avoids hand-written intrinsics.
3. **Hand-written intrinsics with `#[cfg(target_arch)]`** — last resort.
   Must include a scalar fallback for non-x86/non-ARM targets.

---

## Doc-Comment Rules

### Backtick Hygiene (`doc_markdown`)

All identifiers, types, trait names, field names, crate names, and
file-format names in doc comments must be wrapped in backticks so that
rustdoc renders them as inline code and Clippy's `doc_markdown` lint passes.

Applies to: struct/enum/field names, method names (`fn foo`), types
(`Vec<T>`, `Option<f32>`), crate names (`safetensors`, `half`),
file extensions (`.npy`, `.npz`, `.safetensors`), and acronyms that double
as types (`DType`, `NaN`, `FP8`, `E4M3`, `BF16`, `GPTQ`).

> ✅ `` /// Parses the header of a `.safetensors` file. ``
> ❌ `/// Parses the header of a .safetensors file.`

### Intra-Doc Link Safety

Rustdoc intra-doc links must resolve under all feature-flag combinations
(enforced by `#![deny(warnings)]` → `rustdoc::broken_intra_doc_links`).

Feature-gated items (e.g., NPZ types behind `npz` feature, GPTQ types
behind `gptq` feature) must use plain backtick text, not link syntax:

> ✅ `` /// See `NpzTensor` (requires `npz` feature). ``
> ❌ `` /// See [`NpzTensor`](crate::parse::npz::NpzTensor). ``

### Field-Level Docs

Every field of every `pub` struct must carry a `///` doc comment describing:
1. what the field represents,
2. its unit or valid range where applicable.

> Example:
> ```rust
> pub struct NpzTensor {
>     /// Tensor dimensions (e.g., `[2304, 2048]`).
>     pub shape: Vec<usize>,
>     /// Element data type (e.g., `F32`, `F64`, `BF16`).
>     pub dtype: NpzDtype,
>     /// Raw bytes in row-major order. Length = product(shape) * dtype.byte_size().
>     pub data: Vec<u8>,
> }
> ```

---

## Control-Flow Rules

### `if let` vs `match` (`match_like_matches_macro`, `single_match`)

Use the most specific construct for the pattern at hand:

| Situation | Preferred form |
|---|---|
| Testing a single variant, no binding needed | `matches!(expr, Pat)` |
| Testing a single variant, binding needed | `if let Pat(x) = expr { … }` |
| Two or more variants with different bodies | `match expr { … }` |
| Exhaustive dispatch over an enum | `match expr { … }` (never `if let` chains) |

Never use a `match` with a single non-`_` arm and a no-op `_ => {}` where
`if let` or `matches!` would be clearer. Conversely, never chain three or
more `if let … else if let …` arms where a `match` would be exhaustive.

---

## Function Signature Rules

### `const fn`

Declare a function `const fn` when **all** of the following hold:
1. The body contains no heap allocation, I/O, or `dyn` dispatch.
2. All called functions are themselves `const fn`.
3. There are no trait-method calls that are not yet `const`.

This applies to constructors, accessors, and pure arithmetic helpers.
When in doubt, annotate and let the compiler reject it — do not omit `const`
preemptively.

> ✅ `pub const fn byte_size(&self) -> usize { ... }`
> ❌ `pub fn byte_size(&self) -> usize { ... }`

### Pass by Value vs Reference (`needless_pass_by_ref_mut`, `trivially_copy_pass_by_ref`)

| Type | Rule |
|---|---|
| `Copy` type ≤ 2 words (`usize`, `f32`, `bool`, small `enum`) | Pass by value |
| `Copy` type > 2 words | Pass by reference |
| Non-`Copy`, not mutated | Pass by `&T` or `&[T]` |
| Non-`Copy`, mutated | Pass by `&mut T` |
| Owned, consumed by callee | Pass by value (move semantics) |
| `&mut T` not actually mutated in body | Change to `&T` |

---

## `#[non_exhaustive]` Policy

- Public enums that may gain new variants: `#[non_exhaustive]`.
  `Dtype`, `QuantScheme`, `TargetDtype`, and `TensorRole` are all
  non-exhaustive — new formats and dtypes will be added over time.
- Internal dispatch enums matched exhaustively by this crate:
  `#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: <reason>`.

---

## `#[must_use]` Policy

All public functions and methods that return a value and have no side effects
must be annotated `#[must_use]`. This includes constructors, accessors,
and pure queries (`byte_size`, `is_quantized`, `tensor_count`).

The `clippy::must_use_candidate` lint enforces this at `warn` level
(promoted to error by `#![deny(warnings)]`).

---

## `# Errors` Doc Section

All public fallible methods (`-> Result<T>`) must include an `# Errors` section.
Each bullet uses the format:

    /// # Errors
    /// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
    /// Returns [`AnamnesisError::Io`] if the file cannot be read.

Rules:
- Start each bullet with `Returns` followed by the variant in rustdoc link
  syntax, e.g., `` [`AnamnesisError::Parse`] ``.
- Follow with `if` (condition), `on` (event), or `when` (circumstance).
- One bullet per distinct error path.

---

## Error Message Wording

Error strings passed to `AnamnesisError` variants follow two patterns:

- **External failures** (I/O, serde): `"failed to <verb>: {e}"`
  > Example: `AnamnesisError::Parse { reason: format!("failed to parse safetensors header: {e}") }`
- **Validation failures** (format, range): `"<noun> <problem> (<context>)"`
  > Example: `AnamnesisError::Parse { reason: format!("unexpected dtype {dtype} for tensor {name}") }`
  > Example: `AnamnesisError::Unsupported { format: "GPTQ".into(), detail: "3-bit quantization not yet supported".into() }`

Rules:
- Use lowercase, no trailing period.
- Include the offending value and the valid range or constraint when applicable.
- Wrap external errors with `: {e}`, not `.to_string()`.

---

## `# Memory` Doc Section

Public methods that process large files (safetensors, NPZ archives) must include
a `# Memory` section documenting:

1. **Peak allocation** — how much memory the method allocates at its peak.
2. **Lifetime** — whether allocations are dropped before the method returns
   or persist in the returned value.

Format:

    /// # Memory
    /// Reads the entire safetensors file into memory (~2× model size during
    /// dequantization: source FP8 + destination BF16). The source buffer is
    /// dropped before returning.

---

## HashMap Grouping Idiom

When operations must be batched by a key (e.g., grouping tensors by
quantization scheme), use the `Entry` API:

```rust
let mut by_scheme: HashMap<QuantScheme, Vec<TensorInfo>> = HashMap::new();
for tensor in tensors {
    by_scheme.entry(tensor.scheme()).or_default().push(tensor);
}
```

Rules:
- Name the map `by_<grouping_key>` (e.g., `by_scheme`, `by_dtype`).
- Use `.entry(key).or_default().push()` — never `if let Some` + `else insert`.
