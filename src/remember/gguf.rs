// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GGUF` K-quant dequantization — scalar reference kernels producing `BF16`.
//!
//! This module consumes raw block-encoded tensor bytes (as produced by
//! [`parse_gguf`](crate::parse::gguf::parse_gguf)) and decodes them into
//! `BF16` output bytes, matching the output format of the `FP8`, `GPTQ`,
//! `AWQ`, and `BitsAndBytes` dequantisers.
//!
//! # Supported types
//!
//! - **Legacy block quants** (32-element blocks): `Q4_0`, `Q4_1`, `Q5_0`,
//!   `Q5_1`, `Q8_0`, `Q8_1`.
//! - **K-quants** (256-element super-blocks): `Q2_K`, `Q3_K`, `Q4_K`,
//!   `Q5_K`, `Q6_K`, `Q8_K`.
//!
//! `IQ*`, `TQ*`, and `MXFP4` are recognised by the parser but **not yet
//! dequantised**; the dispatcher returns
//! [`AnamnesisError::Unsupported`] for those types.
//!
//! # Two public entry points
//!
//! - [`dequantize_gguf_to_bf16`] materialises the whole tensor into an
//!   owned `Vec<u8>`. Convenient for small-to-medium tensors (anything
//!   that comfortably fits in RAM alongside its `BF16` expansion).
//! - [`dequantize_gguf_blocks_to_bf16`] is the **streaming** variant: the
//!   caller supplies a sink closure that receives one block's worth of
//!   `BF16` bytes at a time (64 B for legacy, 512 B for K-quants), so
//!   peak heap is O(one scratch buffer) regardless of tensor size. Use
//!   this for dequantising 70 B-parameter models on modest RAM.
//!
//! Both entry points share the same validation logic and the same scalar
//! kernels; the `Vec`-returning variant is a thin wrapper that pushes
//! into a pre-sized `Vec::with_capacity` sink.
//!
//! # Algorithm
//!
//! Every kernel follows the two-pass loop-fission pattern from
//! `CONVENTIONS.md`: for each block,
//!
//! 1. **Pass 1** unpacks the packed-bit storage into a stack-resident
//!    `[f32; QK]` scratch buffer (one block at a time, L1-resident).
//! 2. **Pass 2** walks the scratch buffer paired with a per-block
//!    `[u8; QK × 2]` output buffer and emits `BF16` bytes via the shared
//!    [`f32_bits_to_bf16_bits`](crate::remember::fp8) helper. The inner
//!    loop is branch-free and `chunks_exact_mut(2)`-based, giving the
//!    compiler a clean vectorisation target.
//!
//! The formulas are ported verbatim from `ggml-quants.c`'s scalar
//! `dequantize_row_*` reference implementations. Bit-for-bit
//! cross-validation against `llama.cpp` output is Phase 4 step 4.
//!
//! # Output format
//!
//! All kernels emit `BF16` bytes in little-endian order, length
//! `n_elements × 2`. Round-to-nearest-even is performed inside
//! `f32_bits_to_bf16_bits`.

use crate::error::AnamnesisError;
use crate::parse::gguf::GgufType;
use crate::remember::fp8::f32_bits_to_bf16_bits;

// ---------------------------------------------------------------------------
// Block-size constants
// ---------------------------------------------------------------------------

/// Element count per legacy block quant (`Q4_0`..`Q8_1`).
const QK_SMALL: usize = 32;

/// Element count per K-quant super-block (`Q2_K`..`Q8_K`).
const QK_K: usize = 256;

// ---------------------------------------------------------------------------
// Infallible byte readers
// ---------------------------------------------------------------------------

/// Reads a `ggml_half` (IEEE 754 binary16) from a 2-byte array and widens
/// it to `f32`. Infallible by construction — callers slice fixed-length
/// arrays out of their already-validated block slices.
#[inline]
fn read_f16_bytes(bytes: [u8; 2]) -> f32 {
    f32::from(half::f16::from_le_bytes(bytes))
}

/// Reads a little-endian `f32` from a 4-byte array. Used only by the
/// `Q8_K` block, whose super-block scale is `f32` rather than `f16`.
#[inline]
fn read_f32_bytes(bytes: [u8; 4]) -> f32 {
    f32::from_le_bytes(bytes)
}

/// Converts a block's worth of `f32` scratch values to `BF16` bytes.
///
/// This is the hot-path pass 2 for every kernel. Branch-free, contiguous
/// reads, contiguous writes — exactly the shape the compiler needs to
/// auto-vectorise the inner `f32 → BF16` conversion. A future commit may
/// add an AVX2-gated intrinsic variant behind a `simd` feature flag; the
/// scalar version here is the canonical fallback.
///
/// # Preconditions
///
/// `out_block.len() == 2 × scratch.len()`. `chunks_exact_mut(2)` silently
/// truncates any remainder, so a caller that passes the wrong length
/// produces short output rather than a panic — every kernel sizes both
/// buffers at compile time so this never happens in practice.
#[inline]
fn write_scratch_to_bf16(scratch: &[f32], out_block: &mut [u8]) {
    // VECTORIZED: pending cargo-show-asm verification on the scalar path.
    // TODO(phase4-followup): AVX2 `f32x8 → bf16x8` gated behind a `simd`
    // feature per CONVENTIONS.md's accepted-unsafe table — would give a
    // ~4-8× speedup on pass 2, which is ~40-60% of total dequant time.
    for (&val, out_pair) in scratch.iter().zip(out_block.chunks_exact_mut(2)) {
        let bf16 = f32_bits_to_bf16_bits(val.to_bits());
        out_pair.copy_from_slice(&bf16.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Shared validation
// ---------------------------------------------------------------------------

/// Validates dispatcher inputs and returns the output byte length
/// (`n_elements × 2`), with an overflow guard that catches 32-bit targets
/// where the `BF16` output would exceed `usize::MAX`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if
/// - `n_elements` is not a multiple of `dtype.block_size()`,
/// - `n_elements × 2` overflows `usize` (32-bit targets with > 2 GiB
///   output), or
/// - `data.len()` does not equal `n_blocks × dtype.type_size()`.
///
/// Returns [`AnamnesisError::Unsupported`] if `dtype.type_size()` is
/// `None` (`IQ*`/`TQ*`/`MXFP4` — deferred to a later phase).
fn validate_dequant_input(data: &[u8], dtype: GgufType, n_elements: usize) -> crate::Result<usize> {
    let block_size = dtype.block_size();
    if !n_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF dequant: n_elements {n_elements} is not a multiple of block size \
                 {block_size} for {dtype}"
            ),
        });
    }
    // Output byte length: `n_elements × 2`. Guarded against overflow so a
    // 32-bit target with > 2 GiB of BF16 output produces a clean `Parse`
    // error rather than silently truncating the allocation.
    let out_byte_len = n_elements
        .checked_mul(2)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("GGUF dequant: output byte length {n_elements}×2 overflows usize"),
        })?;
    let n_blocks = n_elements / block_size;
    let type_size = dtype
        .type_size()
        .ok_or_else(|| AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("dequantisation not yet supported for {dtype}"),
        })?;
    let expected_bytes = n_blocks
        .checked_mul(type_size)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "GGUF dequant: expected byte count overflows usize".into(),
        })?;
    if data.len() != expected_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF dequant: data length {} != expected {expected_bytes} for \
                 {n_blocks} blocks of {dtype}",
                data.len()
            ),
        });
    }
    Ok(out_byte_len)
}

// ---------------------------------------------------------------------------
// Kernel runners (generic over pass-1 unpack closure + sink closure)
// ---------------------------------------------------------------------------

/// Outer-loop runner for 32-element legacy quant blocks.
///
/// Walks `data` in `type_size` chunks via `chunks_exact`, invokes the
/// caller-supplied `unpack` closure to fill the stack `[f32; 32]`
/// scratch, then writes `BF16` bytes into a stack `[u8; 64]` buffer and
/// forwards it to `sink`.
///
/// Generic + `#[inline]` so the per-type kernel's closure body fully
/// inlines into the outer loop with no indirect-call overhead.
#[inline]
fn run_legacy_kernel<F, P>(
    data: &[u8],
    type_size: usize,
    mut sink: F,
    mut unpack: P,
) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
    P: FnMut(&[u8], &mut [f32; QK_SMALL]),
{
    let mut scratch = [0.0_f32; QK_SMALL];
    let mut block_out = [0u8; QK_SMALL * 2];
    for in_block in data.chunks_exact(type_size) {
        unpack(in_block, &mut scratch);
        write_scratch_to_bf16(&scratch, &mut block_out);
        sink(&block_out)?;
    }
    Ok(())
}

/// Outer-loop runner for 256-element K-quant super-blocks.
///
/// Structurally identical to [`run_legacy_kernel`] but with a 1 KB
/// scratch buffer and a 512 B output buffer. Both still fit comfortably
/// in L1 cache.
#[inline]
fn run_super_kernel<F, P>(
    data: &[u8],
    type_size: usize,
    mut sink: F,
    mut unpack: P,
) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
    P: FnMut(&[u8], &mut [f32; QK_K]),
{
    let mut scratch = [0.0_f32; QK_K];
    let mut block_out = [0u8; QK_K * 2];
    for in_block in data.chunks_exact(type_size) {
        unpack(in_block, &mut scratch);
        write_scratch_to_bf16(&scratch, &mut block_out);
        sink(&block_out)?;
    }
    Ok(())
}

/// Routes a validated `(data, dtype)` pair to the correct per-type
/// streaming kernel. Called by both public entry points.
fn dispatch_streaming<F>(data: &[u8], dtype: GgufType, sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    // EXHAUSTIVE: internal dispatch over GgufType. IQ*/TQ*/MXFP4 have no
    // implemented kernel yet; scalar (non-block) types are structurally
    // dequantised (reinterpret bytes), not in scope here.
    #[allow(clippy::wildcard_enum_match_arm)]
    match dtype {
        GgufType::Q4_0 => dequant_q4_0(data, sink),
        GgufType::Q4_1 => dequant_q4_1(data, sink),
        GgufType::Q5_0 => dequant_q5_0(data, sink),
        GgufType::Q5_1 => dequant_q5_1(data, sink),
        GgufType::Q8_0 => dequant_q8_0(data, sink),
        GgufType::Q8_1 => dequant_q8_1(data, sink),
        GgufType::Q2_K => dequant_q2_k(data, sink),
        GgufType::Q3_K => dequant_q3_k(data, sink),
        GgufType::Q4_K => dequant_q4_k(data, sink),
        GgufType::Q5_K => dequant_q5_k(data, sink),
        GgufType::Q6_K => dequant_q6_k(data, sink),
        GgufType::Q8_K => dequant_q8_k(data, sink),
        _ => Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("dequantisation not yet supported for {dtype}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Dequantises a `GGUF`-encoded block-quantised tensor into an owned
/// `Vec<u8>` of `BF16` bytes.
///
/// Convenience wrapper around [`dequantize_gguf_blocks_to_bf16`] that
/// pushes each block's output into a pre-allocated `Vec::with_capacity`.
/// For very large tensors, prefer the streaming variant — this variant's
/// peak heap is O(`n_elements × 2`).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `n_elements` is not a multiple of
/// the block size for `dtype`, if `n_elements × 2` overflows `usize`
/// (32-bit targets only), or if `data.len()` does not equal the expected
/// byte count (`n_blocks × type_size`).
///
/// Returns [`AnamnesisError::Unsupported`] if `dtype` is one of the
/// recognised-but-not-yet-implemented types (`IQ*`, `TQ*`, `MXFP4`) or a
/// scalar type that is not a quantised block format.
///
/// # Memory
///
/// Allocates a single `Vec<u8>` of length `n_elements × 2` for the `BF16`
/// output. Uses `Vec::with_capacity` + `extend_from_slice` to avoid the
/// `vec![0u8; n]` zero-init memset that would otherwise touch every byte
/// of the output before the dequant loop overwrites them. Peak heap is
/// the output buffer itself — O(n).
pub fn dequantize_gguf_to_bf16(
    data: &[u8],
    dtype: GgufType,
    n_elements: usize,
) -> crate::Result<Vec<u8>> {
    if n_elements == 0 {
        return Ok(Vec::new());
    }
    let out_byte_len = validate_dequant_input(data, dtype, n_elements)?;
    let mut out: Vec<u8> = Vec::with_capacity(out_byte_len);
    dispatch_streaming(data, dtype, |block_out| {
        out.extend_from_slice(block_out);
        Ok(())
    })?;
    Ok(out)
}

/// Streaming `GGUF` dequantisation: invokes `sink` once per block with
/// that block's `BF16` bytes. Peak heap is O(one block) regardless of
/// tensor size.
///
/// The `sink` closure receives `64` bytes per call for legacy quants
/// (`Q4_0`–`Q8_1`) and `512` bytes per call for K-quants. Sink errors
/// abort the stream and are propagated unchanged as the return value.
///
/// This is the canonical form. [`dequantize_gguf_to_bf16`] is a thin
/// wrapper that sinks into a `Vec::with_capacity`.
///
/// # Errors
///
/// Returns the same validation errors as [`dequantize_gguf_to_bf16`]
/// plus any [`AnamnesisError`] that `sink` itself returns.
///
/// # Memory
///
/// Stack only: one `[f32; QK]` scratch buffer (128 B for legacy, 1 KB
/// for K-quants) and one `[u8; QK × 2]` block output buffer (64 B /
/// 512 B). No heap allocation in this function's frame.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "gguf")]
/// # fn _doc() -> anamnesis::Result<()> {
/// use anamnesis::{dequantize_gguf_blocks_to_bf16, GgufType};
///
/// let data: &[u8] = &[];
/// let n_elements = 0;
/// let mut total_bytes = 0usize;
/// dequantize_gguf_blocks_to_bf16(data, GgufType::Q4_0, n_elements, |block| {
///     total_bytes += block.len();
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
pub fn dequantize_gguf_blocks_to_bf16<F>(
    data: &[u8],
    dtype: GgufType,
    n_elements: usize,
    sink: F,
) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    if n_elements == 0 {
        return Ok(());
    }
    // Discarding the returned `out_byte_len`: the streaming variant does
    // not allocate an output buffer, so it only needs the validation side
    // effects.
    let _ = validate_dequant_input(data, dtype, n_elements)?;
    dispatch_streaming(data, dtype, sink)
}

// ---------------------------------------------------------------------------
// Legacy block quants — pass-1 closures
// ---------------------------------------------------------------------------

/// `Q4_0` kernel — 18-byte blocks: `d: f16` + `qs[16]` (4-bit packed).
///
/// Formula: `y[j] = d * (qs[j] nibble - 8)`. Low nibbles of `qs[0..16]`
/// fill output positions `0..16`, high nibbles fill `16..32`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q4_0<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 18, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        // BITWISE: split each qs byte into two 4-bit nibbles, bias by -8
        // CAST: i32 → f32, lossless for values in [-8, 7]
        for j in 0..16 {
            let lo = i32::from(in_block[2 + j] & 0x0F) - 8;
            let hi = i32::from(in_block[2 + j] >> 4) - 8;
            scratch[j] = lo as f32 * d;
            scratch[j + 16] = hi as f32 * d;
        }
    })
}

/// `Q4_1` kernel — 20-byte blocks: `d: f16` + `m: f16` + `qs[16]` (4-bit).
///
/// Formula: `y[j] = d * qs[j] nibble + m`. No `-8` bias.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q4_1<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 20, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let m = read_f16_bytes([in_block[2], in_block[3]]);
        // BITWISE: split each qs byte into two unsigned 4-bit nibbles
        // CAST: i32 → f32, lossless for [0, 15]
        for j in 0..16 {
            let lo = i32::from(in_block[4 + j] & 0x0F);
            let hi = i32::from(in_block[4 + j] >> 4);
            scratch[j] = lo as f32 * d + m;
            scratch[j + 16] = hi as f32 * d + m;
        }
    })
}

/// `Q5_0` kernel — 22-byte blocks: `d: f16` + `qh[4]` (u32) + `qs[16]`.
///
/// Formula: `y[j] = d * ((5-bit value) - 16)`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q5_0<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 22, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let qh = u32::from_le_bytes([in_block[2], in_block[3], in_block[4], in_block[5]]);
        // BITWISE: merge low 4 bits from qs with bit 4 from qh, bias by -16
        // CAST: u32/i32 → f32, lossless for [-16, 15]
        for j in 0..16 {
            // CAST: usize → u32 for the shift amount
            #[allow(clippy::cast_possible_truncation)]
            let j_u32 = j as u32;
            let xh_0 = ((qh >> j_u32) << 4) & 0x10;
            let xh_1 = (qh >> (j_u32 + 12)) & 0x10;
            #[allow(clippy::cast_possible_wrap)]
            let x0 = (i32::from(in_block[6 + j] & 0x0F) | xh_0 as i32) - 16;
            #[allow(clippy::cast_possible_wrap)]
            let x1 = (i32::from(in_block[6 + j] >> 4) | xh_1 as i32) - 16;
            scratch[j] = x0 as f32 * d;
            scratch[j + 16] = x1 as f32 * d;
        }
    })
}

/// `Q5_1` kernel — 24-byte blocks: `d: f16` + `m: f16` + `qh[4]` + `qs[16]`.
///
/// Formula: `y[j] = d * (5-bit value) + m`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q5_1<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 24, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let m = read_f16_bytes([in_block[2], in_block[3]]);
        let qh = u32::from_le_bytes([in_block[4], in_block[5], in_block[6], in_block[7]]);
        for j in 0..16 {
            #[allow(clippy::cast_possible_truncation)]
            let j_u32 = j as u32;
            // BITWISE: merge low 4 bits from qs with bit 4 from qh
            let xh_0 = ((qh >> j_u32) << 4) & 0x10;
            let xh_1 = (qh >> (j_u32 + 12)) & 0x10;
            #[allow(clippy::cast_possible_wrap)]
            let x0 = i32::from(in_block[8 + j] & 0x0F) | xh_0 as i32;
            #[allow(clippy::cast_possible_wrap)]
            let x1 = i32::from(in_block[8 + j] >> 4) | xh_1 as i32;
            // CAST: i32 → f32, lossless for [0, 31]
            scratch[j] = x0 as f32 * d + m;
            scratch[j + 16] = x1 as f32 * d + m;
        }
    })
}

/// `Q8_0` kernel — 34-byte blocks: `d: f16` + `qs[32]` (`i8`).
///
/// Formula: `y[j] = d * qs[j]`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_wrap
)]
fn dequant_q8_0<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 34, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        // CAST: u8 → i8 intentional signed reinterpret, then i8 → f32 lossless
        for j in 0..QK_SMALL {
            let signed = in_block[2 + j] as i8;
            scratch[j] = f32::from(signed) * d;
        }
    })
}

/// `Q8_1` kernel — 36-byte blocks: `d: f16` + `s: f16` (aux) + `qs[32]`.
///
/// The `s` field stores `d × Σ qs[i]` as a matmul accelerator and is
/// **not** used for reconstruction. Formula: `y[j] = d * qs[j]`.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_wrap
)]
fn dequant_q8_1<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_legacy_kernel(data, 36, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        // in_block[2..4] is `s` (aux, ignored)
        for j in 0..QK_SMALL {
            let signed = in_block[4 + j] as i8;
            scratch[j] = f32::from(signed) * d;
        }
    })
}

// ---------------------------------------------------------------------------
// K-quants — pass-1 closures
// ---------------------------------------------------------------------------

/// `Q8_K` kernel — 292-byte blocks: `d: f32` (**not f16!**) + `qs[256]: i8`
/// + `bsums[16]: i16` (aux, ignored for reconstruction).
///
/// Formula: `y[j] = d * qs[j]`. The simplest K-quant.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_wrap
)]
fn dequant_q8_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 292, sink, |in_block, scratch| {
        let d = read_f32_bytes([in_block[0], in_block[1], in_block[2], in_block[3]]);
        // CAST: u8 → i8 intentional signed reinterpret, i8 → f32 lossless
        for j in 0..QK_K {
            let signed = in_block[4 + j] as i8;
            scratch[j] = f32::from(signed) * d;
        }
    })
}

/// `Q2_K` kernel — 84-byte blocks: `scales[16]` (packed 4/4-bit) +
/// `qs[64]` (2-bit) + `d: f16` + `dmin: f16`.
///
/// Each `scales[is]` byte packs a 4-bit scale (low nibble) and a 4-bit
/// min (high nibble). The 64-byte `qs` holds 256 elements as 2-bit values
/// packed 4-per-byte. Iteration walks 8 sub-blocks of 32 elements each,
/// driven by a `shift` variable taking values 0, 2, 4, 6.
///
/// Formula: `y = d * (sc & 0xF) * q2 - dmin * (sc >> 4)` where
/// `q2 = (qs[l] >> shift) & 3`.
#[allow(
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q2_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 84, sink, |in_block, scratch| {
        // Field offsets: scales [0..16], qs [16..80], d [80..82], dmin [82..84]
        let scales = &in_block[0..16];
        let qs = &in_block[16..80];
        let d = read_f16_bytes([in_block[80], in_block[81]]);
        let dmin = read_f16_bytes([in_block[82], in_block[83]]);

        // 2 halves × 4 shifts × 2 sub-groups × 16 elements = 256
        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        for _n in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                // BITWISE: unpack (scale_nibble, min_nibble) from one scales byte
                let sc_a = scales[is];
                is += 1;
                let dl_a = d * f32::from(sc_a & 0x0F);
                let ml_a = dmin * f32::from(sc_a >> 4);
                for l in 0..16 {
                    // BITWISE: 2-bit value at `shift` from qs[q_off + l]
                    // CAST: i32 → f32, lossless for [0, 3]
                    let q2 = i32::from((qs[q_off + l] >> shift) & 0x03);
                    scratch[y_off + l] = dl_a * (q2 as f32) - ml_a;
                }
                let sc_b = scales[is];
                is += 1;
                let dl_b = d * f32::from(sc_b & 0x0F);
                let ml_b = dmin * f32::from(sc_b >> 4);
                for l in 0..16 {
                    let q2 = i32::from((qs[q_off + l + 16] >> shift) & 0x03);
                    scratch[y_off + l + 16] = dl_b * (q2 as f32) - ml_b;
                }
                y_off += 32;
                shift += 2;
            }
            q_off += 32;
        }
    })
}

/// `Q3_K` kernel — 110-byte blocks: `hmask[32]` + `qs[64]` (2-bit low) +
/// `scales[12]` (6-bit packed) + `d: f16`.
///
/// Scales are 6-bit signed values packed into 12 bytes via a `kmask1`/
/// `kmask2` bit-permute. The 3-bit values are reconstructed by combining
/// 2 low bits from `qs` with 1 high bit from `hmask`, then subtracting 4
/// when `hmask` is zero. Formula: `y = d * (scale - 32) * (q2 - hi)`.
#[allow(
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q3_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 110, sink, |in_block, scratch| {
        // Field offsets: hmask [0..32], qs [32..96], scales [96..108], d [108..110]
        let hmask = &in_block[0..32];
        let qs = &in_block[32..96];
        // `.try_into()` on a fixed-length slice cannot fail; the internal
        // fallback error is structurally dead code but kept to satisfy the
        // slice-to-array API.
        let packed_scales: [u8; 12] = [
            in_block[96],
            in_block[97],
            in_block[98],
            in_block[99],
            in_block[100],
            in_block[101],
            in_block[102],
            in_block[103],
            in_block[104],
            in_block[105],
            in_block[106],
            in_block[107],
        ];
        let d_all = read_f16_bytes([in_block[108], in_block[109]]);
        let scales = q3_k_unpack_scales(&packed_scales);

        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        let mut m: u8 = 1;
        for _n in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                // CAST: i8 → f32 (lossless), bias by -32 per ggml reference
                let sc_a = scales[is];
                is += 1;
                let dl_a = d_all * (f32::from(sc_a) - 32.0);
                for l in 0..16 {
                    // BITWISE: 2-bit low from qs + 3rd bit from hmask (offset -4 if absent)
                    let q2 = i32::from((qs[q_off + l] >> shift) & 0x03);
                    let hi = i32::from(hmask[l] & m == 0) * 4;
                    scratch[y_off + l] = dl_a * ((q2 - hi) as f32);
                }
                let sc_b = scales[is];
                is += 1;
                let dl_b = d_all * (f32::from(sc_b) - 32.0);
                for l in 0..16 {
                    let q2 = i32::from((qs[q_off + l + 16] >> shift) & 0x03);
                    let hi = i32::from(hmask[l + 16] & m == 0) * 4;
                    scratch[y_off + l + 16] = dl_b * ((q2 - hi) as f32);
                }
                y_off += 32;
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
    })
}

/// `Q4_K` kernel — 144-byte blocks: `d: f16` + `dmin: f16` +
/// `scales[12]` (6-bit packed) + `qs[128]` (4-bit).
///
/// Scales are extracted via [`get_scale_min_k4`]. Formula:
/// `y = d * sc * (qs nibble) - dmin * m`. Processed in 4 groups of 64,
/// each producing 32 low-nibble outputs then 32 high-nibble outputs.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q4_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 144, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let dmin = read_f16_bytes([in_block[2], in_block[3]]);
        let scales: [u8; 12] = [
            in_block[4],
            in_block[5],
            in_block[6],
            in_block[7],
            in_block[8],
            in_block[9],
            in_block[10],
            in_block[11],
            in_block[12],
            in_block[13],
            in_block[14],
            in_block[15],
        ];
        let qs = &in_block[16..144];

        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        for _j in 0..4 {
            let (sc_lo, min_lo) = get_scale_min_k4(is, &scales);
            let (sc_hi, min_hi) = get_scale_min_k4(is + 1, &scales);
            let d_lo = d * f32::from(sc_lo);
            let off_lo = dmin * f32::from(min_lo);
            let d_hi = d * f32::from(sc_hi);
            let off_hi = dmin * f32::from(min_hi);
            for l in 0..32 {
                // BITWISE: extract low/high 4-bit nibbles
                // CAST: i32 → f32, lossless for [0, 15]
                let lo = i32::from(qs[q_off + l] & 0x0F);
                let hi = i32::from(qs[q_off + l] >> 4);
                scratch[y_off + l] = d_lo * (lo as f32) - off_lo;
                scratch[y_off + l + 32] = d_hi * (hi as f32) - off_hi;
            }
            q_off += 32;
            y_off += 64;
            is += 2;
        }
    })
}

/// `Q5_K` kernel — 176-byte blocks: `d: f16` + `dmin: f16` +
/// `scales[12]` + `qh[32]` (5th-bit store) + `qs[128]` (4-bit low).
///
/// Like [`dequant_q4_k`], but each 4-bit nibble gets an additional high bit
/// from `qh[l]` selected by a rotating `u1`/`u2` mask. Formula:
/// `y = d * sc * ((ql & 0xF) + (qh & mask ? 16 : 0)) - dmin * m`.
#[allow(
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q5_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 176, sink, |in_block, scratch| {
        let d = read_f16_bytes([in_block[0], in_block[1]]);
        let dmin = read_f16_bytes([in_block[2], in_block[3]]);
        let scales: [u8; 12] = [
            in_block[4],
            in_block[5],
            in_block[6],
            in_block[7],
            in_block[8],
            in_block[9],
            in_block[10],
            in_block[11],
            in_block[12],
            in_block[13],
            in_block[14],
            in_block[15],
        ];
        let qh = &in_block[16..48];
        let ql = &in_block[48..176];

        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _j in 0..4 {
            let (sc_lo, min_lo) = get_scale_min_k4(is, &scales);
            let (sc_hi, min_hi) = get_scale_min_k4(is + 1, &scales);
            let d_lo = d * f32::from(sc_lo);
            let off_lo = dmin * f32::from(min_lo);
            let d_hi = d * f32::from(sc_hi);
            let off_hi = dmin * f32::from(min_hi);
            for l in 0..32 {
                // BITWISE: merge 4-bit low from ql with high bit from qh (rotating mask)
                let lo = i32::from(ql[q_off + l] & 0x0F) + i32::from(qh[l] & u1 != 0) * 16;
                let hi = i32::from(ql[q_off + l] >> 4) + i32::from(qh[l] & u2 != 0) * 16;
                // CAST: i32 → f32, lossless for [0, 31]
                scratch[y_off + l] = d_lo * (lo as f32) - off_lo;
                scratch[y_off + l + 32] = d_hi * (hi as f32) - off_hi;
            }
            q_off += 32;
            y_off += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    })
}

/// `Q6_K` kernel — 210-byte blocks: `ql[128]` (4-bit low) + `qh[64]`
/// (2-bit high) + `scales[16]: i8` + `d: f16`.
///
/// Each 6-bit element is reconstructed as `(ql & 0xF) | ((qh >> shift) &
/// 3) << 4`, biased by `-32`. Processed in 2 halves of 128 elements.
#[allow(
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn dequant_q6_k<F>(data: &[u8], sink: F) -> crate::Result<()>
where
    F: FnMut(&[u8]) -> crate::Result<()>,
{
    run_super_kernel(data, 210, sink, |in_block, scratch| {
        // Field offsets: ql [0..128], qh [128..192], scales [192..208], d [208..210]
        let ql_all = &in_block[0..128];
        let qh_all = &in_block[128..192];
        let sc_all = &in_block[192..208];
        let d = read_f16_bytes([in_block[208], in_block[209]]);

        let mut y_off: usize = 0;
        let mut ql_off: usize = 0;
        let mut qh_off: usize = 0;
        let mut sc_off: usize = 0;
        for _n in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                // BITWISE: 4-bit low + 2-bit high (shift 0,2,4,6 for four values)
                let q1 =
                    i32::from((ql_all[ql_off + l] & 0x0F) | ((qh_all[qh_off + l] & 0x03) << 4))
                        - 32;
                let q2 = i32::from(
                    (ql_all[ql_off + l + 32] & 0x0F) | (((qh_all[qh_off + l] >> 2) & 0x03) << 4),
                ) - 32;
                let q3 = i32::from(
                    (ql_all[ql_off + l] >> 4) | (((qh_all[qh_off + l] >> 4) & 0x03) << 4),
                ) - 32;
                let q4 = i32::from(
                    (ql_all[ql_off + l + 32] >> 4) | (((qh_all[qh_off + l] >> 6) & 0x03) << 4),
                ) - 32;

                // CAST: u8 → i8 intentional signed reinterpret for scales
                let s0 = sc_all[sc_off + is] as i8;
                let s2 = sc_all[sc_off + is + 2] as i8;
                let s4 = sc_all[sc_off + is + 4] as i8;
                let s6 = sc_all[sc_off + is + 6] as i8;
                // CAST: i32 → f32, lossless for the narrow ranges in use
                scratch[y_off + l] = d * f32::from(s0) * (q1 as f32);
                scratch[y_off + l + 32] = d * f32::from(s2) * (q2 as f32);
                scratch[y_off + l + 64] = d * f32::from(s4) * (q3 as f32);
                scratch[y_off + l + 96] = d * f32::from(s6) * (q4 as f32);
            }
            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    })
}

// ---------------------------------------------------------------------------
// K-quant scale extractors
// ---------------------------------------------------------------------------

/// Extracts the 6-bit `(scale, min)` pair for sub-block `j` from the
/// packed `scales[12]` field used by `Q4_K` and `Q5_K`.
///
/// Ported verbatim from ggml-quants.c's `get_scale_min_k4` helper.
#[inline]
#[allow(clippy::indexing_slicing)]
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Unpacks the 12-byte packed `scales` field of a `Q3_K` block into 16
/// values (stored as `i8` but the caller biases by `-32` in the dequant
/// formula, so the actual range here is `0..=63`).
///
/// Ported verbatim from ggml-quants.c's `kmask1`/`kmask2` permute. The
/// `>> 0` in the first permute line is kept for symmetry with the
/// `>> 2`/`>> 4`/`>> 6` siblings, so `clippy::identity_op` is allowed.
#[inline]
#[allow(
    clippy::indexing_slicing,
    clippy::identity_op,
    clippy::as_conversions,
    clippy::cast_possible_wrap
)]
fn q3_k_unpack_scales(packed: &[u8; 12]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let aux0_in = u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
    let aux1_in = u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
    let aux2_in = u32::from_le_bytes([packed[8], packed[9], packed[10], packed[11]]);

    let tmp = aux2_in;
    // BITWISE: two-level permute; see ggml-quants.c `dequantize_row_q3_K`
    let aux0 = (aux0_in & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    let aux1 = (aux1_in & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let aux2 = ((aux0_in >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    let aux3 = ((aux1_in >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);

    // Reinterpret the 16 bytes (4 × u32 LE) as [i8; 16] following the
    // ggml reference's `(int8_t*)aux` cast.
    let mut out = [0i8; 16];
    for (i, &word) in [aux0, aux1, aux2, aux3].iter().enumerate() {
        let bytes = word.to_le_bytes();
        // CAST: u8 → i8, intentional signed reinterpret
        out[i * 4] = bytes[0] as i8;
        out[i * 4 + 1] = bytes[1] as i8;
        out[i * 4 + 2] = bytes[2] as i8;
        out[i * 4 + 3] = bytes[3] as i8;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::wildcard_enum_match_arm,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop
)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Helpers shared by test fixtures
    // -----------------------------------------------------------------

    /// Decodes a little-endian `BF16` byte pair back to `f32` for assertions.
    fn bf16_pair_to_f32(bytes: &[u8]) -> f32 {
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        f32::from(half::bf16::from_bits(bits))
    }

    /// Packs an `f32` into the 2-byte little-endian `ggml_half` form.
    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    // -----------------------------------------------------------------
    // Dispatcher validation
    // -----------------------------------------------------------------

    #[test]
    fn zero_elements_returns_empty() {
        let out = dequantize_gguf_to_bf16(&[], GgufType::Q4_0, 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn rejects_non_multiple_of_block_size() {
        let data = vec![0u8; 18];
        let err = dequantize_gguf_to_bf16(&data, GgufType::Q4_0, 17).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn rejects_wrong_data_length() {
        // Q4_0: 1 block = 18 bytes; give it 17.
        let data = vec![0u8; 17];
        let err = dequantize_gguf_to_bf16(&data, GgufType::Q4_0, 32).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn rejects_unsupported_dtype() {
        // IQ4_XS is recognised by GgufType but has type_size() == None.
        let err = dequantize_gguf_to_bf16(&[], GgufType::IQ4_XS, 256).unwrap_err();
        assert!(matches!(err, AnamnesisError::Unsupported { .. }));
    }

    #[test]
    fn overflow_guard_rejects_huge_n_elements() {
        // Largest multiple of 32 that's strictly > usize::MAX / 2 so that
        // `n_elements * 2` overflows. Works on both 64-bit and 32-bit
        // because the calculation scales to `usize::MAX`.
        let n_elements = (usize::MAX / 32) * 32;
        let err = dequantize_gguf_to_bf16(&[], GgufType::Q4_0, n_elements).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(
                    reason.contains("overflows usize"),
                    "expected overflow error, got: {reason}"
                );
            }
            other => panic!("expected Parse overflow, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Streaming API
    // -----------------------------------------------------------------

    #[test]
    fn streaming_calls_sink_once_per_block() {
        // 2 Q4_0 blocks, d = 0.0, qs = 0 — each sink call should receive 64 bytes
        let mut data = vec![0u8; 36];
        data[0..2].copy_from_slice(&f16_bytes(0.0));
        data[18..20].copy_from_slice(&f16_bytes(0.0));
        let mut block_count = 0;
        let mut total_bytes = 0;
        dequantize_gguf_blocks_to_bf16(&data, GgufType::Q4_0, 64, |block| {
            block_count += 1;
            total_bytes += block.len();
            Ok(())
        })
        .unwrap();
        assert_eq!(block_count, 2);
        assert_eq!(total_bytes, 128);
    }

    #[test]
    fn streaming_propagates_sink_error() {
        let mut data = vec![0u8; 18];
        data[0..2].copy_from_slice(&f16_bytes(0.0));
        let err = dequantize_gguf_blocks_to_bf16(&data, GgufType::Q4_0, 32, |_block| {
            Err(AnamnesisError::Parse {
                reason: "sink aborted".into(),
            })
        })
        .unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert_eq!(reason, "sink aborted"),
            other => panic!("expected Parse from sink, got {other:?}"),
        }
    }

    #[test]
    fn streaming_matches_vec_variant() {
        // Verify the two entry points produce byte-identical output for
        // a non-trivial Q4_0 fixture.
        let mut data = vec![0u8; 36];
        data[0..2].copy_from_slice(&f16_bytes(1.0));
        for j in 0..16 {
            data[2 + j] = (j as u8) << 4 | (j as u8);
        }
        data[18..20].copy_from_slice(&f16_bytes(-0.5));
        for j in 0..16 {
            data[20 + j] = 0xF0;
        }

        let vec_out = dequantize_gguf_to_bf16(&data, GgufType::Q4_0, 64).unwrap();

        let mut streamed = Vec::with_capacity(vec_out.len());
        dequantize_gguf_blocks_to_bf16(&data, GgufType::Q4_0, 64, |block| {
            streamed.extend_from_slice(block);
            Ok(())
        })
        .unwrap();

        assert_eq!(vec_out, streamed);
    }

    #[test]
    fn streaming_zero_elements_makes_no_sink_calls() {
        let mut calls = 0;
        dequantize_gguf_blocks_to_bf16(&[], GgufType::Q4_0, 0, |_block| {
            calls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 0);
    }

    // -----------------------------------------------------------------
    // Q8_0
    // -----------------------------------------------------------------

    #[test]
    fn q8_0_all_zero_block() {
        // d = 0.0, qs = 0 → all output 0.0
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16_bytes(0.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q8_0, 32).unwrap();
        assert_eq!(out.len(), 64);
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn q8_0_identity_scale() {
        // d = 1.0, qs = [-16..16]. Expect y[j] = qs[j].
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        for j in 0..32 {
            let v: i8 = (j as i8).wrapping_sub(16);
            block[2 + j] = v as u8;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q8_0, 32).unwrap();
        for j in 0..32 {
            let expected = f32::from((j as i8).wrapping_sub(16));
            let got = bf16_pair_to_f32(&out[j * 2..j * 2 + 2]);
            assert_eq!(got, expected, "Q8_0[{j}]");
        }
    }

    #[test]
    fn q8_0_multi_block() {
        // 2 blocks: block 0 d=2.0 qs=1 → 2.0; block 1 d=-1.0 qs=3 → -3.0
        let mut data = vec![0u8; 68];
        data[0..2].copy_from_slice(&f16_bytes(2.0));
        for j in 0..32 {
            data[2 + j] = 1;
        }
        data[34..36].copy_from_slice(&f16_bytes(-1.0));
        for j in 0..32 {
            data[36 + j] = 3;
        }
        let out = dequantize_gguf_to_bf16(&data, GgufType::Q8_0, 64).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 2.0);
        }
        for j in 32..64 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), -3.0);
        }
    }

    // -----------------------------------------------------------------
    // Q4_0
    // -----------------------------------------------------------------

    #[test]
    fn q4_0_zeroed_nibbles_centers_at_minus_8() {
        // d = 1.0, qs = all 0x00 → every nibble = 0, y = (0 - 8) * 1 = -8
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_0, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), -8.0);
        }
    }

    #[test]
    fn q4_0_identity_nibble_round_trip() {
        // d = 1.0, qs[j] = (j << 4) | j for j < 16.
        // Expect y[j] = (j - 8) and y[j + 16] = (j - 8).
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        for j in 0..16usize {
            block[2 + j] = ((j as u8) << 4) | (j as u8);
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_0, 32).unwrap();
        for j in 0..16 {
            let expected = j as f32 - 8.0;
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), expected);
            assert_eq!(
                bf16_pair_to_f32(&out[(j + 16) * 2..(j + 16) * 2 + 2]),
                expected
            );
        }
    }

    // -----------------------------------------------------------------
    // Q4_1
    // -----------------------------------------------------------------

    #[test]
    fn q4_1_affine_transform() {
        // d = 1.0, m = 10.0, qs = 0xFF → y = 15 * 1 + 10 = 25 at every position
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(10.0));
        for j in 0..16 {
            block[4 + j] = 0xFF;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_1, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 25.0);
        }
    }

    // -----------------------------------------------------------------
    // Q5_0 / Q5_1
    // -----------------------------------------------------------------

    #[test]
    fn q5_0_all_high_bits_set() {
        // d = 1.0, qh = 0xFFFFFFFF, qs = 0 → 5-bit value 16 everywhere,
        // y = (16 - 16) * 1 = 0
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..6].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q5_0, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 0.0);
        }
    }

    #[test]
    fn q5_1_identity_nibbles_no_high_bits() {
        // d = 1.0, m = 5.0, qh = 0, qs[j] = 0x11 → 5-bit = 1, y = 1 + 5 = 6
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(5.0));
        for j in 0..16 {
            block[8 + j] = 0x11;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q5_1, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 6.0);
        }
    }

    // -----------------------------------------------------------------
    // Q8_1
    // -----------------------------------------------------------------

    #[test]
    fn q8_1_ignores_s_field() {
        // d=1.0, s=999.0 (ignored), qs sequential i8
        let mut block = vec![0u8; 36];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(999.0));
        for j in 0..32 {
            let v: i8 = j as i8;
            block[4 + j] = v as u8;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q8_1, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), f32::from(j as i8));
        }
    }

    // -----------------------------------------------------------------
    // Q8_K
    // -----------------------------------------------------------------

    #[test]
    fn q8_k_f32_scale() {
        // d = 0.5 (f32), qs sequential i8 pattern
        let mut block = vec![0u8; 292];
        block[0..4].copy_from_slice(&0.5_f32.to_le_bytes());
        for j in 0..256 {
            let v: i8 = ((j as i32) - 128) as i8;
            block[4 + j] = v as u8;
        }
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q8_K, 256).unwrap();
        for j in 0..256 {
            let expected = f32::from(((j as i32) - 128) as i8) * 0.5;
            assert_eq!(
                bf16_pair_to_f32(&out[j * 2..j * 2 + 2]),
                expected,
                "Q8_K[{j}]"
            );
        }
    }

    // -----------------------------------------------------------------
    // Q2_K
    // -----------------------------------------------------------------

    #[test]
    fn q2_k_zero_scales_and_mins() {
        let block = vec![0u8; 84];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q2_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn q2_k_uniform_scale_uniform_qs() {
        // d = 1.0, dmin = 0.0, scales = 0x02 (sc_lo=2, sc_hi=0), qs = 0x55
        // → 2-bit value 1 at every shift, y = 1 * 2 * 1 - 0 = 2
        let mut block = vec![0u8; 84];
        for j in 0..16 {
            block[j] = 0x02;
        }
        for j in 0..64 {
            block[16 + j] = 0x55;
        }
        block[80..82].copy_from_slice(&f16_bytes(1.0));
        block[82..84].copy_from_slice(&f16_bytes(0.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q2_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 2.0);
        }
    }

    // -----------------------------------------------------------------
    // Q3_K: helper tests first
    // -----------------------------------------------------------------

    #[test]
    fn q3_k_unpack_scales_zero() {
        let scales = q3_k_unpack_scales(&[0u8; 12]);
        assert_eq!(scales, [0i8; 16]);
    }

    #[test]
    fn q3_k_unpack_scales_low_half_trivial() {
        let mut packed = [0u8; 12];
        packed[0] = 0x00;
        packed[1] = 0x01;
        packed[2] = 0x02;
        packed[3] = 0x03;
        let scales = q3_k_unpack_scales(&packed);
        assert_eq!(scales[0], 0);
        assert_eq!(scales[1], 1);
        assert_eq!(scales[2], 2);
        assert_eq!(scales[3], 3);
    }

    #[test]
    fn q3_k_all_zero_block() {
        let block = vec![0u8; 110];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q3_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    // -----------------------------------------------------------------
    // Q4_K: helper tests first
    // -----------------------------------------------------------------

    #[test]
    fn get_scale_min_k4_low_half() {
        let scales = [
            0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(get_scale_min_k4(0, &scales), (0x00, 0x10));
        assert_eq!(get_scale_min_k4(1, &scales), (0x01, 0x11));
        assert_eq!(get_scale_min_k4(2, &scales), (0x02, 0x12));
        assert_eq!(get_scale_min_k4(3, &scales), (0x03, 0x13));
    }

    #[test]
    fn get_scale_min_k4_high_half_zero() {
        let scales = [0u8; 12];
        for j in 0..8 {
            assert_eq!(get_scale_min_k4(j, &scales), (0, 0));
        }
    }

    #[test]
    fn q4_k_all_zero_block() {
        let block = vec![0u8; 144];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    // -----------------------------------------------------------------
    // Q5_K
    // -----------------------------------------------------------------

    #[test]
    fn q5_k_all_zero_block() {
        let block = vec![0u8; 176];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q5_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    // -----------------------------------------------------------------
    // Q6_K
    // -----------------------------------------------------------------

    #[test]
    fn q6_k_all_zero_block() {
        let block = vec![0u8; 210];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q6_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn q6_k_identity_centers_at_minus_32() {
        // d = 1.0, scales = 1, ql/qh = 0 → 6-bit value 0 - 32 = -32, y = -32
        let mut block = vec![0u8; 210];
        for j in 0..16 {
            block[192 + j] = 1;
        }
        block[208..210].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q6_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), -32.0);
        }
    }
}
