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
//! # Algorithm
//!
//! Every kernel follows the two-pass loop-fission pattern from
//! `CONVENTIONS.md`: for each block,
//!
//! 1. **Pass 1** unpacks the packed-bit storage into a stack-resident
//!    `[f32; QK]` scratch buffer (one block at a time, L1-resident).
//! 2. **Pass 2** walks the scratch buffer paired with output chunks and
//!    emits `BF16` bytes via the shared
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
// Byte readers
// ---------------------------------------------------------------------------

/// Reads a `ggml_half` (IEEE 754 binary16) from a 2-byte slice at
/// `offset` and widens it to `f32`.
///
/// # Preconditions
///
/// `bytes.len() >= offset + 2`. Callers pre-validate each block's full
/// slice once at the top of their outer loop, so the inner `.get()` can
/// never fail in practice — the defensive `ok_or_else` only exists to
/// keep the function infallible-shaped in the type system.
#[inline]
fn read_f16_le(bytes: &[u8], offset: usize) -> crate::Result<f32> {
    let pair = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("GGUF dequant: f16 read out of bounds at offset {offset}"),
        })?;
    // BORROW: explicit `.try_into()` converts `&[u8]` to `[u8; 2]` for the
    // `half::f16::from_le_bytes` constructor.
    let arr: [u8; 2] = pair.try_into().map_err(|_| AnamnesisError::Parse {
        reason: "GGUF dequant: f16 slice length != 2".into(),
    })?;
    Ok(f32::from(half::f16::from_le_bytes(arr)))
}

/// Reads a little-endian `f32` from a 4-byte slice at `offset`. Used only
/// by the `Q8_K` block, whose super-block scale is `f32` rather than
/// `f16`.
#[inline]
fn read_f32_le(bytes: &[u8], offset: usize) -> crate::Result<f32> {
    let quad = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("GGUF dequant: f32 read out of bounds at offset {offset}"),
        })?;
    // BORROW: explicit `.try_into()` converts `&[u8]` to `[u8; 4]` for `from_le_bytes`.
    let arr: [u8; 4] = quad.try_into().map_err(|_| AnamnesisError::Parse {
        reason: "GGUF dequant: f32 slice length != 4".into(),
    })?;
    Ok(f32::from_le_bytes(arr))
}

/// Converts a block's-worth of `f32` scratch values to `BF16` bytes.
///
/// This is the hot-path pass 2 for every kernel. Branch-free, contiguous
/// reads, contiguous writes — exactly the shape the compiler needs to
/// auto-vectorise the inner `f32 → BF16` conversion.
///
/// # Preconditions
///
/// `out_block.len() == 2 × scratch.len()`. `chunks_exact_mut(2)` silently
/// truncates any remainder, so a caller that passes the wrong length will
/// produce short output rather than a panic — every kernel validates its
/// block slice upfront so this never happens in practice.
#[inline]
fn write_scratch_to_bf16(scratch: &[f32], out_block: &mut [u8]) {
    // VECTORIZED: pure f32 → BF16 inner loop, branch-free, expected AVX2 target
    // (verify with `cargo-show-asm` after the full kernel set lands).
    for (&val, out_pair) in scratch.iter().zip(out_block.chunks_exact_mut(2)) {
        let bf16 = f32_bits_to_bf16_bits(val.to_bits());
        out_pair.copy_from_slice(&bf16.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Dequantises a `GGUF`-encoded block-quantised tensor to `BF16` bytes.
///
/// Accepts the raw on-disk bytes from
/// [`GgufTensor::data`](crate::parse::gguf::GgufTensor) and returns
/// `n_elements × 2` bytes of `BF16` in little-endian order.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `n_elements` is not a multiple of
/// the block size for `dtype`, or if `data.len()` does not equal the
/// expected byte count (`n_blocks × type_size`).
///
/// Returns [`AnamnesisError::Unsupported`] if `dtype` is one of the
/// recognised-but-not-yet-implemented types (`IQ*`, `TQ*`, `MXFP4`) or a
/// scalar type that is not a quantised block format.
///
/// # Memory
///
/// Allocates a single `Vec<u8>` of length `n_elements × 2` for the `BF16`
/// output plus a stack-resident `[f32; 32]` or `[f32; 256]` scratch
/// buffer (≤ 1 KB). Peak heap is the output buffer itself — O(n).
pub fn dequantize_gguf_to_bf16(
    data: &[u8],
    dtype: GgufType,
    n_elements: usize,
) -> crate::Result<Vec<u8>> {
    // Fast-path for zero-length tensors — accept as a valid no-op.
    if n_elements == 0 {
        return Ok(Vec::new());
    }

    let block_size = dtype.block_size();
    if !n_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF dequant: n_elements {n_elements} is not a multiple of block size \
                 {block_size} for {dtype}"
            ),
        });
    }
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

    // EXHAUSTIVE: internal dispatch over GgufType. IQ*/TQ*/MXFP4 have no
    // implemented kernel yet; scalar (non-block) types (F32/F16/BF16/I*)
    // are structurally dequantised (reinterpret bytes), not in scope here.
    #[allow(clippy::wildcard_enum_match_arm)]
    match dtype {
        GgufType::Q4_0 => dequant_q4_0(data, n_blocks),
        GgufType::Q4_1 => dequant_q4_1(data, n_blocks),
        GgufType::Q5_0 => dequant_q5_0(data, n_blocks),
        GgufType::Q5_1 => dequant_q5_1(data, n_blocks),
        GgufType::Q8_0 => dequant_q8_0(data, n_blocks),
        GgufType::Q8_1 => dequant_q8_1(data, n_blocks),
        GgufType::Q2_K => dequant_q2_k(data, n_blocks),
        GgufType::Q3_K => dequant_q3_k(data, n_blocks),
        GgufType::Q4_K => dequant_q4_k(data, n_blocks),
        GgufType::Q5_K => dequant_q5_k(data, n_blocks),
        GgufType::Q6_K => dequant_q6_k(data, n_blocks),
        GgufType::Q8_K => dequant_q8_k(data, n_blocks),
        _ => Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("dequantisation not yet supported for {dtype}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Legacy block quants
// ---------------------------------------------------------------------------

/// `Q4_0` kernel — 18-byte blocks: `d: f16` + `qs[16]` (4-bit packed).
///
/// Formula: `y[j] = d * (qs[j] 4-bit nibble - 8)`. Low nibbles of `qs[0..16]`
/// fill output positions `0..16`, high nibbles fill `16..32`.
fn dequant_q4_0(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 18;
    let mut out = vec![0u8; n_blocks * QK_SMALL * 2];
    let mut scratch = [0.0_f32; QK_SMALL];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q4_0: block {b} slice out of bounds"),
            })?;
        let d = read_f16_le(block, 0)?;
        // INDEX: block.len() == 18, compile-time range 2..18 is in bounds
        #[allow(clippy::indexing_slicing)]
        let qs = &block[2..18];

        // PASS 1: unpack 4-bit nibbles into f32 scratch.
        // INDEX: qs.len() == 16, j iterates 0..16
        #[allow(clippy::indexing_slicing)]
        for j in 0..16 {
            // BITWISE: extract low and high 4-bit nibbles from qs[j], bias by -8
            let lo = i32::from(qs[j] & 0x0F) - 8;
            let hi = i32::from(qs[j] >> 4) - 8;
            // CAST: i32 → f32, lossless for values in [-8, 7]
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            {
                scratch[j] = lo as f32 * d;
                scratch[j + 16] = hi as f32 * d;
            }
        }

        // PASS 2: f32 scratch → BF16 output bytes.
        let out_block = out
            .get_mut(b * QK_SMALL * 2..(b + 1) * QK_SMALL * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q4_0: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q4_1` kernel — 20-byte blocks: `d: f16` + `m: f16` + `qs[16]` (4-bit).
///
/// Formula: `y[j] = d * qs[j] 4-bit nibble + m`. No `-8` bias.
fn dequant_q4_1(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 20;
    let mut out = vec![0u8; n_blocks * QK_SMALL * 2];
    let mut scratch = [0.0_f32; QK_SMALL];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q4_1: block {b} slice out of bounds"),
            })?;
        let d = read_f16_le(block, 0)?;
        let m = read_f16_le(block, 2)?;
        // INDEX: block.len() == 20, compile-time range 4..20 is in bounds
        #[allow(clippy::indexing_slicing)]
        let qs = &block[4..20];

        // PASS 1: unpack 4-bit nibbles + apply (d, m) affine transform.
        // INDEX: qs.len() == 16, j iterates 0..16
        #[allow(clippy::indexing_slicing)]
        for j in 0..16 {
            // BITWISE: extract low/high 4-bit nibbles (unsigned 0..=15)
            let lo = i32::from(qs[j] & 0x0F);
            let hi = i32::from(qs[j] >> 4);
            // CAST: i32 → f32, lossless for values in [0, 15]
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            {
                scratch[j] = lo as f32 * d + m;
                scratch[j + 16] = hi as f32 * d + m;
            }
        }

        let out_block = out
            .get_mut(b * QK_SMALL * 2..(b + 1) * QK_SMALL * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q4_1: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q5_0` kernel — 22-byte blocks: `d: f16` + `qh[4]` + `qs[16]` (4-bit low).
///
/// `qh` is a little-endian `u32` holding the 5th (high) bit of each of the
/// 32 elements. Formula: `y[j] = d * ((5-bit value) - 16)`.
fn dequant_q5_0(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 22;
    let mut out = vec![0u8; n_blocks * QK_SMALL * 2];
    let mut scratch = [0.0_f32; QK_SMALL];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q5_0: block {b} slice out of bounds"),
            })?;
        let d = read_f16_le(block, 0)?;
        // INDEX: block.len() == 22, offsets 2..6 and 6..22 are in bounds
        #[allow(clippy::indexing_slicing)]
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        #[allow(clippy::indexing_slicing)]
        let qs = &block[6..22];

        // PASS 1: merge low 4 bits from qs with bit 4 from qh, bias by -16.
        // INDEX: j iterates 0..16, qs.len() == 16
        #[allow(clippy::indexing_slicing)]
        for j in 0..16 {
            // BITWISE: extract the j-th high bit from qh and place it as bit 4
            // for the low-nibble value, and the (j+16)-th bit for the high-nibble
            // value. Mirrors the ggml reference exactly.
            // CAST: usize → u32 for the shift amount, j is 0..16 so lossless
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let j_u32 = j as u32;
            let xh_0 = ((qh >> j_u32) << 4) & 0x10;
            let xh_1 = (qh >> (j_u32 + 12)) & 0x10;
            // CAST: u32 → i32, xh_0/xh_1 are at most 0x10 so lossless
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let x0 = (i32::from(qs[j] & 0x0F) | xh_0 as i32) - 16;
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let x1 = (i32::from(qs[j] >> 4) | xh_1 as i32) - 16;
            // CAST: i32 → f32, lossless for values in [-16, 15]
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            {
                scratch[j] = x0 as f32 * d;
                scratch[j + 16] = x1 as f32 * d;
            }
        }

        let out_block = out
            .get_mut(b * QK_SMALL * 2..(b + 1) * QK_SMALL * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q5_0: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q5_1` kernel — 24-byte blocks: `d: f16` + `m: f16` + `qh[4]` + `qs[16]`.
///
/// Formula: `y[j] = d * (5-bit value) + m`.
fn dequant_q5_1(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 24;
    let mut out = vec![0u8; n_blocks * QK_SMALL * 2];
    let mut scratch = [0.0_f32; QK_SMALL];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q5_1: block {b} slice out of bounds"),
            })?;
        let d = read_f16_le(block, 0)?;
        let m = read_f16_le(block, 2)?;
        // INDEX: block.len() == 24, offsets 4..8 and 8..24 are in bounds
        #[allow(clippy::indexing_slicing)]
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        #[allow(clippy::indexing_slicing)]
        let qs = &block[8..24];

        // PASS 1: merge 5-bit values + apply (d, m) affine.
        // INDEX: j iterates 0..16, qs.len() == 16
        #[allow(clippy::indexing_slicing)]
        for j in 0..16 {
            // CAST: usize → u32 for the shift amount, j is 0..16 so lossless
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let j_u32 = j as u32;
            // BITWISE: extract the j-th high bit from qh and place it as bit 4
            let xh_0 = ((qh >> j_u32) << 4) & 0x10;
            let xh_1 = (qh >> (j_u32 + 12)) & 0x10;
            // CAST: u32 → i32, xh_0/xh_1 are at most 0x10 so lossless
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let x0 = i32::from(qs[j] & 0x0F) | xh_0 as i32;
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let x1 = i32::from(qs[j] >> 4) | xh_1 as i32;
            // CAST: i32 → f32, lossless for values in [0, 31]
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            {
                scratch[j] = x0 as f32 * d + m;
                scratch[j + 16] = x1 as f32 * d + m;
            }
        }

        let out_block = out
            .get_mut(b * QK_SMALL * 2..(b + 1) * QK_SMALL * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q5_1: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q8_0` kernel — 34-byte blocks: `d: f16` + `qs[32]` (`i8`).
///
/// Formula: `y[j] = d * qs[j]`. The simplest kernel in the family.
fn dequant_q8_0(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 34;
    let mut out = vec![0u8; n_blocks * QK_SMALL * 2];
    let mut scratch = [0.0_f32; QK_SMALL];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q8_0: block {b} slice out of bounds"),
            })?;
        let d = read_f16_le(block, 0)?;
        // INDEX: block.len() == 34, range 2..34 is 32 bytes of qs
        #[allow(clippy::indexing_slicing)]
        let qs = &block[2..34];

        // PASS 1: reinterpret each u8 as i8, widen to f32, multiply by d.
        // INDEX: scratch.len() == 32, qs.len() == 32
        #[allow(clippy::indexing_slicing)]
        for j in 0..QK_SMALL {
            // CAST: u8 → i8, intentional signed reinterpret
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let signed = qs[j] as i8;
            // CAST: i8 → f32, lossless for the full i8 range
            #[allow(clippy::as_conversions)]
            {
                scratch[j] = f32::from(signed) * d;
            }
        }

        let out_block = out
            .get_mut(b * QK_SMALL * 2..(b + 1) * QK_SMALL * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q8_0: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q8_1` kernel — 36-byte blocks: `d: f16` + `s: f16` (aux) + `qs[32]`.
///
/// The `s` field stores `d × Σ qs[i]` as a matmul accelerator and is
/// **not** used for reconstruction. Formula: `y[j] = d * qs[j]`.
fn dequant_q8_1(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 36;
    let mut out = vec![0u8; n_blocks * QK_SMALL * 2];
    let mut scratch = [0.0_f32; QK_SMALL];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q8_1: block {b} slice out of bounds"),
            })?;
        let d = read_f16_le(block, 0)?;
        // block[2..4] is `s` (aux, ignored)
        // INDEX: block.len() == 36, range 4..36 is 32 bytes of qs
        #[allow(clippy::indexing_slicing)]
        let qs = &block[4..36];

        // PASS 1: same as Q8_0.
        // INDEX: scratch.len() == 32, qs.len() == 32
        #[allow(clippy::indexing_slicing)]
        for j in 0..QK_SMALL {
            // CAST: u8 → i8, intentional signed reinterpret
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let signed = qs[j] as i8;
            // CAST: i8 → f32, lossless
            #[allow(clippy::as_conversions)]
            {
                scratch[j] = f32::from(signed) * d;
            }
        }

        let out_block = out
            .get_mut(b * QK_SMALL * 2..(b + 1) * QK_SMALL * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q8_1: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// K-quants
// ---------------------------------------------------------------------------

/// `Q8_K` kernel — 292-byte blocks: `d: f32` (**not f16!**) + `qs[256]: i8`
/// + `bsums[16]: i16` (aux, ignored for reconstruction).
///
/// Formula: `y[j] = d * qs[j]`. The simplest K-quant; primarily exists as
/// the activation input to K-quant matmuls.
fn dequant_q8_k(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 292;
    let mut out = vec![0u8; n_blocks * QK_K * 2];
    let mut scratch = [0.0_f32; QK_K];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q8_K: block {b} slice out of bounds"),
            })?;
        let d = read_f32_le(block, 0)?;
        // INDEX: block.len() == 292, range 4..260 is 256 bytes of qs
        #[allow(clippy::indexing_slicing)]
        let qs = &block[4..260];

        // PASS 1: reinterpret each u8 as i8, widen to f32, scale by d.
        // INDEX: scratch.len() == 256, qs.len() == 256
        #[allow(clippy::indexing_slicing)]
        for j in 0..QK_K {
            // CAST: u8 → i8, intentional signed reinterpret
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let signed = qs[j] as i8;
            // CAST: i8 → f32, lossless
            #[allow(clippy::as_conversions)]
            {
                scratch[j] = f32::from(signed) * d;
            }
        }

        let out_block = out
            .get_mut(b * QK_K * 2..(b + 1) * QK_K * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q8_K: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
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
///
/// # Indexing rationale
///
/// INDEX: every indexing operation inside the pass-1 inner loops has a
/// statically bounded offset: `is < 16` (scales), `q_off + l < 64` and
/// `q_off + l + 16 < 64` (qs), `y_off + l < 256` and `y_off + l + 16 < 256`
/// (scratch). All four are guaranteed by the `0..2` / `0..4` / `0..16`
/// outer-loop structure, so a function-level `indexing_slicing` allow is
/// applied rather than duplicating the identical annotation on every site.
#[allow(
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q2_k(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 84;
    let mut out = vec![0u8; n_blocks * QK_K * 2];
    let mut scratch = [0.0_f32; QK_K];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q2_K: block {b} slice out of bounds"),
            })?;
        // Field offsets: scales [0..16], qs [16..80], d [80..82], dmin [82..84]
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = read_f16_le(block, 80)?;
        let dmin = read_f16_le(block, 82)?;

        // PASS 1: unpack the 8 sub-blocks of 32 elements each.
        // Layout mirrors ggml-quants.c: two halves of 128 elements, each
        // with 4 shift positions (0,2,4,6) and two 16-element groups per
        // shift. Total scales consumed per block: 16.
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

        let out_block = out
            .get_mut(b * QK_K * 2..(b + 1) * QK_K * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q2_K: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q3_K` kernel — 110-byte blocks: `hmask[32]` + `qs[64]` (2-bit low) +
/// `scales[12]` (6-bit packed) + `d: f16`.
///
/// Scales are 6-bit signed values packed into 12 bytes via a `kmask1`/
/// `kmask2` bit-permute. The 3-bit values are reconstructed by combining
/// 2 low bits from `qs` with 1 high bit from `hmask`, then subtracting 4
/// when `hmask` is zero (implementing the `-4` offset on missing high
/// bits, not 0). Formula: `y = d * (scale - 32) * ((q2 | hi_offset))`.
///
/// # Indexing rationale
///
/// INDEX: as with `dequant_q2_k`, every hot-path index is statically
/// bounded by the `0..2` / `0..4` / `0..16` outer-loop structure:
/// `is < 16`, `q_off + l < 64`, `l < 32` (hmask), `y_off + l < 256`.
/// Function-level `indexing_slicing` allow replaces per-site annotations.
#[allow(
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q3_k(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 110;
    let mut out = vec![0u8; n_blocks * QK_K * 2];
    let mut scratch = [0.0_f32; QK_K];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q3_K: block {b} slice out of bounds"),
            })?;
        // Field offsets: hmask [0..32], qs [32..96], scales [96..108], d [108..110]
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let packed_scales: [u8; 12] =
            block[96..108]
                .try_into()
                .map_err(|_| AnamnesisError::Parse {
                    reason: format!("GGUF Q3_K: block {b} scales slice != 12 bytes"),
                })?;
        let d_all = read_f16_le(block, 108)?;
        let scales = q3_k_unpack_scales(&packed_scales);

        // PASS 1: 256 elements, 2 halves of 128, 4 shifts per half.
        let mut is: usize = 0;
        let mut y_off: usize = 0;
        let mut q_off: usize = 0;
        let mut m: u8 = 1;
        for _n in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                // CAST: i8 → f32, bias by -32 per the ggml reference
                let sc_a = scales[is];
                is += 1;
                let dl_a = d_all * (f32::from(sc_a) - 32.0);
                for l in 0..16 {
                    // BITWISE: 2-bit low from qs + 3rd bit from hmask (offset -4 if absent)
                    let q2 = i32::from((qs[q_off + l] >> shift) & 0x03);
                    let hi = i32::from(hmask[l] & m == 0) * 4;
                    let val = q2 - hi;
                    scratch[y_off + l] = dl_a * (val as f32);
                }
                let sc_b = scales[is];
                is += 1;
                let dl_b = d_all * (f32::from(sc_b) - 32.0);
                for l in 0..16 {
                    let q2 = i32::from((qs[q_off + l + 16] >> shift) & 0x03);
                    let hi = i32::from(hmask[l + 16] & m == 0) * 4;
                    let val = q2 - hi;
                    scratch[y_off + l + 16] = dl_b * (val as f32);
                }
                y_off += 32;
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }

        let out_block = out
            .get_mut(b * QK_K * 2..(b + 1) * QK_K * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q3_K: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q4_K` kernel — 144-byte blocks: `d: f16` + `dmin: f16` +
/// `scales[12]` (6-bit packed) + `qs[128]` (4-bit).
///
/// Scales are extracted via [`get_scale_min_k4`]. Formula:
/// `y = d * sc * (qs 4-bit) - dmin * m`. Processed in 4 groups of 64,
/// each producing 32 low-nibble outputs then 32 high-nibble outputs.
///
/// INDEX: `q_off + l + 32 < q_off + 64 ≤ 128` (qs) and
/// `y_off + l + 32 < y_off + 64 ≤ 256` (scratch) are guaranteed by the
/// `0..4` outer loop advancing `q_off` by 32 and `y_off` by 64.
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q4_k(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 144;
    let mut out = vec![0u8; n_blocks * QK_K * 2];
    let mut scratch = [0.0_f32; QK_K];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q4_K: block {b} slice out of bounds"),
            })?;
        let d = read_f16_le(block, 0)?;
        let dmin = read_f16_le(block, 2)?;
        // Field offsets: d/dmin [0..4], scales [4..16] (12 B), qs [16..144] (128 B)
        let scales: [u8; 12] = block[4..16].try_into().map_err(|_| AnamnesisError::Parse {
            reason: format!("GGUF Q4_K: block {b} scales slice != 12 bytes"),
        })?;
        let qs = &block[16..144];

        // PASS 1: 4 groups of 64 elements (32 low nibbles + 32 high nibbles each).
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
                let lo = i32::from(qs[q_off + l] & 0x0F);
                let hi = i32::from(qs[q_off + l] >> 4);
                // CAST: i32 → f32, lossless for [0, 15]
                scratch[y_off + l] = d_lo * (lo as f32) - off_lo;
                scratch[y_off + l + 32] = d_hi * (hi as f32) - off_hi;
            }
            q_off += 32;
            y_off += 64;
            is += 2;
        }

        let out_block = out
            .get_mut(b * QK_K * 2..(b + 1) * QK_K * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q4_K: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q5_K` kernel — 176-byte blocks: `d: f16` + `dmin: f16` +
/// `scales[12]` + `qh[32]` (5th-bit store) + `qs[128]` (4-bit low).
///
/// Like [`dequant_q4_k`], but each 4-bit nibble gets an additional high bit
/// from `qh[l]` selected by a rotating `u1`/`u2` mask. Formula:
/// `y = d * sc * ((ql & 0xF) + (qh & mask ? 16 : 0)) - dmin * m`.
///
/// INDEX: `q_off + l < q_off + 32 ≤ 128` (ql), `l < 32` (qh),
/// `y_off + l + 32 < 256` (scratch) — guaranteed by the `0..4` outer loop.
#[allow(
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
fn dequant_q5_k(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 176;
    let mut out = vec![0u8; n_blocks * QK_K * 2];
    let mut scratch = [0.0_f32; QK_K];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q5_K: block {b} slice out of bounds"),
            })?;
        let d = read_f16_le(block, 0)?;
        let dmin = read_f16_le(block, 2)?;
        // Field offsets: d/dmin [0..4], scales [4..16], qh [16..48], ql [48..176]
        let scales: [u8; 12] = block[4..16].try_into().map_err(|_| AnamnesisError::Parse {
            reason: format!("GGUF Q5_K: block {b} scales slice != 12 bytes"),
        })?;
        let qh = &block[16..48];
        let ql = &block[48..176];

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

        let out_block = out
            .get_mut(b * QK_K * 2..(b + 1) * QK_K * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q5_K: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

/// `Q6_K` kernel — 210-byte blocks: `ql[128]` (4-bit low) + `qh[64]`
/// (2-bit high) + `scales[16]: i8` + `d: f16`.
///
/// Each 6-bit element is reconstructed as `(ql[l] & 0xF) | ((qh >> s) &
/// 3) << 4`, biased by `-32`. Processed in 2 halves of 128 elements. Per
/// half, 4 sub-groups of 32 elements each use a `qh` shift of 0, 2, 4, 6.
///
/// INDEX: all hot-loop offsets are statically bounded by the half-count
/// walk: `ql_off + l + 32 < ql_off + 64 ≤ 128`, `qh_off + l < qh_off + 32
/// ≤ 64`, `sc_off + is + 6 < sc_off + 8 ≤ 16`, `y_off + l + 96 < 256`.
#[allow(
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
fn dequant_q6_k(data: &[u8], n_blocks: usize) -> crate::Result<Vec<u8>> {
    const TS: usize = 210;
    let mut out = vec![0u8; n_blocks * QK_K * 2];
    let mut scratch = [0.0_f32; QK_K];

    for b in 0..n_blocks {
        let block = data
            .get(b * TS..(b + 1) * TS)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q6_K: block {b} slice out of bounds"),
            })?;
        // Field offsets: ql [0..128], qh [128..192], scales [192..208], d [208..210]
        let ql_all = &block[0..128];
        let qh_all = &block[128..192];
        let sc_all = &block[192..208];
        let d = read_f16_le(block, 208)?;

        // PASS 1: walk 2 halves of 128 elements each. Within each half,
        // per position `l in 0..32`, extract four 6-bit values (at
        // l, l+32, l+64, l+96 in the output half) and scale them by
        // per-subblock scales `sc[is]`.
        let mut y_off: usize = 0;
        let mut ql_off: usize = 0;
        let mut qh_off: usize = 0;
        let mut sc_off: usize = 0;
        for _n in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                // BITWISE: 4-bit low + 2-bit high (shift 0,2,4,6 for four values)
                let q1 = i32::from(
                    (ql_all[ql_off + l] & 0x0F) | ((qh_all[qh_off + l] & 0x03) << 4),
                ) - 32;
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

        let out_block = out
            .get_mut(b * QK_K * 2..(b + 1) * QK_K * 2)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF Q6_K: output block {b} out of bounds"),
            })?;
        write_scratch_to_bf16(&scratch, out_block);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// K-quant scale extractors
// ---------------------------------------------------------------------------

/// Extracts the 6-bit `(scale, min)` pair for sub-block `j` from the
/// packed `scales[12]` field used by `Q4_K` and `Q5_K`.
///
/// Ported verbatim from ggml-quants.c's `get_scale_min_k4` helper. The
/// packing interleaves 8 scales and 8 mins across 12 bytes: for `j < 4`
/// they live in the low 6 bits of `scales[j]` and `scales[j+4]`; for
/// `j >= 4` the low 4 bits come from `scales[j+4]` and the top 2 bits
/// come from the upper 2 bits of `scales[j-4]` and `scales[j]`.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    // INDEX: all reads target a valid byte in `scales[12]`; the branches
    // on `j < 4` partition the input domain.
    #[allow(clippy::indexing_slicing)]
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Unpacks the 12-byte packed `scales` field of a `Q3_K` block into 16
/// signed 6-bit values (0..=63, still unsigned here — the caller biases
/// by `-32` in the dequant formula).
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
        // d = 1.0, qs = [-128..127] (cycle). Expect y[j] = qs[j].
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
        // 2 blocks: block 0 with d=2.0 all qs=1, block 1 with d=-1.0 all qs=3
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
        // d = 1.0, qs = all 0x00 → every nibble is 0, value = (0 - 8)*1 = -8
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q4_0, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), -8.0);
        }
    }

    #[test]
    fn q4_0_identity_nibble_round_trip() {
        // d = 1.0, qs[j] = (hi << 4) | lo where lo = j, hi = j (j < 16).
        // Expect y[j] = lo - 8 and y[j + 16] = hi - 8 = j - 8.
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
        // d = 1.0, m = 10.0, qs = all 0xF (high and low nibble = 15).
        // Expect y[j] = 15 * 1 + 10 = 25 for all 32 positions.
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
        // d = 1.0, qh = 0xFFFFFFFF (every 5th bit set), qs = 0x00 (low = 0).
        // For element j (0..16): xh_0 = ((0xFFFFFFFF >> j) << 4) & 0x10 = 0x10
        // 5-bit value = 0 | 0x10 = 16. y = (16 - 16) * 1 = 0.
        // For element j+16: xh_1 = (0xFFFFFFFF >> (j+12)) & 0x10 = 0x10. Same.
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..6].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        // qs[6..22] left as 0
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q5_0, 32).unwrap();
        for j in 0..32 {
            assert_eq!(bf16_pair_to_f32(&out[j * 2..j * 2 + 2]), 0.0);
        }
    }

    #[test]
    fn q5_1_identity_nibbles_no_high_bits() {
        // d = 1.0, m = 5.0, qh = 0 (no high bits), qs[j] = 0x11 (lo=1, hi=1)
        // 5-bit = 1, y = 1*1 + 5 = 6 for every element.
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&f16_bytes(1.0));
        block[2..4].copy_from_slice(&f16_bytes(5.0));
        // qh already 0
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
        // d=1.0, s=999.0 (should be ignored), qs = sequential i8
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
        // d = 0.5 (f32), qs[0..256] sequential i8 pattern.
        // bsums is ignored by the dequant loop.
        let mut block = vec![0u8; 292];
        block[0..4].copy_from_slice(&0.5_f32.to_le_bytes());
        for j in 0..256 {
            let v: i8 = ((j as i32) - 128) as i8;
            block[4 + j] = v as u8;
        }
        // bsums at [260..292]: leave as zero
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
        // All-zero block: scales=0, qs=0, d=0, dmin=0 → all output 0.
        let block = vec![0u8; 84];
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q2_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), 0.0);
        }
    }

    #[test]
    fn q2_k_uniform_scale_uniform_qs() {
        // d = 1.0, dmin = 0.0, scales[*] = 0x02 (sc_low = 2, sc_high = 0),
        // qs[*] = 0b01010101 = 0x55 (every 2-bit value = 1 at every shift)
        // y = 1.0 * 2 * 1 - 0.0 * 0 = 2.0 everywhere.
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
    // Q3_K: helper tests first, then full kernel
    // -----------------------------------------------------------------

    #[test]
    fn q3_k_unpack_scales_zero() {
        let scales = q3_k_unpack_scales(&[0u8; 12]);
        assert_eq!(scales, [0i8; 16]);
    }

    #[test]
    fn q3_k_unpack_scales_low_half_trivial() {
        // Set the first 4 bytes of the packed buffer to 0, 1, 2, 3 — these
        // should appear unchanged in the low nibble of the first 4 output
        // bytes (high bits come from bytes 8..12 which we leave zero).
        let mut packed = [0u8; 12];
        packed[0] = 0x00;
        packed[1] = 0x01;
        packed[2] = 0x02;
        packed[3] = 0x03;
        let scales = q3_k_unpack_scales(&packed);
        // The permute keeps `aux[0] & KMASK2` in the low nibble of bytes 0..3.
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
    // Q4_K: helper first, then full kernel
    // -----------------------------------------------------------------

    #[test]
    fn get_scale_min_k4_low_half() {
        // For j < 4, `get_scale_min_k4` returns (scales[j] & 63, scales[j+4] & 63).
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
        // All-zero `scales` must yield (0, 0) for every `j`.
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
        // d = 1.0, scales[*] = 1 (signed i8), ql/qh all 0.
        // 6-bit value = 0, biased = 0 - 32 = -32. y = 1*1*(-32) = -32.
        let mut block = vec![0u8; 210];
        // ql and qh already zero
        for j in 0..16 {
            block[192 + j] = 1; // signed scale of 1
        }
        block[208..210].copy_from_slice(&f16_bytes(1.0));
        let out = dequantize_gguf_to_bf16(&block, GgufType::Q6_K, 256).unwrap();
        for chunk in out.chunks_exact(2) {
            assert_eq!(bf16_pair_to_f32(chunk), -32.0);
        }
    }
}
