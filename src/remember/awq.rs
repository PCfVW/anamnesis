// SPDX-License-Identifier: MIT OR Apache-2.0

//! `AWQ` dequantization (activation-aware, INT4/INT8 with per-group scales) to `BF16`.
//!
//! Converts packed integer weights with per-group scale factors and zero-points
//! into `BF16` output bytes. `AWQ` packs along `out_features` (columns), unlike
//! `GPTQ` which packs along `in_features` (rows). No `.g_idx` — groups are always
//! sequential.
//!
//! Reference: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
//! Compression and Acceleration", `MLSys` 2024 (arXiv:2306.00978).

use crate::error::AnamnesisError;
use crate::parse::safetensors::Dtype;
use crate::remember::fp8::f32_bits_to_bf16_bits;

// ---------------------------------------------------------------------------
// Helpers (reuse patterns from GPTQ)
// ---------------------------------------------------------------------------

/// Reads a little-endian `u32` from a byte slice at the given byte offset.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the slice is too short.
fn read_u32_le(data: &[u8], byte_offset: usize) -> crate::Result<u32> {
    let end = byte_offset
        .checked_add(4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "u32 byte offset overflow".into(),
        })?;
    let slice = data
        .get(byte_offset..end)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "u32 read out of bounds: need bytes {byte_offset}..{end}, have {}",
                data.len()
            ),
        })?;
    let arr: [u8; 4] = slice.try_into().map_err(|_| AnamnesisError::Parse {
        reason: "u32 slice is not 4 bytes".into(),
    })?;
    Ok(u32::from_le_bytes(arr))
}

/// Reads a scale factor as `f32` from a byte slice at the given byte offset.
///
/// Supports `F16`, `BF16`, and `F32` scale dtypes.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the slice is too short or the dtype
/// is unsupported for scale factors.
fn read_scale_f32(data: &[u8], byte_offset: usize, dtype: Dtype) -> crate::Result<f32> {
    match dtype {
        Dtype::F16 => {
            let end = byte_offset + 2;
            let slice = data
                .get(byte_offset..end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("F16 scale read out of bounds at offset {byte_offset}"),
                })?;
            let arr: [u8; 2] = slice.try_into().map_err(|_| AnamnesisError::Parse {
                reason: "F16 scale slice is not 2 bytes".into(),
            })?;
            // BITWISE: F16 → f32 via half crate's IEEE 754 conversion
            Ok(half::f16::from_le_bytes(arr).to_f32())
        }
        Dtype::BF16 => {
            let end = byte_offset + 2;
            let slice = data
                .get(byte_offset..end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("BF16 scale read out of bounds at offset {byte_offset}"),
                })?;
            let arr: [u8; 2] = slice.try_into().map_err(|_| AnamnesisError::Parse {
                reason: "BF16 scale slice is not 2 bytes".into(),
            })?;
            // BITWISE: BF16 → f32 by shifting into upper 16 bits of IEEE 754
            Ok(f32::from_bits(u32::from(u16::from_le_bytes(arr)) << 16))
        }
        Dtype::F32 => {
            let end = byte_offset + 4;
            let slice = data
                .get(byte_offset..end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("F32 scale read out of bounds at offset {byte_offset}"),
                })?;
            let arr: [u8; 4] = slice.try_into().map_err(|_| AnamnesisError::Parse {
                reason: "F32 scale slice is not 4 bytes".into(),
            })?;
            Ok(f32::from_le_bytes(arr))
        }
        Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::F64
        | Dtype::Bool
        | Dtype::U8
        | Dtype::I8
        | Dtype::U16
        | Dtype::I16
        | Dtype::U32
        | Dtype::I32
        | Dtype::U64
        | Dtype::I64 => Err(AnamnesisError::Unsupported {
            format: dtype.to_string(),
            detail: "AWQ scale dtype must be F16, BF16, or F32".into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Precomputation
// ---------------------------------------------------------------------------

/// Precompute all zero-point values (unpacked, NO +1 offset) as `f32`.
///
/// Returns a flat `Vec<f32>` of length `num_groups × out_features`.
///
/// Unlike `GPTQ`, `AWQ` does NOT add +1 to zero-points. The stored values
/// are used directly: `dequant = (qw - qz) × scale`.
///
/// `AWQ` packs `qzeros` along `out_features`: shape `[num_groups, out_features / pack_factor]`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `qzeros_data` is too short.
fn precompute_zeros(
    qzeros_data: &[u8],
    num_groups: usize,
    out_features: usize,
    bits: u8,
) -> crate::Result<Vec<f32>> {
    // CAST: u8 → u32, bits is 4 or 8
    #[allow(clippy::as_conversions)]
    let bits_u32 = u32::from(bits);
    // BITWISE: mask for one quantized value
    let mask = (1u32 << bits_u32) - 1;
    // CAST: u8 → usize, bits is 4 or 8
    #[allow(clippy::as_conversions)]
    let pack_factor = 32 / bits as usize;
    let packed_cols =
        out_features
            .checked_div(pack_factor)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "pack_factor is zero".into(),
            })?;

    let total = num_groups
        .checked_mul(out_features)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "num_groups × out_features overflow".into(),
        })?;
    let mut zeros = Vec::with_capacity(total);

    for g in 0..num_groups {
        for j in 0..out_features {
            let packed_col = j / pack_factor;
            let pos = j % pack_factor;
            // CAST: usize → u32, pos is at most 7 (4-bit) or 3 (8-bit)
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let shift = bits_u32 * (pos as u32);

            let byte_offset = (g * packed_cols + packed_col)
                .checked_mul(4)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "qzeros byte offset overflow".into(),
                })?;
            let packed = read_u32_le(qzeros_data, byte_offset)?;
            // BITWISE: extract unsigned zero-point — NO +1 offset (AWQ convention)
            let qz = (packed >> shift) & mask;
            // CAST: u32 → f32, qz is at most 15 (4-bit) or 255 (8-bit), exact in f32
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            let zero_f32 = qz as f32;
            zeros.push(zero_f32);
        }
    }

    Ok(zeros)
}

/// Precompute all scale factors as `f32` for all groups.
///
/// Returns a flat `Vec<f32>` of length `num_groups × out_features`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `scales_data` is too short.
fn precompute_scales(
    scales_data: &[u8],
    num_groups: usize,
    out_features: usize,
    scale_dtype: Dtype,
) -> crate::Result<Vec<f32>> {
    let bps = scale_dtype.byte_size();
    let total = num_groups
        .checked_mul(out_features)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "num_groups × out_features overflow".into(),
        })?;
    let mut scales = Vec::with_capacity(total);

    for idx in 0..total {
        let byte_offset = idx.checked_mul(bps).ok_or_else(|| AnamnesisError::Parse {
            reason: "scale byte offset overflow".into(),
        })?;
        scales.push(read_scale_f32(scales_data, byte_offset, scale_dtype)?);
    }

    Ok(scales)
}

// ---------------------------------------------------------------------------
// Main dequantization (public API)
// ---------------------------------------------------------------------------

/// Dequantizes an `AWQ`-quantized weight tensor to `BF16`.
///
/// `AWQ` packs along `out_features` (columns): `qweight` shape is
/// `[in_features, out_features / pack_factor]`. The dequantization formula is:
/// `dequant[i, j] = (qweight[i, j] - qzeros[g, j]) × scales[g, j]`
///
/// Unlike `GPTQ`, there is no +1 offset on zero-points and no `g_idx` —
/// groups are always sequential (`g = i / group_size`).
///
/// # Arguments
///
/// * `qweight_data` — packed `I32` weight bytes, row-major `[in_features, out_features/pack_factor]`.
/// * `scales_data` — scale factor bytes, row-major `[num_groups, out_features]`.
/// * `qzeros_data` — packed `I32` zero-point bytes, row-major `[num_groups, out_features/pack_factor]`.
/// * `in_features` — number of input features (weight rows).
/// * `out_features` — number of output features (unpacked weight columns).
/// * `group_size` — number of input features per group (typically 128).
/// * `bits` — quantization bit width (4 or 8).
/// * `scale_dtype` — dtype of the scales tensor (`F16`, `BF16`, or `F32`).
///
/// # Returns
///
/// A `Vec<u8>` of length `in_features × out_features × 2`, containing `BF16`
/// values in little-endian byte order. Shape: `[in_features, out_features]`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if tensor dimensions are inconsistent.
/// Returns [`AnamnesisError::Unsupported`] if the bit width or scale dtype
/// is not supported.
///
/// # Memory
///
/// Allocates precomputed zero-point and scale arrays (`num_groups × out_features × 4`
/// bytes each), a scratch buffer (`out_features × 4` bytes), plus the output
/// buffer (`in_features × out_features × 2` bytes).
#[allow(clippy::too_many_arguments)]
pub fn dequantize_awq_to_bf16(
    qweight_data: &[u8],
    scales_data: &[u8],
    qzeros_data: &[u8],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    bits: u8,
    scale_dtype: Dtype,
) -> crate::Result<Vec<u8>> {
    // --- Validate bit width ---
    if bits != 4 && bits != 8 {
        return Err(AnamnesisError::Unsupported {
            format: "AWQ".into(),
            detail: format!("{bits}-bit quantization not supported (expected 4 or 8)"),
        });
    }

    // CAST: u8 → usize, bits is 4 or 8
    #[allow(clippy::as_conversions)]
    let pack_factor = 32 / bits as usize;

    // --- Validate dimensions ---
    if in_features == 0 || out_features == 0 || group_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "zero dimension: in_features={in_features}, out_features={out_features}, \
                 group_size={group_size}"
            ),
        });
    }
    if !out_features.is_multiple_of(pack_factor) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "out_features {out_features} is not a multiple of pack_factor {pack_factor}"
            ),
        });
    }
    if !in_features.is_multiple_of(group_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "in_features {in_features} is not a multiple of group_size {group_size}"
            ),
        });
    }

    let packed_cols = out_features / pack_factor;
    let num_groups = in_features / group_size;

    // --- Validate tensor sizes ---
    let expected_qw_len = in_features
        .checked_mul(packed_cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "qweight byte length overflow".into(),
        })?;
    if qweight_data.len() != expected_qw_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "qweight data length {} != expected {expected_qw_len}",
                qweight_data.len()
            ),
        });
    }

    let expected_scales_len = num_groups
        .checked_mul(out_features)
        .and_then(|n| n.checked_mul(scale_dtype.byte_size()))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "scales byte length overflow".into(),
        })?;
    if scales_data.len() != expected_scales_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "scales data length {} != expected {expected_scales_len}",
                scales_data.len()
            ),
        });
    }

    let expected_qzeros_len = num_groups
        .checked_mul(packed_cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "qzeros byte length overflow".into(),
        })?;
    if qzeros_data.len() != expected_qzeros_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "qzeros data length {} != expected {expected_qzeros_len}",
                qzeros_data.len()
            ),
        });
    }

    // --- Precompute group data ---
    let all_zeros = precompute_zeros(qzeros_data, num_groups, out_features, bits)?;
    let all_scales = precompute_scales(scales_data, num_groups, out_features, scale_dtype)?;

    // --- Allocate output + scratch buffer ---
    let out_byte_len = in_features
        .checked_mul(out_features)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "output size overflow".into(),
        })?;
    let mut output = vec![0u8; out_byte_len];

    // Scratch buffer for unpacked f32 values (one row, reused across iterations).
    let mut unpacked_buf = vec![0.0_f32; out_features];

    // --- Precompute constants ---
    // CAST: u8 → u32, bits is 4 or 8
    #[allow(clippy::as_conversions)]
    let bits_u32 = u32::from(bits);
    // BITWISE: mask for one quantized value
    let mask = (1u32 << bits_u32) - 1;

    // --- Hot loop: row-by-row dequantization ---
    // Two-level bounds checking per CONVENTIONS.md: validate slices ONCE
    // before the inner loop, then iterate branch-free inside.
    // Loop fission per CONVENTIONS.md: separate byte unpacking from f32 arithmetic.
    for i in 0..in_features {
        let g = i / group_size;

        // --- Pre-validate slices ONCE ---

        // qweight row: packed_cols contiguous I32 values for this row.
        // AWQ packs along out_features: qweight[i, 0..packed_cols]
        let qw_row_start = i
            .checked_mul(packed_cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "qweight row byte offset overflow".into(),
            })?;
        let qw_row_end = qw_row_start + packed_cols * 4;
        let qw_row =
            qweight_data
                .get(qw_row_start..qw_row_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("qweight row {i} out of bounds"),
                })?;

        // Group zeros and scales: contiguous f32 slices, pre-validated.
        let group_start = g * out_features;
        let group_end = group_start + out_features;
        let zeros_row =
            all_zeros
                .get(group_start..group_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("zeros row for group {g} out of bounds"),
                })?;
        let scales_row =
            all_scales
                .get(group_start..group_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("scales row for group {g} out of bounds"),
                })?;

        // Output row.
        let out_row_start = i * out_features * 2;
        let out_row_end = out_row_start + out_features * 2;
        let out_row =
            output
                .get_mut(out_row_start..out_row_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("output row {i} out of bounds"),
                })?;

        // Scratch buffer.
        let unpacked_row =
            unpacked_buf
                .get_mut(..out_features)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "unpacked buffer too short".into(),
                })?;

        // --- Pass 1: unpack qweight row → f32 scratch buffer ---
        // AWQ packs along out_features: each I32 contains pack_factor consecutive
        // output features. Unpack all of them.
        #[allow(clippy::indexing_slicing)]
        for (packed_col, qw_chunk) in qw_row.chunks_exact(4).enumerate() {
            // INDEX: chunks_exact(4) guarantees exactly 4 bytes per chunk
            let packed = u32::from_le_bytes([qw_chunk[0], qw_chunk[1], qw_chunk[2], qw_chunk[3]]);

            // Unpack pack_factor values from this I32
            for pos in 0..pack_factor {
                // CAST: usize → u32, pos is at most 7 (4-bit) or 3 (8-bit)
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                let shift = bits_u32 * (pos as u32);
                // BITWISE: extract unsigned quantized value
                let qw = (packed >> shift) & mask;
                // CAST: u32 → f32, exact for values ≤ 255
                #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
                let qw_f32 = qw as f32;
                // INDEX: packed_col * pack_factor + pos < out_features
                // (packed_col < packed_cols, pos < pack_factor, packed_cols * pack_factor = out_features)
                unpacked_row[packed_col * pack_factor + pos] = qw_f32;
            }
        }

        // --- Pass 2: pure f32 arithmetic → BF16 output (VECTORIZES to AVX2) ---
        // Contiguous f32 reads (unpacked, zeros, scales) and contiguous BF16 writes.
        // No byte manipulation — just sub + mul + bf16 convert.
        // VECTORIZED: pending cargo-show-asm verification
        for (((out_pair, &qw), &zero), &scale) in out_row
            .chunks_exact_mut(2)
            .zip(unpacked_row.iter())
            .zip(zeros_row.iter())
            .zip(scales_row.iter())
        {
            let val = (qw - zero) * scale;
            let bf16 = f32_bits_to_bf16_bits(val.to_bits());
            out_pair.copy_from_slice(&bf16.to_le_bytes());
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    // -- Small dequantization with known values ------------------------------

    /// Build a minimal AWQ 4-bit test case.
    ///
    /// 8 input features, 8 output features, `group_size`=8, `bits`=4.
    /// `AWQ` packs along `out_features`: `qweight` shape `[8, 1]` (8/8=1 packed col).
    /// All qweight nibbles = 10, all qzeros nibbles = 3 (NO +1), all scales = 2.0.
    ///
    /// Expected: (10 - 3) × 2.0 = 14.0 for every element.
    #[test]
    fn dequant_4bit_uniform() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let bits: u8 = 4;

        // qweight: [8, 1] (8 rows, 1 packed col per row)
        // Each I32 packs 8 output features. All nibbles = 10 (0xA).
        let qweight_i32 = 0xAAAA_AAAAu32;
        let mut qweight_data = Vec::new();
        for _i in 0..in_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: [1, 8], all 2.0 in F16
        let scale_f16 = half::f16::from_f32(2.0).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_f16);
        }

        // qzeros: [1, 1] (1 group, 1 packed col)
        // All nibbles = 3 (no +1 offset for AWQ)
        let qzeros_i32 = 0x3333_3333u32;
        let qzeros_data = qzeros_i32.to_le_bytes().to_vec();

        let output = dequantize_awq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        assert_eq!(output.len(), in_features * out_features * 2);

        // Expected: (10 - 3) × 2.0 = 14.0 → BF16 0x4160 → LE [0x60, 0x41]
        let bf16_14 = f32_bits_to_bf16_bits(14.0_f32.to_bits());
        for chunk in output.chunks_exact(2) {
            let actual = u16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(actual, bf16_14, "expected BF16 14.0");
        }
    }

    /// Verify that AWQ does NOT apply the +1 zero-point offset that GPTQ uses.
    #[test]
    fn dequant_4bit_no_plus_one_offset() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let bits: u8 = 4;

        // qweight: all nibbles = 5
        let qweight_i32 = 0x5555_5555u32;
        let mut qweight_data = Vec::new();
        for _i in 0..in_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: all 1.0 in F16 (identity scale)
        let scale_f16 = half::f16::from_f32(1.0).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_f16);
        }

        // qzeros: all nibbles = 3
        let qzeros_i32 = 0x3333_3333u32;
        let qzeros_data = qzeros_i32.to_le_bytes().to_vec();

        let output = dequantize_awq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        // AWQ: (5 - 3) × 1.0 = 2.0 (NOT (5 - 4) × 1.0 = 1.0 like GPTQ would give)
        let bf16_2 = f32_bits_to_bf16_bits(2.0_f32.to_bits());
        for chunk in output.chunks_exact(2) {
            let actual = u16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(actual, bf16_2, "expected BF16 2.0 (AWQ: no +1 offset)");
        }
    }

    #[test]
    fn dequant_8bit_uniform() {
        let in_features = 4;
        let out_features = 4;
        let group_size = 4;
        let bits: u8 = 8;

        // qweight: [4, 1] (4 rows, 4/4=1 packed col)
        // Each I32 packs 4 output features. All bytes = 100.
        let qweight_i32 = 0x6464_6464u32; // 0x64 = 100
        let mut qweight_data = Vec::new();
        for _i in 0..in_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: [1, 4], all 0.5 in F16
        let scale_half = half::f16::from_f32(0.5).to_le_bytes();
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&scale_half);
        }

        // qzeros: [1, 1] (1 group, 1 packed col)
        // All bytes = 50
        let qzeros_i32 = 0x3232_3232u32; // 0x32 = 50
        let qzeros_data = qzeros_i32.to_le_bytes().to_vec();

        let output = dequantize_awq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        // Expected: (100 - 50) × 0.5 = 25.0
        let bf16_25 = f32_bits_to_bf16_bits(25.0_f32.to_bits());
        for chunk in output.chunks_exact(2) {
            let actual = u16::from_le_bytes([chunk[0], chunk[1]]);
            assert_eq!(actual, bf16_25, "expected BF16 25.0");
        }
    }

    #[test]
    fn dequant_4bit_two_groups() {
        let in_features = 8;
        let out_features = 8;
        let group_size = 4;
        let bits: u8 = 4;

        // 2 groups: rows 0-3 → group 0, rows 4-7 → group 1
        // qweight: all nibbles = 8
        let qweight_i32 = 0x8888_8888u32;
        let mut qweight_data = Vec::new();
        for _i in 0..in_features {
            qweight_data.extend_from_slice(&qweight_i32.to_le_bytes());
        }

        // scales: [2, 8]
        // Group 0: scale = 1.0, Group 1: scale = 3.0
        let mut scales_data = Vec::new();
        for _ in 0..out_features {
            scales_data.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
        for _ in 0..out_features {
            scales_data.extend_from_slice(&half::f16::from_f32(3.0).to_le_bytes());
        }

        // qzeros: [2, 1]
        // Group 0: zero = 6, Group 1: zero = 2
        let qz_g0 = 0x6666_6666u32;
        let qz_g1 = 0x2222_2222u32;
        let mut qzeros_data = Vec::new();
        qzeros_data.extend_from_slice(&qz_g0.to_le_bytes());
        qzeros_data.extend_from_slice(&qz_g1.to_le_bytes());

        let output = dequantize_awq_to_bf16(
            &qweight_data,
            &scales_data,
            &qzeros_data,
            in_features,
            out_features,
            group_size,
            bits,
            Dtype::F16,
        )
        .unwrap();

        // Group 0 (rows 0-3): (8 - 6) × 1.0 = 2.0
        let bf16_2 = f32_bits_to_bf16_bits(2.0_f32.to_bits());
        for i in 0..4 {
            for j in 0..out_features {
                let offset = (i * out_features + j) * 2;
                let actual = u16::from_le_bytes([output[offset], output[offset + 1]]);
                assert_eq!(actual, bf16_2, "element [{i},{j}]: expected BF16 2.0");
            }
        }

        // Group 1 (rows 4-7): (8 - 2) × 3.0 = 18.0
        let bf16_18 = f32_bits_to_bf16_bits(18.0_f32.to_bits());
        for i in 4..8 {
            for j in 0..out_features {
                let offset = (i * out_features + j) * 2;
                let actual = u16::from_le_bytes([output[offset], output[offset + 1]]);
                assert_eq!(actual, bf16_18, "element [{i},{j}]: expected BF16 18.0");
            }
        }
    }

    // -- Validation errors ---------------------------------------------------

    #[test]
    fn validation_unsupported_bits() {
        let result = dequantize_awq_to_bf16(&[], &[], &[], 8, 8, 8, 3, Dtype::F16);
        assert!(result.is_err());
    }

    #[test]
    fn validation_zero_dimensions() {
        let result = dequantize_awq_to_bf16(&[], &[], &[], 0, 8, 8, 4, Dtype::F16);
        assert!(result.is_err());
    }

    #[test]
    fn validation_qweight_length_mismatch() {
        // in_features=8, out_features=8, bits=4 → qweight should be 8×1×4=32 bytes
        let result = dequantize_awq_to_bf16(
            &[0u8; 16], // wrong: should be 32
            &[0u8; 16], // scales: 1 group × 8 × 2 = 16
            &[0u8; 4],  // qzeros: 1 group × 1 × 4 = 4
            8,
            8,
            8,
            4,
            Dtype::F16,
        );
        assert!(result.is_err());
    }
}
