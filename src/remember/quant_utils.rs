// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers for GPTQ and AWQ dequantization.
//!
//! Both schemes pack quantized weights into `u32` words and use per-group
//! scale factors stored as `F16`, `BF16`, or `F32`. These utilities are
//! byte-for-byte identical across both modules, so they live here to avoid
//! duplication.

use crate::error::AnamnesisError;
use crate::parse::safetensors::Dtype;

/// Reads a little-endian `u32` from a byte slice at the given byte offset.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the slice is too short.
pub(crate) fn read_u32_le(data: &[u8], byte_offset: usize) -> crate::Result<u32> {
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
pub(crate) fn read_scale_f32(data: &[u8], byte_offset: usize, dtype: Dtype) -> crate::Result<f32> {
    match dtype {
        Dtype::F16 => {
            let end = byte_offset
                .checked_add(2)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "F16 scale byte offset overflow".into(),
                })?;
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
            let end = byte_offset
                .checked_add(2)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "BF16 scale byte offset overflow".into(),
                })?;
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
            let end = byte_offset
                .checked_add(4)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "F32 scale byte offset overflow".into(),
                })?;
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
            detail: "scale dtype must be F16, BF16, or F32".into(),
        }),
    }
}

/// Tile edge (in elements) for the cache-blocked `BF16` transpose.
///
/// 32×32 `u16` elements = 2 KiB per tile side — both the read and the
/// write tile fit in L1 regardless of the matrix's leading dimension.
const TRANSPOSE_TILE: usize = 32;

/// Transposes a row-major `[rows, cols]` `BF16` matrix to row-major
/// `[cols, rows]`.
///
/// Used by `ParsedModel::remember` / `remember_to_bytes` to convert the
/// GEMM-native `[in_features, out_features]` orientation the `GPTQ` / `AWQ`
/// dequant kernels produce (the canonical libraries' own kernel orientation,
/// which the cross-validation fixtures anchor) into the standard
/// `nn.Linear.weight` `[out_features, in_features]` orientation a
/// loadable-by-any-framework safetensors requires — the same boundary
/// transpose `GPTQModel`'s `dequantize_model` applies (`.T`) when assigning
/// to a plain `nn.Linear`.
///
/// Elements are copied as opaque `u16` words (no float interpretation), in
/// [`TRANSPOSE_TILE`]² cache-blocked tiles so both access streams stay
/// L1-resident.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if `rows × cols × 2` overflows or does
/// not equal `data.len()`.
///
/// # Memory
///
/// Allocates one `rows × cols × 2`-byte output buffer; the input is
/// unmodified. Callers replacing their input with the result hold both
/// buffers only for the duration of the call.
pub(crate) fn transpose_bf16(data: &[u8], rows: usize, cols: usize) -> crate::Result<Vec<u8>> {
    let n_elements = rows
        .checked_mul(cols)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "transpose element count overflow".into(),
        })?;
    let byte_len = n_elements
        .checked_mul(2)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "transpose byte count overflow".into(),
        })?;
    if data.len() != byte_len {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "transpose input length {} != rows × cols × 2 = {byte_len} \
                 (rows={rows}, cols={cols})",
                data.len()
            ),
        });
    }

    let mut output = vec![0u8; byte_len];

    // Cache-blocked transpose over TRANSPOSE_TILE² element tiles. All
    // indices below are bounded by `n_elements`: r < rows, c < cols, and
    // both `r * cols + c` and `c * rows + r` are < rows × cols, with the
    // ×2 byte offsets < byte_len (validated against data.len() above and
    // matching output.len() by construction).
    // EXPLICIT: imperative tile loops — the 2-D blocked traversal has no
    // iterator-chain equivalent that preserves the tiling.
    #[allow(clippy::indexing_slicing)]
    for row_tile in (0..rows).step_by(TRANSPOSE_TILE) {
        for col_tile in (0..cols).step_by(TRANSPOSE_TILE) {
            let row_end = (row_tile + TRANSPOSE_TILE).min(rows);
            let col_end = (col_tile + TRANSPOSE_TILE).min(cols);
            for r in row_tile..row_end {
                for c in col_tile..col_end {
                    // INDEX: src/dst element indices < rows × cols (see the
                    // block-level bound argument above)
                    let src = (r * cols + c) * 2;
                    let dst = (c * rows + r) * 2;
                    output[dst] = data[src];
                    output[dst + 1] = data[src + 1];
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    // -- read_scale_f32 ------------------------------------------------------

    #[test]
    fn read_scale_f16() {
        // F16 1.0 = 0x3C00
        let data = 0x3C00u16.to_le_bytes();
        let val = read_scale_f32(&data, 0, Dtype::F16).unwrap();
        assert_eq!(val, 1.0);
    }

    #[test]
    fn read_scale_bf16() {
        // BF16 1.0 = 0x3F80
        let data = 0x3F80u16.to_le_bytes();
        let val = read_scale_f32(&data, 0, Dtype::BF16).unwrap();
        assert_eq!(val, 1.0);
    }

    #[test]
    fn read_scale_f32_dtype() {
        let data = 2.0_f32.to_le_bytes();
        let val = read_scale_f32(&data, 0, Dtype::F32).unwrap();
        assert_eq!(val, 2.0);
    }

    // -- transpose_bf16 ------------------------------------------------------

    /// Build BF16 LE bytes where element k carries the value k (exact for
    /// the small test sizes used here).
    fn bf16_ramp(n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|k| {
                let bits = ((k as f32).to_bits() >> 16) as u16;
                bits.to_le_bytes()
            })
            .collect()
    }

    fn bf16_value(data: &[u8], idx: usize) -> f32 {
        let bits = u16::from_le_bytes([data[idx * 2], data[idx * 2 + 1]]);
        f32::from_bits(u32::from(bits) << 16)
    }

    #[test]
    fn transpose_maps_elements_exactly() {
        // 2×3 matrix [[0,1,2],[3,4,5]] → 3×2 [[0,3],[1,4],[2,5]].
        let data = bf16_ramp(6);
        let out = transpose_bf16(&data, 2, 3).unwrap();
        let expected = [0.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        for (idx, &exp) in expected.iter().enumerate() {
            assert_eq!(bf16_value(&out, idx), exp, "element {idx}");
        }
    }

    #[test]
    fn transpose_round_trips_non_square() {
        // Sizes straddling the 32-element tile edge, including odd remainders.
        for &(rows, cols) in &[(1usize, 7usize), (5, 33), (33, 5), (64, 96), (40, 40)] {
            let data = bf16_ramp(rows * cols);
            let once = transpose_bf16(&data, rows, cols).unwrap();
            let twice = transpose_bf16(&once, cols, rows).unwrap();
            assert_eq!(twice, data, "transpose² != identity for {rows}×{cols}");
        }
    }

    #[test]
    fn transpose_rejects_length_mismatch() {
        let data = bf16_ramp(6);
        assert!(transpose_bf16(&data, 2, 4).is_err());
        assert!(transpose_bf16(&data, 7, 1).is_err());
    }
}
