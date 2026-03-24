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
}
