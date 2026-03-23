// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BitsAndBytes` dequantization (`NF4`/`FP4` 4-bit and `INT8`) to `BF16`.
//!
//! `NF4`/`FP4` uses a 16-entry lookup table + per-block absmax scaling.
//! `INT8` (`LLM.int8()`) uses per-row absmax with linear `I8` quantization.
//!
//! # References
//!
//! - Dettmers et al., "`LLM.int8()`: 8-bit Matrix Multiplication for
//!   Transformers at Scale", `NeurIPS` 2022 (`arXiv:2208.07339`)
//! - Dettmers et al., "`QLoRA`: Efficient Finetuning of Quantized Large
//!   Language Models", `NeurIPS` 2023 (`arXiv:2305.14314`)

use crate::error::AnamnesisError;
use crate::remember::fp8::f32_bits_to_bf16_bits;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reads a little-endian `f32` from a byte slice at the given offset.
///
/// # Errors
///
/// Returns `None` if the slice does not contain 4 bytes at `offset`.
fn read_f32_le(data: &[u8], offset: usize) -> Option<f32> {
    let bytes: &[u8] = data.get(offset..offset + 4)?;
    let arr: [u8; 4] = bytes.try_into().ok()?;
    Some(f32::from_le_bytes(arr))
}

// ---------------------------------------------------------------------------
// NF4/FP4 dequantization (4-bit, lookup-table based)
// ---------------------------------------------------------------------------

/// Dequantizes `BitsAndBytes` `NF4`/`FP4` quantized weights to `BF16`.
///
/// Each byte in `weight_data` packs two 4-bit values: low nibble first
/// (`byte & 0x0F`), high nibble second (`byte >> 4`). Each nibble indexes
/// into `quant_map_data` (a 16-entry `F32` lookup table). The looked-up
/// value is then scaled by the block's absmax.
///
/// # Arguments
///
/// - `weight_data` — `U8` bytes, two `NF4`/`FP4` values per byte.
/// - `absmax_data` — `F32` per-block absmax values.
/// - `quant_map_data` — `F32[16]` lookup table.
/// - `total_elements` — total number of dequantized elements (= weight bytes × 2).
/// - `block_size` — elements per absmax block (typically 64).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if tensor dimensions are inconsistent.
///
/// # Memory
///
/// Allocates `total_elements × 2` bytes for `BF16` output, plus a scratch
/// buffer of `block_size × 4` bytes for loop fission (fits in L1 cache).
pub fn dequantize_bnb4_to_bf16(
    weight_data: &[u8],
    absmax_data: &[u8],
    quant_map_data: &[u8],
    total_elements: usize,
    block_size: usize,
) -> crate::Result<Vec<u8>> {
    // --- Validation ---
    if block_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: "BnB block_size must be > 0".into(),
        });
    }
    let expected_weight_bytes =
        total_elements
            .checked_mul(1)
            .and_then(|n| if n % 2 == 0 { Some(n / 2) } else { None });
    if expected_weight_bytes != Some(weight_data.len()) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 weight byte count mismatch: expected {} for {} elements, got {}",
                expected_weight_bytes.unwrap_or(0),
                total_elements,
                weight_data.len()
            ),
        });
    }
    if !total_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 total_elements ({total_elements}) not divisible by block_size ({block_size})"
            ),
        });
    }
    let num_blocks = total_elements / block_size;
    let expected_absmax_bytes = num_blocks
        .checked_mul(4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "absmax byte count overflow".into(),
        })?;
    if absmax_data.len() != expected_absmax_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 absmax byte count mismatch: expected {expected_absmax_bytes}, got {}",
                absmax_data.len()
            ),
        });
    }
    // quant_map must be exactly 16 F32 values = 64 bytes
    if quant_map_data.len() != 64 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 quant_map must be 64 bytes (16×F32), got {}",
                quant_map_data.len()
            ),
        });
    }

    // --- Pre-load quant_map (16 entries) ---
    let mut quant_map = [0.0f32; 16];
    for (i, val) in quant_map.iter_mut().enumerate() {
        *val = read_f32_le(quant_map_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 quant_map read out of bounds".into(),
        })?;
    }

    // --- Allocate output ---
    let out_byte_len = total_elements
        .checked_mul(2)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 output byte count overflow".into(),
        })?;
    let mut output = vec![0u8; out_byte_len];

    // --- Per-block dequantization with loop fission ---
    let bytes_per_block = block_size / 2;
    // Scratch buffer for unpacked f32 values (one block at a time, fits in L1)
    let mut scratch = vec![0.0f32; block_size];

    for block_idx in 0..num_blocks {
        let absmax =
            read_f32_le(absmax_data, block_idx * 4).ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB4 absmax read out of bounds at block {block_idx}"),
            })?;

        // Pre-slice validated ranges (two-level bounds checking per CONVENTIONS.md)
        let w_start = block_idx * bytes_per_block;
        let w_end = w_start + bytes_per_block;
        let weight_block =
            weight_data
                .get(w_start..w_end)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("BnB4 weight block {block_idx} out of bounds"),
                })?;
        let o_start = block_idx * block_size * 2;
        let o_end = o_start + block_size * 2;
        let out_block = output
            .get_mut(o_start..o_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB4 output block {block_idx} out of bounds"),
            })?;

        // --- Pass 1 (unpack): byte → 2 nibbles → table lookup → f32 scratch ---
        // Each byte produces two f32 values via the quant_map lookup.
        // VECTORIZED: pass 1 extracts nibbles and performs table lookup;
        // pass 2 does pure f32 multiply + BF16 convert (verified: vmulps ymm
        // with target-cpu=native).
        // INDEX: scratch.len() == block_size, guaranteed by vec![0.0f32; block_size]
        #[allow(clippy::indexing_slicing)]
        let scratch_block = &mut scratch[..block_size];
        for (&byte, pair) in weight_block.iter().zip(scratch_block.chunks_exact_mut(2)) {
            // BITWISE: extract low nibble (bits [3:0]) and high nibble (bits [7:4])
            // CAST: u8 → usize, nibble values 0-15 used as lookup indices
            #[allow(clippy::as_conversions)]
            let low = (byte & 0x0F) as usize;
            #[allow(clippy::as_conversions)]
            let high = (byte >> 4) as usize;
            // INDEX: low and high are 0-15, quant_map has 16 entries
            #[allow(clippy::indexing_slicing)]
            {
                pair[0] = quant_map[low];
                pair[1] = quant_map[high];
            }
        }

        // --- Pass 2 (scale): f32 scratch × absmax → BF16 output ---
        // Pure float multiply + BF16 integer rounding — vectorizes to AVX2.
        // INDEX: scratch.len() == block_size, guaranteed by vec![0.0f32; block_size]
        #[allow(clippy::indexing_slicing)]
        let scratch_view = &scratch[..block_size];
        for (val, out_pair) in scratch_view.iter().zip(out_block.chunks_exact_mut(2)) {
            let scaled = val * absmax;
            let bf16 = f32_bits_to_bf16_bits(scaled.to_bits());
            out_pair.copy_from_slice(&bf16.to_le_bytes());
        }
    }

    Ok(output)
}

/// Dequantizes `BitsAndBytes` `NF4`/`FP4` with double quantization to `BF16`.
///
/// First dequantizes the nested absmax values (themselves quantized to `U8`),
/// then uses the recovered `F32` absmax values for the main `NF4`/`FP4` dequant.
///
/// # Arguments
///
/// - `weight_data` — `U8` bytes, two 4-bit values per byte.
/// - `absmax_data` — `U8` quantized absmax values (one per block).
/// - `quant_map_data` — `F32[16]` main lookup table.
/// - `nested_absmax_data` — `F32` absmax for the nested quantization.
/// - `nested_quant_map_data` — `F32[256]` lookup table for the nested quantization.
/// - `total_elements` — total number of dequantized elements.
/// - `block_size` — elements per absmax block (typically 64).
/// - `nested_block_size` — elements per nested absmax block (typically 256).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if tensor dimensions are inconsistent.
///
/// # Memory
///
/// Allocates `total_elements × 2` bytes for `BF16` output, plus intermediate
/// `F32` absmax array (`num_blocks × 4` bytes).
#[allow(clippy::too_many_arguments)]
pub fn dequantize_bnb4_double_quant_to_bf16(
    weight_data: &[u8],
    absmax_data: &[u8],
    quant_map_data: &[u8],
    nested_absmax_data: &[u8],
    nested_quant_map_data: &[u8],
    total_elements: usize,
    block_size: usize,
    nested_block_size: usize,
) -> crate::Result<Vec<u8>> {
    // --- Validation ---
    if block_size == 0 || nested_block_size == 0 {
        return Err(AnamnesisError::Parse {
            reason: "BnB block_size and nested_block_size must be > 0".into(),
        });
    }
    if !total_elements.is_multiple_of(block_size) {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 total_elements ({total_elements}) not divisible by block_size ({block_size})"
            ),
        });
    }
    let num_blocks = total_elements / block_size;
    if absmax_data.len() != num_blocks {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 double-quant: absmax byte count mismatch: expected {num_blocks}, got {}",
                absmax_data.len()
            ),
        });
    }
    // nested_quant_map must be exactly 256 F32 values = 1024 bytes
    if nested_quant_map_data.len() != 1024 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 nested_quant_map must be 1024 bytes (256×F32), got {}",
                nested_quant_map_data.len()
            ),
        });
    }

    // --- Pre-load nested quant_map (256 entries) ---
    let mut nested_quant_map = [0.0f32; 256];
    for (i, val) in nested_quant_map.iter_mut().enumerate() {
        *val = read_f32_le(nested_quant_map_data, i * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB4 nested_quant_map read out of bounds".into(),
        })?;
    }

    // --- Dequantize nested absmax: U8 → F32 via nested lookup × nested_absmax ---
    let num_nested_blocks = if num_blocks.is_multiple_of(nested_block_size) {
        num_blocks / nested_block_size
    } else {
        // Partial last block: round up
        num_blocks / nested_block_size + 1
    };
    let expected_nested_absmax_bytes =
        num_nested_blocks
            .checked_mul(4)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "nested absmax byte count overflow".into(),
            })?;
    if nested_absmax_data.len() != expected_nested_absmax_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB4 nested_absmax byte count mismatch: expected {expected_nested_absmax_bytes}, got {}",
                nested_absmax_data.len()
            ),
        });
    }

    let mut dequantized_absmax = vec![0.0f32; num_blocks];
    for (i, &absmax_byte) in absmax_data.iter().enumerate() {
        let nested_block_idx = i / nested_block_size;
        let nested_absmax_val =
            read_f32_le(nested_absmax_data, nested_block_idx * 4).ok_or_else(|| {
                AnamnesisError::Parse {
                    reason: format!(
                        "BnB4 nested_absmax read out of bounds at block {nested_block_idx}"
                    ),
                }
            })?;
        // CAST: u8 → usize, absmax_byte is 0-255 used as lookup index
        #[allow(clippy::as_conversions)]
        let idx = absmax_byte as usize;
        // INDEX: idx is 0-255, nested_quant_map has 256 entries;
        //        i < num_blocks, dequantized_absmax has num_blocks entries
        #[allow(clippy::indexing_slicing)]
        {
            dequantized_absmax[i] = nested_quant_map[idx] * nested_absmax_val;
        }
    }

    // --- Build F32 absmax byte slice from dequantized values ---
    let mut absmax_f32_bytes = vec![0u8; num_blocks * 4];
    for (&val, chunk) in dequantized_absmax
        .iter()
        .zip(absmax_f32_bytes.chunks_exact_mut(4))
    {
        chunk.copy_from_slice(&val.to_le_bytes());
    }

    // --- Delegate to plain NF4/FP4 dequant with recovered absmax ---
    dequantize_bnb4_to_bf16(
        weight_data,
        &absmax_f32_bytes,
        quant_map_data,
        total_elements,
        block_size,
    )
}

// ---------------------------------------------------------------------------
// INT8 dequantization (LLM.int8(), per-row absmax)
// ---------------------------------------------------------------------------

/// Dequantizes `BitsAndBytes` `INT8` (`LLM.int8()`) quantized weights to `BF16`.
///
/// Each `I8` weight value is dequantized via: `value = weight_i8 × (SCB / 127)`,
/// where `SCB` is the per-row absolute maximum.
///
/// # Arguments
///
/// - `weight_data` — `I8` bytes, one per element.
/// - `scb_data` — `F32` per-row absmax values (one per `out_features`).
/// - `out_features` — number of output rows.
/// - `in_features` — number of input columns.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if tensor dimensions are inconsistent.
///
/// # Memory
///
/// Allocates `out_features × in_features × 2` bytes for `BF16` output.
pub fn dequantize_bnb_int8_to_bf16(
    weight_data: &[u8],
    scb_data: &[u8],
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<u8>> {
    // --- Validation ---
    let total_elements =
        out_features
            .checked_mul(in_features)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "BnB INT8 element count overflow".into(),
            })?;
    if weight_data.len() != total_elements {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB INT8 weight byte count mismatch: expected {total_elements}, got {}",
                weight_data.len()
            ),
        });
    }
    let expected_scb_bytes = out_features
        .checked_mul(4)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "SCB byte count overflow".into(),
        })?;
    if scb_data.len() != expected_scb_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "BnB INT8 SCB byte count mismatch: expected {expected_scb_bytes}, got {}",
                scb_data.len()
            ),
        });
    }

    // --- Allocate output ---
    let out_byte_len = total_elements
        .checked_mul(2)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "BnB INT8 output byte count overflow".into(),
        })?;
    let mut output = vec![0u8; out_byte_len];

    // --- Per-row dequantization ---
    // Scale is constant per row → hoisted. Inner loop is 1:1 (I8 → BF16),
    // should vectorize without loop fission (like FP8 per-channel).
    // VECTORIZED: single-pass i8→f32 multiply + BF16 convert; verified
    // vcvtdq2ps + vmulps ymm with target-cpu=native.
    for row in 0..out_features {
        let scb_val = read_f32_le(scb_data, row * 4).ok_or_else(|| AnamnesisError::Parse {
            reason: format!("BnB INT8 SCB read out of bounds at row {row}"),
        })?;
        // Precompute row scale: SCB / 127.0
        let scale = scb_val / 127.0;

        // Pre-slice for branch-free inner loop (two-level bounds checking)
        let w_start = row * in_features;
        let w_end = w_start + in_features;
        let row_weights = weight_data
            .get(w_start..w_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB INT8 weight row {row} out of bounds"),
            })?;
        let o_start = row * in_features * 2;
        let o_end = o_start + in_features * 2;
        let out_row = output
            .get_mut(o_start..o_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("BnB INT8 output row {row} out of bounds"),
            })?;

        // Hot loop: 1:1 byte → BF16, scale hoisted, contiguous I/O
        for (&w_byte, out_pair) in row_weights.iter().zip(out_row.chunks_exact_mut(2)) {
            // CAST: u8 (from I8 two's complement) → i8 → f32
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let w_i8 = w_byte as i8;
            let w_f32 = f32::from(w_i8);
            let val = w_f32 * scale;
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
    clippy::float_cmp,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    /// Helper: build F32 LE bytes from a slice of f32 values.
    fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Helper: read a BF16 value from output bytes at element index.
    fn read_bf16(output: &[u8], idx: usize) -> f32 {
        let offset = idx * 2;
        let bits = u16::from_le_bytes([output[offset], output[offset + 1]]);
        let f32_bits = u32::from(bits) << 16;
        f32::from_bits(f32_bits)
    }

    // --- NF4/FP4 tests ---

    #[test]
    fn bnb4_uniform_lookup() {
        // All bytes = 0x00 → both nibbles = 0 → quant_map[0] * absmax
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);
        let block_size = 4;
        let num_bytes = 2; // 4 elements = 2 bytes
        let weight_data = vec![0x00u8; num_bytes];
        let absmax_bytes = f32_to_bytes(&[2.0]); // 1 block

        let out =
            dequantize_bnb4_to_bf16(&weight_data, &absmax_bytes, &quant_map_bytes, 4, block_size)
                .unwrap();

        // quant_map[0] = 0.0, so all outputs should be 0.0
        for i in 0..4 {
            assert_eq!(read_bf16(&out, i), 0.0, "element {i}");
        }
    }

    #[test]
    fn bnb4_nibble_extraction() {
        // Byte 0x31 → low nibble = 1, high nibble = 3
        // Byte 0x42 → low nibble = 2, high nibble = 4
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);
        let weight_data = vec![0x31, 0x42];
        let absmax_bytes = f32_to_bytes(&[1.0]); // 1 block, scale=1.0

        let out =
            dequantize_bnb4_to_bf16(&weight_data, &absmax_bytes, &quant_map_bytes, 4, 4).unwrap();

        // Element 0: quant_map[1] * 1.0 = 1.0
        assert_eq!(read_bf16(&out, 0), 1.0);
        // Element 1: quant_map[3] * 1.0 = 3.0
        assert_eq!(read_bf16(&out, 1), 3.0);
        // Element 2: quant_map[2] * 1.0 = 2.0
        assert_eq!(read_bf16(&out, 2), 2.0);
        // Element 3: quant_map[4] * 1.0 = 4.0
        assert_eq!(read_bf16(&out, 3), 4.0);
    }

    #[test]
    fn bnb4_absmax_scaling() {
        // quant_map[5] = 0.5, absmax = 4.0 → result = 2.0
        let mut quant_map = [0.0f32; 16];
        quant_map[5] = 0.5;
        let quant_map_bytes = f32_to_bytes(&quant_map);
        let weight_data = vec![0x55]; // both nibbles = 5
        let absmax_bytes = f32_to_bytes(&[4.0]);

        let out =
            dequantize_bnb4_to_bf16(&weight_data, &absmax_bytes, &quant_map_bytes, 2, 2).unwrap();

        assert_eq!(read_bf16(&out, 0), 2.0); // 0.5 * 4.0
        assert_eq!(read_bf16(&out, 1), 2.0);
    }

    #[test]
    fn bnb4_multi_block() {
        // Two blocks with different absmax values
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);
        // Block 0: byte 0x10 → nibbles 0,1; Block 1: byte 0x10 → nibbles 0,1
        let weight_data = vec![0x10, 0x10];
        let absmax_bytes = f32_to_bytes(&[1.0, 3.0]);

        let out = dequantize_bnb4_to_bf16(
            &weight_data,
            &absmax_bytes,
            &quant_map_bytes,
            4,
            2, // block_size = 2
        )
        .unwrap();

        // Block 0: quant_map[0]*1.0=0.0, quant_map[1]*1.0=1.0
        assert_eq!(read_bf16(&out, 0), 0.0);
        assert_eq!(read_bf16(&out, 1), 1.0);
        // Block 1: quant_map[0]*3.0=0.0, quant_map[1]*3.0=3.0
        assert_eq!(read_bf16(&out, 2), 0.0);
        assert_eq!(read_bf16(&out, 3), 3.0);
    }

    #[test]
    fn bnb4_validation_errors() {
        let quant_map_bytes = f32_to_bytes(&[0.0; 16]);
        let absmax_bytes = f32_to_bytes(&[1.0]);

        // block_size = 0
        assert!(dequantize_bnb4_to_bf16(&[0], &absmax_bytes, &quant_map_bytes, 2, 0).is_err());

        // Mismatched weight length
        assert!(dequantize_bnb4_to_bf16(&[0, 0], &absmax_bytes, &quant_map_bytes, 2, 2).is_err());

        // Wrong quant_map size
        assert!(dequantize_bnb4_to_bf16(&[0], &absmax_bytes, &[0; 32], 2, 2).is_err());
    }

    // --- Double-quant tests ---

    #[test]
    fn bnb4_double_quant_basic() {
        // Nested: absmax U8 value 2 → nested_quant_map[2] * nested_absmax[0]
        let quant_map: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let quant_map_bytes = f32_to_bytes(&quant_map);

        let mut nested_quant_map = [0.0f32; 256];
        nested_quant_map[2] = 0.5; // absmax byte=2 → lookup=0.5
        let nested_quant_map_bytes = f32_to_bytes(&nested_quant_map);

        let nested_absmax_bytes = f32_to_bytes(&[4.0]); // nested scale = 4.0
                                                        // Recovered absmax = nested_quant_map[2] * nested_absmax[0] = 0.5 * 4.0 = 2.0

        let absmax_data = vec![2u8]; // 1 block, absmax byte = 2
        let weight_data = vec![0x10]; // nibbles: 0, 1

        let out = dequantize_bnb4_double_quant_to_bf16(
            &weight_data,
            &absmax_data,
            &quant_map_bytes,
            &nested_absmax_bytes,
            &nested_quant_map_bytes,
            2,   // total_elements
            2,   // block_size
            256, // nested_block_size
        )
        .unwrap();

        // quant_map[0] * 2.0 = 0.0
        assert_eq!(read_bf16(&out, 0), 0.0);
        // quant_map[1] * 2.0 = 2.0
        assert_eq!(read_bf16(&out, 1), 2.0);
    }

    // --- INT8 tests ---

    #[test]
    fn bnb_int8_basic() {
        // 2×2 matrix, SCB = [127.0, 254.0]
        // weight_i8 = [[1, -1], [2, -2]]
        // dequant = weight_i8 * SCB[row] / 127.0
        let weight_data: Vec<u8> = vec![
            1u8,  // i8 = 1
            0xFF, // i8 = -1
            2u8,  // i8 = 2
            0xFE, // i8 = -2
        ];
        let scb_bytes = f32_to_bytes(&[127.0, 254.0]);

        let out = dequantize_bnb_int8_to_bf16(&weight_data, &scb_bytes, 2, 2).unwrap();

        // Row 0: scale = 127.0/127.0 = 1.0
        assert_eq!(read_bf16(&out, 0), 1.0); // 1 * 1.0
        assert_eq!(read_bf16(&out, 1), -1.0); // -1 * 1.0
                                              // Row 1: scale = 254.0/127.0 = 2.0
        assert_eq!(read_bf16(&out, 2), 4.0); // 2 * 2.0
        assert_eq!(read_bf16(&out, 3), -4.0); // -2 * 2.0
    }

    #[test]
    fn bnb_int8_zero_scale() {
        // SCB = 0.0 → all outputs should be 0.0
        let weight_data = vec![127u8, 1u8]; // i8 = 127, 1
        let scb_bytes = f32_to_bytes(&[0.0]);

        let out = dequantize_bnb_int8_to_bf16(&weight_data, &scb_bytes, 1, 2).unwrap();

        assert_eq!(read_bf16(&out, 0), 0.0);
        assert_eq!(read_bf16(&out, 1), 0.0);
    }

    #[test]
    fn bnb_int8_validation_errors() {
        let scb_bytes = f32_to_bytes(&[1.0]);

        // Mismatched weight length (2 elements but only 1 byte)
        assert!(dequantize_bnb_int8_to_bf16(&[0], &scb_bytes, 1, 2).is_err());

        // Mismatched SCB length (2 rows but only 1 SCB value)
        assert!(dequantize_bnb_int8_to_bf16(&[0; 4], &scb_bytes, 2, 2).is_err());
    }
}
