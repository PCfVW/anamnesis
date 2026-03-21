// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-level parse-first API.
//!
//! [`parse`] reads a `.safetensors` file, returning a [`ParsedModel`] that
//! holds both the header metadata and the raw byte data. All subsequent
//! operations ([`ParsedModel::inspect`], [`ParsedModel::remember`]) work
//! from this parsed representation — no file is re-read.

use std::fmt;
use std::path::Path;

use crate::error::AnamnesisError;
use crate::inspect::InspectInfo;
use crate::parse::safetensors::{
    parse_safetensors_header, Dtype, QuantScheme, SafetensorsHeader, TensorRole,
};
use crate::remember::fp8::{dequantize_fp8_to_bf16, dequantize_per_tensor_fp8_to_bf16};

/// Target dtype for dequantization output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TargetDtype {
    /// `BF16` (bfloat16) — 2 bytes per element. The standard research/training dtype.
    BF16,
}

impl fmt::Display for TargetDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BF16 => f.write_str("BF16"),
        }
    }
}

/// A parsed `.safetensors` model, holding both header metadata and raw tensor
/// data in memory.
///
/// Created by [`parse`]. All operations work from this in-memory representation.
pub struct ParsedModel {
    /// Parsed header metadata (tensor names, dtypes, shapes, roles, scheme).
    pub header: SafetensorsHeader,
    /// Raw file bytes. Tensor data starts at offset `header_size + 8`.
    buffer: Vec<u8>,
}

/// Parses a `.safetensors` file, returning a [`ParsedModel`] holding both
/// header metadata and raw tensor data.
///
/// This is the entry point for all anamnesis operations. The file is read
/// once; all subsequent operations ([`ParsedModel::inspect`],
/// [`ParsedModel::remember`]) work from the in-memory representation.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be read.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
pub fn parse(path: impl AsRef<Path>) -> crate::Result<ParsedModel> {
    let buffer = std::fs::read(path.as_ref())?;
    let header = parse_safetensors_header(&buffer)?;
    Ok(ParsedModel { header, buffer })
}

impl ParsedModel {
    /// Returns inspection info (format, tensor counts, size estimates).
    ///
    /// Delegates to [`InspectInfo::from`]. No I/O — purely derived from
    /// the parsed header.
    pub fn inspect(&self) -> InspectInfo {
        InspectInfo::from(&self.header)
    }

    /// Returns the raw bytes for a tensor from the in-memory buffer.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the tensor's data offsets are
    /// out of bounds.
    fn tensor_data(&self, start: usize, end: usize) -> crate::Result<&[u8]> {
        let data_offset = self.header.header_size + 8;
        let abs_start = data_offset
            .checked_add(start)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "tensor data start offset overflow".into(),
            })?;
        let abs_end = data_offset
            .checked_add(end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "tensor data end offset overflow".into(),
            })?;
        self.buffer
            .get(abs_start..abs_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "tensor data offsets {abs_start}..{abs_end} out of bounds (buffer len {})",
                    self.buffer.len()
                ),
            })
    }

    /// Reads a scalar scale value from raw bytes, handling both `F32` and
    /// `BF16` scale dtypes.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the data is too short for the
    /// given dtype.
    /// Returns [`AnamnesisError::Unsupported`] if the scale dtype is not
    /// `F32` or `BF16`.
    fn read_scalar_scale(data: &[u8], dtype: Dtype, weight_name: &str) -> crate::Result<f32> {
        match dtype {
            Dtype::F32 => {
                let arr: [u8; 4] =
                    data.get(..4)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!(
                                "per-tensor F32 scale for `{weight_name}` is not 4 bytes"
                            ),
                        })?;
                Ok(f32::from_le_bytes(arr))
            }
            Dtype::BF16 => {
                let arr: [u8; 2] =
                    data.get(..2)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!(
                                "per-tensor BF16 scale for `{weight_name}` is not 2 bytes"
                            ),
                        })?;
                // BITWISE: BF16 → f32 by shifting into upper 16 bits of IEEE 754
                Ok(f32::from_bits(u32::from(u16::from_le_bytes(arr)) << 16))
            }
            Dtype::F8E4M3
            | Dtype::F8E5M2
            | Dtype::F16
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
                detail: format!("per-tensor scale for `{weight_name}` has unsupported dtype"),
            }),
        }
    }

    /// Extracts `(rows, cols)` from a tensor shape for the fine-grained
    /// dequantization function.
    ///
    /// - 2D: `(shape[0], shape[1])`
    /// - >2D: `(product of all dims except last, last dim)`
    fn shape_to_rows_cols(shape: &[usize]) -> crate::Result<(usize, usize)> {
        match shape.len() {
            0 | 1 => Err(AnamnesisError::Parse {
                reason: format!(
                    "quantized tensor has {}-D shape, expected >= 2D",
                    shape.len()
                ),
            }),
            2 => {
                let rows = shape.first().copied().unwrap_or(0);
                let cols = shape.get(1).copied().unwrap_or(0);
                Ok((rows, cols))
            }
            _ => {
                let cols = shape.last().copied().unwrap_or(0);
                let rows: usize = shape.get(..shape.len() - 1).unwrap_or(&[]).iter().product();
                Ok((rows, cols))
            }
        }
    }

    /// Dequantizes all quantized tensors and writes a standard `.safetensors`
    /// file loadable by any Rust ML framework.
    ///
    /// - **Quantized tensors**: dequantized to the target dtype using the
    ///   detected quantization scheme and companion scale factors.
    /// - **Scale tensors**: consumed during dequantization, not written.
    /// - **Passthrough tensors**: copied as-is (zero-copy from the buffer).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    ///
    /// # Memory
    ///
    /// Peak memory is roughly `input_file_size + dequantized_output_size`
    /// (~3× the `FP8` file size when dequantizing to `BF16`). The input
    /// buffer is held for the duration of the call (passthrough tensors
    /// borrow from it). Dequantized tensors are allocated individually
    /// and freed after serialization.
    pub fn remember(
        &self,
        output_path: impl AsRef<Path>,
        target: TargetDtype,
    ) -> crate::Result<()> {
        match target {
            TargetDtype::BF16 => self.remember_bf16(output_path.as_ref()),
        }
    }

    /// Internal: dequantize to `BF16` and write.
    fn remember_bf16(&self, output_path: &Path) -> crate::Result<()> {
        // Collect dequantized data (owned) for quantized tensors.
        // Passthrough tensors borrow from self.buffer.
        let mut dequantized_data: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        let mut passthrough_refs: Vec<(&str, &[u8], &[usize])> = Vec::new();

        for entry in &self.header.tensors {
            match entry.role {
                TensorRole::Quantized => {
                    let weight_data =
                        self.tensor_data(entry.data_offsets.0, entry.data_offsets.1)?;

                    let bf16_bytes = match self.header.scheme {
                        QuantScheme::FineGrainedFp8 => {
                            let scale_entry =
                                self.header.find_scale_for(&entry.name).ok_or_else(|| {
                                    AnamnesisError::Parse {
                                        reason: format!(
                                            "no scale tensor found for quantized weight `{}`",
                                            entry.name
                                        ),
                                    }
                                })?;
                            let scale_data = self.tensor_data(
                                scale_entry.data_offsets.0,
                                scale_entry.data_offsets.1,
                            )?;
                            let (rows, cols) = Self::shape_to_rows_cols(&entry.shape)?;
                            dequantize_fp8_to_bf16(
                                weight_data,
                                scale_data,
                                rows,
                                cols,
                                scale_entry.dtype,
                            )?
                        }
                        QuantScheme::PerTensorFp8 => {
                            // Look for a companion scale tensor; default to 1.0 if none.
                            let scale = if let Some(scale_entry) =
                                self.header.find_scale_for(&entry.name)
                            {
                                let scale_data = self.tensor_data(
                                    scale_entry.data_offsets.0,
                                    scale_entry.data_offsets.1,
                                )?;
                                Self::read_scalar_scale(scale_data, scale_entry.dtype, &entry.name)?
                            } else {
                                1.0
                            };
                            dequantize_per_tensor_fp8_to_bf16(weight_data, scale)
                        }
                        QuantScheme::Unquantized => {
                            // Shouldn't have quantized tensors in an unquantized model,
                            // but treat as passthrough to be safe.
                            passthrough_refs.push((&entry.name, weight_data, &entry.shape));
                            continue;
                        }
                    };

                    dequantized_data.push((entry.name.clone(), bf16_bytes, entry.shape.clone()));
                }
                TensorRole::Scale => {
                    // Scale tensors are consumed during dequantization; skip.
                }
                TensorRole::Passthrough => {
                    let data = self.tensor_data(entry.data_offsets.0, entry.data_offsets.1)?;
                    passthrough_refs.push((&entry.name, data, &entry.shape));
                }
            }
        }

        // Build TensorView list for serialization.
        // Dequantized tensors use safetensors::Dtype::BF16.
        // Passthrough tensors keep their original dtype.
        let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();

        for (name, data, shape) in &dequantized_data {
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.clone(), data)
                    .map_err(|e| AnamnesisError::Parse {
                        reason: format!("failed to create TensorView for `{name}`: {e}"),
                    })?;
            views.push((name.clone(), view));
        }

        for &(name, data, shape) in &passthrough_refs {
            // Look up the original dtype for this passthrough tensor.
            let entry = self
                .header
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("passthrough tensor `{name}` not found in header"),
                })?;
            let st_dtype = entry.dtype.to_safetensors_dtype()?;
            let view = safetensors::tensor::TensorView::new(st_dtype, shape.to_vec(), data)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("failed to create TensorView for `{name}`: {e}"),
                })?;
            views.push((name.to_owned(), view));
        }

        // Serialize to file.
        let metadata = self.header.metadata.clone();
        safetensors::tensor::serialize_to_file(views, &metadata, output_path).map_err(|e| {
            AnamnesisError::Parse {
                reason: format!("failed to write safetensors file: {e}"),
            }
        })?;

        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    /// Build a minimal safetensors file in memory with the given tensors.
    fn build_safetensors(tensors: &[(&str, safetensors::Dtype, &[usize], &[u8])]) -> Vec<u8> {
        let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                let view =
                    safetensors::tensor::TensorView::new(*dtype, shape.to_vec(), data).unwrap();
                (*name, view)
            })
            .collect();
        safetensors::tensor::serialize(views, &None).unwrap()
    }

    #[test]
    fn parse_and_inspect_unquantized() {
        // 2 BF16 tensors
        let bf16_data = vec![0x80, 0x3F]; // BF16 1.0
        let file = build_safetensors(&[
            ("weight", safetensors::Dtype::BF16, &[1], &bf16_data),
            ("norm", safetensors::Dtype::BF16, &[1], &bf16_data),
        ]);

        let tmp = std::env::temp_dir().join("test_unquant.safetensors");
        std::fs::write(&tmp, &file).unwrap();

        let model = parse(&tmp).unwrap();
        let info = model.inspect();

        assert_eq!(info.format, QuantScheme::Unquantized);
        assert_eq!(info.quantized, 0);
        assert_eq!(info.passthrough, 2);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn parse_nonexistent_file() {
        let result = parse("/tmp/nonexistent_anamnesis_test.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn parse_invalid_data() {
        let tmp = std::env::temp_dir().join("test_invalid.safetensors");
        std::fs::write(&tmp, b"not a safetensors file").unwrap();

        let result = parse(&tmp);
        assert!(result.is_err());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn remember_passthrough_only() {
        // BF16 tensor with known value: 2.0 = 0x4000 in BF16
        let bf16_data = vec![0x00, 0x40, 0x00, 0x40]; // two BF16 2.0
        let file = build_safetensors(&[("weight", safetensors::Dtype::BF16, &[2], &bf16_data)]);

        let tmp_in = std::env::temp_dir().join("test_pass_in.safetensors");
        let tmp_out = std::env::temp_dir().join("test_pass_out.safetensors");
        std::fs::write(&tmp_in, &file).unwrap();

        let model = parse(&tmp_in).unwrap();
        model.remember(&tmp_out, TargetDtype::BF16).unwrap();

        // Read output and verify bytes match
        let out_data = std::fs::read(&tmp_out).unwrap();
        let out_model = parse(&tmp_out).unwrap();
        let out_info = out_model.inspect();
        assert_eq!(out_info.passthrough, 1);

        // Verify the tensor data is preserved
        let entry = &out_model.header.tensors[0];
        let data_offset = out_model.header.header_size + 8;
        let tensor_bytes =
            &out_data[data_offset + entry.data_offsets.0..data_offset + entry.data_offsets.1];
        assert_eq!(tensor_bytes, &bf16_data);

        std::fs::remove_file(&tmp_in).ok();
        std::fs::remove_file(&tmp_out).ok();
    }

    #[test]
    fn remember_fp8_round_trip() {
        // Build a safetensors file with:
        // - FP8 weight tensor: 2x2 matrix of 0x38 (1.0 in E4M3)
        // - F32 scale tensor: single scale = 2.0 (per-tensor)
        // - BF16 passthrough: norm tensor

        let fp8_data = vec![0x38u8; 4]; // 2x2 of 1.0 in E4M3
        let scale_data = 2.0_f32.to_le_bytes().to_vec();
        let norm_data = vec![0x80, 0x3F]; // BF16 1.0

        // Build header JSON manually since safetensors crate may not support F8_E4M3
        // for serialization. Instead, build the raw file.
        let mut header_map = serde_json::Map::new();

        // FP8 weight at offset 0, length 4
        let mut w_info = serde_json::Map::new();
        w_info.insert("dtype".into(), "F8_E4M3".into());
        w_info.insert("shape".into(), serde_json::json!([2, 2]));
        w_info.insert("data_offsets".into(), serde_json::json!([0, 4]));
        header_map.insert("layer.weight".into(), w_info.into());

        // F32 scale at offset 4, length 4
        let mut s_info = serde_json::Map::new();
        s_info.insert("dtype".into(), "F32".into());
        s_info.insert("shape".into(), serde_json::json!([1]));
        s_info.insert("data_offsets".into(), serde_json::json!([4, 8]));
        header_map.insert("layer.weight_scale".into(), s_info.into());

        // BF16 norm at offset 8, length 2
        let mut n_info = serde_json::Map::new();
        n_info.insert("dtype".into(), "BF16".into());
        n_info.insert("shape".into(), serde_json::json!([1]));
        n_info.insert("data_offsets".into(), serde_json::json!([8, 10]));
        header_map.insert("norm.weight".into(), n_info.into());

        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();

        // Build raw safetensors file: 8-byte length + header + data
        // CAST: usize → u64, header length fits in u64
        #[allow(clippy::as_conversions)]
        let header_len = header_bytes.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&fp8_data);
        file_bytes.extend_from_slice(&scale_data);
        file_bytes.extend_from_slice(&norm_data);

        let tmp_in = std::env::temp_dir().join("test_fp8_in.safetensors");
        let tmp_out = std::env::temp_dir().join("test_fp8_out.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();

        let model = parse(&tmp_in).unwrap();
        assert_eq!(model.header.scheme, QuantScheme::PerTensorFp8);
        assert_eq!(model.inspect().quantized, 1);

        model.remember(&tmp_out, TargetDtype::BF16).unwrap();

        // Read output and verify
        let out_model = parse(&tmp_out).unwrap();
        let out_info = out_model.inspect();
        // Output should have: 1 passthrough (was FP8, now BF16) + 1 passthrough (norm)
        // Scale tensor should be absent
        assert_eq!(out_info.passthrough, 2); // both are now BF16
        assert_eq!(out_info.quantized, 0);

        // Verify the weight values: 1.0 * 2.0 = 2.0 → BF16 0x4000 → LE [0x00, 0x40]
        let w_entry = out_model
            .header
            .tensors
            .iter()
            .find(|t| t.name == "layer.weight")
            .unwrap();
        let data_start = out_model.header.header_size + 8;
        let out_bytes = std::fs::read(&tmp_out).unwrap();
        let w_data =
            &out_bytes[data_start + w_entry.data_offsets.0..data_start + w_entry.data_offsets.1];
        // 4 elements × 2 bytes = 8 bytes
        assert_eq!(w_data.len(), 8);
        for chunk in w_data.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }

        std::fs::remove_file(&tmp_in).ok();
        std::fs::remove_file(&tmp_out).ok();
    }

    #[test]
    fn target_dtype_display() {
        assert_eq!(TargetDtype::BF16.to_string(), "BF16");
    }
}
