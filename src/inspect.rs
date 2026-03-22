// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use crate::parse::safetensors::{Dtype, QuantScheme, SafetensorsHeader, TensorRole};

/// Summary information produced by inspecting a parsed `.safetensors` file.
///
/// Built on [`SafetensorsHeader`] — no file I/O, no re-read. All fields are
/// derived from the parsed header metadata.
#[derive(Debug, Clone)]
#[must_use]
pub struct InspectInfo {
    /// Detected quantization scheme (e.g., `FineGrainedFp8`, `PerTensorFp8`).
    pub format: QuantScheme,
    /// Number of quantized weight tensors.
    pub quantized: usize,
    /// Number of scale factor tensors (non-zero only for fine-grained `FP8`).
    pub scales: usize,
    /// Number of passthrough tensors (norms, embeddings, `lm_head`).
    pub passthrough: usize,
    /// Unique dtypes of scale factor tensors, in order of first occurrence.
    pub scale_dtypes: Vec<Dtype>,
    /// Number of zero-point tensors (`GPTQ` `.qzeros`).
    pub zeropoints: usize,
    /// Number of group-index tensors (`GPTQ` `.g_idx`).
    pub group_indices: usize,
    /// Total tensor data size in bytes (as stored in the file).
    pub current_size: u64,
    /// Estimated tensor data size in bytes after dequantization to `BF16`.
    pub dequantized_size: u64,
}

impl InspectInfo {
    /// Returns the number of bytes of precision that Lethe took
    /// (difference between dequantized and current size).
    ///
    /// Zero when the model is unquantized.
    #[must_use]
    pub fn lethe_took(&self) -> u64 {
        self.dequantized_size.saturating_sub(self.current_size)
    }
}

impl From<&SafetensorsHeader> for InspectInfo {
    fn from(header: &SafetensorsHeader) -> Self {
        let quantized = header.quantized_count();
        let scales = header.scale_count();
        let passthrough = header.passthrough_count();
        let zeropoints = header.zeropoint_count();
        let group_indices = header.group_index_count();

        let mut scale_dtypes: Vec<Dtype> = Vec::new();
        for entry in header.scale_tensors() {
            if !scale_dtypes.contains(&entry.dtype) {
                scale_dtypes.push(entry.dtype);
            }
        }

        let mut current_size: u64 = 0;
        let mut dequantized_size: u64 = 0;

        for entry in &header.tensors {
            // CAST: usize → u64, byte lengths fit in u64 for any realistic model
            #[allow(clippy::as_conversions)]
            let byte_len = entry.byte_len() as u64;
            current_size += byte_len;

            match entry.role {
                TensorRole::Quantized => {
                    // CAST: usize → u64, element count fits in u64 for any realistic model
                    #[allow(clippy::as_conversions)]
                    let deq_bytes = entry.num_elements() as u64 * 2;
                    dequantized_size += deq_bytes;
                }
                TensorRole::Scale | TensorRole::ZeroPoint | TensorRole::GroupIndex => {
                    // Companion tensors are consumed during dequantization,
                    // not written to the output file.
                }
                TensorRole::Passthrough => {
                    // Passthrough tensors are copied as-is.
                    dequantized_size += byte_len;
                }
            }
        }

        Self {
            format: header.scheme,
            quantized,
            scales,
            passthrough,
            scale_dtypes,
            zeropoints,
            group_indices,
            current_size,
            dequantized_size,
        }
    }
}

impl fmt::Display for InspectInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Format:      {}", self.format)?;

        if self.scales > 0 {
            let dtype_list: String = self
                .scale_dtypes
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            write!(
                f,
                "\nQuantized:   {} tensors (weights) + {} scale tensors ({dtype_list})",
                self.quantized, self.scales,
            )?;
        } else {
            write!(f, "\nQuantized:   {} tensors (weights)", self.quantized)?;
        }

        write!(
            f,
            "\nPassthrough: {} tensors (norms, embeddings)",
            self.passthrough,
        )?;

        if self.zeropoints > 0 {
            write!(f, "\nZero-points: {} tensors", self.zeropoints,)?;
        }

        if self.group_indices > 0 {
            write!(
                f,
                "\nGroup index: {} tensors (activation-order)",
                self.group_indices,
            )?;
        }

        let scheme_label = match self.format {
            QuantScheme::Gptq => "GPTQ",
            QuantScheme::Unquantized => "unquantized",
            QuantScheme::FineGrainedFp8
            | QuantScheme::PerChannelFp8
            | QuantScheme::PerTensorFp8 => "FP8",
        };
        write!(
            f,
            "\nSize:        {} ({scheme_label}) -> {} (BF16)",
            format_bytes(self.current_size),
            format_bytes(self.dequantized_size),
        )?;

        if self.format != QuantScheme::Unquantized {
            write!(
                f,
                "\nLethe took:  ~{} of precision",
                format_bytes(self.lethe_took()),
            )?;
        }

        Ok(())
    }
}

/// Format a byte count as a human-readable string.
///
/// Examples: `"0 B"`, `"512 B"`, `"45.6 KB"`, `"302 MB"`, `"4.35 GB"`.
#[must_use]
#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    // CAST: u64 → f64 throughout; model sizes are well within f64 mantissa range
    // (52-bit mantissa covers exact integers up to 2^53 ≈ 9 PB).
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::parse::safetensors::TensorEntry;

    fn make_entry(name: &str, dtype: Dtype, role: TensorRole, shape: &[usize]) -> TensorEntry {
        let num_elements: usize = shape.iter().product();
        let byte_len = num_elements * dtype.byte_size();
        TensorEntry {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
            data_offsets: (0, byte_len),
            role,
        }
    }

    // -- format_bytes --------------------------------------------------------

    #[test]
    fn format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn format_bytes_small() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
    }

    #[test]
    fn format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1 MB");
        assert_eq!(format_bytes(302 * 1024 * 1024), "302 MB");
    }

    #[test]
    fn format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        // 4.35 GB ≈ 4672 MB
        assert_eq!(format_bytes(4_672 * 1024 * 1024), "4.56 GB");
    }

    // -- InspectInfo from SafetensorsHeader -----------------------------------

    #[test]
    fn inspect_unquantized() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry("norm.weight", Dtype::BF16, TensorRole::Passthrough, &[2048]),
                make_entry(
                    "lm_head.weight",
                    Dtype::BF16,
                    TensorRole::Passthrough,
                    &[32000, 2048],
                ),
            ],
            scheme: QuantScheme::Unquantized,
            metadata: None,
            header_size: 0,
            gptq_config: None,
        };
        let info = InspectInfo::from(&header);

        assert_eq!(info.format, QuantScheme::Unquantized);
        assert_eq!(info.quantized, 0);
        assert_eq!(info.scales, 0);
        assert_eq!(info.passthrough, 2);
        assert_eq!(info.current_size, info.dequantized_size);
        assert_eq!(info.lethe_took(), 0);
    }

    #[test]
    fn inspect_fine_grained_fp8() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry(
                    "layer.weight",
                    Dtype::F8E4M3,
                    TensorRole::Quantized,
                    &[2048, 2048],
                ),
                make_entry(
                    "layer.weight_scale_inv",
                    Dtype::F32,
                    TensorRole::Scale,
                    &[16, 16],
                ),
                make_entry("norm.weight", Dtype::BF16, TensorRole::Passthrough, &[2048]),
            ],
            scheme: QuantScheme::FineGrainedFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
        };
        let info = InspectInfo::from(&header);

        assert_eq!(info.quantized, 1);
        assert_eq!(info.scales, 1);
        assert_eq!(info.passthrough, 1);

        // Quantized: 2048×2048 = 4_194_304 elements × 1 byte = 4_194_304 bytes
        // Scale: 16×16 = 256 × 4 bytes = 1024 bytes
        // Passthrough: 2048 × 2 bytes = 4096 bytes
        let expected_current = 4_194_304 + 1024 + 4096;
        assert_eq!(info.current_size, expected_current);

        // Dequantized: quantized → 4_194_304 × 2 = 8_388_608, scale → 0, passthrough → 4096
        let expected_deq = 8_388_608 + 4096;
        assert_eq!(info.dequantized_size, expected_deq);

        assert!(info.lethe_took() > 0);
    }

    #[test]
    fn inspect_per_tensor_fp8() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry(
                    "layer.weight",
                    Dtype::F8E4M3,
                    TensorRole::Quantized,
                    &[1024, 1024],
                ),
                make_entry("norm.weight", Dtype::BF16, TensorRole::Passthrough, &[1024]),
            ],
            scheme: QuantScheme::PerTensorFp8,
            metadata: None,
            header_size: 0,
            gptq_config: None,
        };
        let info = InspectInfo::from(&header);

        assert_eq!(info.quantized, 1);
        assert_eq!(info.scales, 0);
        assert_eq!(info.passthrough, 1);

        // Quantized: 1024×1024 = 1_048_576 × 1 byte
        // Passthrough: 1024 × 2 bytes = 2048
        assert_eq!(info.current_size, 1_048_576 + 2048);
        // Dequantized: 1_048_576 × 2 + 2048
        assert_eq!(info.dequantized_size, 2_097_152 + 2048);
    }

    // -- Display output ------------------------------------------------------

    #[test]
    fn display_per_tensor_fp8() {
        let info = InspectInfo {
            format: QuantScheme::PerTensorFp8,
            quantized: 224,
            scales: 0,
            passthrough: 53,
            scale_dtypes: vec![],
            zeropoints: 0,
            group_indices: 0,
            current_size: 4_672 * 1024 * 1024,
            dequantized_size: 8_269 * 1024 * 1024,
        };
        let output = info.to_string();

        assert!(output.contains("Per-tensor FP8 (E4M3)"));
        assert!(output.contains("224 tensors (weights)"));
        assert!(!output.contains("scale tensors"));
        assert!(output.contains("53 tensors"));
        assert!(output.contains("Lethe took"));
    }

    #[test]
    fn display_fine_grained_fp8() {
        let info = InspectInfo {
            format: QuantScheme::FineGrainedFp8,
            quantized: 180,
            scales: 180,
            passthrough: 31,
            scale_dtypes: vec![Dtype::F32],
            zeropoints: 0,
            group_indices: 0,
            current_size: 1_310 * 1024 * 1024,
            dequantized_size: 2_580 * 1024 * 1024,
        };
        let output = info.to_string();

        assert!(output.contains("Fine-grained FP8 (E4M3), 128x128 blocks"));
        assert!(output.contains("180 tensors (weights) + 180 scale tensors (F32)"));
        assert!(output.contains("31 tensors"));
        assert!(output.contains("Lethe took"));
    }

    #[test]
    fn display_fine_grained_fp8_bf16_scales() {
        let info = InspectInfo {
            format: QuantScheme::FineGrainedFp8,
            quantized: 180,
            scales: 180,
            passthrough: 31,
            scale_dtypes: vec![Dtype::BF16],
            zeropoints: 0,
            group_indices: 0,
            current_size: 1_310 * 1024 * 1024,
            dequantized_size: 2_580 * 1024 * 1024,
        };
        let output = info.to_string();

        assert!(output.contains("180 scale tensors (BF16)"));
        assert!(!output.contains("(F32)"));
    }

    #[test]
    fn display_unquantized_omits_lethe() {
        let info = InspectInfo {
            format: QuantScheme::Unquantized,
            quantized: 0,
            scales: 0,
            passthrough: 100,
            scale_dtypes: vec![],
            zeropoints: 0,
            group_indices: 0,
            current_size: 1024 * 1024 * 1024,
            dequantized_size: 1024 * 1024 * 1024,
        };
        let output = info.to_string();

        assert!(output.contains("Unquantized"));
        assert!(!output.contains("Lethe took"));
    }
}
