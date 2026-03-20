use std::collections::HashMap;
use std::fmt;

use crate::error::AnamnesisError;

// ---------------------------------------------------------------------------
// Dtype
// ---------------------------------------------------------------------------

/// Element data type as parsed from a `.safetensors` header.
///
/// This is anamnesis's own enum, decoupled from `safetensors::Dtype`, so that
/// we can add helper methods and remain insulated from upstream changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Dtype {
    /// 8-bit floating point, 4-bit exponent, 3-bit mantissa.
    F8E4M3,
    /// 8-bit floating point, 5-bit exponent, 2-bit mantissa.
    F8E5M2,
    /// 16-bit brain floating point.
    BF16,
    /// 16-bit IEEE 754 half-precision.
    F16,
    /// 32-bit IEEE 754 single-precision.
    F32,
    /// 64-bit IEEE 754 double-precision.
    F64,
    /// Boolean (1 byte per element in safetensors).
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
}

impl Dtype {
    /// Returns the number of bytes per element for this dtype.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    /// Returns `true` if this dtype represents a quantized format requiring
    /// dequantization (`F8_E4M3` or `F8_E5M2`).
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::F8E4M3 | Self::F8E5M2)
    }

    /// Returns `true` if this dtype is a floating-point type.
    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        matches!(
            self,
            Self::F8E4M3 | Self::F8E5M2 | Self::BF16 | Self::F16 | Self::F32 | Self::F64
        )
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::BF16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U64 => "U64",
            Self::I64 => "I64",
        };
        f.write_str(s)
    }
}

impl TryFrom<safetensors::Dtype> for Dtype {
    type Error = AnamnesisError;

    /// Converts a `safetensors::Dtype` into anamnesis's own `Dtype`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if the upstream crate introduces
    /// a dtype variant that anamnesis does not yet handle.
    fn try_from(st: safetensors::Dtype) -> std::result::Result<Self, Self::Error> {
        match st {
            safetensors::Dtype::F8_E4M3 => Ok(Self::F8E4M3),
            safetensors::Dtype::F8_E5M2 => Ok(Self::F8E5M2),
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F16 => Ok(Self::F16),
            safetensors::Dtype::F32 => Ok(Self::F32),
            safetensors::Dtype::F64 => Ok(Self::F64),
            safetensors::Dtype::BOOL => Ok(Self::Bool),
            safetensors::Dtype::U8 => Ok(Self::U8),
            safetensors::Dtype::I8 => Ok(Self::I8),
            safetensors::Dtype::U16 => Ok(Self::U16),
            safetensors::Dtype::I16 => Ok(Self::I16),
            safetensors::Dtype::U32 => Ok(Self::U32),
            safetensors::Dtype::I32 => Ok(Self::I32),
            safetensors::Dtype::U64 => Ok(Self::U64),
            safetensors::Dtype::I64 => Ok(Self::I64),
            // safetensors::Dtype is #[non_exhaustive]; handle future additions.
            unknown => Err(AnamnesisError::Unsupported {
                format: "safetensors".into(),
                detail: format!("unknown dtype {unknown:?}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorRole
// ---------------------------------------------------------------------------

/// Classification of a tensor's role in a quantized model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorRole {
    /// Quantized weight tensor requiring dequantization.
    Quantized,
    /// Scale factor tensor (companion to a quantized weight).
    Scale,
    /// Passthrough tensor (norms, embeddings, `lm_head`) — already full-precision.
    Passthrough,
}

/// Classify a tensor based on its name and dtype.
fn classify_tensor(name: &str, dtype: Dtype) -> TensorRole {
    if name.ends_with("_scale_inv") || name.ends_with("_scale") {
        TensorRole::Scale
    } else if dtype.is_quantized() {
        TensorRole::Quantized
    } else {
        TensorRole::Passthrough
    }
}

// ---------------------------------------------------------------------------
// QuantScheme
// ---------------------------------------------------------------------------

/// Detected quantization scheme for a `.safetensors` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuantScheme {
    /// Fine-grained `FP8` with 128×128 block scale factors (`_scale_inv` companions).
    FineGrainedFp8,
    /// Per-tensor `FP8` with a single scale factor per tensor (or no explicit companion).
    PerTensorFp8,
    /// No quantization detected — all tensors are passthrough.
    Unquantized,
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::FineGrainedFp8 => "Fine-grained FP8 (E4M3), 128x128 blocks",
            Self::PerTensorFp8 => "Per-tensor FP8 (E4M3)",
            Self::Unquantized => "Unquantized",
        };
        f.write_str(s)
    }
}

/// Detect the quantization scheme from a list of classified tensor entries.
fn detect_scheme(entries: &[TensorEntry]) -> QuantScheme {
    let has_quantized = entries.iter().any(|e| e.role == TensorRole::Quantized);
    if !has_quantized {
        return QuantScheme::Unquantized;
    }

    // Check whether any quantized tensor has a matching `_scale_inv` companion.
    let scale_names: std::collections::HashSet<&str> = entries
        .iter()
        .filter(|e| e.role == TensorRole::Scale)
        .map(|e| e.name.as_str()) // BORROW: explicit .as_str() for HashSet<&str>
        .collect();

    let has_fine_grained_scales = entries
        .iter()
        .filter(|e| e.role == TensorRole::Quantized)
        .any(|e| {
            let expected = format!("{}_scale_inv", e.name);
            scale_names.contains(expected.as_str()) // BORROW: explicit .as_str() for HashSet lookup
        });

    if has_fine_grained_scales {
        QuantScheme::FineGrainedFp8
    } else {
        QuantScheme::PerTensorFp8
    }
}

// ---------------------------------------------------------------------------
// TensorEntry
// ---------------------------------------------------------------------------

/// Metadata for a single tensor parsed from a `.safetensors` header.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Tensor name as it appears in the header
    /// (e.g., `"model.layers.0.self_attn.q_proj.weight"`).
    pub name: String,
    /// Element data type (e.g., `F8E4M3`, `BF16`).
    pub dtype: Dtype,
    /// Tensor dimensions (e.g., `[2048, 2048]`).
    pub shape: Vec<usize>,
    /// Byte offset range `[start, end)` within the data section of the file.
    pub data_offsets: (usize, usize),
    /// Classification of this tensor's role in the model.
    pub role: TensorRole,
}

impl TensorEntry {
    /// Returns the total number of elements in the tensor.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the byte length of the tensor's data (`end - start` offset).
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }
}

// ---------------------------------------------------------------------------
// SafetensorsHeader
// ---------------------------------------------------------------------------

/// Parsed `.safetensors` header with tensor metadata and quantization scheme.
///
/// Produced by [`parse_safetensors_header`]. Contains all the information
/// needed to decide how to dequantize (remember) or inspect the file, without
/// having read any tensor data yet.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// All tensors found in the header, sorted by name.
    pub tensors: Vec<TensorEntry>,
    /// Detected quantization scheme for the file.
    pub scheme: QuantScheme,
    /// Raw metadata from the `__metadata__` section, if present.
    pub metadata: Option<HashMap<String, String>>,
    /// Size of the JSON header in bytes (data begins at `header_size + 8`).
    pub header_size: usize,
}

impl SafetensorsHeader {
    /// Returns an iterator over quantized tensors.
    pub fn quantized_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::Quantized)
    }

    /// Returns an iterator over scale factor tensors.
    pub fn scale_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::Scale)
    }

    /// Returns an iterator over passthrough tensors.
    pub fn passthrough_tensors(&self) -> impl Iterator<Item = &TensorEntry> {
        self.tensors
            .iter()
            .filter(|e| e.role == TensorRole::Passthrough)
    }

    /// Returns the number of quantized tensors.
    #[must_use]
    pub fn quantized_count(&self) -> usize {
        self.quantized_tensors().count()
    }

    /// Returns the number of scale factor tensors.
    #[must_use]
    pub fn scale_count(&self) -> usize {
        self.scale_tensors().count()
    }

    /// Returns the number of passthrough tensors.
    #[must_use]
    pub fn passthrough_count(&self) -> usize {
        self.passthrough_tensors().count()
    }

    /// Finds the scale tensor for a given weight tensor name.
    ///
    /// Looks for `{weight_name}_scale_inv` first, then `{weight_name}_scale`.
    #[must_use]
    pub fn find_scale_for(&self, weight_name: &str) -> Option<&TensorEntry> {
        let scale_inv = format!("{weight_name}_scale_inv");
        let scale = format!("{weight_name}_scale");
        self.tensors
            .iter()
            .find(|e| e.name == scale_inv)
            .or_else(|| self.tensors.iter().find(|e| e.name == scale))
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parses the header of a `.safetensors` file from a byte buffer.
///
/// Extracts all tensor metadata (names, shapes, dtypes, byte offsets),
/// classifies each tensor (quantized, scale, passthrough), and detects
/// the quantization scheme (fine-grained `FP8`, per-tensor `FP8`, or unquantized).
///
/// The buffer must contain at least the 8-byte length prefix and the full
/// JSON header. It may also contain the tensor data (the data is not read).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
///
/// Returns [`AnamnesisError::Unsupported`] if a tensor uses an unrecognized dtype.
///
/// # Memory
///
/// Allocates a `Vec<TensorEntry>` proportional to the number of tensors in the
/// header (typically hundreds). No tensor data is copied or read.
pub fn parse_safetensors_header(buffer: &[u8]) -> crate::Result<SafetensorsHeader> {
    let (header_size, metadata) =
        safetensors::SafeTensors::read_metadata(buffer).map_err(AnamnesisError::from)?;

    let st_tensors = metadata.tensors();
    let mut entries = Vec::with_capacity(st_tensors.len());

    for (name, info) in &st_tensors {
        let dtype = Dtype::try_from(info.dtype)?;
        let role = classify_tensor(name, dtype);
        entries.push(TensorEntry {
            name: name.clone(),
            dtype,
            shape: info.shape.clone(),
            data_offsets: info.data_offsets,
            role,
        });
    }

    // Sort by name for deterministic ordering (HashMap iteration is arbitrary).
    entries.sort_by(|a, b| a.name.cmp(&b.name));

    let scheme = detect_scheme(&entries);
    let file_metadata = metadata.metadata().clone();

    Ok(SafetensorsHeader {
        tensors: entries,
        scheme,
        metadata: file_metadata,
        header_size,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    // -- Dtype ---------------------------------------------------------------

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(Dtype::F8E4M3.byte_size(), 1);
        assert_eq!(Dtype::F8E5M2.byte_size(), 1);
        assert_eq!(Dtype::U8.byte_size(), 1);
        assert_eq!(Dtype::I8.byte_size(), 1);
        assert_eq!(Dtype::Bool.byte_size(), 1);
        assert_eq!(Dtype::BF16.byte_size(), 2);
        assert_eq!(Dtype::F16.byte_size(), 2);
        assert_eq!(Dtype::U16.byte_size(), 2);
        assert_eq!(Dtype::I16.byte_size(), 2);
        assert_eq!(Dtype::F32.byte_size(), 4);
        assert_eq!(Dtype::U32.byte_size(), 4);
        assert_eq!(Dtype::I32.byte_size(), 4);
        assert_eq!(Dtype::F64.byte_size(), 8);
        assert_eq!(Dtype::U64.byte_size(), 8);
        assert_eq!(Dtype::I64.byte_size(), 8);
    }

    #[test]
    fn dtype_is_quantized() {
        assert!(Dtype::F8E4M3.is_quantized());
        assert!(Dtype::F8E5M2.is_quantized());
        assert!(!Dtype::BF16.is_quantized());
        assert!(!Dtype::F32.is_quantized());
        assert!(!Dtype::U8.is_quantized());
    }

    #[test]
    fn dtype_is_floating_point() {
        assert!(Dtype::F8E4M3.is_floating_point());
        assert!(Dtype::BF16.is_floating_point());
        assert!(Dtype::F32.is_floating_point());
        assert!(Dtype::F64.is_floating_point());
        assert!(!Dtype::U8.is_floating_point());
        assert!(!Dtype::I32.is_floating_point());
        assert!(!Dtype::Bool.is_floating_point());
    }

    #[test]
    fn dtype_display() {
        assert_eq!(Dtype::F8E4M3.to_string(), "F8_E4M3");
        assert_eq!(Dtype::BF16.to_string(), "BF16");
        assert_eq!(Dtype::F32.to_string(), "F32");
    }

    #[test]
    fn dtype_try_from_safetensors() {
        assert_eq!(
            Dtype::try_from(safetensors::Dtype::F8_E4M3).ok(),
            Some(Dtype::F8E4M3)
        );
        assert_eq!(
            Dtype::try_from(safetensors::Dtype::BF16).ok(),
            Some(Dtype::BF16)
        );
        assert_eq!(
            Dtype::try_from(safetensors::Dtype::F32).ok(),
            Some(Dtype::F32)
        );
    }

    // -- Classification ------------------------------------------------------

    #[test]
    fn classify_quantized_weight() {
        let role = classify_tensor("model.layers.0.self_attn.q_proj.weight", Dtype::F8E4M3);
        assert_eq!(role, TensorRole::Quantized);
    }

    #[test]
    fn classify_scale_inv() {
        let role =
            classify_tensor("model.layers.0.self_attn.q_proj.weight_scale_inv", Dtype::F32);
        assert_eq!(role, TensorRole::Scale);
    }

    #[test]
    fn classify_scale() {
        let role = classify_tensor("model.layers.0.self_attn.q_proj.weight_scale", Dtype::F32);
        assert_eq!(role, TensorRole::Scale);
    }

    #[test]
    fn classify_passthrough_norm() {
        let role = classify_tensor("model.norm.weight", Dtype::BF16);
        assert_eq!(role, TensorRole::Passthrough);
    }

    #[test]
    fn classify_passthrough_embedding() {
        let role = classify_tensor("model.embed_tokens.weight", Dtype::BF16);
        assert_eq!(role, TensorRole::Passthrough);
    }

    // -- Scheme detection ----------------------------------------------------

    fn make_entry(name: &str, dtype: Dtype, role: TensorRole) -> TensorEntry {
        TensorEntry {
            name: name.to_owned(),
            dtype,
            shape: vec![128, 128],
            data_offsets: (0, 128 * 128),
            role,
        }
    }

    #[test]
    fn detect_unquantized() {
        let entries = vec![
            make_entry("model.norm.weight", Dtype::BF16, TensorRole::Passthrough),
            make_entry("lm_head.weight", Dtype::BF16, TensorRole::Passthrough),
        ];
        assert_eq!(detect_scheme(&entries), QuantScheme::Unquantized);
    }

    #[test]
    fn detect_fine_grained_fp8() {
        let entries = vec![
            make_entry("layer.0.weight", Dtype::F8E4M3, TensorRole::Quantized),
            make_entry("layer.0.weight_scale_inv", Dtype::F32, TensorRole::Scale),
            make_entry("model.norm.weight", Dtype::BF16, TensorRole::Passthrough),
        ];
        assert_eq!(detect_scheme(&entries), QuantScheme::FineGrainedFp8);
    }

    #[test]
    fn detect_per_tensor_fp8() {
        let entries = vec![
            make_entry("layer.0.weight", Dtype::F8E4M3, TensorRole::Quantized),
            make_entry("model.norm.weight", Dtype::BF16, TensorRole::Passthrough),
        ];
        assert_eq!(detect_scheme(&entries), QuantScheme::PerTensorFp8);
    }

    // -- find_scale_for ------------------------------------------------------

    #[test]
    fn find_scale_for_prefers_scale_inv() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry("w", Dtype::F8E4M3, TensorRole::Quantized),
                make_entry("w_scale", Dtype::F32, TensorRole::Scale),
                make_entry("w_scale_inv", Dtype::F32, TensorRole::Scale),
            ],
            scheme: QuantScheme::FineGrainedFp8,
            metadata: None,
            header_size: 0,
        };
        let found = header.find_scale_for("w");
        assert_eq!(found.map(|e| e.name.as_str()), Some("w_scale_inv"));
    }

    #[test]
    fn find_scale_for_falls_back_to_scale() {
        let header = SafetensorsHeader {
            tensors: vec![
                make_entry("w", Dtype::F8E4M3, TensorRole::Quantized),
                make_entry("w_scale", Dtype::F32, TensorRole::Scale),
            ],
            scheme: QuantScheme::PerTensorFp8,
            metadata: None,
            header_size: 0,
        };
        let found = header.find_scale_for("w");
        assert_eq!(found.map(|e| e.name.as_str()), Some("w_scale"));
    }

    #[test]
    fn find_scale_for_returns_none_when_missing() {
        let header = SafetensorsHeader {
            tensors: vec![make_entry("w", Dtype::F8E4M3, TensorRole::Quantized)],
            scheme: QuantScheme::PerTensorFp8,
            metadata: None,
            header_size: 0,
        };
        assert!(header.find_scale_for("w").is_none());
    }

    // -- Full parse round-trip -----------------------------------------------

    #[test]
    fn parse_minimal_safetensors() {
        use safetensors::tensor::serialize;

        // Build a minimal safetensors buffer with one BF16 tensor.
        let data: Vec<u8> = vec![0; 4]; // 2 elements × 2 bytes
        let tensors = vec![(
            "test_tensor",
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                vec![2],
                &data,
            )
            .unwrap_or_else(|e| panic!("failed to create TensorView: {e}")),
        )];
        let buffer = serialize(tensors, &None).unwrap_or_else(|e| panic!("serialize: {e}"));

        let header =
            parse_safetensors_header(&buffer).unwrap_or_else(|e| panic!("parse: {e}"));

        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.tensors[0].name, "test_tensor"); // INDEX: single element, bounds checked by len() assert above
        assert_eq!(header.tensors[0].dtype, Dtype::BF16);
        assert_eq!(header.tensors[0].shape, vec![2]);
        assert_eq!(header.tensors[0].role, TensorRole::Passthrough);
        assert_eq!(header.scheme, QuantScheme::Unquantized);
    }

    #[test]
    fn parse_fp8_with_scale() {
        use safetensors::tensor::serialize;

        let weight_data: Vec<u8> = vec![0; 4]; // 4 FP8 elements
        let scale_data: Vec<u8> = vec![0; 4]; // 1 F32 scale value

        let tensors = vec![
            (
                "layer.weight",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F8_E4M3,
                    vec![2, 2],
                    &weight_data,
                )
                .unwrap_or_else(|e| panic!("weight TensorView: {e}")),
            ),
            (
                "layer.weight_scale_inv",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 1],
                    &scale_data,
                )
                .unwrap_or_else(|e| panic!("scale TensorView: {e}")),
            ),
        ];
        let buffer = serialize(tensors, &None).unwrap_or_else(|e| panic!("serialize: {e}"));

        let header =
            parse_safetensors_header(&buffer).unwrap_or_else(|e| panic!("parse: {e}"));

        assert_eq!(header.tensors.len(), 2);
        assert_eq!(header.quantized_count(), 1);
        assert_eq!(header.scale_count(), 1);
        assert_eq!(header.passthrough_count(), 0);
        assert_eq!(header.scheme, QuantScheme::FineGrainedFp8);

        let scale = header.find_scale_for("layer.weight");
        assert_eq!(scale.map(|e| e.name.as_str()), Some("layer.weight_scale_inv"));
    }
}
