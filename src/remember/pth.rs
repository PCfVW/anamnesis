// SPDX-License-Identifier: MIT OR Apache-2.0

//! `.pth` → safetensors lossless format conversion.
//!
//! Unlike the other `remember` submodules (`FP8`, `GPTQ`, `AWQ`, `BnB`), this module
//! performs **no dequantization** — `.pth` tensors are already full-precision.
//! The conversion is a pure format change: tensor names, shapes, dtypes, and
//! raw bytes are preserved exactly.

use std::path::Path;

use crate::error::AnamnesisError;
use crate::parse::pth::PthTensor;

/// Converts parsed `.pth` tensors to a safetensors file.
///
/// Each tensor is written with its original dtype — no dequantization or
/// dtype conversion. Tensor names are preserved as-is from the `state_dict`
/// keys (e.g., `"rnn.weight_ih_l0"`, `"linear.weight"`).
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if a tensor dtype has no
/// safetensors equivalent (currently all `PthDtype` variants map
/// successfully).
///
/// Returns [`AnamnesisError::Io`] if the output file cannot be written.
///
/// Returns [`AnamnesisError::Parse`] if safetensors serialization fails
/// (e.g., duplicate tensor names, shape/data mismatch).
pub fn pth_to_safetensors(tensors: &[PthTensor], output: impl AsRef<Path>) -> crate::Result<()> {
    let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        Vec::with_capacity(tensors.len());

    for tensor in tensors {
        let st_dtype = tensor.dtype.to_dtype()?.to_safetensors_dtype()?;
        let view =
            safetensors::tensor::TensorView::new(st_dtype, tensor.shape.clone(), &tensor.data)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("failed to create TensorView for `{}`: {e}", tensor.name),
                })?;
        views.push((tensor.name.clone(), view));
    }

    safetensors::tensor::serialize_to_file(views, &None, output.as_ref()).map_err(
        // EXHAUSTIVE: SafeTensorError is a foreign type that may gain variants;
        // we extract IoError and treat everything else as a parse/format error.
        #[allow(clippy::wildcard_enum_match_arm)]
        |e| match e {
            safetensors::SafeTensorError::IoError(io_err) => AnamnesisError::Io(io_err),
            other => AnamnesisError::Parse {
                reason: format!("failed to write safetensors file: {other}"),
            },
        },
    )?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::parse::pth::PthDtype;

    #[test]
    fn roundtrip_simple() {
        let tensors = vec![
            PthTensor {
                name: "weight".into(),
                shape: vec![2, 2],
                dtype: PthDtype::F32,
                // 4 floats: 1.0, 2.0, 3.0, 4.0
                data: vec![
                    0x00, 0x00, 0x80, 0x3F, // 1.0f32 LE
                    0x00, 0x00, 0x00, 0x40, // 2.0f32 LE
                    0x00, 0x00, 0x40, 0x40, // 3.0f32 LE
                    0x00, 0x00, 0x80, 0x40, // 4.0f32 LE
                ],
            },
            PthTensor {
                name: "bias".into(),
                shape: vec![2],
                dtype: PthDtype::F32,
                data: vec![
                    0x00, 0x00, 0x00, 0x3F, // 0.5f32 LE
                    0x00, 0x00, 0x00, 0xBF, // -0.5f32 LE
                ],
            },
        ];

        let tmp = tempfile::NamedTempFile::new().unwrap();
        pth_to_safetensors(&tensors, tmp.path()).unwrap();

        // Read back with safetensors crate and verify.
        let data = std::fs::read(tmp.path()).unwrap();
        let st = safetensors::SafeTensors::deserialize(&data).unwrap();

        assert_eq!(st.len(), 2);

        let w = st.tensor("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.dtype(), safetensors::Dtype::F32);
        assert_eq!(w.data(), &tensors[0].data);

        let b = st.tensor("bias").unwrap();
        assert_eq!(b.shape(), &[2]);
        assert_eq!(b.dtype(), safetensors::Dtype::F32);
        assert_eq!(b.data(), &tensors[1].data);
    }

    #[test]
    fn empty_tensors() {
        let tensors: Vec<PthTensor> = vec![];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        pth_to_safetensors(&tensors, tmp.path()).unwrap();

        let data = std::fs::read(tmp.path()).unwrap();
        let st = safetensors::SafeTensors::deserialize(&data).unwrap();
        assert_eq!(st.len(), 0);
    }
}
