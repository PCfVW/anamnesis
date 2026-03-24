// SPDX-License-Identifier: MIT OR Apache-2.0

/// `NPZ`/`NPY` archive parsing — custom `NPY` header parser with bulk `read_exact`.
#[cfg(feature = "npz")]
pub mod npz;

/// Safetensors header parsing, tensor classification, and quantization scheme detection.
pub mod safetensors;

#[cfg(feature = "npz")]
pub use npz::{parse_npz, NpzDtype, NpzTensor};
pub use safetensors::{
    AwqCompanions, AwqConfig, Bnb4Companions, BnbConfig, Dtype, GptqCompanions, GptqConfig,
    QuantScheme, SafetensorsHeader, TensorEntry, TensorRole,
};
