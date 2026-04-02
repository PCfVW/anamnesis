// SPDX-License-Identifier: MIT OR Apache-2.0

/// `NPZ`/`NPY` archive parsing — custom `NPY` header parser with bulk `read_exact`.
#[cfg(feature = "npz")]
pub mod npz;

/// Safetensors header parsing, tensor classification, and quantization scheme detection.
pub mod safetensors;

/// PyTorch `.pth` state_dict parsing — minimal pickle VM.
#[cfg(feature = "pth")]
pub mod pth;

/// Shared parsing utilities (byte-swap, etc.) used by multiple format parsers.
#[cfg(any(feature = "npz", feature = "pth"))]
pub(crate) mod utils;

#[cfg(feature = "npz")]
pub use npz::{parse_npz, NpzDtype, NpzTensor};
#[cfg(feature = "pth")]
pub use pth::{parse_pth, PthDtype, PthTensor};
pub use safetensors::{
    AwqCompanions, AwqConfig, Bnb4Companions, BnbConfig, Dtype, GptqCompanions, GptqConfig,
    QuantScheme, SafetensorsHeader, TensorEntry, TensorRole,
};
