// SPDX-License-Identifier: MIT OR Apache-2.0

/// `NPZ`/`NPY` archive parsing — custom `NPY` header parser with bulk `read_exact`.
#[cfg(feature = "npz")]
pub mod npz;

/// Safetensors header parsing, tensor classification, and quantization scheme detection.
pub mod safetensors;

/// PyTorch `.pth` state_dict parsing — minimal pickle VM.
#[cfg(feature = "pth")]
pub mod pth;

/// `GGUF` file parsing — header, metadata key-value pairs, and tensor info table.
#[cfg(feature = "gguf")]
pub mod gguf;

/// Shared parsing utilities (byte-swap, etc.) used by multiple format parsers.
#[cfg(any(feature = "npz", feature = "pth"))]
pub(crate) mod utils;

#[cfg(feature = "gguf")]
pub use gguf::{
    inspect_gguf_from_reader, parse_gguf, GgufInspectInfo, GgufMetadataArray, GgufMetadataValue,
    GgufTensor, GgufTensorInfo, GgufType, ParsedGguf,
};
#[cfg(feature = "npz")]
pub use npz::{
    inspect_npz, inspect_npz_from_reader, parse_npz, NpzDtype, NpzInspectInfo, NpzTensor,
    NpzTensorInfo,
};
#[cfg(feature = "pth")]
pub use pth::{parse_pth, ParsedPth, PthDtype, PthInspectInfo, PthTensor, PthTensorInfo};
pub use safetensors::{
    parse_safetensors_header, parse_safetensors_header_from_reader, AwqCompanions, AwqConfig,
    Bnb4Companions, BnbConfig, Dtype, GptqCompanions, GptqConfig, QuantScheme, SafetensorsHeader,
    TensorEntry, TensorRole,
};
