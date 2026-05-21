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

/// `GGUF` file writing — the format-symmetric inverse of [`gguf`]. Scalar
/// dtype passthrough only in Phase 6; quantised emit lands in Phase 7.5.
#[cfg(feature = "gguf")]
pub mod gguf_write;

/// `Ollama` model-cache path resolver — turns `llama3.2:1b` into the
/// local `GGUF` blob path. Feature-gated behind `ollama` (which implies
/// [`gguf`] because every Ollama blob is a `GGUF`).
#[cfg(feature = "ollama")]
pub mod ollama;

/// Shared parsing utilities (byte-swap, etc.) used by multiple format parsers.
#[cfg(any(feature = "npz", feature = "pth"))]
pub(crate) mod utils;

#[cfg(feature = "gguf")]
pub use gguf::{
    inspect_gguf_from_reader, parse_gguf, GgufInspectInfo, GgufMetadataArray, GgufMetadataValue,
    GgufTensor, GgufTensorInfo, GgufType, ParsedGguf,
};
#[cfg(feature = "gguf")]
pub use gguf_write::{write_gguf, write_gguf_to_writer, GgufWriteTensor};
#[cfg(feature = "npz")]
pub use npz::{
    inspect_npz, inspect_npz_from_reader, parse_npz, NpzDtype, NpzInspectInfo, NpzTensor,
    NpzTensorInfo,
};
#[cfg(feature = "ollama")]
pub use ollama::resolve_ollama_model;
#[cfg(feature = "pth")]
pub use pth::{
    inspect_pth_from_reader, parse_pth, ParsedPth, PthDtype, PthInspectInfo, PthTensor,
    PthTensorInfo,
};
pub use safetensors::{
    parse_safetensors_header, parse_safetensors_header_from_reader, AwqCompanions, AwqConfig,
    Bnb4Companions, BnbConfig, Dtype, GptqCompanions, GptqConfig, QuantScheme, SafetensorsHeader,
    TensorEntry, TensorRole,
};
