// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(warnings)]

pub mod error;
pub mod inspect;
pub mod model;
pub mod parse;
pub mod remember;

pub use error::{AnamnesisError, Result};
pub use inspect::{format_bytes, InspectInfo};
pub use model::{parse, ParsedModel, TargetDtype};
pub use parse::{
    AwqCompanions, AwqConfig, Dtype, GptqCompanions, GptqConfig, QuantScheme, SafetensorsHeader,
    TensorEntry, TensorRole,
};
#[cfg(feature = "awq")]
pub use remember::dequantize_awq_to_bf16;
#[cfg(feature = "gptq")]
pub use remember::dequantize_gptq_to_bf16;
pub use remember::{
    dequantize_fp8_to_bf16, dequantize_per_channel_fp8_to_bf16, dequantize_per_tensor_fp8_to_bf16,
};
