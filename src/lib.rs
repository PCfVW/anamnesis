// SPDX-License-Identifier: MIT OR Apache-2.0

//! **ἀνάμνησις** — parse any tensor format, recover any precision.
//!
//! `anamnesis` is a framework-agnostic Rust library for dequantizing
//! quantized model weights and parsing tensor archives. It handles
//! `.safetensors` (read once, classify, dequantize to `BF16`) and
//! `.npz` (bulk extraction at near-I/O speed) — all without `unsafe` code.
//!
//! # Supported Quantization Schemes
//!
//! | Scheme | Feature gate | Speedup vs `PyTorch` CPU (AVX2) |
//! |--------|-------------|-------------------------------|
//! | `FP8` `E4M3` (fine-grained, per-channel, per-tensor) | *(always on)* | 2.7–9.7× |
//! | `GPTQ` (`INT4`/`INT8`, group-wise, `g_idx`) | `gptq` | 6.5–12.2× |
//! | `AWQ` (`INT4`, per-group, activation-aware) | `awq` | 4.7–5.7× |
//! | `BitsAndBytes` `NF4`/`FP4` (lookup + per-block absmax) | `bnb` | 18–54× |
//! | `BitsAndBytes` `INT8` (`LLM.int8()`, per-row absmax) | `bnb` | 1.2× |
//!
//! All schemes produce **bit-exact** output (0 ULP difference) against
//! `PyTorch` reference implementations, verified on real models.
//!
//! # `NPZ`/`NPY` Parsing
//!
//! Feature-gated behind `npz`. Custom `NPY` header parser with bulk
//! `read_exact` — zero per-element deserialization for LE data on LE
//! machines. Supports `F16`, `BF16`, `F32`, `F64`, all integer types,
//! and `Bool`. **3,586 MB/s** on a 302 MB file (1.3× raw I/O overhead).
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use anamnesis::{parse, TargetDtype};
//!
//! let model = parse("model-fp8.safetensors")?;
//! let info = model.inspect();
//! println!("{info}");
//!
//! model.remember("model-bf16.safetensors", TargetDtype::BF16)?;
//! # Ok::<(), anamnesis::AnamnesisError>(())
//! ```
//!
//! # Architecture
//!
//! - [`parse()`] — read a `.safetensors` file into a [`ParsedModel`]
//! - [`ParsedModel::inspect`] — derive format, tensor counts, and size
//!   estimates from the header (zero I/O)
//! - [`ParsedModel::remember`] — dequantize all quantized tensors to `BF16`
//!   and write a standard `.safetensors` file
//! - `parse_npz()` — read an `.npz` archive into a `HashMap<String, NpzTensor>`
//!   (requires `npz` feature)
//!
//! The [`remember`] module contains one submodule per quantization family
//! ([`remember::fp8`], [`remember::gptq`], [`remember::awq`],
//! [`remember::bnb`]), each feature-gated independently.

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
#[cfg(feature = "npz")]
pub use parse::{parse_npz, NpzDtype, NpzTensor};
pub use parse::{
    AwqCompanions, AwqConfig, Bnb4Companions, BnbConfig, Dtype, GptqCompanions, GptqConfig,
    QuantScheme, SafetensorsHeader, TensorEntry, TensorRole,
};
#[cfg(feature = "awq")]
pub use remember::dequantize_awq_to_bf16;
#[cfg(feature = "gptq")]
pub use remember::dequantize_gptq_to_bf16;
#[cfg(feature = "bnb")]
pub use remember::{dequantize_bnb4_to_bf16, dequantize_bnb_int8_to_bf16};
pub use remember::{
    dequantize_fp8_to_bf16, dequantize_per_channel_fp8_to_bf16, dequantize_per_tensor_fp8_to_bf16,
};
