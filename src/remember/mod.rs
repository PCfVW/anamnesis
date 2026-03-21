// SPDX-License-Identifier: MIT OR Apache-2.0

//! Precision recovery (dequantization) — built on [`parse`](crate::parse).
//!
//! Each submodule handles one quantization family. All operations take raw byte
//! slices from the parsed `.safetensors` file and produce dequantized output as
//! raw `BF16` bytes suitable for writing back to a standard `.safetensors` file.

pub mod fp8;

pub use fp8::dequantize_fp8_to_bf16;
