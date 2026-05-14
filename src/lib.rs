// SPDX-License-Identifier: MIT OR Apache-2.0

//! **ἀνάμνησις** — parse any tensor format, recover any precision.
//!
//! `anamnesis` is a framework-agnostic Rust library for dequantizing
//! quantized model weights and parsing tensor archives. It handles
//! `.safetensors` (memory-mapped, classify, dequantize to `BF16`),
//! `.npz` (bulk extraction at near-I/O speed), and `PyTorch` `.pth`
//! (zero-copy mmap with lossless safetensors conversion, 11–31× faster
//! than `torch.load()`).
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
//! # `PyTorch` `.pth` Parsing
//!
//! Feature-gated behind `pth`. Minimal pickle VM (~36 opcodes) with
//! security allowlist. Memory-mapped I/O with zero-copy `Cow::Borrowed`
//! tensor data. Lossless `.pth` → `.safetensors` conversion.
//! **11–31× faster** than `torch.load()` on torchvision models.
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
//! - [`parse()`] — memory-map a `.safetensors` file into a
//!   [`ParsedModel`]. Inspect-only workflows touch only the header
//!   (~1 MiB) regardless of file size; full dequantisation pages
//!   tensor bytes in lazily.
//! - [`ParsedModel::inspect`] — derive format, tensor counts, and size
//!   estimates from the parsed header (zero further I/O)
//! - [`ParsedModel::remember`] — dequantize all quantized tensors to `BF16`
//!   and write a standard `.safetensors` file
//! - [`parse_safetensors_header`] / [`parse_safetensors_header_from_reader`]
//!   — header-only safetensors parsing. The reader-generic variant accepts
//!   any `Read` substrate (in-memory `Cursor`, `HTTP`-range-backed adapter,
//!   …) and reads only the 8-byte length prefix plus the `JSON` header,
//!   so a multi-GB shard's metadata can be inspected with a single
//!   ~1 MiB sequential fetch.
//! - `parse_npz()` — read an `.npz` archive into a `HashMap<String, NpzTensor>`
//!   (requires `npz` feature)
//! - `inspect_npz()` / `inspect_npz_from_reader()` — header-only `NPZ`
//!   inspection. The reader-generic variant accepts any `Read + Seek`
//!   substrate (in-memory `Cursor`, HTTP-range-backed adapter, …) so callers
//!   can extract tensor metadata without materialising the data segment
//!   (requires `npz` feature)
//! - `parse_gguf()` / `inspect_gguf_from_reader()` — `GGUF` parsing /
//!   inspection. The path-based variant memory-maps the file and returns a
//!   `ParsedGguf` with zero-copy tensor views; the reader-generic variant
//!   accepts any `Read + Seek` substrate and returns just the
//!   `GgufInspectInfo` summary, so a multi-GB quantised `GGUF`'s metadata
//!   can be inspected in a few range fetches over the front-loaded header
//!   without downloading the data section (requires `gguf` feature)
//! - `parse_pth()` / `inspect_pth_from_reader()` — `PyTorch` `.pth` parsing
//!   / inspection. The path-based variant memory-maps the file and returns
//!   a `ParsedPth` with zero-copy `tensors()`; the reader-generic variant
//!   accepts any `Read + Seek` substrate and returns just the
//!   `PthInspectInfo` summary, so a torchvision-class `.pth` is inspectable
//!   in a single `<100 KiB` range fetch over the ZIP central directory and
//!   `data.pkl` entry — no tensor-data files inside the archive are read
//!   (requires `pth` feature)
//! - `pth_to_safetensors()` — lossless `.pth` → `.safetensors` conversion
//!   (requires `pth` feature)
//!
//! The [`remember`] module contains one submodule per quantization family
//! ([`remember::fp8`], [`remember::gptq`], [`remember::awq`],
//! [`remember::bnb`]), each feature-gated independently.

// `deny` (not `forbid`) allows feature-gated modules to opt in to unsafe
// where required by external APIs (e.g., memmap2 in the `pth` module).
// See CONVENTIONS.md "// SAFETY:" rules for the policy.
#![deny(unsafe_code)]
#![deny(warnings)]
// Allow unknown lint names so that `#[allow(clippy::newer_lint)]` in test
// modules does not become an error when built with MSRV clippy (which may
// not recognise lints added in later releases). Without this, every new
// clippy lint suppression is a potential MSRV CI break.
#![allow(unknown_lints)]

pub mod error;
pub mod inspect;
pub mod model;
pub mod parse;
pub mod remember;

pub use error::{AnamnesisError, Result};
pub use inspect::{format_bytes, InspectInfo};
pub use model::{parse, ParsedModel, TargetDtype};
#[cfg(feature = "gguf")]
pub use parse::{
    inspect_gguf_from_reader, parse_gguf, GgufInspectInfo, GgufMetadataArray, GgufMetadataValue,
    GgufTensor, GgufTensorInfo, GgufType, ParsedGguf,
};
#[cfg(feature = "npz")]
pub use parse::{
    inspect_npz, inspect_npz_from_reader, parse_npz, NpzDtype, NpzInspectInfo, NpzTensor,
    NpzTensorInfo,
};
#[cfg(feature = "pth")]
pub use parse::{
    inspect_pth_from_reader, parse_pth, ParsedPth, PthDtype, PthInspectInfo, PthTensor,
    PthTensorInfo,
};
pub use parse::{
    parse_safetensors_header, parse_safetensors_header_from_reader, AwqCompanions, AwqConfig,
    Bnb4Companions, BnbConfig, Dtype, GptqCompanions, GptqConfig, QuantScheme, SafetensorsHeader,
    TensorEntry, TensorRole,
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
#[cfg(feature = "gguf")]
pub use remember::{dequantize_gguf_blocks_to_bf16, dequantize_gguf_to_bf16};
#[cfg(feature = "pth")]
pub use remember::{pth_to_safetensors, pth_to_safetensors_bytes};
