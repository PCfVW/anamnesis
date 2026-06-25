// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-level parse-first API.
//!
//! [`parse`] memory-maps a `.safetensors` file, returning a
//! [`ParsedModel`] that holds the parsed header metadata and the file's
//! bytes. All subsequent operations ([`ParsedModel::inspect`],
//! [`ParsedModel::remember`]) work from this parsed representation — no
//! second open, no eager copy. On the memory-mapped path the kernel pages
//! bytes in lazily on access, so `inspect()` on a multi-GB shard only faults
//! the header (~1 MiB).
//!
//! # Trusted vs untrusted input
//!
//! [`parse`] / [`parse_with_limits`] memory-map the file — the
//! **trusted-local-file fast path**. A memory map can fault with `SIGBUS` if
//! the file is truncated or written concurrently, an OS signal the caller
//! cannot catch. For **untrusted input** (a user upload, a network / FUSE
//! path) prefer the copy-based [`parse_bytes`] / [`parse_from_reader`] entry
//! points: they read the artefact into an owned buffer (bounded by
//! [`ParseLimits`]), parse with no mmap and no `unsafe`, and fail with a clean
//! `Err` rather than a `SIGBUS`.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use crate::backing::Backing;
use crate::error::AnamnesisError;
use crate::inspect::InspectInfo;
use crate::parse::safetensors::{
    parse_safetensors_header_with_limits, Dtype, QuantScheme, SafetensorsHeader, TensorRole,
};
use crate::parse::utils::checked_num_elements;
#[cfg(feature = "awq")]
use crate::remember::awq::dequantize_awq_to_bf16;
#[cfg(feature = "bnb")]
use crate::remember::bnb::{
    dequantize_bnb4_double_quant_to_bf16, dequantize_bnb4_to_bf16, dequantize_bnb_int8_to_bf16,
};
use crate::remember::fp8::{
    dequantize_fp8_to_bf16, dequantize_per_channel_fp8_to_bf16, dequantize_per_tensor_fp8_to_bf16,
};
#[cfg(feature = "gptq")]
use crate::remember::gptq::dequantize_gptq_to_bf16;
#[cfg(any(feature = "gptq", feature = "awq"))]
use crate::remember::quant_utils::transpose_bf16;
use crate::ParseLimits;

/// Target dtype for dequantization output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TargetDtype {
    /// `BF16` (bfloat16) — 2 bytes per element. The standard research/training dtype.
    BF16,
}

impl fmt::Display for TargetDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BF16 => f.write_str("BF16"),
        }
    }
}

impl FromStr for TargetDtype {
    type Err = AnamnesisError;

    /// Parses a target dtype from a case-insensitive string.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if the string does not match
    /// a known target dtype.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::BF16),
            other => Err(AnamnesisError::Unsupported {
                format: other.to_owned(),
                detail: "supported target dtypes: bf16".to_owned(),
            }),
        }
    }
}

/// A parsed `.safetensors` model, holding parsed header metadata and the
/// file's bytes.
///
/// Created by [`parse`] (memory-mapped) or by [`parse_bytes`] /
/// [`parse_from_reader`] (owned copy). The bytes are reached through a `&[u8]`
/// regardless of backing, so both paths share every method
/// ([`ParsedModel::inspect`], [`ParsedModel::remember`]) and the backing is
/// invisible to callers. On the memory-mapped path the kernel pages bytes in
/// lazily on access, so `inspect()` on a multi-GB shard only faults in the
/// header (~1 MiB).
pub struct ParsedModel {
    /// Parsed header metadata (tensor names, dtypes, shapes, roles, scheme).
    pub header: SafetensorsHeader,
    /// File bytes — either a memory map (path-based [`parse`]) or an owned
    /// `Vec<u8>` (copy-based [`parse_bytes`] / [`parse_from_reader`]). Tensor
    /// data starts at offset `header_size + 8`. On the mmap path the OS pages
    /// bytes in lazily on access, so `parse()` + `inspect()` on a multi-GB
    /// shard touches only the header (~1 MiB) instead of materialising the
    /// whole file.
    buffer: Backing,
}

/// Parses a `.safetensors` file, returning a [`ParsedModel`] holding both
/// header metadata and the file bytes (memory-mapped).
///
/// This is the entry point for all anamnesis operations. The file is
/// memory-mapped once; all subsequent operations
/// ([`ParsedModel::inspect`], [`ParsedModel::remember`]) work from the
/// mmap, so tensor pages are paged in lazily on access.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or
/// mapped.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is
/// malformed.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header exceeds the
/// permanent 100 MiB cap (`MAX_SAFETENSORS_HEADER_BYTES`, always-on).
///
/// # Memory
///
/// Uses `memmap2::Mmap` so the file's bytes do not occupy heap. The
/// kernel pages bytes in on access and may drop them under memory
/// pressure — which means a 70 GiB shard can be inspected on a 32 GiB
/// machine without `OOM`ing the way a `Vec<u8>` allocation would.
/// `parse()` + `inspect()` only touches the header (~1 MiB), so the
/// resident-set growth on inspect-only workflows is bounded by the
/// header size, not the file size.
pub fn parse(path: impl AsRef<Path>) -> crate::Result<ParsedModel> {
    parse_with_limits(path, &ParseLimits::default())
}

/// Parses a `.safetensors` file under a caller-supplied [`ParseLimits`] budget.
///
/// Identical to [`parse`] but enforces every applicable [`ParseLimits`] ceiling
/// (the per-allocation and cumulative-byte budgets — see [`ParseLimits`] for the
/// axes) fail-fast, before the header is allocated. The built-in 100 MiB header
/// cap still applies; `limits` can only tighten it. [`parse`] is the
/// `ParseLimits::default()` (unbounded) special case.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or mapped.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header size exceeds
/// `limits`.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
///
/// # Memory
///
/// Uses `memmap2::Mmap` so the file's bytes do not occupy heap; `parse()` +
/// `inspect()` only touches the header. See [`parse`] for the full rationale.
#[allow(unsafe_code)]
pub fn parse_with_limits(
    path: impl AsRef<Path>,
    limits: &ParseLimits,
) -> crate::Result<ParsedModel> {
    let file = std::fs::File::open(path.as_ref())?;
    // SAFETY: `memmap2::Mmap` requires `unsafe` because the OS could
    // modify the mapped region if another process writes to the
    // underlying file concurrently. Tensor files are read-only artefacts
    // in practice — the same assumption every other tensor parser in this
    // crate (`parse_pth`, `parse_gguf`) and the upstream `safetensors`
    // crate's mmap path rely on. The mapping is released when the
    // returned `ParsedModel` is dropped. Untrusted callers that cannot
    // make the read-only-artefact assumption use `parse_bytes` /
    // `parse_from_reader` instead (no mmap, no `SIGBUS`).
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(AnamnesisError::Io)?;
    parsed_model_from_backing(Backing::Mmap(mmap), limits)
}

/// Builds a [`ParsedModel`] from an already-acquired byte backing — the single
/// construction site shared by the mmap path ([`parse_with_limits`]) and the
/// copy-based paths ([`parse_bytes_with_limits`] /
/// [`parse_from_reader_with_limits`]), so the header parse cannot drift between
/// them.
fn parsed_model_from_backing(buffer: Backing, limits: &ParseLimits) -> crate::Result<ParsedModel> {
    let header = parse_safetensors_header_with_limits(&buffer, limits)?;
    Ok(ParsedModel { header, buffer })
}

/// Parses `.safetensors` bytes already held in memory, returning a
/// [`ParsedModel`] that **owns** them — the copy-based, mmap-free path.
///
/// This is the **recommended entry point for untrusted input** (a user upload,
/// bytes received over the network): unlike [`parse`], it never memory-maps, so
/// a truncated or concurrently-written source cannot fault the process with a
/// `SIGBUS`; a malformed input is a clean `Err`. [`parse_bytes`] is the
/// [`ParseLimits::default`] (unbounded) special case of
/// [`parse_bytes_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header exceeds the
/// permanent 100 MiB cap (`MAX_SAFETENSORS_HEADER_BYTES`, always-on — reachable
/// even at the default limits this wrapper passes).
///
/// # Memory
///
/// Takes ownership of `bytes` (no copy) and holds them for the
/// [`ParsedModel`]'s lifetime — peak heap is the input size. Contrast [`parse`],
/// which memory-maps and pages lazily.
pub fn parse_bytes(bytes: Vec<u8>) -> crate::Result<ParsedModel> {
    parse_bytes_with_limits(bytes, &ParseLimits::default())
}

/// Parses owned `.safetensors` bytes under a caller-supplied [`ParseLimits`]
/// budget — the bounded, mmap-free path for untrusted input.
///
/// Rejects an input larger than [`ParseLimits::max_single_alloc_bytes`] before
/// parsing, then enforces every applicable [`ParseLimits`] ceiling on the header
/// exactly as [`parse_with_limits`] does.
///
/// # Errors
///
/// Returns [`AnamnesisError::LimitExceeded`] if `bytes` exceeds `limits`.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
///
/// # Memory
///
/// Takes ownership of `bytes` (no copy); peak heap is the input size.
pub fn parse_bytes_with_limits(bytes: Vec<u8>, limits: &ParseLimits) -> crate::Result<ParsedModel> {
    let len = u64::try_from(bytes.len()).map_err(|_| AnamnesisError::Parse {
        reason: "safetensors bytes: length overflows u64".into(),
    })?;
    limits.check_alloc(len, "safetensors bytes")?;
    parsed_model_from_backing(Backing::Owned(bytes), limits)
}

/// Parses a `.safetensors` artefact from any reader, returning a [`ParsedModel`]
/// that **owns** the bytes — the copy-based, mmap-free path.
///
/// The **recommended entry point for untrusted streamed input**: the whole
/// stream is read into an owned buffer (bounded by [`ParseLimits`]) and parsed
/// with no mmap, so a truncated or hostile stream is a clean `Err`, never a
/// `SIGBUS`. [`parse_from_reader`] is the [`ParseLimits::default`] (unbounded)
/// special case of [`parse_from_reader_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
/// Returns [`AnamnesisError::LimitExceeded`] if the declared header exceeds the
/// permanent 100 MiB cap (`MAX_SAFETENSORS_HEADER_BYTES`, always-on — reachable
/// even at the default limits this wrapper passes).
///
/// # Memory
///
/// Reads the entire stream into an owned `Vec<u8>`; peak heap is the artefact
/// size.
pub fn parse_from_reader<R: std::io::Read>(reader: R) -> crate::Result<ParsedModel> {
    parse_from_reader_with_limits(reader, &ParseLimits::default())
}

/// Parses a `.safetensors` artefact from any reader under a caller-supplied
/// [`ParseLimits`] budget — the bounded, mmap-free path for untrusted input.
///
/// The read is bounded by [`ParseLimits::max_single_alloc_bytes`] so an
/// unbounded or hostile stream cannot exhaust memory; the header is then parsed
/// under the same `limits` as [`parse_with_limits`].
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the reader fails.
/// Returns [`AnamnesisError::LimitExceeded`] if the bytes read exceed `limits`.
/// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
///
/// # Memory
///
/// Reads the stream into an owned `Vec<u8>` of at most
/// `max_single_alloc_bytes + 1` bytes; peak heap is the artefact size.
pub fn parse_from_reader_with_limits<R: std::io::Read>(
    reader: R,
    limits: &ParseLimits,
) -> crate::Result<ParsedModel> {
    let bytes = limits.read_to_vec_bounded(reader, "safetensors file")?;
    parse_bytes_with_limits(bytes, limits)
}

/// Owned dequantised tensors produced by [`ParsedModel::dequantize_all`]:
/// `(output name, `BF16` bytes, output shape)`.
type DequantizedTensors = Vec<(String, Vec<u8>, Vec<usize>)>;

/// Passthrough tensors that borrow `self.buffer`: `(name, bytes, shape)`. Tied
/// to the `ParsedModel` borrow they were collected under.
type PassthroughRefs<'a> = Vec<(&'a str, &'a [u8], &'a [usize])>;

impl ParsedModel {
    /// Returns inspection info (format, tensor counts, size estimates).
    ///
    /// Delegates to [`InspectInfo::from`]. No I/O — purely derived from
    /// the parsed header.
    pub fn inspect(&self) -> InspectInfo {
        InspectInfo::from(&self.header)
    }

    /// Returns the raw bytes for a tensor from the memory-mapped file
    /// buffer. The slice borrows from the mmap; pages are paged in by
    /// the kernel on first access.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the tensor's data offsets are
    /// out of bounds.
    fn tensor_data(&self, start: usize, end: usize) -> crate::Result<&[u8]> {
        let data_offset = self.header.header_size + 8;
        let abs_start = data_offset
            .checked_add(start)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "tensor data start offset overflow".into(),
            })?;
        let abs_end = data_offset
            .checked_add(end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "tensor data end offset overflow".into(),
            })?;
        self.buffer
            .get(abs_start..abs_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "tensor data offsets {abs_start}..{abs_end} out of bounds (buffer len {})",
                    self.buffer.len()
                ),
            })
    }

    /// Reads a scalar scale value from raw bytes, handling both `F32` and
    /// `BF16` scale dtypes.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the data is too short for the
    /// given dtype.
    /// Returns [`AnamnesisError::Unsupported`] if the scale dtype is not
    /// `F32` or `BF16`.
    fn read_scalar_scale(data: &[u8], dtype: Dtype, weight_name: &str) -> crate::Result<f32> {
        match dtype {
            Dtype::F32 => {
                let arr: [u8; 4] =
                    data.get(..4)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!(
                                "per-tensor F32 scale for `{weight_name}` is not 4 bytes"
                            ),
                        })?;
                Ok(f32::from_le_bytes(arr))
            }
            Dtype::BF16 => {
                let arr: [u8; 2] =
                    data.get(..2)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!(
                                "per-tensor BF16 scale for `{weight_name}` is not 2 bytes"
                            ),
                        })?;
                // BITWISE: BF16 → f32 by shifting into upper 16 bits of IEEE 754
                Ok(f32::from_bits(u32::from(u16::from_le_bytes(arr)) << 16))
            }
            Dtype::F16 => {
                let arr: [u8; 2] =
                    data.get(..2)
                        .and_then(|s| s.try_into().ok())
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!(
                                "per-tensor F16 scale for `{weight_name}` is not 2 bytes"
                            ),
                        })?;
                // BITWISE: F16 → f32 via half crate's IEEE 754 conversion
                Ok(half::f16::from_le_bytes(arr).to_f32())
            }
            Dtype::F8E4M3
            | Dtype::F8E5M2
            | Dtype::F64
            | Dtype::Bool
            | Dtype::U8
            | Dtype::I8
            | Dtype::U16
            | Dtype::I16
            | Dtype::U32
            | Dtype::I32
            | Dtype::U64
            | Dtype::I64 => Err(AnamnesisError::Unsupported {
                format: dtype.to_string(),
                detail: format!("per-tensor scale for `{weight_name}` has unsupported dtype"),
            }),
        }
    }

    /// Extracts `(rows, cols)` from a tensor shape for the fine-grained
    /// dequantization function.
    ///
    /// - 2D: `(shape[0], shape[1])`
    /// - >2D: `(product of all dims except last, last dim)`
    fn shape_to_rows_cols(shape: &[usize]) -> crate::Result<(usize, usize)> {
        match shape.len() {
            0 | 1 => Err(AnamnesisError::Parse {
                reason: format!(
                    "quantized tensor has {}-D shape, expected >= 2D",
                    shape.len()
                ),
            }),
            2 => {
                // INDEX: shape.len() == 2 guaranteed by match arm
                #[allow(clippy::indexing_slicing)]
                Ok((shape[0], shape[1]))
            }
            _ => {
                // shape.len() >= 3 guaranteed by match arms above
                let cols = shape.last().copied().ok_or_else(|| AnamnesisError::Parse {
                    reason: "shape has no last dimension".into(),
                })?;
                let leading =
                    shape
                        .get(..shape.len() - 1)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "shape slice out of bounds".into(),
                        })?;
                let rows = checked_num_elements(leading).ok_or_else(|| AnamnesisError::Parse {
                    reason: "shape row-count product overflows usize".into(),
                })?;
                Ok((rows, cols))
            }
        }
    }

    /// Dequantizes all quantized tensors and writes a standard `.safetensors`
    /// file loadable by any Rust ML framework.
    ///
    /// See [`remember_to_bytes`](Self::remember_to_bytes) for the in-memory
    /// variant that returns the bytes instead of writing a file.
    ///
    /// - **Quantized tensors**: dequantized to the target dtype using the
    ///   detected quantization scheme and companion scale factors. `GPTQ` /
    ///   `AWQ` projection weights are additionally transposed from the
    ///   GEMM-native `[in_features, out_features]` kernel orientation to the
    ///   standard `nn.Linear` `[out_features, in_features]` — the layout a
    ///   standard consumer (candle, `transformers` as a plain model)
    ///   expects, and the same boundary transpose `GPTQModel`'s
    ///   `dequantize_model` applies. `BnB` and `FP8` weights are already
    ///   stored / recovered in standard orientation.
    /// - **Scale tensors**: consumed during dequantization, not written.
    /// - **Passthrough tensors**: copied as-is (zero-copy from the buffer).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    ///
    /// # Memory
    ///
    /// Peak heap is `O(total_dequantised_output_size)` ≈ `2 × n_parameters`
    /// bytes (the output `BF16` tensors). The input file is memory-mapped
    /// — pages are paged in by the kernel on access and may be dropped
    /// under memory pressure — so the input side does not contribute to
    /// the heap. **Every dequantised tensor's `Vec<u8>` is retained
    /// simultaneously** until the underlying `safetensors::serialize_to_file`
    /// call returns: the safetensors crate's writer itself streams tensor
    /// bodies one at a time, but the eager buffering happens in this
    /// method's caller-side `Vec` collection. The `GPTQ` / `AWQ`
    /// orientation transpose holds one extra tensor-sized buffer
    /// transiently (per tensor, dropped immediately) — the peak class is
    /// unchanged. Comfortable for `≤ 7 B` models on a 32 GB system; tight
    /// at 13 B; `OOM`s at 70 B+. A streaming output path (planned ROADMAP
    /// Phase 10) will drop this to `O(largest_tensor_BF16)`.
    pub fn remember(
        &self,
        output_path: impl AsRef<Path>,
        target: TargetDtype,
    ) -> crate::Result<()> {
        match target {
            TargetDtype::BF16 => self.remember_bf16(output_path.as_ref()),
        }
    }

    /// Dequantizes all quantized tensors with per-tensor progress reporting,
    /// and writes a standard `.safetensors` file loadable by any Rust ML
    /// framework.
    ///
    /// Behaves identically to [`remember`](Self::remember), but calls
    /// `on_tensor` after each quantized tensor is dequantized. Use this to
    /// drive a progress bar in CLI contexts.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    pub fn remember_with_progress<F>(
        &self,
        output_path: impl AsRef<Path>,
        target: TargetDtype,
        on_tensor: F,
    ) -> crate::Result<()>
    where
        F: FnMut(),
    {
        match target {
            TargetDtype::BF16 => self.remember_bf16_inner(output_path.as_ref(), on_tensor),
        }
    }

    /// Dequantizes all quantized tensors and returns the standard `.safetensors`
    /// bytes in memory, instead of writing a file.
    ///
    /// The in-memory twin of [`remember`](Self::remember): identical dequant and
    /// companion-grouping, but returns the serialized `BF16` safetensors as a
    /// `Vec<u8>` so an embedder can load the dequantised model without a disk
    /// round-trip (e.g. candle-mi's quantized loader → `from_buffered_safetensors`).
    /// Completes the file/bytes pairing the crate's other serializers already
    /// have ([`ParsedPth::to_safetensors_bytes`](crate::ParsedPth::to_safetensors_bytes),
    /// `write_bnb_nf4_safetensors_bytes`).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if tensor data is malformed or
    /// shapes are inconsistent, or if serialization fails.
    /// Returns [`AnamnesisError::Unsupported`] if the quantization scheme
    /// is not yet implemented.
    ///
    /// # Memory
    ///
    /// Peak heap is **higher** than [`remember`](Self::remember)'s file path.
    /// Both dequantize every tensor into owned `BF16` `Vec`s (`O(2 × n_parameters)`),
    /// but where [`remember`](Self::remember) streams those bodies to disk one at
    /// a time via `safetensors::serialize_to_file`, this method calls
    /// `safetensors::serialize`, which copies every tensor into one contiguous
    /// output buffer — so the per-tensor `Vec`s **and** the full output `Vec` are
    /// live simultaneously (~`2 ×` the dequantised set transiently) before the
    /// per-tensor `Vec`s drop. Comfortable for `≤ 7 B` models on a 32 GB system;
    /// the streaming, peak-bounded `remember_to_writer` / `remember_to_sink`
    /// variants are planned for ROADMAP Phase 10.
    pub fn remember_to_bytes(&self, target: TargetDtype) -> crate::Result<Vec<u8>> {
        match target {
            TargetDtype::BF16 => self.remember_to_bytes_bf16(),
        }
    }

    /// Internal: dequantize to `BF16` and write (no progress callback).
    fn remember_bf16(&self, output_path: &Path) -> crate::Result<()> {
        self.remember_bf16_inner(output_path, || {})
    }

    /// Internal: run the per-scheme dequant for every tensor, returning the
    /// owned `BF16` results plus the passthrough tensors (which borrow
    /// `self.buffer`). Shared by `remember_bf16_inner` (→ file) and
    /// `remember_to_bytes_bf16` (→ bytes); `on_tensor` fires after each
    /// quantized tensor so callers can drive a progress bar.
    fn dequantize_all<F>(
        &self,
        mut on_tensor: F,
    ) -> crate::Result<(DequantizedTensors, PassthroughRefs<'_>)>
    where
        F: FnMut(),
    {
        // Collect dequantized data (owned) for quantized tensors.
        // Passthrough tensors borrow from self.buffer.
        let mut dequantized_data: DequantizedTensors = Vec::new();
        let mut passthrough_refs: PassthroughRefs<'_> = Vec::new();

        for entry in &self.header.tensors {
            match entry.role {
                TensorRole::Quantized => {
                    let weight_data =
                        self.tensor_data(entry.data_offsets.0, entry.data_offsets.1)?;

                    let bf16_bytes = match self.header.scheme {
                        QuantScheme::FineGrainedFp8 => {
                            let scale_entry =
                                self.header.find_scale_for(&entry.name).ok_or_else(|| {
                                    AnamnesisError::Parse {
                                        reason: format!(
                                            "no scale tensor found for quantized weight `{}`",
                                            entry.name
                                        ),
                                    }
                                })?;
                            let scale_data = self.tensor_data(
                                scale_entry.data_offsets.0,
                                scale_entry.data_offsets.1,
                            )?;
                            let (rows, cols) = Self::shape_to_rows_cols(&entry.shape)?;
                            dequantize_fp8_to_bf16(
                                weight_data,
                                scale_data,
                                rows,
                                cols,
                                scale_entry.dtype,
                            )?
                        }
                        QuantScheme::PerChannelFp8 => {
                            let scale_entry =
                                self.header.find_scale_for(&entry.name).ok_or_else(|| {
                                    AnamnesisError::Parse {
                                        reason: format!(
                                            "no scale tensor found for quantized weight `{}`",
                                            entry.name
                                        ),
                                    }
                                })?;
                            let scale_data = self.tensor_data(
                                scale_entry.data_offsets.0,
                                scale_entry.data_offsets.1,
                            )?;
                            let (rows, cols) = Self::shape_to_rows_cols(&entry.shape)?;
                            dequantize_per_channel_fp8_to_bf16(
                                weight_data,
                                scale_data,
                                rows,
                                cols,
                                scale_entry.dtype,
                            )?
                        }
                        QuantScheme::PerTensorFp8 => {
                            // Look for a companion scale tensor; default to 1.0 if none.
                            let scale = if let Some(scale_entry) =
                                self.header.find_scale_for(&entry.name)
                            {
                                let scale_data = self.tensor_data(
                                    scale_entry.data_offsets.0,
                                    scale_entry.data_offsets.1,
                                )?;
                                Self::read_scalar_scale(scale_data, scale_entry.dtype, &entry.name)?
                            } else {
                                1.0
                            };
                            dequantize_per_tensor_fp8_to_bf16(weight_data, scale)?
                        }
                        #[cfg(feature = "gptq")]
                        QuantScheme::Gptq => {
                            let config =
                                self.header
                                    .gptq_config
                                    .ok_or_else(|| AnamnesisError::Parse {
                                        reason: format!(
                                            "GPTQ config not available for `{}`",
                                            entry.name
                                        ),
                                    })?;
                            let companions = self
                                .header
                                .find_gptq_companions(&entry.name)
                                .ok_or_else(|| AnamnesisError::Parse {
                                    reason: format!(
                                        "GPTQ companions not found for `{}`",
                                        entry.name
                                    ),
                                })?;

                            let scales_data = self.tensor_data(
                                companions.scales.data_offsets.0,
                                companions.scales.data_offsets.1,
                            )?;
                            let qzeros_data = self.tensor_data(
                                companions.qzeros.data_offsets.0,
                                companions.qzeros.data_offsets.1,
                            )?;
                            let g_idx_data = companions
                                .g_idx
                                .map(|e| self.tensor_data(e.data_offsets.0, e.data_offsets.1))
                                .transpose()?;

                            // Derive in_features and out_features from qweight shape.
                            // qweight shape: [in_features/pack_factor, out_features]
                            let (packed_rows, out_features) =
                                Self::shape_to_rows_cols(&entry.shape)?;
                            // CAST: u8 → usize, bits is 4 or 8
                            #[allow(clippy::as_conversions)]
                            let pack_factor = 32 / config.bits as usize;
                            let in_features =
                                packed_rows.checked_mul(pack_factor).ok_or_else(|| {
                                    AnamnesisError::Parse {
                                        reason: "in_features overflow".into(),
                                    }
                                })?;

                            let bf16_native = dequantize_gptq_to_bf16(
                                weight_data,
                                scales_data,
                                qzeros_data,
                                g_idx_data,
                                in_features,
                                out_features,
                                config.group_size,
                                config.bits,
                                companions.scales.dtype,
                            )?;
                            // The kernel returns the GEMM-native
                            // [in_features, out_features] orientation (the
                            // canonical GPTQModel kernel layout the
                            // cross-validation fixtures anchor). A standard
                            // nn.Linear safetensors is [out, in] — apply the
                            // same boundary transpose GPTQModel's
                            // dequantize_model applies (`.T`).
                            let bf16_data =
                                transpose_bf16(&bf16_native, in_features, out_features)?;

                            // Output tensor: strip ".qweight" suffix, use ".weight".
                            let output_name = entry.name.strip_suffix(".qweight").map_or_else(
                                || entry.name.clone(),
                                |base| format!("{base}.weight"),
                            );
                            let output_shape = vec![out_features, in_features];

                            dequantized_data.push((output_name, bf16_data, output_shape));
                            on_tensor();
                            continue;
                        }
                        #[cfg(not(feature = "gptq"))]
                        QuantScheme::Gptq => {
                            return Err(AnamnesisError::Unsupported {
                                format: "GPTQ".into(),
                                detail: "GPTQ dequantization requires the `gptq` feature".into(),
                            });
                        }
                        #[cfg(feature = "awq")]
                        QuantScheme::Awq => {
                            let config =
                                self.header
                                    .awq_config
                                    .ok_or_else(|| AnamnesisError::Parse {
                                        reason: format!(
                                            "AWQ config not available for `{}`",
                                            entry.name
                                        ),
                                    })?;
                            let companions = self
                                .header
                                .find_awq_companions(&entry.name)
                                .ok_or_else(|| AnamnesisError::Parse {
                                    reason: format!(
                                        "AWQ companions not found for `{}`",
                                        entry.name
                                    ),
                                })?;

                            let scales_data = self.tensor_data(
                                companions.scales.data_offsets.0,
                                companions.scales.data_offsets.1,
                            )?;
                            let qzeros_data = self.tensor_data(
                                companions.qzeros.data_offsets.0,
                                companions.qzeros.data_offsets.1,
                            )?;

                            // Derive in_features and out_features from qweight + scales shapes.
                            // AWQ qweight: [in_features, out_features/pack_factor]
                            // scales: [num_groups, out_features]
                            let in_features = entry.shape.first().copied().ok_or_else(|| {
                                AnamnesisError::Parse {
                                    reason: "AWQ qweight has no first dimension".into(),
                                }
                            })?;
                            let out_features =
                                companions.scales.shape.last().copied().ok_or_else(|| {
                                    AnamnesisError::Parse {
                                        reason: "AWQ scales has no last dimension".into(),
                                    }
                                })?;

                            let bf16_native = dequantize_awq_to_bf16(
                                weight_data,
                                scales_data,
                                qzeros_data,
                                in_features,
                                out_features,
                                config.group_size,
                                config.bits,
                                companions.scales.dtype,
                            )?;
                            // The kernel returns the GEMM-native
                            // [in_features, out_features] orientation (the
                            // canonical AutoAWQ kernel layout the
                            // cross-validation fixtures anchor). A standard
                            // nn.Linear safetensors is [out, in] — transpose
                            // at the output-contract boundary, exactly as
                            // GPTQModel's dequantize_model does for its
                            // GEMM-native dequant (`.T`).
                            let bf16_data =
                                transpose_bf16(&bf16_native, in_features, out_features)?;

                            // Output tensor: strip ".qweight" suffix, use ".weight".
                            let output_name = entry.name.strip_suffix(".qweight").map_or_else(
                                || entry.name.clone(),
                                |base| format!("{base}.weight"),
                            );
                            let output_shape = vec![out_features, in_features];

                            dequantized_data.push((output_name, bf16_data, output_shape));
                            on_tensor();
                            continue;
                        }
                        #[cfg(not(feature = "awq"))]
                        QuantScheme::Awq => {
                            return Err(AnamnesisError::Unsupported {
                                format: "AWQ".into(),
                                detail: "AWQ dequantization requires the `awq` feature".into(),
                            });
                        }
                        #[cfg(feature = "bnb")]
                        QuantScheme::Bnb4 => {
                            let config =
                                self.header
                                    .bnb_config
                                    .ok_or_else(|| AnamnesisError::Parse {
                                        reason: format!(
                                            "BnB config not available for `{}`",
                                            entry.name
                                        ),
                                    })?;
                            let companions = self
                                .header
                                .find_bnb4_companions(&entry.name)
                                .ok_or_else(|| AnamnesisError::Parse {
                                    reason: format!(
                                        "BnB4 companions not found for `{}`",
                                        entry.name
                                    ),
                                })?;

                            let absmax_data = self.tensor_data(
                                companions.absmax.data_offsets.0,
                                companions.absmax.data_offsets.1,
                            )?;
                            let quant_map_data = self.tensor_data(
                                companions.quant_map.data_offsets.0,
                                companions.quant_map.data_offsets.1,
                            )?;

                            let total_elements =
                                entry.byte_len().checked_mul(2).ok_or_else(|| {
                                    AnamnesisError::Parse {
                                        reason: "BnB4 total_elements overflow".into(),
                                    }
                                })?;

                            // Read the quant_state JSON blob once: the
                            // double-quant path needs `nested_offset` from it
                            // BEFORE dequantizing, and the shape recovery
                            // below needs `shape`.
                            let quant_state_data = companions
                                .quant_state
                                .map(|qs_entry| {
                                    self.tensor_data(
                                        qs_entry.data_offsets.0,
                                        qs_entry.data_offsets.1,
                                    )
                                })
                                .transpose()?;

                            let bf16_data = if config.double_quant {
                                let nested_absmax = companions.nested_absmax.ok_or_else(|| {
                                    AnamnesisError::Parse {
                                        reason: format!(
                                            "BnB4 double-quant: nested_absmax not found for `{}`",
                                            entry.name
                                        ),
                                    }
                                })?;
                                let nested_quant_map =
                                    companions.nested_quant_map.ok_or_else(|| {
                                        AnamnesisError::Parse {
                                            reason: format!(
                                            "BnB4 double-quant: nested_quant_map not found for `{}`",
                                            entry.name
                                        ),
                                        }
                                    })?;
                                let nested_absmax_data = self.tensor_data(
                                    nested_absmax.data_offsets.0,
                                    nested_absmax.data_offsets.1,
                                )?;
                                let nested_quant_map_data = self.tensor_data(
                                    nested_quant_map.data_offsets.0,
                                    nested_quant_map.data_offsets.1,
                                )?;

                                // Infer nested_block_size from absmax count / nested_absmax count
                                let absmax_count = companions.absmax.num_elements();
                                let nested_absmax_count = nested_absmax.num_elements();
                                let nested_block_size = if nested_absmax_count > 0 {
                                    absmax_count.div_ceil(nested_absmax_count)
                                } else {
                                    256
                                };

                                // The nested_offset is mandatory for the
                                // double-quant absmax recovery; a DQ tensor
                                // without a quant_state blob cannot be
                                // decoded correctly.
                                let nested_offset = match quant_state_data {
                                    Some(qs_data) => {
                                        parse_bnb_quant_state_nested_offset(qs_data, &entry.name)?
                                    }
                                    None => {
                                        return Err(AnamnesisError::Parse {
                                            reason: format!(
                                                "BnB4 double-quant: quant_state blob not found \
                                                 for `{}` (required for nested_offset)",
                                                entry.name
                                            ),
                                        })
                                    }
                                };

                                dequantize_bnb4_double_quant_to_bf16(
                                    weight_data,
                                    absmax_data,
                                    quant_map_data,
                                    nested_absmax_data,
                                    nested_quant_map_data,
                                    nested_offset,
                                    total_elements,
                                    config.block_size,
                                    nested_block_size,
                                )?
                            } else {
                                dequantize_bnb4_to_bf16(
                                    weight_data,
                                    absmax_data,
                                    quant_map_data,
                                    total_elements,
                                    config.block_size,
                                )?
                            };

                            // BnB4 weights are stored flattened to [N, 1]. Recover the original
                            // 2D shape from the quant_state companion tensor (JSON blob with
                            // "shape" field), falling back to flat [total_elements] if absent.
                            let output_shape = if let Some(qs_data) = quant_state_data {
                                parse_bnb_quant_state_shape(qs_data, total_elements, &entry.name)?
                            } else {
                                vec![total_elements]
                            };

                            dequantized_data.push((entry.name.clone(), bf16_data, output_shape));
                            on_tensor();
                            continue;
                        }
                        #[cfg(feature = "bnb")]
                        QuantScheme::BnbInt8 => {
                            let scb_entry =
                                self.header.find_bnb_int8_scb(&entry.name).ok_or_else(|| {
                                    AnamnesisError::Parse {
                                        reason: format!(
                                            "BnB INT8 SCB companion not found for `{}`",
                                            entry.name
                                        ),
                                    }
                                })?;
                            let scb_data = self
                                .tensor_data(scb_entry.data_offsets.0, scb_entry.data_offsets.1)?;

                            // INT8 keeps its 2D shape [out_features, in_features].
                            let (out_features, in_features) =
                                Self::shape_to_rows_cols(&entry.shape)?;

                            let bf16_data = dequantize_bnb_int8_to_bf16(
                                weight_data,
                                scb_data,
                                out_features,
                                in_features,
                            )?;

                            // Output tensor: keep name, keep shape.
                            dequantized_data.push((
                                entry.name.clone(),
                                bf16_data,
                                entry.shape.clone(),
                            ));
                            on_tensor();
                            continue;
                        }
                        #[cfg(not(feature = "bnb"))]
                        QuantScheme::Bnb4 | QuantScheme::BnbInt8 => {
                            return Err(AnamnesisError::Unsupported {
                                format: "BnB".into(),
                                detail: "BnB dequantization requires the `bnb` feature".into(),
                            });
                        }
                        QuantScheme::Unquantized => {
                            // Shouldn't have quantized tensors in an unquantized model,
                            // but treat as passthrough to be safe.
                            passthrough_refs.push((&entry.name, weight_data, &entry.shape));
                            continue;
                        }
                    };

                    dequantized_data.push((entry.name.clone(), bf16_bytes, entry.shape.clone()));
                    on_tensor();
                }
                TensorRole::Scale
                | TensorRole::ZeroPoint
                | TensorRole::GroupIndex
                | TensorRole::QuantMap
                | TensorRole::NestedScale
                | TensorRole::QuantState => {
                    // Companion tensors are consumed during dequantization; skip.
                }
                TensorRole::Passthrough => {
                    let data = self.tensor_data(entry.data_offsets.0, entry.data_offsets.1)?;
                    passthrough_refs.push((&entry.name, data, &entry.shape));
                }
            }
        }

        Ok((dequantized_data, passthrough_refs))
    }

    /// Internal: build the `safetensors` `TensorView` list from the dequantised
    /// (owned) tensors and the passthrough (borrowed) tensors. Shared by both
    /// `remember` destinations; the views borrow `dequantized_data`, so the
    /// caller must keep it alive until serialization completes.
    fn build_views<'a>(
        &'a self,
        dequantized_data: &'a [(String, Vec<u8>, Vec<usize>)],
        passthrough_refs: &[(&'a str, &'a [u8], &'a [usize])],
    ) -> crate::Result<Vec<(String, safetensors::tensor::TensorView<'a>)>> {
        // Build TensorView list for serialization.
        // Dequantized tensors use safetensors::Dtype::BF16.
        // Passthrough tensors keep their original dtype.
        let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();

        for (name, data, shape) in dequantized_data {
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.clone(), data)
                    .map_err(|e| AnamnesisError::Parse {
                        reason: format!("failed to create TensorView for `{name}`: {e}"),
                    })?;
            views.push((name.clone(), view));
        }

        for &(name, data, shape) in passthrough_refs {
            // Look up the original dtype for this passthrough tensor.
            let entry = self
                .header
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("passthrough tensor `{name}` not found in header"),
                })?;
            let st_dtype = entry.dtype.to_safetensors_dtype()?;
            let view = safetensors::tensor::TensorView::new(st_dtype, shape.to_vec(), data)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("failed to create TensorView for `{name}`: {e}"),
                })?;
            views.push((name.to_owned(), view));
        }

        Ok(views)
    }

    /// Internal: dequantize to `BF16` and write, with optional progress callback.
    fn remember_bf16_inner<F>(&self, output_path: &Path, on_tensor: F) -> crate::Result<()>
    where
        F: FnMut(),
    {
        let (dequantized_data, passthrough_refs) = self.dequantize_all(on_tensor)?;
        let views = self.build_views(&dequantized_data, &passthrough_refs)?;

        // Serialize to file. The safetensors writer streams tensor bodies one at
        // a time, so the file path's peak stays at the dequantised set — unlike
        // `remember_to_bytes`, which holds the whole serialized `Vec`.
        let metadata = self.header.metadata.clone();
        safetensors::tensor::serialize_to_file(views, metadata, output_path).map_err(
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

    /// Internal: dequantize to `BF16` and return the serialized safetensors bytes.
    fn remember_to_bytes_bf16(&self) -> crate::Result<Vec<u8>> {
        let (dequantized_data, passthrough_refs) = self.dequantize_all(|| {})?;
        let views = self.build_views(&dequantized_data, &passthrough_refs)?;

        let metadata = self.header.metadata.clone();
        safetensors::tensor::serialize(views, metadata).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to serialize safetensors bytes: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// BnB4 quant_state shape recovery
// ---------------------------------------------------------------------------

/// Parses the original tensor shape from a `BnB` `quant_state` companion tensor.
///
/// The `quant_state.bitsandbytes__nf4` (or `__fp4`) tensor stores a `JSON` blob
/// as raw `U8` bytes. The blob contains a `"shape"` field with the original
/// 2D tensor dimensions (e.g., `[2048, 8192]`).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the `JSON` is malformed, the `"shape"`
/// field is missing, or the recovered shape does not match `total_elements`.
#[cfg(feature = "bnb")]
fn parse_bnb_quant_state_shape(
    qs_data: &[u8],
    total_elements: usize,
    weight_name: &str,
) -> crate::Result<Vec<usize>> {
    let qs_str = std::str::from_utf8(qs_data).map_err(|e| AnamnesisError::Parse {
        reason: format!("quant_state for `{weight_name}` is not valid UTF-8: {e}"),
    })?;

    let qs_json: serde_json::Value =
        serde_json::from_str(qs_str).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to parse quant_state JSON for `{weight_name}`: {e}"),
        })?;

    let shape_arr = qs_json
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("quant_state for `{weight_name}` missing \"shape\" array"),
        })?;

    let shape: Vec<usize> = shape_arr
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!(
                        "quant_state shape dimension not a valid usize for `{weight_name}`"
                    ),
                })
        })
        .collect::<crate::Result<_>>()?;

    // Validate: product of recovered shape must equal total_elements.
    let product: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("quant_state shape overflow for `{weight_name}`"),
        })?;

    if product != total_elements {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "quant_state shape {shape:?} product {product} != total_elements {total_elements} \
                 for `{weight_name}`"
            ),
        });
    }

    Ok(shape)
}

/// Parses the double-quant `nested_offset` from a `BnB` `quant_state`
/// companion tensor.
///
/// `bitsandbytes` double quantization subtracts the mean of the per-block
/// absmax values before nested-quantizing them, and stores that mean in the
/// `quant_state` `JSON` blob as `"nested_offset"`. Recovery must add it back
/// (`absmax = nested_dequant(...) + nested_offset`); omitting it biases every
/// recovered absmax low by the offset.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the `JSON` is malformed or the
/// `"nested_offset"` field is missing or not a number. The field is
/// mandatory for double-quant states: every `bitsandbytes` serialization
/// that emits `nested_absmax` / `nested_quant_map` also emits it, so its
/// absence indicates a malformed or truncated `quant_state`.
#[cfg(feature = "bnb")]
fn parse_bnb_quant_state_nested_offset(qs_data: &[u8], weight_name: &str) -> crate::Result<f32> {
    let qs_str = std::str::from_utf8(qs_data).map_err(|e| AnamnesisError::Parse {
        reason: format!("quant_state for `{weight_name}` is not valid UTF-8: {e}"),
    })?;

    let qs_json: serde_json::Value =
        serde_json::from_str(qs_str).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to parse quant_state JSON for `{weight_name}`: {e}"),
        })?;

    let offset_f64 = qs_json
        .get("nested_offset")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "quant_state for `{weight_name}` missing \"nested_offset\" (required for \
                 double-quant absmax recovery)"
            ),
        })?;

    // CAST: the JSON value is the decimal rendering of a bitsandbytes f32;
    // narrowing f64 → f32 recovers it exactly.
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    Ok(offset_f64 as f32)
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::float_cmp
)]
mod tests {
    use super::*;

    /// Build a minimal safetensors file in memory with the given tensors.
    fn build_safetensors(tensors: &[(&str, safetensors::Dtype, &[usize], &[u8])]) -> Vec<u8> {
        let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                let view =
                    safetensors::tensor::TensorView::new(*dtype, shape.to_vec(), data).unwrap();
                (*name, view)
            })
            .collect();
        safetensors::tensor::serialize(views, None).unwrap()
    }

    #[test]
    fn parse_and_inspect_unquantized() {
        // 2 BF16 tensors
        let bf16_data = vec![0x80, 0x3F]; // BF16 1.0
        let file = build_safetensors(&[
            ("weight", safetensors::Dtype::BF16, &[1], &bf16_data),
            ("norm", safetensors::Dtype::BF16, &[1], &bf16_data),
        ]);

        let tmp = std::env::temp_dir().join("test_unquant.safetensors");
        std::fs::write(&tmp, &file).unwrap();

        let model = parse(&tmp).unwrap();
        let info = model.inspect();

        assert_eq!(info.format, QuantScheme::Unquantized);
        assert_eq!(info.quantized, 0);
        assert_eq!(info.passthrough, 2);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn parse_nonexistent_file() {
        let result = parse("/tmp/nonexistent_anamnesis_test.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn parse_invalid_data() {
        let tmp = std::env::temp_dir().join("test_invalid.safetensors");
        std::fs::write(&tmp, b"not a safetensors file").unwrap();

        let result = parse(&tmp);
        assert!(result.is_err());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn remember_passthrough_only() {
        // BF16 tensor with known value: 2.0 = 0x4000 in BF16
        let bf16_data = vec![0x00, 0x40, 0x00, 0x40]; // two BF16 2.0
        let file = build_safetensors(&[("weight", safetensors::Dtype::BF16, &[2], &bf16_data)]);

        let tmp_in = std::env::temp_dir().join("test_pass_in.safetensors");
        let tmp_out = std::env::temp_dir().join("test_pass_out.safetensors");
        std::fs::write(&tmp_in, &file).unwrap();

        let model = parse(&tmp_in).unwrap();
        model.remember(&tmp_out, TargetDtype::BF16).unwrap();

        // Read output and verify bytes match
        let out_data = std::fs::read(&tmp_out).unwrap();
        let out_model = parse(&tmp_out).unwrap();
        let out_info = out_model.inspect();
        assert_eq!(out_info.passthrough, 1);

        // Verify the tensor data is preserved
        let entry = &out_model.header.tensors[0];
        let data_offset = out_model.header.header_size + 8;
        let tensor_bytes =
            &out_data[data_offset + entry.data_offsets.0..data_offset + entry.data_offsets.1];
        assert_eq!(tensor_bytes, &bf16_data);

        std::fs::remove_file(&tmp_in).ok();
        std::fs::remove_file(&tmp_out).ok();
    }

    /// Builds a raw per-tensor-FP8 safetensors fixture in memory: a 2×2 `F8_E4M3`
    /// weight (`1.0`), a scalar `F32` scale (`2.0`), and a `BF16` passthrough
    /// norm (`1.0`). Built by hand because the `safetensors` crate may not
    /// serialize `F8_E4M3`. Shared by the file and bytes `remember` round-trips.
    fn build_fp8_per_tensor_fixture() -> Vec<u8> {
        let fp8_data = vec![0x38u8; 4]; // 2x2 of 1.0 in E4M3
        let scale_data = 2.0_f32.to_le_bytes().to_vec();
        let norm_data = vec![0x80, 0x3F]; // BF16 1.0

        let mut header_map = serde_json::Map::new();

        // FP8 weight at offset 0, length 4
        let mut w_info = serde_json::Map::new();
        w_info.insert("dtype".into(), "F8_E4M3".into());
        w_info.insert("shape".into(), serde_json::json!([2, 2]));
        w_info.insert("data_offsets".into(), serde_json::json!([0, 4]));
        header_map.insert("layer.weight".into(), w_info.into());

        // F32 scale at offset 4, length 4
        let mut s_info = serde_json::Map::new();
        s_info.insert("dtype".into(), "F32".into());
        s_info.insert("shape".into(), serde_json::json!([1]));
        s_info.insert("data_offsets".into(), serde_json::json!([4, 8]));
        header_map.insert("layer.weight_scale".into(), s_info.into());

        // BF16 norm at offset 8, length 2
        let mut n_info = serde_json::Map::new();
        n_info.insert("dtype".into(), "BF16".into());
        n_info.insert("shape".into(), serde_json::json!([1]));
        n_info.insert("data_offsets".into(), serde_json::json!([8, 10]));
        header_map.insert("norm.weight".into(), n_info.into());

        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();

        // Build raw safetensors file: 8-byte length + header + data
        // CAST: usize → u64, header length fits in u64
        #[allow(clippy::as_conversions)]
        let header_len = header_bytes.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&fp8_data);
        file_bytes.extend_from_slice(&scale_data);
        file_bytes.extend_from_slice(&norm_data);
        file_bytes
    }

    #[test]
    fn remember_fp8_round_trip() {
        // FP8 weight (1.0) × per-tensor scale (2.0) → BF16 2.0, plus a BF16
        // passthrough norm; the scale tensor is consumed (absent from output).
        let file_bytes = build_fp8_per_tensor_fixture();

        let tmp_in = std::env::temp_dir().join("test_fp8_in.safetensors");
        let tmp_out = std::env::temp_dir().join("test_fp8_out.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();

        let model = parse(&tmp_in).unwrap();
        assert_eq!(model.header.scheme, QuantScheme::PerTensorFp8);
        assert_eq!(model.inspect().quantized, 1);

        model.remember(&tmp_out, TargetDtype::BF16).unwrap();

        // Read output and verify
        let out_model = parse(&tmp_out).unwrap();
        let out_info = out_model.inspect();
        // Output should have: 1 passthrough (was FP8, now BF16) + 1 passthrough (norm)
        // Scale tensor should be absent
        assert_eq!(out_info.passthrough, 2); // both are now BF16
        assert_eq!(out_info.quantized, 0);

        // Verify the weight values: 1.0 * 2.0 = 2.0 → BF16 0x4000 → LE [0x00, 0x40]
        let w_entry = out_model
            .header
            .tensors
            .iter()
            .find(|t| t.name == "layer.weight")
            .unwrap();
        let data_start = out_model.header.header_size + 8;
        let out_bytes = std::fs::read(&tmp_out).unwrap();
        let w_data =
            &out_bytes[data_start + w_entry.data_offsets.0..data_start + w_entry.data_offsets.1];
        // 4 elements × 2 bytes = 8 bytes
        assert_eq!(w_data.len(), 8);
        for chunk in w_data.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }

        std::fs::remove_file(&tmp_in).ok();
        std::fs::remove_file(&tmp_out).ok();
    }

    #[test]
    fn remember_to_bytes_fp8_round_trip() {
        let file_bytes = build_fp8_per_tensor_fixture();

        let tmp_in = std::env::temp_dir().join("test_fp8_bytes_in.safetensors");
        let tmp_out = std::env::temp_dir().join("test_fp8_bytes_out.safetensors");
        std::fs::write(&tmp_in, &file_bytes).unwrap();

        let model = parse(&tmp_in).unwrap();
        assert_eq!(model.header.scheme, QuantScheme::PerTensorFp8);

        // Core pairing invariant: the in-memory bytes are byte-identical to what
        // the file path writes (same views, same metadata, same serialization).
        let bytes = model.remember_to_bytes(TargetDtype::BF16).unwrap();
        model.remember(&tmp_out, TargetDtype::BF16).unwrap();
        let file_out = std::fs::read(&tmp_out).unwrap();
        assert_eq!(
            bytes, file_out,
            "remember_to_bytes must match remember's file bytes"
        );

        // Round-trip: parse the returned bytes back and verify the dequant.
        std::fs::write(&tmp_out, &bytes).unwrap();
        let out_model = parse(&tmp_out).unwrap();
        let out_info = out_model.inspect();
        assert_eq!(out_info.passthrough, 2); // weight (now BF16) + norm
        assert_eq!(out_info.quantized, 0); // scale consumed

        // Weight values: 1.0 × 2.0 = 2.0 → BF16 0x4000 → LE [0x00, 0x40].
        let w_entry = out_model
            .header
            .tensors
            .iter()
            .find(|t| t.name == "layer.weight")
            .unwrap();
        let data_start = out_model.header.header_size + 8;
        let w_data =
            &bytes[data_start + w_entry.data_offsets.0..data_start + w_entry.data_offsets.1];
        assert_eq!(w_data.len(), 8); // 4 elements × 2 bytes
        for chunk in w_data.chunks_exact(2) {
            assert_eq!(chunk, &[0x00, 0x40], "expected BF16 2.0");
        }

        std::fs::remove_file(&tmp_in).ok();
        std::fs::remove_file(&tmp_out).ok();
    }

    #[test]
    fn target_dtype_display() {
        assert_eq!(TargetDtype::BF16.to_string(), "BF16");
    }
}
