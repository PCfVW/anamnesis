// SPDX-License-Identifier: MIT OR Apache-2.0

//! `NPZ`/`NPY` archive parsing — zero-copy bulk read for near-I/O-speed extraction.
//!
//! This module implements a lean `NPY` header parser and bulk data reader that
//! bypasses per-element deserialization entirely. For little-endian data on
//! little-endian machines (>99% of ML files on x86/ARM), the raw bytes in the
//! `NPY` file ARE the correct in-memory representation — no per-element
//! processing is needed.
//!
//! The `ZIP` layer is handled by the `zip` crate directly. `NPZ` archives
//! typically use the `STORE` method (no compression) for large arrays, making
//! the `ZIP` layer a pure passthrough.
//!
//! # Performance
//!
//! On a 302 MB `NPZ` file (Gemma Scope 2B SAE, 5 `F32` arrays):
//! - Raw `fs::read`: ~64 ms (I/O baseline)
//! - This parser: near I/O baseline (single bulk read, zero per-element work)
//! - Previous `npyz`-backed parser: ~1,500 ms (per-element deserialization)

use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Seek};
use std::path::Path;

use crate::error::AnamnesisError;
use crate::parse::utils::byteswap_inplace;

// ---------------------------------------------------------------------------
// NPY magic
// ---------------------------------------------------------------------------

/// `NPY` magic bytes: `\x93NUMPY`.
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

// ---------------------------------------------------------------------------
// NpzDtype
// ---------------------------------------------------------------------------

/// Element data type for tensors parsed from `NPZ`/`NPY` archives.
///
/// Includes `BF16` which `NumPy` cannot represent natively. When a `BF16`
/// tensor is detected (stored as void/`V2` by `JAX`), anamnesis reads the
/// raw bytes and interprets them as `half::bf16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NpzDtype {
    /// Boolean (1 byte per element).
    Bool,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// 16-bit IEEE 754 half-precision (`f2` in `NumPy`).
    F16,
    /// 16-bit brain floating point (`BF16`).
    ///
    /// `NumPy` has no native `BF16` dtype. `JAX` stores `BF16` as void (`V2`)
    /// with `bfloat16` metadata. This variant represents that interpretation.
    BF16,
    /// 32-bit IEEE 754 single-precision.
    F32,
    /// 64-bit IEEE 754 double-precision.
    F64,
}

impl NpzDtype {
    /// Returns the number of bytes per element for this dtype.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// Displays the canonical uppercase name (e.g., `"F32"`, `"BF16"`, `"BOOL"`).
///
/// This is the string used in inspection output and cross-validation tests.
impl fmt::Display for NpzDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::U16 => "U16",
            Self::I16 => "I16",
            Self::U32 => "U32",
            Self::I32 => "I32",
            Self::U64 => "U64",
            Self::I64 => "I64",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// NpzTensor
// ---------------------------------------------------------------------------

/// A single tensor extracted from an `NPZ` archive.
///
/// Contains the tensor name, shape, dtype, and raw byte data in little-endian,
/// row-major (C) order. Framework consumers can interpret `data` directly
/// according to `dtype` and `shape`.
#[derive(Debug, Clone)]
pub struct NpzTensor {
    /// Tensor name as stored in the archive (without `.npy` extension).
    /// Matches the `HashMap` key returned by [`parse_npz`].
    pub name: String,
    /// Tensor dimensions (e.g., `[16384, 2304]`).
    pub shape: Vec<usize>,
    /// Element data type (e.g., `F32`, `BF16`).
    pub dtype: NpzDtype,
    /// Raw bytes in row-major (C) order, little-endian.
    /// Length equals `product(shape) × dtype.byte_size()`.
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// NPY header parsing
// ---------------------------------------------------------------------------

/// Parsed `NPY` header: dtype, endianness, memory order, and shape.
struct NpyHeader {
    /// Parsed element data type.
    dtype: NpzDtype,
    /// `true` if the data is stored big-endian (descr prefix `>`).
    big_endian: bool,
    /// `true` if the data is in Fortran (column-major) order.
    fortran_order: bool,
    /// Array shape (e.g., `[16384, 2304]`).
    shape: Vec<usize>,
}

/// Parses the `NPY` header from a reader, consuming the header bytes.
///
/// Supports `NPY` format versions 1.0, 2.0, and 3.0.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the magic bytes, version, or header
/// dict are malformed.
fn parse_npy_header(reader: &mut impl Read) -> crate::Result<NpyHeader> {
    // Read magic (6 bytes) + major (1) + minor (1) = 8 bytes.
    let mut preamble = [0u8; 8];
    reader
        .read_exact(&mut preamble)
        .map_err(|e| AnamnesisError::Parse {
            reason: format!("NPY preamble read failed: {e}"),
        })?;

    // INDEX: preamble is exactly 8 bytes, slicing [..6] is safe
    #[allow(clippy::indexing_slicing)]
    if &preamble[..6] != NPY_MAGIC {
        return Err(AnamnesisError::Parse {
            reason: "invalid NPY magic bytes".into(),
        });
    }

    // INDEX: preamble[6] is safe (8-byte array)
    // EXPLICIT: preamble[7] (minor version) is read but unused — the NPY spec
    // defines only minor version 0 for all major versions (1, 2, 3).
    #[allow(clippy::indexing_slicing)]
    let major = preamble[6];

    // Read header length (version-dependent).
    let header_len: usize = match major {
        1 => {
            let mut buf = [0u8; 2];
            reader
                .read_exact(&mut buf)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("NPY v1 header length read failed: {e}"),
                })?;
            usize::from(u16::from_le_bytes(buf))
        }
        2 | 3 => {
            let mut buf = [0u8; 4];
            reader
                .read_exact(&mut buf)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("NPY v{major} header length read failed: {e}"),
                })?;
            // CAST: u32 → usize, NPY headers are always small
            #[allow(clippy::as_conversions)]
            let len = u32::from_le_bytes(buf) as usize;
            len
        }
        _ => {
            return Err(AnamnesisError::Unsupported {
                format: "NPY".into(),
                detail: format!("unsupported NPY version {major}"),
            });
        }
    };

    // Read header string.
    let mut header_buf = vec![0u8; header_len];
    reader
        .read_exact(&mut header_buf)
        .map_err(|e| AnamnesisError::Parse {
            reason: format!("NPY header data read failed: {e}"),
        })?;

    let header_str = std::str::from_utf8(&header_buf).map_err(|e| AnamnesisError::Parse {
        reason: format!("NPY header is not valid UTF-8: {e}"),
    })?;

    // Parse the Python dict literal.
    let (dtype, big_endian) = extract_descr(header_str)?;
    let fortran_order = extract_fortran_order(header_str);
    let shape = extract_shape(header_str)?;

    Ok(NpyHeader {
        dtype,
        big_endian,
        fortran_order,
        shape,
    })
}

/// Extracts the `descr` field from the `NPY` header dict and maps it to
/// `(NpzDtype, is_big_endian)`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the descr field is missing.
/// Returns [`AnamnesisError::Unsupported`] if the dtype string is not recognized.
fn extract_descr(header: &str) -> crate::Result<(NpzDtype, bool)> {
    // Find 'descr': then extract the quoted value after it.
    let descr_start = header.find("'descr'").or_else(|| header.find("\"descr\""));
    let descr_start = descr_start.ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header missing 'descr' field".into(),
    })?;

    // Skip past 'descr': to find the value.
    let after_key = header
        .get(descr_start..)
        .and_then(|s| s.find(':').map(|i| descr_start + i + 1))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header 'descr' field has no value".into(),
        })?;

    let value_str = header
        .get(after_key..)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header truncated after 'descr'".into(),
        })?;

    // Extract the string between quotes. Detect the quote character from the
    // first quote found in the value portion (not the entire header tail),
    // so mixed-quote headers like {'descr': "<f4", 'other': ...} work.
    let trimmed = value_str.trim_start();
    let quote_char = match trimmed.as_bytes().first() {
        Some(b'\'') => '\'',
        Some(b'"') => '"',
        _ => {
            return Err(AnamnesisError::Parse {
                reason: "NPY header 'descr' value not quoted".into(),
            });
        }
    };
    let inner = trimmed.get(1..).ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header 'descr' value truncated after opening quote".into(),
    })?;
    let closing = inner
        .find(quote_char)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header 'descr' value missing closing quote".into(),
        })?;
    let descr = inner.get(..closing).ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header 'descr' extraction failed".into(),
    })?;

    parse_descr(descr)
}

/// Maps a `NumPy` dtype descriptor string (e.g., `<f4`, `>u2`, `|V2`) to
/// `(NpzDtype, is_big_endian)`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if the descriptor is not recognized.
fn parse_descr(descr: &str) -> crate::Result<(NpzDtype, bool)> {
    // First character is endianness: '<' = LE, '>' = BE, '|' = N/A, '=' = native.
    // BORROW: explicit .as_bytes() to inspect endianness prefix byte
    let bytes = descr.as_bytes();
    if bytes.len() < 2 {
        return Err(AnamnesisError::Unsupported {
            format: "NPY".into(),
            detail: format!("dtype descriptor too short: '{descr}'"),
        });
    }

    // INDEX: bytes.len() >= 2, so [0] and [1..] are safe
    #[allow(clippy::indexing_slicing)]
    let endian_char = bytes[0];
    #[allow(clippy::indexing_slicing)]
    let type_str = &descr[1..];

    // EXPLICIT: '=' (native endian) is treated as little-endian. All modern ML
    // platforms (x86-64, ARM64) are LE; a BE-native machine would need '>' explicitly.
    let big_endian = endian_char == b'>';

    let dtype = match type_str {
        // Boolean
        "b1" => NpzDtype::Bool,
        // Unsigned integers
        "u1" => NpzDtype::U8,
        "u2" => NpzDtype::U16,
        "u4" => NpzDtype::U32,
        "u8" => NpzDtype::U64,
        // Signed integers
        "i1" => NpzDtype::I8,
        "i2" => NpzDtype::I16,
        "i4" => NpzDtype::I32,
        "i8" => NpzDtype::I64,
        // Floats
        "f2" => NpzDtype::F16,
        "f4" => NpzDtype::F32,
        "f8" => NpzDtype::F64,
        // Void (BF16 via JAX convention)
        "V2" => NpzDtype::BF16,
        _ => {
            return Err(AnamnesisError::Unsupported {
                format: "NPY".into(),
                detail: format!("unsupported dtype descriptor '{descr}'"),
            });
        }
    };

    Ok((dtype, big_endian))
}

/// Extracts the `fortran_order` field from the `NPY` header dict.
///
/// Returns `false` if the field is not found or has any value other than
/// `True` (defaults to C-order).
// EXPLICIT: returns false (C-order) for missing or malformed fortran_order
// fields. The NPY spec mandates this field, but defaulting to C-order is
// the safe choice — Fortran-order is rejected by parse_npz anyway.
fn extract_fortran_order(header: &str) -> bool {
    // Look for 'fortran_order': True
    header
        .find("'fortran_order'")
        .or_else(|| header.find("\"fortran_order\""))
        .is_some_and(|pos| {
            header
                .get(pos..)
                .and_then(|s| s.find(':').and_then(|i| s.get(i + 1..)))
                .is_some_and(|val| val.trim_start().starts_with("True"))
        })
}

/// Extracts the `shape` field from the `NPY` header dict and parses it as
/// a tuple of `usize` dimensions.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the shape field is missing or
/// contains invalid dimension values.
fn extract_shape(header: &str) -> crate::Result<Vec<usize>> {
    let shape_start = header.find("'shape'").or_else(|| header.find("\"shape\""));
    let shape_start = shape_start.ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header missing 'shape' field".into(),
    })?;

    // Find the opening paren after 'shape':
    let after_key = header
        .get(shape_start..)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header truncated at 'shape'".into(),
        })?;
    let paren_open = after_key.find('(').ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header 'shape' value missing opening paren".into(),
    })?;
    let inner_start = after_key
        .get(paren_open + 1..)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header 'shape' truncated after paren".into(),
        })?;
    let paren_close = inner_start.find(')').ok_or_else(|| AnamnesisError::Parse {
        reason: "NPY header 'shape' value missing closing paren".into(),
    })?;
    let inner = inner_start
        .get(..paren_close)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "NPY header 'shape' extraction failed".into(),
        })?;

    inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("NPY shape dimension parse error: {e}"),
                })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bulk data extraction
// ---------------------------------------------------------------------------

/// Reads array data as raw little-endian bytes into a freshly allocated `Vec`.
///
/// For little-endian data on a little-endian machine, the raw bytes are the
/// correct in-memory representation — zero per-element processing. For
/// big-endian data, a byte-swap pass is applied in-place after the bulk read.
///
/// Allocates the destination buffer with [`Vec::with_capacity`] and fills it
/// via `reader.take(data_bytes).read_to_end(...)`, avoiding the full
/// zero-init `memset` that `vec![0u8; data_bytes]` would otherwise perform
/// before `read_exact` overwrites every byte. On the 302 MB Gemma Scope
/// `params.npz` this eliminates a 302 MB write that previously sat directly
/// on the parse hot path.
///
/// `read_to_end` does not error on a short read, so this function explicitly
/// validates that exactly `data_bytes` bytes were read.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the element count or byte count
/// overflows `usize`, if the read fails, or if the underlying reader
/// returns fewer than `data_bytes` bytes (truncated entry).
fn read_array_data(reader: &mut impl Read, header: &NpyHeader) -> crate::Result<Vec<u8>> {
    let n_elements: usize = header
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "element count overflow".into(),
        })?;

    let data_bytes = n_elements
        .checked_mul(header.dtype.byte_size())
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "data byte count overflow".into(),
        })?;

    // CAST: usize → u64, byte counts always fit in u64 (widening on every
    // supported target — 32-bit usize ≤ u32::MAX, 64-bit usize ≤ u64::MAX).
    #[allow(clippy::as_conversions)]
    let limit = data_bytes as u64;

    let mut buf: Vec<u8> = Vec::with_capacity(data_bytes);
    let bytes_read =
        reader
            .take(limit)
            .read_to_end(&mut buf)
            .map_err(|e| AnamnesisError::Parse {
                reason: format!("array data read failed ({data_bytes} bytes): {e}"),
            })?;

    if bytes_read != data_bytes {
        return Err(AnamnesisError::Parse {
            reason: format!("array data truncated: expected {data_bytes} bytes, got {bytes_read}"),
        });
    }

    // Byte-swap for big-endian data with multi-byte elements.
    if header.big_endian && header.dtype.byte_size() > 1 {
        byteswap_inplace(&mut buf, header.dtype.byte_size());
    }

    Ok(buf)
}

// ---------------------------------------------------------------------------
// NpzTensorInfo / NpzInspectInfo (lightweight, header-only)
// ---------------------------------------------------------------------------

/// Lightweight per-tensor metadata from an `NPZ` archive.
///
/// Produced by [`inspect_npz`]. Contains only `NPY` header information —
/// no tensor data is read from the archive.
#[derive(Debug, Clone)]
pub struct NpzTensorInfo {
    /// Tensor name (without `.npy` extension).
    pub name: String,
    /// Tensor dimensions (e.g., `[16384, 2304]`).
    pub shape: Vec<usize>,
    /// Element data type (e.g., `F32`, `BF16`).
    pub dtype: NpzDtype,
    /// Total byte length (`product(shape) * dtype.byte_size()`).
    pub byte_len: usize,
}

/// Summary information about an `NPZ` archive, derived from headers only.
///
/// Produced by [`inspect_npz`]. No tensor data is loaded — peak memory
/// is proportional to the number of tensors (metadata only), not the
/// file size.
#[derive(Debug, Clone)]
#[must_use]
pub struct NpzInspectInfo {
    /// Per-tensor metadata.
    pub tensors: Vec<NpzTensorInfo>,
    /// Total size of all tensor data in bytes.
    pub total_bytes: u64,
    /// Distinct dtypes found (in order of first occurrence).
    pub dtypes: Vec<NpzDtype>,
}

impl fmt::Display for NpzInspectInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Format:      NPZ archive")?;
        write!(f, "\nTensors:     {}", self.tensors.len())?;
        write!(
            f,
            "\nTotal size:  {}",
            crate::inspect::format_bytes(self.total_bytes)
        )?;
        let dtype_list: String = self
            .dtypes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "\nDtypes:      {dtype_list}")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Inspects an `NPZ` archive on disk, returning metadata for all arrays
/// without reading tensor data.
///
/// Reads only `NPY` headers (~128 bytes per array) — no bulk data
/// extraction. For a 300 MB file this uses kilobytes of memory instead
/// of 300 MB.
///
/// This is a thin convenience wrapper that opens `path` as a [`std::fs::File`]
/// and delegates to [`inspect_npz_from_reader`]. Callers that need to inspect
/// an `NPZ` from any other `Read + Seek` substrate (in-memory `Cursor`,
/// HTTP-range-backed adapter, custom transport) should call
/// [`inspect_npz_from_reader`] directly.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or read.
///
/// Returns [`AnamnesisError::Parse`] if the `ZIP` archive is malformed or
/// an `NPY` header is invalid.
///
/// Returns [`AnamnesisError::Unsupported`] if an array uses Fortran order
/// or an unsupported dtype.
///
/// # Memory
///
/// Allocates only per-tensor metadata (name, shape, dtype). No tensor
/// data is read or allocated. Peak memory ≈ kilobytes for typical models.
///
/// **Note:** If a shape's element count overflows `usize`, `byte_len`
/// saturates to `usize::MAX` and `total_bytes` saturates to `u64::MAX`.
/// This differs from `parse_npz`, which returns `Err` on the same overflow.
/// The distinction is intentional: `inspect_npz` is best-effort metadata
/// extraction, while `parse_npz` must validate before allocating buffers.
pub fn inspect_npz(path: impl AsRef<Path>) -> crate::Result<NpzInspectInfo> {
    let file = std::fs::File::open(path.as_ref())?;
    inspect_npz_from_reader(file)
}

/// Inspects an `NPZ` archive from any `Read + Seek` source, returning
/// metadata for all arrays without reading tensor data.
///
/// This is the reader-generic core of [`inspect_npz`]: the path-based
/// variant is a two-line wrapper that opens a file and delegates here. By
/// accepting any `Read + Seek` substrate, callers can supply alternative
/// I/O backings — in-memory cursors (`std::io::Cursor`), shared-buffer
/// adapters, or HTTP-range-backed transports that lazily fetch only the
/// bytes they need.
///
/// # Range-read access pattern
///
/// `NPZ` is a `ZIP` archive whose central directory lives at the *end* of
/// the file. An HTTP-range-backed adapter typically only needs three
/// logical fetches to satisfy this function:
///
/// 1. **End-of-file scan for the EOCD record** (~64 KiB worst case) — the
///    `zip` crate seeks to the file's end and scans backwards for the
///    end-of-central-directory signature.
/// 2. **One read for the central directory** (a few KiB for typical ML
///    `NPZ` archives, since each `STORE`d entry produces one fixed-size
///    record plus its UTF-8 name) — the `zip` crate seeks to the offset
///    recorded in the EOCD and reads the full directory.
/// 3. **One read per entry for the local file header + `NPY` header**
///    (~512 B per `.npy` entry) — for each entry, the `zip` crate seeks
///    to the local file header offset, then this function reads the
///    `NPY` preamble + dict header (~128 B in practice).
///
/// A naive adapter that issues one HTTP range request per `read`/`seek`
/// call will still work but may be inefficient. Adapters that prefetch
/// and cache the EOCD region and the central directory on first access
/// amortise the round trips effectively to two HTTP requests plus one
/// per entry. For a typical 5-array Gemma Scope `params.npz`, that is
/// ~7 small range requests covering well under 100 KiB instead of the
/// full ~300 MiB download.
///
/// Anamnesis does not ship an HTTP transport itself — the network layer
/// belongs in downstream crates (e.g., `hf-fm`'s safetensors range-reader
/// extended to `NPZ`). This function defines the I/O contract such an
/// adapter must satisfy.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if a `read` or `seek` on the supplied
/// reader fails.
///
/// Returns [`AnamnesisError::Parse`] if the `ZIP` archive is malformed or
/// an `NPY` header is invalid.
///
/// Returns [`AnamnesisError::Unsupported`] if an array uses Fortran order
/// or an unsupported dtype.
///
/// # Memory
///
/// Allocates only per-tensor metadata (name, shape, dtype). No tensor
/// data is read or allocated. Peak memory is proportional to the number
/// of entries (a few hundred bytes per tensor plus the transient `NPY`
/// header buffer during parsing), independent of the archive's
/// data-segment size.
///
/// **Saturation note:** if a shape's element count overflows `usize`,
/// `byte_len` saturates to `usize::MAX` and `total_bytes` saturates to
/// `u64::MAX`. Behaviour matches [`inspect_npz`] and differs from
/// `parse_npz`, which returns `Err` on the same overflow.
pub fn inspect_npz_from_reader<R: Read + Seek>(reader: R) -> crate::Result<NpzInspectInfo> {
    let mut archive = zip::ZipArchive::new(reader)?;

    let mut tensors = Vec::with_capacity(archive.len());
    let mut total_bytes: u64 = 0;
    let mut dtypes: Vec<NpzDtype> = Vec::new();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to read ZIP entry {i}: {e}"),
        })?;

        // Strip .npy suffix; skip non-.npy entries (e.g., __MACOSX/).
        // BORROW: .to_owned() converts &str from zip entry to owned String
        let full_name = entry.name().to_owned();
        let name = match full_name.strip_suffix(".npy") {
            // BORROW: .to_owned() converts &str slice to owned String
            Some(n) => n.to_owned(),
            None => continue,
        };

        let header = parse_npy_header(&mut entry)?;

        if header.fortran_order {
            return Err(AnamnesisError::Unsupported {
                format: "NPZ".into(),
                detail: format!(
                    "fortran-order arrays not supported (array '{name}'). \
                     ML frameworks save C-order by default"
                ),
            });
        }

        let n_elements: usize = header
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .unwrap_or(usize::MAX);
        let byte_len = n_elements.saturating_mul(header.dtype.byte_size());

        // CAST: usize → u64, byte lengths fit in u64
        #[allow(clippy::as_conversions)]
        {
            total_bytes = total_bytes.saturating_add(byte_len as u64);
        }

        if !dtypes.contains(&header.dtype) {
            dtypes.push(header.dtype);
        }

        tensors.push(NpzTensorInfo {
            name,
            shape: header.shape,
            dtype: header.dtype,
            byte_len,
        });
    }

    Ok(NpzInspectInfo {
        tensors,
        total_bytes,
        dtypes,
    })
}

/// Parses an `NPZ` archive, returning all arrays as a name-to-tensor map.
///
/// Implements a lean `NPY` header parser with bulk data extraction. For
/// little-endian data on a little-endian machine (the common case for ML
/// weight files), the raw bytes are returned directly — zero per-element
/// deserialization.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or read.
///
/// Returns [`AnamnesisError::Parse`] if the `ZIP` archive is malformed, an
/// `NPY` header is invalid, or array data is truncated.
///
/// Returns [`AnamnesisError::Unsupported`] if an array uses Fortran order
/// or an unsupported dtype.
///
/// # Memory
///
/// Allocates one `Vec<u8>` per array (the raw data). Peak memory equals the
/// sum of all arrays. No intermediate typed buffers — data goes directly
/// from the `ZIP` entry to the output `Vec<u8>`.
pub fn parse_npz(path: impl AsRef<Path>) -> crate::Result<HashMap<String, NpzTensor>> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut archive = zip::ZipArchive::new(file)?;

    let mut result = HashMap::with_capacity(archive.len());

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to read ZIP entry {i}: {e}"),
        })?;

        // Strip .npy suffix; skip non-.npy entries (e.g., __MACOSX/).
        // BORROW: .to_owned() converts &str from zip entry to owned String
        let full_name = entry.name().to_owned();
        let name = match full_name.strip_suffix(".npy") {
            // BORROW: .to_owned() converts &str slice to owned String for HashMap key
            Some(n) => n.to_owned(),
            None => continue,
        };

        let header = parse_npy_header(&mut entry)?;

        if header.fortran_order {
            return Err(AnamnesisError::Unsupported {
                format: "NPZ".into(),
                detail: format!(
                    "fortran-order arrays not supported (array '{name}'). \
                     ML frameworks save C-order by default"
                ),
            });
        }

        let data = read_array_data(&mut entry, &header)?;

        result.insert(
            name.clone(),
            NpzTensor {
                name,
                shape: header.shape,
                dtype: header.dtype,
                data,
            },
        );
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]
mod tests {
    use std::io::Write;

    use super::*;

    // -- NpzDtype::byte_size -------------------------------------------------

    #[test]
    fn byte_size_1() {
        assert_eq!(NpzDtype::Bool.byte_size(), 1);
        assert_eq!(NpzDtype::U8.byte_size(), 1);
        assert_eq!(NpzDtype::I8.byte_size(), 1);
    }

    #[test]
    fn byte_size_2() {
        assert_eq!(NpzDtype::U16.byte_size(), 2);
        assert_eq!(NpzDtype::I16.byte_size(), 2);
        assert_eq!(NpzDtype::F16.byte_size(), 2);
        assert_eq!(NpzDtype::BF16.byte_size(), 2);
    }

    #[test]
    fn byte_size_4() {
        assert_eq!(NpzDtype::U32.byte_size(), 4);
        assert_eq!(NpzDtype::I32.byte_size(), 4);
        assert_eq!(NpzDtype::F32.byte_size(), 4);
    }

    #[test]
    fn byte_size_8() {
        assert_eq!(NpzDtype::U64.byte_size(), 8);
        assert_eq!(NpzDtype::I64.byte_size(), 8);
        assert_eq!(NpzDtype::F64.byte_size(), 8);
    }

    // -- NpzDtype Display ----------------------------------------------------

    #[test]
    fn display() {
        assert_eq!(NpzDtype::F32.to_string(), "F32");
        assert_eq!(NpzDtype::BF16.to_string(), "BF16");
        assert_eq!(NpzDtype::I64.to_string(), "I64");
        assert_eq!(NpzDtype::Bool.to_string(), "BOOL");
    }

    // -- parse_descr ---------------------------------------------------------

    #[test]
    fn parse_descr_float_types() {
        assert_eq!(parse_descr("<f2").unwrap(), (NpzDtype::F16, false));
        assert_eq!(parse_descr("<f4").unwrap(), (NpzDtype::F32, false));
        assert_eq!(parse_descr("<f8").unwrap(), (NpzDtype::F64, false));
        assert_eq!(parse_descr(">f4").unwrap(), (NpzDtype::F32, true));
    }

    #[test]
    fn parse_descr_int_types() {
        assert_eq!(parse_descr("|i1").unwrap(), (NpzDtype::I8, false));
        assert_eq!(parse_descr("<i2").unwrap(), (NpzDtype::I16, false));
        assert_eq!(parse_descr("<i4").unwrap(), (NpzDtype::I32, false));
        assert_eq!(parse_descr("<i8").unwrap(), (NpzDtype::I64, false));
        assert_eq!(parse_descr(">i4").unwrap(), (NpzDtype::I32, true));
    }

    #[test]
    fn parse_descr_uint_types() {
        assert_eq!(parse_descr("|u1").unwrap(), (NpzDtype::U8, false));
        assert_eq!(parse_descr("<u2").unwrap(), (NpzDtype::U16, false));
        assert_eq!(parse_descr("<u4").unwrap(), (NpzDtype::U32, false));
        assert_eq!(parse_descr("<u8").unwrap(), (NpzDtype::U64, false));
    }

    #[test]
    fn parse_descr_bool() {
        assert_eq!(parse_descr("|b1").unwrap(), (NpzDtype::Bool, false));
    }

    #[test]
    fn parse_descr_bf16_void() {
        assert_eq!(parse_descr("|V2").unwrap(), (NpzDtype::BF16, false));
    }

    #[test]
    fn parse_descr_unsupported() {
        assert!(parse_descr("<c8").is_err()); // complex
        assert!(parse_descr("<U4").is_err()); // unicode string
        assert!(parse_descr("x").is_err()); // too short
    }

    // -- extract_descr -------------------------------------------------------

    #[test]
    fn extract_descr_from_header() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        let (dtype, be) = extract_descr(header).unwrap();
        assert_eq!(dtype, NpzDtype::F32);
        assert!(!be);
    }

    #[test]
    fn extract_descr_double_quotes() {
        let header = "{\"descr\": \"<i4\", \"fortran_order\": False, \"shape\": (10,), }";
        let (dtype, _) = extract_descr(header).unwrap();
        assert_eq!(dtype, NpzDtype::I32);
    }

    #[test]
    fn extract_descr_mixed_quotes() {
        // Double-quoted descr value followed by single-quoted keys.
        // Previously broken: the quote-char detection scanned the entire
        // header tail, picking up the wrong quote character.
        let header = "{'descr': \"<f4\", 'fortran_order': False, 'shape': (2, 3), }";
        let (dtype, be) = extract_descr(header).unwrap();
        assert_eq!(dtype, NpzDtype::F32);
        assert!(!be);
    }

    // -- extract_fortran_order ------------------------------------------------

    #[test]
    fn fortran_order_false() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        assert!(!extract_fortran_order(header));
    }

    #[test]
    fn fortran_order_true() {
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }";
        assert!(extract_fortran_order(header));
    }

    #[test]
    fn fortran_order_missing() {
        let header = "{'descr': '<f4', 'shape': (2, 3), }";
        assert!(!extract_fortran_order(header));
    }

    // -- extract_shape -------------------------------------------------------

    #[test]
    fn shape_scalar() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (), }";
        let shape = extract_shape(header).unwrap();
        assert!(shape.is_empty());
    }

    #[test]
    fn shape_1d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (16384,), }";
        let shape = extract_shape(header).unwrap();
        assert_eq!(shape, vec![16384]);
    }

    #[test]
    fn shape_2d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2304, 16384), }";
        let shape = extract_shape(header).unwrap();
        assert_eq!(shape, vec![2304, 16384]);
    }

    #[test]
    fn shape_3d() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 4), }";
        let shape = extract_shape(header).unwrap();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    // -- NPY header roundtrip ------------------------------------------------

    /// Build a minimal NPY v1 file with the given header and data bytes.
    fn make_npy_v1(header_str: &str, data: &[u8]) -> Vec<u8> {
        let header_bytes = header_str.as_bytes();
        // Pad header to 64-byte alignment (magic=6 + version=2 + len=2 = 10).
        let total_before_pad = 10 + header_bytes.len();
        let padding = (64 - (total_before_pad % 64)) % 64;
        let padded_len = header_bytes.len() + padding;

        let mut npy = Vec::new();
        npy.extend_from_slice(NPY_MAGIC);
        npy.push(1); // major
        npy.push(0); // minor
        npy.extend_from_slice(&(padded_len as u16).to_le_bytes());
        npy.extend_from_slice(header_bytes);
        // Pad with spaces, ending with newline.
        if padding > 0 {
            npy.extend(std::iter::repeat_n(b' ', padding - 1));
            npy.push(b'\n');
        }
        npy.extend_from_slice(data);
        npy
    }

    #[test]
    fn roundtrip_f32_npy_v1() {
        let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let npy = make_npy_v1(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }",
            &data,
        );

        let mut reader = std::io::Cursor::new(&npy);
        let header = parse_npy_header(&mut reader).unwrap();
        assert_eq!(header.dtype, NpzDtype::F32);
        assert!(!header.big_endian);
        assert!(!header.fortran_order);
        assert_eq!(header.shape, vec![2, 2]);

        let result = read_array_data(&mut reader, &header).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn roundtrip_f32_big_endian() {
        // Big-endian f32: 1.0 = 0x3F800000 → bytes [3F, 80, 00, 00]
        let data_be: Vec<u8> = vec![0x3F, 0x80, 0x00, 0x00];

        let npy = make_npy_v1(
            "{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }",
            &data_be,
        );

        let mut reader = std::io::Cursor::new(&npy);
        let header = parse_npy_header(&mut reader).unwrap();
        assert!(header.big_endian);

        let result = read_array_data(&mut reader, &header).unwrap();
        // After byteswap: [00, 00, 80, 3F] = 1.0 in LE
        assert_eq!(result, vec![0x00, 0x00, 0x80, 0x3F]);
        let val = f32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(val, 1.0);
    }

    #[test]
    fn npy_v2_header() {
        let header_str = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }";
        let header_bytes = header_str.as_bytes();
        let total_before_pad = 12 + header_bytes.len();
        let padding = (64 - (total_before_pad % 64)) % 64;
        let padded_len = header_bytes.len() + padding;

        let mut npy = Vec::new();
        npy.extend_from_slice(NPY_MAGIC);
        npy.push(2); // major
        npy.push(0); // minor
        npy.extend_from_slice(&(padded_len as u32).to_le_bytes());
        npy.extend_from_slice(header_bytes);
        if padding > 0 {
            npy.extend(std::iter::repeat_n(b' ', padding - 1));
            npy.push(b'\n');
        }
        // One f64 value
        npy.extend_from_slice(&42.5_f64.to_le_bytes());

        let mut reader = std::io::Cursor::new(&npy);
        let header = parse_npy_header(&mut reader).unwrap();
        assert_eq!(header.dtype, NpzDtype::F64);
        assert_eq!(header.shape, vec![1]);

        let result = read_array_data(&mut reader, &header).unwrap();
        assert_eq!(result, 42.5_f64.to_le_bytes());
    }

    /// `read_array_data` reports a clean `Parse` error when the underlying
    /// reader runs out of bytes before `data_bytes` are available. With the
    /// previous `vec![0u8; n]` + `read_exact` pattern this fell out of
    /// `UnexpectedEof`; the new `Vec::with_capacity` + `take().read_to_end`
    /// pattern would silently return a short `Vec` without this guard, so
    /// the explicit length-check is correctness-load-bearing.
    #[test]
    fn read_array_data_rejects_truncated_input() {
        // Header claims 4 × f32 = 16 bytes, but we feed only 8 bytes of data.
        let truncated_data = [0u8; 8];
        let npy = make_npy_v1(
            "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }",
            &truncated_data,
        );
        let mut reader = std::io::Cursor::new(&npy);
        let header = parse_npy_header(&mut reader).unwrap();
        assert_eq!(header.dtype, NpzDtype::F32);
        assert_eq!(header.shape, vec![4]);

        let err = read_array_data(&mut reader, &header).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("truncated") || msg.contains("read failed"),
            "expected truncation error, got: {msg}"
        );
    }

    #[test]
    fn invalid_magic_rejected() {
        let data = b"NOT_NUMPY_DATA_AT_ALL";
        let mut reader = std::io::Cursor::new(data);
        assert!(parse_npy_header(&mut reader).is_err());
    }

    #[test]
    fn fortran_order_rejected_in_parse_npz() {
        // We can't easily test parse_npz with Fortran order without creating
        // a real NPZ file, but we can verify the extraction logic.
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }";
        assert!(extract_fortran_order(header));
    }

    // -- Gap tests (review findings G32–G36) ---------------------------------

    // G32: Fortran-order rejection through parse_npz end-to-end
    #[test]
    fn fortran_order_rejected_end_to_end() {
        // Build a minimal NPZ containing a single Fortran-order NPY entry.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("arr.npy", options).unwrap();

            // Build NPY v1 with fortran_order: True
            let header_str = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }";
            let npy = make_npy_v1(header_str, &[0u8; 16]); // 4 f32 zeros
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }
        let err = parse_npz(tmp.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Fortran-order") || msg.contains("fortran"),
            "expected Fortran-order error, got: {msg}"
        );
    }

    // G33: Empty NPZ archive
    #[test]
    fn empty_npz_archive() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let zip = zip::ZipWriter::new(file);
            zip.finish().unwrap();
        }
        let result = parse_npz(tmp.path()).unwrap();
        assert!(result.is_empty(), "empty NPZ should return empty map");

        // NN4: also verify inspect_npz handles empty archives
        let info = inspect_npz(tmp.path()).unwrap();
        assert!(info.tensors.is_empty());
        assert_eq!(info.total_bytes, 0);
    }

    // G35: Native-endian '=' prefix in parse_descr
    #[test]
    fn parse_descr_native_endian() {
        // '=' means native endian — treated as LE on all modern platforms
        let (dtype, be) = parse_descr("=f4").unwrap();
        assert_eq!(dtype, NpzDtype::F32);
        assert!(!be, "'=' should not be treated as big-endian");
    }

    // G34: Big-endian array through parse_npz end-to-end
    #[test]
    fn big_endian_through_parse_npz() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("val.npy", options).unwrap();

            // NPY with big-endian f32: 1.0 = [3F, 80, 00, 00] BE
            let npy = make_npy_v1(
                "{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }",
                &[0x3F, 0x80, 0x00, 0x00],
            );
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }
        let tensors = parse_npz(tmp.path()).unwrap();
        let t = tensors.get("val").expect("val not found");
        assert_eq!(t.dtype, NpzDtype::F32);
        // After byteswap: [00, 00, 80, 3F] = 1.0 LE
        assert_eq!(t.data, vec![0x00, 0x00, 0x80, 0x3F]);
    }

    // G36: inspect_npz overflow — large shape values saturate gracefully
    #[test]
    fn inspect_npz_overflow_saturates() {
        // Build NPZ with an array whose shape would overflow usize when
        // multiplied. inspect_npz should saturate rather than panic.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(tmp.path()).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("huge.npy", options).unwrap();

            // Shape with dimensions that overflow when multiplied:
            // (usize::MAX, 2) → usize::MAX * 2 overflows
            // We encode this as NPY v1 header with huge shape.
            // But we can't actually store usize::MAX elements — the shape
            // in the header can claim anything. inspect_npz only reads
            // the header, not the data, so the file can be tiny.
            let shape_str = format!(
                "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 2), }}",
                usize::MAX / 2 + 1
            );
            let npy = make_npy_v1(&shape_str, &[]); // no actual data
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }

        let info = inspect_npz(tmp.path()).unwrap();
        assert_eq!(info.tensors.len(), 1);
        // Element count overflows → unwrap_or(usize::MAX) → saturating_mul
        // The byte_len should be usize::MAX (saturated)
        assert_eq!(info.tensors[0].byte_len, usize::MAX);
    }

    // -- Phase 4.7: reader-generic inspection --------------------------------

    /// Build a minimal in-memory NPZ archive containing a single STORED .npy
    /// entry with the given header and (zero-filled) data bytes. Returns the
    /// raw archive bytes — callers can wrap them in a Cursor to test the
    /// reader-generic API.
    fn make_in_memory_npz(arr_name: &str, header_str: &str, data: &[u8]) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            let entry_name = format!("{arr_name}.npy");
            zip.start_file(&entry_name, options).unwrap();
            let npy = make_npy_v1(header_str, data);
            zip.write_all(&npy).unwrap();
            zip.finish().unwrap();
        }
        buf
    }

    /// `inspect_npz_from_reader` over an in-memory `Cursor` returns the same
    /// `NpzInspectInfo` as `inspect_npz` over the same archive on disk.
    /// Locks the contract that the reader-generic and path-based APIs are
    /// substrate-equivalent — the substrate (file vs. cursor) cannot change
    /// the metadata. This is what downstream HTTP-range adapters rely on.
    #[test]
    fn inspect_from_reader_matches_path() {
        // Build a multi-array NPZ in memory: one F32 [2, 3] and one I64 [4].
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut zip = zip::ZipWriter::new(cursor);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);

            // Array 1: F32 [2, 3] = 24 bytes
            zip.start_file("weights.npy", options).unwrap();
            let npy1 = make_npy_v1(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
                &[0u8; 24],
            );
            zip.write_all(&npy1).unwrap();

            // Array 2: I64 [4] = 32 bytes
            zip.start_file("indices.npy", options).unwrap();
            let npy2 = make_npy_v1(
                "{'descr': '<i8', 'fortran_order': False, 'shape': (4,), }",
                &[0u8; 32],
            );
            zip.write_all(&npy2).unwrap();

            zip.finish().unwrap();
        }

        // Write the same bytes to a temp file for the path-based comparison.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &buf).unwrap();

        let path_info = inspect_npz(tmp.path()).unwrap();
        let reader_info = inspect_npz_from_reader(std::io::Cursor::new(&buf)).unwrap();

        // Substrate-equivalence: every field of NpzInspectInfo matches.
        assert_eq!(path_info.tensors.len(), reader_info.tensors.len());
        assert_eq!(path_info.total_bytes, reader_info.total_bytes);
        assert_eq!(path_info.dtypes, reader_info.dtypes);
        for (a, b) in path_info.tensors.iter().zip(reader_info.tensors.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.dtype, b.dtype);
            assert_eq!(a.byte_len, b.byte_len);
        }

        // And spot-check the actual values to make sure we are not just
        // comparing two equal-but-wrong outputs.
        assert_eq!(reader_info.tensors.len(), 2);
        assert_eq!(reader_info.total_bytes, 24 + 32);
    }

    /// `inspect_npz_from_reader` returns `Ok` with no entries when handed a
    /// well-formed empty ZIP archive (i.e., the archive parses but contains
    /// no `.npy` payloads). Mirrors the existing `empty_npz_archive` test
    /// for the path-based variant.
    #[test]
    fn inspect_from_reader_empty_archive() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let zip = zip::ZipWriter::new(cursor);
            zip.finish().unwrap();
        }
        let info = inspect_npz_from_reader(std::io::Cursor::new(&buf)).unwrap();
        assert!(info.tensors.is_empty());
        assert_eq!(info.total_bytes, 0);
        assert!(info.dtypes.is_empty());
    }

    /// `inspect_npz_from_reader` propagates Fortran-order rejection through
    /// the same code path as `inspect_npz`. Confirms the refactor did not
    /// silently lose the unsupported-format guard.
    #[test]
    fn inspect_from_reader_rejects_fortran_order() {
        let buf = make_in_memory_npz(
            "arr",
            "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }",
            &[0u8; 16],
        );
        let err = inspect_npz_from_reader(std::io::Cursor::new(&buf)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Fortran-order") || msg.contains("fortran"),
            "expected Fortran-order error, got: {msg}"
        );
    }
}
