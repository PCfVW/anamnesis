// SPDX-License-Identifier: MIT OR Apache-2.0

//! `NPZ`/`NPY` archive parsing — wraps [`npyz`] to provide an anamnesis-native API.
//!
//! This module delegates format parsing (`ZIP` extraction, `NPY` header decoding,
//! endianness handling) entirely to the `npyz` crate. It adds:
//!
//! - An anamnesis-native dtype enum ([`NpzDtype`]) that includes `BF16`
//! - A `BF16` interpretation layer (read as raw bytes, reinterpret as `half::bf16`)
//! - Shape conversion from `npyz`'s `&[u64]` to `Vec<usize>`

use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::path::Path;

use crate::error::AnamnesisError;

// ---------------------------------------------------------------------------
// NpzDtype
// ---------------------------------------------------------------------------

/// Element data type for tensors parsed from `NPZ`/`NPY` archives.
///
/// Includes `BF16` which `npyz` cannot represent natively. When a `BF16`
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
// Dtype classification
// ---------------------------------------------------------------------------

/// Maps an `npyz` `DType` to an [`NpzDtype`].
///
/// Detects `BF16` from void/`V2` (the convention used by `JAX` / `TensorFlow`
/// for `bfloat16` arrays in `NumPy` archives).
fn classify_dtype(dtype: &npyz::DType, name: &str) -> crate::Result<NpzDtype> {
    match dtype {
        npyz::DType::Plain(ts) => match (ts.type_char(), ts.size_field()) {
            (npyz::TypeChar::Bool, 1) => Ok(NpzDtype::Bool),
            (npyz::TypeChar::Uint, 1) => Ok(NpzDtype::U8),
            (npyz::TypeChar::Int, 1) => Ok(NpzDtype::I8),
            (npyz::TypeChar::Uint, 2) => Ok(NpzDtype::U16),
            (npyz::TypeChar::Int, 2) => Ok(NpzDtype::I16),
            (npyz::TypeChar::Uint, 4) => Ok(NpzDtype::U32),
            (npyz::TypeChar::Int, 4) => Ok(NpzDtype::I32),
            (npyz::TypeChar::Uint, 8) => Ok(NpzDtype::U64),
            (npyz::TypeChar::Int, 8) => Ok(NpzDtype::I64),
            (npyz::TypeChar::Float, 2) => Ok(NpzDtype::F16),
            (npyz::TypeChar::Float, 4) => Ok(NpzDtype::F32),
            (npyz::TypeChar::Float, 8) => Ok(NpzDtype::F64),
            (npyz::TypeChar::RawData, 2) => Ok(NpzDtype::BF16),
            // EXHAUSTIVE: TypeChar is a foreign #[non_exhaustive] enum; catch-all
            // covers complex, timedelta, datetime, bytestr, unicode, and future variants
            (tc, size) => Err(AnamnesisError::Unsupported {
                format: "NPZ".into(),
                detail: format!("unsupported dtype {}{size} for array {name}", tc.to_str()),
            }),
        },
        npyz::DType::Record(_) => Err(AnamnesisError::Unsupported {
            format: "NPZ".into(),
            detail: format!("structured/record arrays not supported (array {name})"),
        }),
        npyz::DType::Array(_, _) => Err(AnamnesisError::Unsupported {
            format: "NPZ".into(),
            detail: format!("nested array dtypes not supported (array {name})"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Shape conversion
// ---------------------------------------------------------------------------

/// Converts `npyz`'s `&[u64]` shape to `Vec<usize>`, checking for overflow.
fn convert_shape(shape: &[u64], name: &str) -> crate::Result<Vec<usize>> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| AnamnesisError::Parse {
                reason: format!("shape dimension {dim} overflows usize for array {name}"),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Typed data reading
// ---------------------------------------------------------------------------

/// Converts a typed slice to little-endian `Vec<u8>` using a per-element
/// write function.
///
/// Peak memory: the input slice plus the output `Vec<u8>` coexist briefly.
fn typed_to_le_bytes<T, F>(vec: &[T], byte_size: usize, write: F) -> crate::Result<Vec<u8>>
where
    F: Fn(&T, &mut [u8]),
{
    let total_bytes = vec
        .len()
        .checked_mul(byte_size)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "array byte count overflow".into(),
        })?;
    let mut bytes = vec![0u8; total_bytes];
    for (val, chunk) in vec.iter().zip(bytes.chunks_exact_mut(byte_size)) {
        write(val, chunk);
    }
    Ok(bytes)
}

/// Reads array data as raw little-endian bytes by dispatching
/// `into_vec::<T>()` for the appropriate type.
///
/// The `header` and `reader` are combined via `NpyFile::with_header` to
/// perform type-safe deserialization through `npyz`. This ensures correct
/// endianness handling (big-endian `NPY` files get byte-swapped by `npyz`).
fn read_typed_array<R: Read>(
    header: npyz::NpyHeader,
    reader: R,
    dtype: NpzDtype,
    name: &str,
) -> crate::Result<Vec<u8>> {
    // Each arm: construct NpyFile, into_vec::<T>(), convert to LE bytes.
    // The typed Vec<T> is dropped when the arm returns.
    match dtype {
        NpzDtype::Bool => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<bool, R>(npy, name)?;
            Ok(vals.iter().map(|&v| u8::from(v)).collect())
        }
        NpzDtype::U8 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            read_into_vec::<u8, R>(npy, name)
        }
        NpzDtype::I8 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<i8, R>(npy, name)?;
            let bytes = vals.iter().map(|&v| v.cast_unsigned()).collect();
            Ok(bytes)
        }
        NpzDtype::U16 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<u16, R>(npy, name)?;
            typed_to_le_bytes(&vals, 2, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::I16 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<i16, R>(npy, name)?;
            typed_to_le_bytes(&vals, 2, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::U32 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<u32, R>(npy, name)?;
            typed_to_le_bytes(&vals, 4, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::I32 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<i32, R>(npy, name)?;
            typed_to_le_bytes(&vals, 4, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::U64 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<u64, R>(npy, name)?;
            typed_to_le_bytes(&vals, 8, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::I64 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<i64, R>(npy, name)?;
            typed_to_le_bytes(&vals, 8, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::F16 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<half::f16, R>(npy, name)?;
            typed_to_le_bytes(&vals, 2, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::F32 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<f32, R>(npy, name)?;
            typed_to_le_bytes(&vals, 4, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::F64 => {
            let npy = npyz::NpyFile::with_header(header, reader);
            let vals = read_into_vec::<f64, R>(npy, name)?;
            typed_to_le_bytes(&vals, 8, |v, out| out.copy_from_slice(&v.to_le_bytes()))
        }
        NpzDtype::BF16 => {
            // BF16 is handled by raw byte reading in the caller, not here.
            Err(AnamnesisError::Parse {
                reason: format!("BF16 arrays must be read as raw bytes (array {name})"),
            })
        }
    }
}

/// Deserializes all elements from an `NpyFile` into a `Vec<T>`.
fn read_into_vec<T: npyz::Deserialize, R: Read>(
    npy: npyz::NpyFile<R>,
    name: &str,
) -> crate::Result<Vec<T>> {
    npy.into_vec().map_err(|e| AnamnesisError::Parse {
        reason: format!("failed to read array {name}: {e}"),
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parses an `NPZ` archive, returning all arrays as a name-to-tensor map.
///
/// Delegates `NPY` format parsing to `npyz`. Each array is deserialized into
/// raw little-endian bytes via type-appropriate `into_vec` calls. `BF16`
/// arrays (stored as `|V2` by `JAX`) are read as raw bytes.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or read.
///
/// Returns [`AnamnesisError::Unsupported`] if an array has an unsupported
/// dtype (e.g., structured records, complex numbers, strings).
///
/// Returns [`AnamnesisError::Parse`] if shape values overflow `usize` or
/// array data cannot be deserialized.
///
/// # Memory
///
/// Allocates one `Vec<u8>` per array (the raw data). Peak memory equals the
/// sum of all parsed arrays plus a transient typed `Vec<T>` buffer during
/// deserialization (one array at a time, dropped before the next). For a
/// typical model, peak is approximately sum(all arrays) + largest array.
pub fn parse_npz(path: impl AsRef<Path>) -> crate::Result<HashMap<String, NpzTensor>> {
    let mut archive = npyz::npz::NpzArchive::open(path.as_ref())?;

    // Collect names first — `zip_archive()` borrows `&mut self`.
    let names: Vec<String> = archive.array_names().map(String::from).collect();

    let mut result = HashMap::with_capacity(names.len());

    for name in &names {
        let zip_name = format!("{name}.npy");
        let mut entry =
            archive
                .zip_archive()
                .by_name(&zip_name)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("failed to read zip entry {zip_name}: {e}"),
                })?;

        let header =
            npyz::NpyHeader::from_reader(&mut entry).map_err(|e| AnamnesisError::Parse {
                reason: format!("failed to parse npy header for array {name}: {e}"),
            })?;

        let npz_dtype = classify_dtype(&header.dtype(), name)?;
        let shape = convert_shape(header.shape(), name)?;

        let data = if npz_dtype == NpzDtype::BF16 {
            // BF16 (V2): read raw bytes directly — header already consumed,
            // entry is positioned at the data section.
            let n_elements: usize = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("element count overflow for array {name}"),
                })?;
            let byte_count = n_elements
                .checked_mul(2)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("BF16 byte count overflow for array {name}"),
                })?;
            let mut buf = vec![0u8; byte_count];
            entry.read_exact(&mut buf)?;
            buf
        } else {
            read_typed_array(header, entry, npz_dtype, name)?
        };

        result.insert(
            name.clone(),
            NpzTensor {
                name: name.clone(),
                shape,
                dtype: npz_dtype,
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
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]
mod tests {
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

    // -- classify_dtype ------------------------------------------------------

    fn make_plain(type_char: char, size: u64) -> npyz::DType {
        let endianness = if size == 1 { '|' } else { '<' };
        let descr = format!("{endianness}{type_char}{size}");
        npyz::DType::Plain(descr.parse().unwrap())
    }

    #[test]
    fn classify_float_types() {
        assert_eq!(
            classify_dtype(&make_plain('f', 2), "x").unwrap(),
            NpzDtype::F16
        );
        assert_eq!(
            classify_dtype(&make_plain('f', 4), "x").unwrap(),
            NpzDtype::F32
        );
        assert_eq!(
            classify_dtype(&make_plain('f', 8), "x").unwrap(),
            NpzDtype::F64
        );
    }

    #[test]
    fn classify_int_types() {
        assert_eq!(
            classify_dtype(&make_plain('i', 1), "x").unwrap(),
            NpzDtype::I8
        );
        assert_eq!(
            classify_dtype(&make_plain('i', 2), "x").unwrap(),
            NpzDtype::I16
        );
        assert_eq!(
            classify_dtype(&make_plain('i', 4), "x").unwrap(),
            NpzDtype::I32
        );
        assert_eq!(
            classify_dtype(&make_plain('i', 8), "x").unwrap(),
            NpzDtype::I64
        );
    }

    #[test]
    fn classify_uint_types() {
        assert_eq!(
            classify_dtype(&make_plain('u', 1), "x").unwrap(),
            NpzDtype::U8
        );
        assert_eq!(
            classify_dtype(&make_plain('u', 2), "x").unwrap(),
            NpzDtype::U16
        );
        assert_eq!(
            classify_dtype(&make_plain('u', 4), "x").unwrap(),
            NpzDtype::U32
        );
        assert_eq!(
            classify_dtype(&make_plain('u', 8), "x").unwrap(),
            NpzDtype::U64
        );
    }

    #[test]
    fn classify_bool() {
        assert_eq!(
            classify_dtype(&make_plain('b', 1), "x").unwrap(),
            NpzDtype::Bool
        );
    }

    #[test]
    fn classify_bf16_void() {
        assert_eq!(
            classify_dtype(&make_plain('V', 2), "x").unwrap(),
            NpzDtype::BF16
        );
    }

    #[test]
    fn classify_unsupported_complex() {
        let result = classify_dtype(&make_plain('c', 8), "z");
        assert!(result.is_err());
    }

    #[test]
    fn classify_unsupported_string() {
        // Unicode string type
        let dtype = npyz::DType::Plain("<U4".parse().unwrap());
        let result = classify_dtype(&dtype, "s");
        assert!(result.is_err());
    }

    #[test]
    fn classify_record_unsupported() {
        let dtype = npyz::DType::Record(vec![]);
        let result = classify_dtype(&dtype, "r");
        assert!(result.is_err());
    }

    // -- convert_shape -------------------------------------------------------

    #[test]
    fn shape_empty_scalar() {
        let shape = convert_shape(&[], "x").unwrap();
        assert!(shape.is_empty());
    }

    #[test]
    fn shape_1d() {
        let shape = convert_shape(&[42], "x").unwrap();
        assert_eq!(shape, vec![42]);
    }

    #[test]
    fn shape_2d() {
        let shape = convert_shape(&[16384, 2304], "W_dec").unwrap();
        assert_eq!(shape, vec![16384, 2304]);
    }
}
