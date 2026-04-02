// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PyTorch` `.pth` `state_dict` parsing — minimal pickle VM with security boundary.
//!
//! Since `PyTorch` 1.6 (July 2020), `torch.save()` produces a ZIP archive
//! containing a pickle stream (`data.pkl`) that describes the `state_dict`
//! structure, plus raw tensor data files (`data/0`, `data/1`, ...).
//!
//! This module implements a minimal pickle interpreter (~36 opcodes) that
//! reconstructs the `state_dict` structure, then extracts tensor metadata
//! (name, shape, dtype) and raw data. An explicit `GLOBAL` allowlist rejects
//! any callable not related to `PyTorch` tensor reconstruction — equivalent
//! to `weights_only=True` but stricter.
//!
//! # Security
//!
//! Unlike Python's `pickle.load()`, this parser **never executes arbitrary
//! code**. Only allowlisted `GLOBAL` references (`torch._utils`,
//! `torch.*Storage`, `collections.OrderedDict`) are accepted. Unrecognized
//! globals produce `AnamnesisError::Parse`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use crate::error::AnamnesisError;
use crate::parse::safetensors::Dtype;
use crate::parse::utils::byteswap_inplace;

// ---------------------------------------------------------------------------
// PthDtype
// ---------------------------------------------------------------------------

/// Element data type derived from `PyTorch` storage class names.
///
/// Maps `torch.FloatStorage` to `F32`, `torch.HalfStorage` to `F16`, etc.
/// Covers all storage types emitted by `torch.save()` for `state_dict` files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PthDtype {
    /// 16-bit IEEE 754 half-precision (`torch.HalfStorage`).
    F16,
    /// 16-bit brain floating point (`torch.BFloat16Storage`).
    BF16,
    /// 32-bit IEEE 754 single-precision (`torch.FloatStorage`).
    F32,
    /// 64-bit IEEE 754 double-precision (`torch.DoubleStorage`).
    F64,
    /// Unsigned 8-bit integer (`torch.ByteStorage`).
    U8,
    /// Signed 8-bit integer (`torch.CharStorage`).
    I8,
    /// Signed 16-bit integer (`torch.ShortStorage`).
    I16,
    /// Signed 32-bit integer (`torch.IntStorage`).
    I32,
    /// Signed 64-bit integer (`torch.LongStorage`).
    I64,
    /// Boolean (`torch.BoolStorage`).
    Bool,
}

impl PthDtype {
    /// Returns the number of bytes per element for this dtype.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }

    /// Converts to the anamnesis `Dtype` used for safetensors output.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if no `safetensors` equivalent
    /// exists (currently all variants map successfully).
    pub fn to_dtype(self) -> crate::Result<Dtype> {
        match self {
            Self::F16 => Ok(Dtype::F16),
            Self::BF16 => Ok(Dtype::BF16),
            Self::F32 => Ok(Dtype::F32),
            Self::F64 => Ok(Dtype::F64),
            Self::U8 => Ok(Dtype::U8),
            Self::I8 => Ok(Dtype::I8),
            Self::I16 => Ok(Dtype::I16),
            Self::I32 => Ok(Dtype::I32),
            Self::I64 => Ok(Dtype::I64),
            Self::Bool => Ok(Dtype::Bool),
        }
    }

    /// Parses a `PyTorch` storage class name into a `PthDtype`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if the storage name is not recognized.
    fn from_storage_class(module: &str, name: &str) -> crate::Result<Self> {
        if module != "torch" {
            return Err(AnamnesisError::Parse {
                reason: format!("unknown storage module `{module}.{name}`"),
            });
        }
        match name {
            "FloatStorage" => Ok(Self::F32),
            "DoubleStorage" => Ok(Self::F64),
            "HalfStorage" => Ok(Self::F16),
            "BFloat16Storage" => Ok(Self::BF16),
            "LongStorage" => Ok(Self::I64),
            "IntStorage" => Ok(Self::I32),
            "ShortStorage" => Ok(Self::I16),
            "CharStorage" => Ok(Self::I8),
            "ByteStorage" => Ok(Self::U8),
            "BoolStorage" => Ok(Self::Bool),
            _ => Err(AnamnesisError::Parse {
                reason: format!("unknown storage class `torch.{name}`"),
            }),
        }
    }
}

impl fmt::Display for PthDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Bool => "BOOL",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// PthTensor
// ---------------------------------------------------------------------------

/// A single tensor view from a parsed `.pth` file.
///
/// Borrows raw data from the memory-mapped file (zero-copy for contiguous
/// little-endian tensors) or owns a copy (non-contiguous / big-endian).
#[derive(Debug, Clone)]
pub struct PthTensor<'a> {
    /// Tensor name (`state_dict` key, e.g. `"linear.weight"`).
    pub name: String,
    /// Tensor shape (e.g. `[16, 10]` for a 16-by-10 matrix).
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: PthDtype,
    /// Raw bytes in row-major, native-endian order.
    ///
    /// `Cow::Borrowed` when the tensor data is contiguous and little-endian
    /// (zero-copy slice from the mmap). `Cow::Owned` when a layout
    /// transformation was required (non-contiguous strides or
    /// big-endian byte-swap).
    pub data: Cow<'a, [u8]>,
}

/// Tensor metadata extracted from the pickle stream (no data).
#[derive(Debug)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    dtype: PthDtype,
    /// Data file index in the ZIP archive (e.g., `"0"` → `data/0`).
    storage_key: String,
    /// Byte offset into the storage file.
    storage_offset: usize,
    strides: Vec<usize>,
}

/// A parsed `.pth` file — owns the memory-mapped data and provides
/// zero-copy tensor access.
///
/// Created by [`parse_pth`]. Call [`tensors()`](ParsedPth::tensors) to get
/// `PthTensor` views that borrow directly from the mapped file region.
#[derive(Debug)]
pub struct ParsedPth {
    /// Memory-mapped file.
    mmap: memmap2::Mmap,
    /// Per-tensor metadata (name, shape, dtype, storage location).
    meta: Vec<TensorMeta>,
    /// ZIP entry index: suffix → `(data_start, data_len)` in the mmap.
    entry_index: HashMap<String, (usize, usize)>,
    /// Whether the file uses big-endian storage.
    big_endian: bool,
}

impl ParsedPth {
    /// Returns tensor views borrowing directly from the mmap.
    ///
    /// For contiguous little-endian tensors (>99% of real files), the
    /// data is a zero-copy `&[u8]` slice from the mmap — no heap
    /// allocation. Non-contiguous or big-endian tensors get an owned copy.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] if a storage entry is missing
    /// or a tensor's byte range exceeds the storage.
    pub fn tensors(&self) -> crate::Result<Vec<PthTensor<'_>>> {
        let mut tensors = Vec::with_capacity(self.meta.len());
        for m in &self.meta {
            let storage_suffix = format!("data/{}", m.storage_key);
            let &(storage_start, storage_len) = self
                .entry_index
                .get(storage_suffix.as_str())
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("ZIP entry `{storage_suffix}` not found"),
                })?;
            let storage = self
                .mmap
                .get(storage_start..storage_start + storage_len)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("storage `{}`: mmap slice out of bounds", m.storage_key),
                })?;

            let elem_size = m.dtype.byte_size();
            let data: Cow<'_, [u8]> = if is_contiguous(&m.shape, &m.strides) && !self.big_endian {
                // Zero-copy: borrow directly from the mmap.
                let n_elements: usize = m
                    .shape
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("tensor `{}`: element count overflow", m.name),
                    })?;
                let n_bytes =
                    n_elements
                        .checked_mul(elem_size)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("tensor `{}`: byte count overflow", m.name),
                        })?;
                let end =
                    m.storage_offset
                        .checked_add(n_bytes)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("tensor `{}`: storage end offset overflow", m.name),
                        })?;
                Cow::Borrowed(storage.get(m.storage_offset..end).ok_or_else(|| {
                    AnamnesisError::Parse {
                        reason: format!(
                            "tensor `{}`: storage read out of bounds \
                             ([{}..{}], storage len = {})",
                            m.name,
                            m.storage_offset,
                            end,
                            storage.len()
                        ),
                    }
                })?)
            } else if is_contiguous(&m.shape, &m.strides) {
                // Contiguous but big-endian: copy + byte-swap.
                let n_elements: usize = m
                    .shape
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("tensor `{}`: element count overflow", m.name),
                    })?;
                let n_bytes =
                    n_elements
                        .checked_mul(elem_size)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("tensor `{}`: byte count overflow", m.name),
                        })?;
                let end =
                    m.storage_offset
                        .checked_add(n_bytes)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("tensor `{}`: storage end offset overflow", m.name),
                        })?;
                let mut buf = storage
                    .get(m.storage_offset..end)
                    .ok_or_else(|| AnamnesisError::Parse {
                        reason: format!("tensor `{}`: storage read out of bounds", m.name),
                    })?
                    .to_vec();
                byteswap_inplace(&mut buf, elem_size);
                Cow::Owned(buf)
            } else {
                // Non-contiguous: copy to contiguous layout.
                let mut buf =
                    copy_to_contiguous(storage, m.storage_offset, &m.shape, &m.strides, elem_size)?;
                if self.big_endian && elem_size > 1 {
                    byteswap_inplace(&mut buf, elem_size);
                }
                Cow::Owned(buf)
            };

            tensors.push(PthTensor {
                name: m.name.clone(),
                shape: m.shape.clone(),
                dtype: m.dtype,
                data,
            });
        }
        Ok(tensors)
    }

    /// Returns the number of tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    /// Returns `true` if the file contained no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// Returns inspection info derived from the parsed metadata.
    ///
    /// No I/O — purely computed from the tensor metadata extracted during
    /// [`parse_pth`].
    pub fn inspect(&self) -> PthInspectInfo {
        let mut total_bytes: u64 = 0;
        let mut dtypes: Vec<PthDtype> = Vec::new();
        for m in &self.meta {
            let n_elements: u64 = m
                .shape
                .iter()
                .try_fold(1u64, |acc, &d| {
                    // CAST: usize → u64, element counts fit in u64
                    #[allow(clippy::as_conversions)]
                    acc.checked_mul(d as u64)
                })
                .unwrap_or(0);
            // CAST: usize → u64, byte sizes fit
            #[allow(clippy::as_conversions)]
            let byte_size = m.dtype.byte_size() as u64;
            total_bytes = total_bytes.saturating_add(n_elements.saturating_mul(byte_size));
            if !dtypes.contains(&m.dtype) {
                dtypes.push(m.dtype);
            }
        }
        PthInspectInfo {
            tensor_count: self.meta.len(),
            total_bytes,
            dtypes,
            big_endian: self.big_endian,
        }
    }

    /// Converts the parsed `.pth` tensors to a safetensors file.
    ///
    /// Equivalent to calling [`tensors()`](Self::tensors) followed by
    /// `pth_to_safetensors` — but as a single convenience method.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Io`] if the output file cannot be written.
    /// Returns [`AnamnesisError::Parse`] if tensor extraction or
    /// serialization fails.
    pub fn to_safetensors(&self, output: impl AsRef<std::path::Path>) -> crate::Result<()> {
        let tensors = self.tensors()?;
        crate::remember::pth::pth_to_safetensors(&tensors, output)
    }
}

/// Summary information about a parsed `.pth` file.
///
/// Produced by [`ParsedPth::inspect`]. No I/O — derived from metadata.
#[derive(Debug, Clone)]
#[must_use]
pub struct PthInspectInfo {
    /// Number of tensors in the `state_dict`.
    pub tensor_count: usize,
    /// Total size of raw tensor data in bytes.
    pub total_bytes: u64,
    /// Distinct dtypes found (in order of first occurrence).
    pub dtypes: Vec<PthDtype>,
    /// Whether the file uses big-endian storage.
    pub big_endian: bool,
}

impl fmt::Display for PthInspectInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Format:      PyTorch state_dict (.pth)")?;
        write!(f, "\nTensors:     {}", self.tensor_count)?;
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
        let endian = if self.big_endian {
            "big-endian"
        } else {
            "little-endian"
        };
        write!(f, "\nByte order:  {endian}")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pickle VM (internal)
// ---------------------------------------------------------------------------

/// Value on the pickle VM stack.
///
/// Not all Python types are represented — only the subset produced by
/// `torch.save()` for `state_dict` files.
// Fields like `Bool(bool)` and `Bytes(Vec<u8>)` are populated by the pickle
// VM when interpreting `state_dict` streams but are not destructured during
// tensor extraction. They must remain for correct pickle interpretation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
// EXHAUSTIVE: private enum — crate owns and matches all variants; wildcard
// arms are used in extraction code where most variants are irrelevant.
#[allow(clippy::wildcard_enum_match_arm)]
enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    String(String),
    Bytes(Vec<u8>),
    Tuple(Vec<PickleValue>),
    List(Vec<PickleValue>),
    /// Key-value pairs preserving insertion order (matches Python `OrderedDict`).
    Dict(Vec<(PickleValue, PickleValue)>),
    /// A `GLOBAL` reference: `module.name`.
    Global {
        module: String,
        name: String,
    },
    /// A persistent ID referencing tensor data in the ZIP archive.
    PersistentId(Box<PickleValue>),
    /// Result of `REDUCE(callable, args)`.
    Reduced {
        callable: Box<PickleValue>,
        args: Box<PickleValue>,
    },
    /// Result of `BUILD(obj, state)` — sets `obj.__dict__` from `state`.
    Built {
        obj: Box<PickleValue>,
        state: Box<PickleValue>,
    },
}

/// Checks whether a `GLOBAL` reference is in the security allowlist.
///
/// Only `PyTorch` tensor-reconstruction callables, storage classes, and
/// `collections.OrderedDict` are permitted.
fn is_allowed_global(module: &str, name: &str) -> bool {
    matches!(
        (module, name),
        (
            "torch._utils",
            "_rebuild_tensor_v2" | "_rebuild_parameter" | "_rebuild_parameter_with_state"
        ) | (
            "torch",
            "FloatStorage"
                | "DoubleStorage"
                | "HalfStorage"
                | "BFloat16Storage"
                | "LongStorage"
                | "IntStorage"
                | "ShortStorage"
                | "CharStorage"
                | "ByteStorage"
                | "BoolStorage"
        ) | ("collections", "OrderedDict")
            | ("torch.nn.parameter", "Parameter")
    )
}

/// Returns `true` if this is a `REDUCE(OrderedDict, ())` call that should
/// produce an empty `Dict` instead of a `Reduced` node.
fn is_ordered_dict_constructor(callable: &PickleValue, args: &PickleValue) -> bool {
    if let PickleValue::Global { module, name } = callable {
        if module == "collections" && name == "OrderedDict" {
            if let PickleValue::Tuple(items) = args {
                return items.is_empty();
            }
        }
    }
    false
}

/// Minimal pickle VM state.
struct PickleVm<'a> {
    /// Raw pickle bytes.
    data: &'a [u8],
    /// Current read position.
    pos: usize,
    /// Value stack.
    stack: Vec<PickleValue>,
    /// Mark stack (positions in the value stack).
    mark_stack: Vec<usize>,
    /// Memo table (protocol 2+ `BINPUT`/`BINGET`, protocol 4+ `MEMOIZE`).
    memo: HashMap<u32, PickleValue>,
    /// Auto-incrementing memo key for `MEMOIZE` opcode.
    next_memo_id: u32,
}

impl<'a> PickleVm<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            mark_stack: Vec::new(),
            memo: HashMap::new(),
            next_memo_id: 0,
        }
    }

    // -- byte reading helpers ------------------------------------------------

    fn read_u8(&mut self) -> crate::Result<u8> {
        let b = self
            .data
            .get(self.pos)
            .copied()
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "unexpected end of pickle stream".into(),
            })?;
        self.pos += 1;
        Ok(b)
    }

    fn read_u16_le(&mut self) -> crate::Result<u16> {
        let bytes: [u8; 2] = self.read_fixed()?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i32_le(&mut self) -> crate::Result<i32> {
        let bytes: [u8; 4] = self.read_fixed()?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_u32_le(&mut self) -> crate::Result<u32> {
        let bytes: [u8; 4] = self.read_fixed()?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64_le(&mut self) -> crate::Result<u64> {
        let bytes: [u8; 8] = self.read_fixed()?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Reads exactly `N` bytes from the pickle stream into a fixed-size array.
    fn read_fixed<const N: usize>(&mut self) -> crate::Result<[u8; N]> {
        let hi = self
            .pos
            .checked_add(N)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "pickle offset overflow".into(),
            })?;
        let slice = self
            .data
            .get(self.pos..hi)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "unexpected end of pickle stream".into(),
            })?;
        self.pos = hi;
        // try_into: slice length == N guaranteed by .get(pos..pos+N)
        let arr: [u8; N] = slice.try_into().map_err(|_| AnamnesisError::Parse {
            reason: "internal: slice-to-array conversion failed".into(),
        })?;
        Ok(arr)
    }

    fn read_bytes(&mut self, n: usize) -> crate::Result<&'a [u8]> {
        let hi = self
            .pos
            .checked_add(n)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "pickle offset overflow".into(),
            })?;
        let slice = self
            .data
            .get(self.pos..hi)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "unexpected end of pickle stream".into(),
            })?;
        self.pos = hi;
        Ok(slice)
    }

    /// Reads a newline-terminated string (for text-mode `GLOBAL` opcode).
    fn read_line(&mut self) -> crate::Result<&'a str> {
        let start = self.pos;
        loop {
            let b = self.read_u8()?;
            if b == b'\n' {
                let line =
                    self.data
                        .get(start..self.pos - 1)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "pickle line read out of bounds".into(),
                        })?;
                return std::str::from_utf8(line).map_err(|e| AnamnesisError::Parse {
                    reason: format!("non-UTF-8 pickle string: {e}"),
                });
            }
        }
    }

    // -- stack helpers -------------------------------------------------------

    fn pop(&mut self) -> crate::Result<PickleValue> {
        self.stack.pop().ok_or_else(|| AnamnesisError::Parse {
            reason: "pickle stack underflow".into(),
        })
    }

    fn pop_mark(&mut self) -> crate::Result<Vec<PickleValue>> {
        let mark_pos = self.mark_stack.pop().ok_or_else(|| AnamnesisError::Parse {
            reason: "pickle mark stack underflow".into(),
        })?;
        let items = self.stack.split_off(mark_pos);
        Ok(items)
    }

    // -- main execution loop -------------------------------------------------

    /// Executes the pickle stream and returns the top-of-stack value.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`] on malformed opcodes, stack
    /// underflow, non-allowlisted globals, or unrecognized opcodes.
    // BORROW: throughout this function, `.to_owned()` and `.to_vec()` convert
    // borrowed slices from the pickle byte stream (which borrows from the mmap)
    // into owned `PickleValue` variants. The pickle VM's stack must own its
    // values because the VM outlives individual opcode reads.
    fn execute(&mut self) -> crate::Result<PickleValue> {
        loop {
            let opcode = self.read_u8()?;
            match opcode {
                // PROTO — protocol version (skip the version byte)
                0x80 => {
                    let _version = self.read_u8()?;
                }
                // FRAME — protocol 4+ frame marker (skip 8-byte length)
                0x95 => {
                    let _frame_len = self.read_u64_le()?;
                }
                // STOP — end of pickle, return top of stack
                b'.' => return self.pop(),

                // -- push constants ------------------------------------------
                b'N' => self.stack.push(PickleValue::None),
                // NEWTRUE
                0x88 => self.stack.push(PickleValue::Bool(true)),
                // NEWFALSE
                0x89 => self.stack.push(PickleValue::Bool(false)),

                // -- integers ------------------------------------------------

                // BININT (4-byte signed)
                b'J' => {
                    let v = self.read_i32_le()?;
                    // CAST: i32 → i64, lossless widening
                    self.stack.push(PickleValue::Int(i64::from(v)));
                }
                // BININT1 (1-byte unsigned)
                b'K' => {
                    let v = self.read_u8()?;
                    // CAST: u8 → i64, lossless widening
                    self.stack.push(PickleValue::Int(i64::from(v)));
                }
                // BININT2 (2-byte unsigned)
                b'M' => {
                    let v = self.read_u16_le()?;
                    // CAST: u16 → i64, lossless widening
                    self.stack.push(PickleValue::Int(i64::from(v)));
                }
                // LONG1 (arbitrary-precision int, 1-byte size prefix)
                0x8a => {
                    let n = self.read_u8()?;
                    // CAST: u8 → usize, lossless on all platforms
                    let bytes = self.read_bytes(usize::from(n))?;
                    let val = long1_to_i64(bytes)?;
                    self.stack.push(PickleValue::Int(val));
                }

                // -- strings -------------------------------------------------

                // SHORT_BINUNICODE (1-byte length prefix)
                0x8c => {
                    let n = self.read_u8()?;
                    // CAST: u8 → usize, lossless
                    let bytes = self.read_bytes(usize::from(n))?;
                    let s = std::str::from_utf8(bytes).map_err(|e| AnamnesisError::Parse {
                        reason: format!("non-UTF-8 pickle string: {e}"),
                    })?;
                    self.stack.push(PickleValue::String(s.to_owned()));
                }
                // BINUNICODE (4-byte length prefix)
                b'X' => {
                    let n = self.read_u32_le()?;
                    let len = usize::try_from(n).map_err(|_| AnamnesisError::Parse {
                        reason: "BINUNICODE length overflow".into(),
                    })?;
                    let bytes = self.read_bytes(len)?;
                    let s = std::str::from_utf8(bytes).map_err(|e| AnamnesisError::Parse {
                        reason: format!("non-UTF-8 pickle string: {e}"),
                    })?;
                    self.stack.push(PickleValue::String(s.to_owned()));
                }
                // SHORT_BINSTRING (1-byte length, protocol 2 — bytes, not str)
                b'U' => {
                    let n = self.read_u8()?;
                    // CAST: u8 → usize, lossless
                    let bytes = self.read_bytes(usize::from(n))?;
                    // Protocol 2 SHORT_BINSTRING is used for ASCII identifiers;
                    // treat as UTF-8 string if valid, otherwise keep as bytes.
                    match std::str::from_utf8(bytes) {
                        Ok(s) => self.stack.push(PickleValue::String(s.to_owned())),
                        Err(_) => self.stack.push(PickleValue::Bytes(bytes.to_vec())),
                    }
                }
                // BINSTRING (4-byte length, protocol 2)
                b'T' => {
                    let n = self.read_i32_le()?;
                    if n < 0 {
                        return Err(AnamnesisError::Parse {
                            reason: "negative BINSTRING length".into(),
                        });
                    }
                    let len = usize::try_from(n).map_err(|_| AnamnesisError::Parse {
                        reason: "BINSTRING length overflow".into(),
                    })?;
                    let bytes = self.read_bytes(len)?;
                    match std::str::from_utf8(bytes) {
                        Ok(s) => self.stack.push(PickleValue::String(s.to_owned())),
                        Err(_) => self.stack.push(PickleValue::Bytes(bytes.to_vec())),
                    }
                }

                // -- bytes ---------------------------------------------------

                // BINBYTES (4-byte length, protocol 3+)
                b'B' => {
                    let n = self.read_u32_le()?;
                    let len = usize::try_from(n).map_err(|_| AnamnesisError::Parse {
                        reason: "BINBYTES length overflow".into(),
                    })?;
                    let bytes = self.read_bytes(len)?;
                    self.stack.push(PickleValue::Bytes(bytes.to_vec()));
                }
                // SHORT_BINBYTES (1-byte length, protocol 3+)
                b'C' => {
                    let n = self.read_u8()?;
                    // CAST: u8 → usize, lossless
                    let bytes = self.read_bytes(usize::from(n))?;
                    self.stack.push(PickleValue::Bytes(bytes.to_vec()));
                }

                // -- containers ----------------------------------------------

                // EMPTY_DICT
                b'}' => self.stack.push(PickleValue::Dict(Vec::new())),
                // EMPTY_LIST
                b']' => self.stack.push(PickleValue::List(Vec::new())),
                // EMPTY_TUPLE
                b')' => self.stack.push(PickleValue::Tuple(Vec::new())),
                // MARK
                b'(' => self.mark_stack.push(self.stack.len()),

                // TUPLE (pop to mark → tuple)
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(PickleValue::Tuple(items));
                }
                // TUPLE1
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a]));
                }
                // TUPLE2
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a, b]));
                }
                // TUPLE3
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![a, b, c]));
                }

                // SETITEMS (pop mark → alternating key/value pairs → dict)
                b'u' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(AnamnesisError::Parse {
                            reason: "SETITEMS: odd number of items on stack".into(),
                        });
                    }
                    let dict = self.stack.last_mut().ok_or_else(|| AnamnesisError::Parse {
                        reason: "SETITEMS: empty stack (no dict)".into(),
                    })?;
                    if let PickleValue::Dict(ref mut pairs) = *dict {
                        let mut iter = items.into_iter();
                        while let Some(key) = iter.next() {
                            // EXPLICIT: the odd-length check above guarantees
                            // a value exists for every key.
                            let val = iter.next().ok_or_else(|| AnamnesisError::Parse {
                                reason: "SETITEMS: missing value for key".into(),
                            })?;
                            pairs.push((key, val));
                        }
                    } else {
                        return Err(AnamnesisError::Parse {
                            reason: "SETITEMS: top of stack is not a dict".into(),
                        });
                    }
                }
                // SETITEM (pop value, key → dict)
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let dict = self.stack.last_mut().ok_or_else(|| AnamnesisError::Parse {
                        reason: "SETITEM: empty stack (no dict)".into(),
                    })?;
                    if let PickleValue::Dict(ref mut pairs) = *dict {
                        pairs.push((key, value));
                    } else {
                        return Err(AnamnesisError::Parse {
                            reason: "SETITEM: top of stack is not a dict".into(),
                        });
                    }
                }
                // APPEND
                b'a' => {
                    let item = self.pop()?;
                    let list = self.stack.last_mut().ok_or_else(|| AnamnesisError::Parse {
                        reason: "APPEND: empty stack (no list)".into(),
                    })?;
                    if let PickleValue::List(ref mut items) = *list {
                        items.push(item);
                    } else {
                        return Err(AnamnesisError::Parse {
                            reason: "APPEND: top of stack is not a list".into(),
                        });
                    }
                }
                // APPENDS
                b'e' => {
                    let new_items = self.pop_mark()?;
                    let list = self.stack.last_mut().ok_or_else(|| AnamnesisError::Parse {
                        reason: "APPENDS: empty stack (no list)".into(),
                    })?;
                    if let PickleValue::List(ref mut items) = *list {
                        items.extend(new_items);
                    } else {
                        return Err(AnamnesisError::Parse {
                            reason: "APPENDS: top of stack is not a list".into(),
                        });
                    }
                }

                // -- object construction -------------------------------------

                // GLOBAL (text mode: "module\nname\n")
                b'c' => {
                    let module = self.read_line()?.to_owned();
                    let name = self.read_line()?.to_owned();
                    if !is_allowed_global(&module, &name) {
                        return Err(AnamnesisError::Parse {
                            reason: format!(
                                "disallowed pickle global `{module}.{name}` \
                                 (potential code execution)"
                            ),
                        });
                    }
                    self.stack.push(PickleValue::Global { module, name });
                }
                // STACK_GLOBAL (protocol 4+: pop name, pop module from stack)
                0x93 => {
                    let name_val = self.pop()?;
                    let module_val = self.pop()?;
                    let (module, name) = match (&module_val, &name_val) {
                        (PickleValue::String(m), PickleValue::String(n)) => {
                            (m.as_str(), n.as_str())
                        }
                        _ => {
                            return Err(AnamnesisError::Parse {
                                reason: "STACK_GLOBAL: module/name are not strings".into(),
                            })
                        }
                    };
                    if !is_allowed_global(module, name) {
                        return Err(AnamnesisError::Parse {
                            reason: format!(
                                "disallowed pickle global `{module}.{name}` \
                                 (potential code execution)"
                            ),
                        });
                    }
                    self.stack.push(PickleValue::Global {
                        module: module.to_owned(),
                        name: name.to_owned(),
                    });
                }
                // REDUCE (pop args, pop callable → Reduced)
                // NEWOBJ (pop args, pop cls → Reduced) — same semantics
                b'R' | 0x81 => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    // Semantic interpretation: REDUCE(OrderedDict, ()) → empty Dict.
                    // Python actually calls OrderedDict() here, producing a real dict
                    // that SETITEMS will later populate. Without this, SETITEMS fails.
                    if is_ordered_dict_constructor(&callable, &args) {
                        self.stack.push(PickleValue::Dict(Vec::new()));
                    } else {
                        self.stack.push(PickleValue::Reduced {
                            callable: Box::new(callable),
                            args: Box::new(args),
                        });
                    }
                }
                // BUILD (pop state, peek obj → Built)
                b'b' => {
                    let state = self.pop()?;
                    let obj = self.pop()?;
                    self.stack.push(PickleValue::Built {
                        obj: Box::new(obj),
                        state: Box::new(state),
                    });
                }
                // BINPERSID (pop id → PersistentId)
                b'Q' => {
                    let pid = self.pop()?;
                    self.stack.push(PickleValue::PersistentId(Box::new(pid)));
                }

                // -- memo operations -----------------------------------------

                // BINPUT (1-byte key)
                b'q' => {
                    let key = self.read_u8()?;
                    let val = self
                        .stack
                        .last()
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "BINPUT: empty stack".into(),
                        })?
                        .clone();
                    // CAST: u8 → u32, lossless
                    self.memo.insert(u32::from(key), val);
                }
                // LONG_BINPUT (4-byte key)
                b'r' => {
                    let key = self.read_u32_le()?;
                    let val = self
                        .stack
                        .last()
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "LONG_BINPUT: empty stack".into(),
                        })?
                        .clone();
                    self.memo.insert(key, val);
                }
                // BINGET (1-byte key)
                b'h' => {
                    let key = self.read_u8()?;
                    // CAST: u8 → u32, lossless
                    let val = self
                        .memo
                        .get(&u32::from(key))
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("BINGET: memo key {key} not found"),
                        })?
                        .clone();
                    self.stack.push(val);
                }
                // LONG_BINGET (4-byte key)
                b'j' => {
                    let key = self.read_u32_le()?;
                    let val = self
                        .memo
                        .get(&key)
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: format!("LONG_BINGET: memo key {key} not found"),
                        })?
                        .clone();
                    self.stack.push(val);
                }
                // MEMOIZE (protocol 4+: auto-assigns next key)
                0x94 => {
                    let val = self
                        .stack
                        .last()
                        .ok_or_else(|| AnamnesisError::Parse {
                            reason: "MEMOIZE: empty stack".into(),
                        })?
                        .clone();
                    self.memo.insert(self.next_memo_id, val);
                    self.next_memo_id += 1;
                }

                _ => {
                    return Err(AnamnesisError::Parse {
                        reason: format!("unsupported pickle opcode 0x{opcode:02x}"),
                    });
                }
            }
        }
    }
}

/// Converts a pickle `LONG1` byte sequence to `i64`.
///
/// Pickle's `LONG1` stores an arbitrary-precision signed integer in
/// little-endian two's-complement. We support values that fit in `i64`.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the value exceeds `i64` range.
fn long1_to_i64(bytes: &[u8]) -> crate::Result<i64> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() > 8 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "LONG1 value too large ({} bytes, max 8 for i64)",
                bytes.len()
            ),
        });
    }
    // Two's complement: if the high bit of the last byte is set, the
    // number is negative. Pad with 0xFF for negative, 0x00 for positive.
    let last = bytes.last().copied().ok_or_else(|| AnamnesisError::Parse {
        reason: "LONG1 empty bytes".into(),
    })?;
    let pad = if last & 0x80 != 0 { 0xFF } else { 0x00 };
    let mut buf = [pad; 8];
    // bytes.len() ≤ 8 checked above; .get_mut returns Some
    let dest = buf
        .get_mut(..bytes.len())
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "LONG1 internal: slice bounds exceeded".into(),
        })?;
    dest.copy_from_slice(bytes);
    Ok(i64::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// Tensor extraction
// ---------------------------------------------------------------------------

/// Intermediate representation of a tensor reference parsed from the pickle.
struct TensorRef {
    name: String,
    /// Data file index in the ZIP archive (e.g., `"0"` → `archive/data/0`).
    storage_key: String,
    dtype: PthDtype,
    /// Byte offset into the storage file.
    storage_offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// Attempts to extract an `i64` from a `PickleValue::Int`.
fn as_i64(val: &PickleValue) -> crate::Result<i64> {
    if let PickleValue::Int(v) = val {
        Ok(*v)
    } else {
        Err(AnamnesisError::Parse {
            reason: format!("expected int, got {val:?}"),
        })
    }
}

/// Attempts to extract a `usize` from a `PickleValue::Int`.
fn as_usize(val: &PickleValue) -> crate::Result<usize> {
    let v = as_i64(val)?;
    usize::try_from(v).map_err(|_| AnamnesisError::Parse {
        reason: format!("integer {v} does not fit in usize"),
    })
}

/// Attempts to extract a `&str` from a `PickleValue::String`.
fn as_str(val: &PickleValue) -> crate::Result<&str> {
    if let PickleValue::String(s) = val {
        Ok(s.as_str())
    } else {
        Err(AnamnesisError::Parse {
            reason: format!("expected string, got {val:?}"),
        })
    }
}

/// Converts a `PickleValue::Tuple` into a `Vec<usize>` (for shape/strides).
fn tuple_to_usize_vec(val: &PickleValue) -> crate::Result<Vec<usize>> {
    if let PickleValue::Tuple(items) = val {
        items.iter().map(as_usize).collect()
    } else {
        Err(AnamnesisError::Parse {
            reason: format!("expected tuple, got {val:?}"),
        })
    }
}

/// Parses a `_rebuild_tensor_v2` call's arguments into a `TensorRef`.
///
/// Expected args tuple:
/// `(PersistentId(storage_info), offset, shape, strides, requires_grad, metadata)`
// EXHAUSTIVE: PickleValue is private; wildcards catch irrelevant variants
#[allow(clippy::wildcard_enum_match_arm)]
fn parse_rebuild_args(name: &str, args: &PickleValue) -> crate::Result<TensorRef> {
    let PickleValue::Tuple(items) = args else {
        return Err(AnamnesisError::Parse {
            reason: format!("tensor `{name}`: expected tuple args for _rebuild_tensor_v2"),
        });
    };

    if items.len() < 4 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "tensor `{name}`: _rebuild_tensor_v2 needs ≥4 args, got {}",
                items.len()
            ),
        });
    }

    // Item 0: PersistentId wrapping a tuple of (tag, storage_type, key, device, n_elements)
    let persistent_id = items.first().ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing args[0]"),
    })?;
    let storage_tuple = match persistent_id {
        PickleValue::PersistentId(inner) => match inner.as_ref() {
            PickleValue::Tuple(t) => t,
            other => {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "tensor `{name}`: PersistentId payload is not a tuple: {other:?}"
                    ),
                })
            }
        },
        other => {
            return Err(AnamnesisError::Parse {
                reason: format!("tensor `{name}`: expected PersistentId, got {other:?}"),
            })
        }
    };

    if storage_tuple.len() < 5 {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "tensor `{name}`: storage tuple needs ≥5 items, got {}",
                storage_tuple.len()
            ),
        });
    }

    // storage_tuple[1] is the storage type Global, [2] is the data file key
    let st1 = storage_tuple.get(1).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing storage_tuple[1]"),
    })?;
    let dtype = match st1 {
        PickleValue::Global { module, name: cls } => PthDtype::from_storage_class(module, cls)?,
        other => {
            return Err(AnamnesisError::Parse {
                reason: format!("tensor `{name}`: expected storage Global, got {other:?}"),
            })
        }
    };
    let st2 = storage_tuple.get(2).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing storage_tuple[2]"),
    })?;
    // BORROW: owned copy needed — TensorMeta outlives the PickleValue borrow
    let storage_key = as_str(st2)?.to_owned();

    // items[1]: storage offset (in elements, not bytes)
    let it1 = items.get(1).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing args[1]"),
    })?;
    let storage_offset_elements = as_usize(it1)?;
    let storage_offset = storage_offset_elements
        .checked_mul(dtype.byte_size())
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("tensor `{name}`: storage offset overflow"),
        })?;

    // items[2], items[3]: shape and strides
    let it2 = items.get(2).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing args[2]"),
    })?;
    let it3 = items.get(3).ok_or_else(|| AnamnesisError::Parse {
        reason: format!("tensor `{name}`: missing args[3]"),
    })?;
    let shape = tuple_to_usize_vec(it2)?;
    let strides = tuple_to_usize_vec(it3)?;

    Ok(TensorRef {
        // BORROW: owned copy — TensorRef outlives the PickleValue borrow
        name: name.to_owned(),
        storage_key,
        dtype,
        storage_offset,
        shape,
        strides,
    })
}

// EXHAUSTIVE: PickleValue is private; wildcards catch irrelevant variants
#[allow(clippy::wildcard_enum_match_arm)]
/// Unwraps nested pickle structures to find the inner `_rebuild_tensor_v2` call.
///
/// Handles:
/// - Direct: `Reduced { _rebuild_tensor_v2, args }`
/// - Parameter-wrapped: `Reduced { _rebuild_parameter, Tuple([Reduced { _rebuild_tensor_v2, args }, ...]) }`
/// - `Built`-wrapped: `Built { obj, state }` → recurse into `obj`
fn unwrap_to_rebuild(val: &PickleValue) -> Option<(&PickleValue, &PickleValue)> {
    match val {
        PickleValue::Reduced { callable, args, .. } => {
            if let PickleValue::Global { module, name } = callable.as_ref() {
                if module == "torch._utils" && name == "_rebuild_tensor_v2" {
                    return Some((callable, args));
                }
                // _rebuild_parameter wraps _rebuild_tensor_v2 as first arg:
                // REDUCE(_rebuild_parameter, TUPLE(REDUCE(_rebuild_tensor_v2, ...), ...))
                if module == "torch._utils"
                    && (name == "_rebuild_parameter" || name == "_rebuild_parameter_with_state")
                {
                    if let PickleValue::Tuple(items) = args.as_ref() {
                        if let Some(first) = items.first() {
                            return unwrap_to_rebuild(first);
                        }
                    }
                }
            }
            None
        }
        PickleValue::Built { obj, .. } => unwrap_to_rebuild(obj),
        _ => None,
    }
}

/// Extracts the dict of key→value pairs from the top-level pickle value.
///
/// Handles both a raw `Dict` and a `Reduced { OrderedDict, ... }`.
// EXHAUSTIVE: PickleValue is private; wildcards catch irrelevant variants
#[allow(clippy::wildcard_enum_match_arm)]
fn extract_dict_pairs(root: &PickleValue) -> crate::Result<&[(PickleValue, PickleValue)]> {
    match root {
        PickleValue::Dict(pairs) => Ok(pairs),
        PickleValue::Reduced { callable, args: _ } => {
            // OrderedDict is constructed as REDUCE(GLOBAL("collections","OrderedDict"), args)
            // where args is a Tuple containing a List of Tuple(key, value) pairs,
            // or an empty Tuple (with SETITEMS populating later — already handled by BUILD).
            if let PickleValue::Global { module, name } = callable.as_ref() {
                if module == "collections" && name == "OrderedDict" {
                    // The result of REDUCE(OrderedDict, ()) + SETITEMS is a Dict
                    // via BUILD. But if it arrived as Reduced, the args may contain
                    // the data. Return empty and let the caller handle Built.
                    return Ok(&[]);
                }
            }
            Err(AnamnesisError::Parse {
                reason: format!("top-level pickle value is not a dict: {root:?}"),
            })
        }
        PickleValue::Built { obj, state: _ } => {
            // BUILD(obj, state) sets obj.__dict__ = state. The tensor data
            // lives in obj (the OrderedDict), not in state (which is the
            // __dict__ containing _metadata). Always recurse into obj.
            extract_dict_pairs(obj)
        }
        _ => Err(AnamnesisError::Parse {
            reason: format!("top-level pickle value is not a dict or OrderedDict: {root:?}"),
        }),
    }
}

/// Computes the expected strides for a contiguous (row-major) tensor.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    // Walk right-to-left, accumulating the product.
    // EXPLICIT: manual indexing used because each stride depends on the
    // previously computed stride[i+1]; an iterator chain cannot express this.
    for i in (0..ndim.saturating_sub(1)).rev() {
        // .get() returns Some because i < ndim-1 ⇒ i+1 < ndim
        if let (Some(&prev), Some(&dim)) = (strides.get(i + 1), shape.get(i + 1)) {
            if let Some(s) = strides.get_mut(i) {
                *s = prev.saturating_mul(dim);
            }
        }
    }
    strides
}

/// Returns `true` if the tensor's strides match contiguous (row-major) layout.
fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let expected = contiguous_strides(shape);
    strides == expected
}

/// Copies a non-contiguous tensor to contiguous (row-major) layout.
///
/// This is the slow path for tensors with non-standard strides (e.g.,
/// transposed views). Rare in `state_dict` files but must be handled for
/// correctness.
fn copy_to_contiguous(
    storage: &[u8],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    elem_size: usize,
) -> crate::Result<Vec<u8>> {
    let n_elements: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "element count overflow".into(),
        })?;
    let out_bytes = n_elements
        .checked_mul(elem_size)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "output size overflow".into(),
        })?;
    let mut out = vec![0u8; out_bytes];

    // Multi-dimensional index iteration: walk every element by incrementing
    // a coordinate vector, computing the source offset from strides.
    let ndim = shape.len();
    let mut coords = vec![0usize; ndim];
    for flat_idx in 0..n_elements {
        // Compute source offset from strides and coordinates.
        let src_elem_offset: usize = coords
            .iter()
            .zip(strides.iter())
            .try_fold(0usize, |acc, (&c, &s)| {
                c.checked_mul(s).and_then(|cs| acc.checked_add(cs))
            })
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "stride offset overflow".into(),
            })?;
        let src_byte = offset
            .checked_add(src_elem_offset.checked_mul(elem_size).ok_or_else(|| {
                AnamnesisError::Parse {
                    reason: "source byte offset overflow".into(),
                }
            })?)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "source byte offset overflow".into(),
            })?;
        let src_end = src_byte
            .checked_add(elem_size)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "source end offset overflow".into(),
            })?;
        let src_slice = storage
            .get(src_byte..src_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "storage read out of bounds at [{src_byte}..{src_end}], \
                     storage len = {}",
                    storage.len()
                ),
            })?;

        let dst_byte = flat_idx
            .checked_mul(elem_size)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "destination offset overflow".into(),
            })?;
        let dst_end = dst_byte
            .checked_add(elem_size)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "destination end offset overflow".into(),
            })?;
        let dst_slice = out
            .get_mut(dst_byte..dst_end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "destination write out of bounds".into(),
            })?;
        dst_slice.copy_from_slice(src_slice);

        // Increment coordinates (rightmost dimension first).
        // EXPLICIT: manual coordinate increment; iterator-based
        // multi-index generation would allocate per-element.
        for d in (0..ndim).rev() {
            // d < ndim guaranteed by loop range; coords and shape have length ndim
            if let (Some(c), Some(&s)) = (coords.get_mut(d), shape.get(d)) {
                *c += 1;
                if *c < s {
                    break;
                }
                *c = 0;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parses a `PyTorch` `.pth` `state_dict` file.
///
/// Returns a `Vec` of `PthTensor` with raw data in native-endian, row-major
/// layout. The order matches the `OrderedDict` insertion order from the
/// original Python `state_dict`.
///
/// # Supported Formats
///
/// Only modern `.pth` files (`PyTorch` ≥ 1.6, ZIP-based) are supported.
/// Legacy raw-pickle files (pre-1.6) are rejected with
/// [`AnamnesisError::Unsupported`].
///
/// # Security
///
/// The pickle interpreter uses an explicit `GLOBAL` allowlist. Non-`PyTorch`
/// callables (e.g., `os.system`, `subprocess.Popen`) are rejected with
/// [`AnamnesisError::Parse`], preventing arbitrary code execution.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the file is not a valid `PyTorch`
/// ZIP archive, uses unsupported pickle opcodes, or contains
/// non-allowlisted globals.
///
/// Returns [`AnamnesisError::Unsupported`] for legacy (pre-1.6) `.pth`
/// files that are raw pickle without ZIP wrapping.
///
/// Returns [`AnamnesisError::Io`] if the file cannot be read.
///
/// # Memory
///
/// Memory-maps the file with `memmap2` (page prefaulting). Tensor data
/// is **not** copied during parsing — [`ParsedPth::tensors()`] slices
/// directly from the mmap (zero-copy for contiguous little-endian tensors,
/// which is >99% of real files). Peak memory during parsing ≈ file size
/// (mapped, not heap-allocated) + ~1 KB metadata per tensor. The mmap is
/// released when the returned `ParsedPth` is dropped.
#[allow(unsafe_code)]
pub fn parse_pth(path: impl AsRef<Path>) -> crate::Result<ParsedPth> {
    let file = std::fs::File::open(path.as_ref())?;
    // SAFETY: memmap2::Mmap requires unsafe because the OS could modify the
    // mapped region if another process writes to the file concurrently.
    // Model files are read-only artifacts — this is the standard assumption
    // for all tensor format parsers (same as safetensors crate's mmap path).
    let raw =
        unsafe { memmap2::MmapOptions::new().populate().map(&file) }.map_err(AnamnesisError::Io)?;

    // Legacy format detection: check ZIP magic before attempting to parse.
    let magic = raw.get(..4).ok_or_else(|| AnamnesisError::Parse {
        reason: "file too small to be a .pth archive".into(),
    })?;
    if magic.first() == Some(&0x80) && magic.get(1).is_some_and(|&b| b <= 0x05) {
        return Err(AnamnesisError::Unsupported {
            format: "pth".into(),
            detail: "legacy .pth format (pre-PyTorch 1.6) is not supported; \
                     re-save with torch.save()"
                .into(),
        });
    }
    if magic != b"PK\x03\x04" {
        return Err(AnamnesisError::Parse {
            reason: "file is not a ZIP archive (missing PK\\x03\\x04 magic)".into(),
        });
    }

    let cursor = std::io::Cursor::new(&raw[..]);
    let mut archive = zip::ZipArchive::new(cursor)?;

    // 1. Pre-index all ZIP entry names → (data_start, size) for O(1) lookup.
    //    This replaces the O(n) find_entry_name scanning per tensor.
    let entry_index = build_entry_index(&mut archive, &raw)?;

    // 2. Read byte order (default to little-endian).
    let big_endian = match entry_index.get("byteorder") {
        Some(&(start, len)) => {
            let bytes = raw
                .get(start..start + len)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: "byteorder entry out of bounds".into(),
                })?;
            let text = std::str::from_utf8(bytes).map_err(|e| AnamnesisError::Parse {
                reason: format!("byteorder entry is not UTF-8: {e}"),
            })?;
            match text.trim() {
                "little" => false,
                "big" => true,
                other => {
                    return Err(AnamnesisError::Parse {
                        reason: format!(
                            "unknown byte order `{other}` (expected `little` or `big`)"
                        ),
                    })
                }
            }
        }
        None => false, // default: little-endian
    };

    // 3. Read and execute the pickle stream.
    let &(pkl_start, pkl_len) =
        entry_index
            .get("data.pkl")
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "ZIP entry `data.pkl` not found".into(),
            })?;
    let pkl_data =
        raw.get(pkl_start..pkl_start + pkl_len)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: "data.pkl slice out of bounds".into(),
            })?;
    let mut vm = PickleVm::new(pkl_data);
    let root = vm.execute()?;

    // 4. Extract tensor metadata from the pickle structure.
    //    Data is NOT copied here — tensors() will borrow from the mmap.
    let dict_pairs = extract_dict_pairs(&root)?;
    let mut meta = Vec::new();
    for (key, value) in dict_pairs {
        let name = as_str(key)?;
        if let Some((_callable, args)) = unwrap_to_rebuild(value) {
            let tref = parse_rebuild_args(name, args)?;
            meta.push(TensorMeta {
                name: tref.name,
                shape: tref.shape,
                dtype: tref.dtype,
                storage_key: tref.storage_key,
                storage_offset: tref.storage_offset,
                strides: tref.strides,
            });
        }
    }

    Ok(ParsedPth {
        mmap: raw,
        meta,
        entry_index,
        big_endian,
    })
}

/// Builds an O(1) index of ZIP entry suffix → `(data_start, data_len)` in `raw`.
///
/// Only indexes STORED entries (uncompressed). The suffix is the part after
/// the archive prefix (e.g., `"data.pkl"`, `"data/0"`, `"byteorder"`).
fn build_entry_index(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    raw: &[u8],
) -> crate::Result<HashMap<String, (usize, usize)>> {
    let mut index = HashMap::with_capacity(archive.len());

    for i in 0..archive.len() {
        let entry = archive.by_index(i).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to read ZIP entry {i}: {e}"),
        })?;

        if entry.compression() != zip::CompressionMethod::Stored {
            continue;
        }

        // CAST: u64 → usize, ZIP offsets/sizes fit in usize for model files.
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let data_start = entry.data_start() as usize;
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let data_len = entry.size() as usize;

        // BORROW: owned copy — entry name borrows from ZipArchive, but
        // the HashMap key and error messages must outlive the entry borrow.
        let full_name = entry.name().to_owned();

        // Validate range.
        let data_end = data_start
            .checked_add(data_len)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("ZIP entry `{full_name}`: data range overflow"),
            })?;
        if data_end > raw.len() {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "ZIP entry `{full_name}`: data range [{data_start}..{data_end}] \
                     exceeds file size {}",
                    raw.len()
                ),
            });
        }

        // Strip the archive prefix to get the suffix key.
        // "archive/data.pkl" → "data.pkl", "my_model/data/0" → "data/0"
        let suffix = full_name
            .find('/')
            .map_or(full_name.as_str(), |pos| {
                full_name.get(pos + 1..).unwrap_or(&full_name)
            })
            .to_owned();

        if !suffix.is_empty() {
            index.insert(suffix, (data_start, data_len));
        }
    }

    Ok(index)
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
    clippy::wildcard_enum_match_arm
)]
mod tests {
    use super::*;

    // -- PthDtype ------------------------------------------------------------

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(PthDtype::Bool.byte_size(), 1);
        assert_eq!(PthDtype::U8.byte_size(), 1);
        assert_eq!(PthDtype::I8.byte_size(), 1);
        assert_eq!(PthDtype::F16.byte_size(), 2);
        assert_eq!(PthDtype::BF16.byte_size(), 2);
        assert_eq!(PthDtype::I16.byte_size(), 2);
        assert_eq!(PthDtype::F32.byte_size(), 4);
        assert_eq!(PthDtype::I32.byte_size(), 4);
        assert_eq!(PthDtype::F64.byte_size(), 8);
        assert_eq!(PthDtype::I64.byte_size(), 8);
    }

    #[test]
    fn dtype_display() {
        assert_eq!(PthDtype::F32.to_string(), "F32");
        assert_eq!(PthDtype::BF16.to_string(), "BF16");
        assert_eq!(PthDtype::Bool.to_string(), "BOOL");
    }

    #[test]
    fn dtype_to_dtype_roundtrip() {
        assert_eq!(PthDtype::F32.to_dtype().unwrap(), Dtype::F32);
        assert_eq!(PthDtype::F16.to_dtype().unwrap(), Dtype::F16);
        assert_eq!(PthDtype::BF16.to_dtype().unwrap(), Dtype::BF16);
        assert_eq!(PthDtype::I64.to_dtype().unwrap(), Dtype::I64);
        assert_eq!(PthDtype::Bool.to_dtype().unwrap(), Dtype::Bool);
    }

    #[test]
    fn dtype_from_storage_class() {
        assert_eq!(
            PthDtype::from_storage_class("torch", "FloatStorage").unwrap(),
            PthDtype::F32
        );
        assert_eq!(
            PthDtype::from_storage_class("torch", "BFloat16Storage").unwrap(),
            PthDtype::BF16
        );
        assert!(PthDtype::from_storage_class("torch", "UnknownStorage").is_err());
        assert!(PthDtype::from_storage_class("numpy", "FloatStorage").is_err());
    }

    // -- LONG1 conversion ----------------------------------------------------

    #[test]
    fn long1_zero() {
        assert_eq!(long1_to_i64(&[]).unwrap(), 0);
    }

    #[test]
    fn long1_positive() {
        // 255 = 0xFF as unsigned, but in two's complement with sign bit clear
        // we need two bytes: 0xFF, 0x00
        assert_eq!(long1_to_i64(&[0xFF, 0x00]).unwrap(), 255);
        assert_eq!(long1_to_i64(&[0x01]).unwrap(), 1);
        assert_eq!(long1_to_i64(&[0x80, 0x00]).unwrap(), 128);
    }

    #[test]
    fn long1_negative() {
        // -1 in two's complement = 0xFF
        assert_eq!(long1_to_i64(&[0xFF]).unwrap(), -1);
        // -128 = 0x80
        assert_eq!(long1_to_i64(&[0x80]).unwrap(), -128);
    }

    #[test]
    fn long1_too_large() {
        let big = vec![0x01; 9]; // 9 bytes > 8
        assert!(long1_to_i64(&big).is_err());
    }

    // -- Contiguous strides --------------------------------------------------

    #[test]
    fn contiguous_strides_2d() {
        assert_eq!(contiguous_strides(&[3, 4]), vec![4, 1]);
        assert_eq!(contiguous_strides(&[16, 10]), vec![10, 1]);
    }

    #[test]
    fn contiguous_strides_1d() {
        assert_eq!(contiguous_strides(&[5]), vec![1]);
    }

    #[test]
    fn contiguous_strides_scalar() {
        assert_eq!(contiguous_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn is_contiguous_true() {
        assert!(is_contiguous(&[3, 4], &[4, 1]));
        assert!(is_contiguous(&[5], &[1]));
    }

    #[test]
    fn is_contiguous_transposed() {
        // A transposed 3×4 matrix would have strides [1, 3]
        assert!(!is_contiguous(&[3, 4], &[1, 3]));
    }

    // -- Pickle VM -----------------------------------------------------------

    #[test]
    fn vm_simple_int() {
        // PROTO 2, BININT1 42, STOP
        let pkl = &[0x80, 0x02, b'K', 42, b'.'];
        let mut vm = PickleVm::new(pkl);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::Int(42)));
    }

    #[test]
    fn vm_string() {
        // PROTO 2, SHORT_BINUNICODE "hi" (len=2), STOP
        let pkl = &[0x80, 0x02, 0x8c, 0x02, b'h', b'i', b'.'];
        let mut vm = PickleVm::new(pkl);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::String(ref s) if s == "hi"));
    }

    #[test]
    fn vm_tuple2() {
        // PROTO 2, BININT1 1, BININT1 2, TUPLE2, STOP
        let pkl = &[0x80, 0x02, b'K', 1, b'K', 2, 0x86, b'.'];
        let mut vm = PickleVm::new(pkl);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert_eq!(items.len(), 2);
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn vm_empty_dict() {
        // PROTO 2, EMPTY_DICT, STOP
        let pkl = &[0x80, 0x02, b'}', b'.'];
        let mut vm = PickleVm::new(pkl);
        let result = vm.execute().unwrap();
        assert!(matches!(result, PickleValue::Dict(ref d) if d.is_empty()));
    }

    #[test]
    fn vm_dict_with_setitem() {
        // PROTO 2, EMPTY_DICT, SHORT_BINUNICODE "k" (len=1), BININT1 7, SETITEM, STOP
        let pkl = &[0x80, 0x02, b'}', 0x8c, 1, b'k', b'K', 7, b's', b'.'];
        let mut vm = PickleVm::new(pkl);
        let result = vm.execute().unwrap();
        if let PickleValue::Dict(pairs) = result {
            assert_eq!(pairs.len(), 1);
        } else {
            panic!("expected Dict");
        }
    }

    #[test]
    fn vm_memo_roundtrip() {
        // PROTO 2, BININT1 99, BINPUT 0, POP (not implemented — use different approach)
        // Instead: BININT1 99, BINPUT 0, BININT1 0, BINGET 0, TUPLE2, STOP
        // This stores 99 in memo[0], pushes 0, gets memo[0] (=99), makes tuple (0, 99)
        let pkl = &[
            0x80, 0x02, // PROTO 2
            b'K', 99, // BININT1 99
            b'q', 0, // BINPUT 0
            b'K', 0, // BININT1 0
            b'h', 0,    // BINGET 0
            0x86, // TUPLE2
            b'.', // STOP
        ];
        let mut vm = PickleVm::new(pkl);
        let result = vm.execute().unwrap();
        if let PickleValue::Tuple(items) = result {
            assert_eq!(items.len(), 2);
            assert!(matches!(&items[0], PickleValue::Int(0)));
            assert!(matches!(&items[1], PickleValue::Int(99)));
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn vm_rejects_disallowed_global() {
        // PROTO 2, GLOBAL "os\nsystem\n", STOP
        let pkl = b"\x80\x02cos\nsystem\n.";
        let mut vm = PickleVm::new(pkl);
        let err = vm.execute().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("disallowed pickle global"), "got: {msg}");
        assert!(msg.contains("os.system"), "got: {msg}");
    }

    #[test]
    fn vm_rejects_unknown_opcode() {
        // PROTO 2, unknown opcode 0xFF, STOP
        let pkl = &[0x80, 0x02, 0xFF, b'.'];
        let mut vm = PickleVm::new(pkl);
        let err = vm.execute().unwrap_err();
        assert!(err.to_string().contains("unsupported pickle opcode 0xff"));
    }

    #[test]
    fn vm_allows_torch_global() {
        // PROTO 2, GLOBAL "torch._utils\n_rebuild_tensor_v2\n", STOP
        let pkl = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n.";
        let mut vm = PickleVm::new(pkl);
        let result = vm.execute().unwrap();
        assert!(matches!(
            result,
            PickleValue::Global { ref module, ref name }
            if module == "torch._utils" && name == "_rebuild_tensor_v2"
        ));
    }

    // -- Legacy detection ----------------------------------------------------

    #[test]
    fn reject_legacy_pth() {
        // Fake legacy pickle: starts with PROTO 2 but no ZIP magic
        let data = vec![0x80, 0x02, 0x00, 0x00, 0x00];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();
        let err = parse_pth(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("legacy .pth format"));
    }

    #[test]
    fn reject_non_zip() {
        // Random bytes, not ZIP, not pickle
        let data = vec![0x00, 0x01, 0x02, 0x03, 0x04];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();
        let err = parse_pth(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("not a ZIP archive"));
    }

    #[test]
    fn reject_too_small() {
        let data = vec![0x50, 0x4B]; // Just "PK" — too short
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &data).unwrap();
        let err = parse_pth(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("too small"));
    }
}
