// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GGUF` file parsing — header, metadata key-value pairs, and tensor info table.
//!
//! `GGUF` is the dominant format for local inference, used by `llama.cpp`,
//! `Ollama`, and `LM Studio`. This module implements a lean, dependency-free
//! parser for `GGUF` versions 2 and 3 that memory-maps the file and exposes
//! tensor views as zero-copy `Cow::Borrowed` slices into the mapping — no
//! per-tensor allocation for little-endian data (the common case).
//!
//! # What this module does
//!
//! - Validates the `GGUF` magic (`"GGUF"`) and version.
//! - Reads every metadata key-value pair into a `HashMap` — supports all 13
//!   value types defined by the `GGUF` specification, including nested
//!   `ARRAY` values.
//! - Reads the tensor info table (`name`, `shape`, `ggml_type`, `offset`) and
//!   resolves each tensor's absolute position inside the memory-mapped
//!   region, honouring the `general.alignment` metadata key (default 32).
//! - Exposes a [`ParsedGguf`] handle with inspection helpers and a
//!   [`tensors`](ParsedGguf::tensors) method that returns tensor views
//!   borrowed from the mmap.
//!
//! # What this module does not do
//!
//! Dequantization of `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, etc. is the job of the
//! `remember::gguf` module (Phase 4 step 2). This parser reports the
//! [`GgufType`] of every tensor but does not decode any packed blocks.
//!
//! # Security
//!
//! The parser enforces cheap upper bounds on `tensor_count`,
//! `metadata_kv_count`, string lengths, array lengths, and array nesting
//! depth so that an adversarial file cannot cause unbounded allocation or
//! stack growth.
//!
//! # Spec reference
//!
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use crate::error::AnamnesisError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `GGUF` magic bytes — spells `"GGUF"` in ASCII.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// `GGUF` magic read as a little-endian `u32`. Useful for detecting
/// byte-swapped (big-endian) files without a full magic-byte comparison.
const GGUF_MAGIC_LE_U32: u32 = u32::from_le_bytes(*GGUF_MAGIC);

/// `GGUF` magic as a big-endian `u32`. A file that begins with this value
/// when interpreted little-endian is actually stored big-endian.
const GGUF_MAGIC_BE_U32: u32 = u32::from_be_bytes(*GGUF_MAGIC);

/// Default tensor-data alignment when `general.alignment` metadata is absent.
const DEFAULT_ALIGNMENT: u32 = 32;

/// Upper bound on `tensor_count` (soft `DoS` guard).
const MAX_TENSOR_COUNT: u64 = 1_000_000;

/// Upper bound on `metadata_kv_count` (soft `DoS` guard).
const MAX_KV_COUNT: u64 = 1_000_000;

/// Upper bound on a single `GGUF` string length (16 MiB).
///
/// The `GGUF` specification caps metadata keys at 65 535 bytes; values are
/// unbounded in theory but in practice rarely exceed a few hundred kilobytes
/// (e.g., a tokenizer vocabulary serialised as a single string).
const MAX_STRING_LEN: u64 = 16 * 1024 * 1024;

/// Upper bound on `ARRAY` element count (soft `DoS` guard).
const MAX_ARRAY_LEN: u64 = 16_000_000;

/// Maximum nesting depth for metadata `ARRAY` values.
const MAX_ARRAY_DEPTH: u32 = 4;

/// Upper bound on `n_dimensions` for a single tensor.
///
/// `ggml` itself caps this at `GGML_MAX_DIMS = 4`; we accept up to 8 for
/// future-proofing.
const MAX_TENSOR_DIMS: u32 = 8;

/// Upper bound on a single tensor's name length, in bytes.
///
/// The `GGUF` specification caps tensor names at 64 bytes, but some encoders
/// produce longer names in practice. 65 535 bytes is the metadata-key cap
/// and is comfortably above anything any real encoder emits, while keeping
/// the per-tensor string allocation bounded for adversarial inputs.
const MAX_TENSOR_NAME_LEN: u64 = 65_535;

/// Upper bound on product-of-dimensions for a single tensor (soft `DoS` guard).
///
/// Real tensors never exceed a few hundred billion elements (a 70B model's
/// embedding matrix tops out around 5·10⁹). One trillion elements is
/// comfortably beyond anything real while rejecting absurd inputs.
const MAX_TENSOR_ELEMENTS: u64 = 1_000_000_000_000;

/// Soft cap on `Vec` / `HashMap` pre-allocation for file-declared counts.
///
/// The parser accepts adversarial headers that claim up to `MAX_KV_COUNT`
/// or `MAX_TENSOR_COUNT` entries (1 M each). Trusting those counts for
/// `with_capacity` calls would allocate ~175 MB of heap before reading a
/// single entry (empirically measured: 114 MB for the metadata `HashMap`
/// plus 61 MB for the `Vec<GgufTensorInfo>` at 1 M cap). Clamping every
/// trust-the-header pre-allocation hint to this constant bounds the
/// worst-case eager allocation to ~34 KB while imposing at most
/// `log₂(MAX / CAP)` extra reallocs on legitimate files — imperceptible
/// given parse is I/O-bound and real `GGUF` files never reach the cap.
const PREALLOC_SOFT_CAP: usize = 256;

/// Total number of [`GgufType`] variants — used to size the per-dtype
/// dedup bitmap in [`ParsedGguf::inspect`]. Must be kept in sync with
/// the match arms of `GgufType::inspect_index`.
const GGUF_TYPE_COUNT: usize = 32;

// ---------------------------------------------------------------------------
// GgufType
// ---------------------------------------------------------------------------

/// Element data type for a `GGUF` tensor — mirrors `ggml_type` in `llama.cpp`.
///
/// The enum is `#[non_exhaustive]` because new `ggml_type` values are added
/// over time (e.g., the `IQ*` family appeared after the original `K`-quants,
/// and `MXFP4` was added in 2024).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum GgufType {
    /// 32-bit IEEE 754 single-precision (`GGML_TYPE_F32 = 0`).
    F32,
    /// 16-bit IEEE 754 half-precision (`GGML_TYPE_F16 = 1`).
    F16,
    /// 32-element block, 4-bit symmetric quantisation (`GGML_TYPE_Q4_0 = 2`).
    Q4_0,
    /// 32-element block, 4-bit asymmetric quantisation (`GGML_TYPE_Q4_1 = 3`).
    Q4_1,
    /// 32-element block, 5-bit symmetric quantisation (`GGML_TYPE_Q5_0 = 6`).
    Q5_0,
    /// 32-element block, 5-bit asymmetric quantisation (`GGML_TYPE_Q5_1 = 7`).
    Q5_1,
    /// 32-element block, 8-bit symmetric quantisation (`GGML_TYPE_Q8_0 = 8`).
    Q8_0,
    /// 32-element block, 8-bit quantisation with sum (`GGML_TYPE_Q8_1 = 9`).
    Q8_1,
    /// 256-element super-block, 2-bit K-quant (`GGML_TYPE_Q2_K = 10`).
    Q2_K,
    /// 256-element super-block, 3-bit K-quant (`GGML_TYPE_Q3_K = 11`).
    Q3_K,
    /// 256-element super-block, 4-bit K-quant (`GGML_TYPE_Q4_K = 12`).
    Q4_K,
    /// 256-element super-block, 5-bit K-quant (`GGML_TYPE_Q5_K = 13`).
    Q5_K,
    /// 256-element super-block, 6-bit K-quant (`GGML_TYPE_Q6_K = 14`).
    Q6_K,
    /// 256-element super-block, 8-bit K-quant (`GGML_TYPE_Q8_K = 15`).
    Q8_K,
    /// 256-element, ~2.0625 bpw (`GGML_TYPE_IQ2_XXS = 16`).
    IQ2_XXS,
    /// 256-element, ~2.3125 bpw (`GGML_TYPE_IQ2_XS = 17`).
    IQ2_XS,
    /// 256-element, ~3.0625 bpw (`GGML_TYPE_IQ3_XXS = 18`).
    IQ3_XXS,
    /// 256-element, ~1.5625 bpw (`GGML_TYPE_IQ1_S = 19`).
    IQ1_S,
    /// 32-element, 4-bit non-linear (`GGML_TYPE_IQ4_NL = 20`).
    IQ4_NL,
    /// 256-element, ~3.4375 bpw (`GGML_TYPE_IQ3_S = 21`).
    IQ3_S,
    /// 256-element, ~2.5 bpw (`GGML_TYPE_IQ2_S = 22`).
    IQ2_S,
    /// 256-element, ~4.25 bpw (`GGML_TYPE_IQ4_XS = 23`).
    IQ4_XS,
    /// Signed 8-bit integer (`GGML_TYPE_I8 = 24`).
    I8,
    /// Signed 16-bit integer (`GGML_TYPE_I16 = 25`).
    I16,
    /// Signed 32-bit integer (`GGML_TYPE_I32 = 26`).
    I32,
    /// Signed 64-bit integer (`GGML_TYPE_I64 = 27`).
    I64,
    /// 64-bit IEEE 754 double-precision (`GGML_TYPE_F64 = 28`).
    F64,
    /// 256-element, ~1.75 bpw (`GGML_TYPE_IQ1_M = 29`).
    IQ1_M,
    /// 16-bit brain floating point (`GGML_TYPE_BF16 = 30`).
    BF16,
    /// Ternary 1-bit packing variant `0` (`GGML_TYPE_TQ1_0 = 34`).
    TQ1_0,
    /// Ternary 2-bit packing variant `0` (`GGML_TYPE_TQ2_0 = 35`).
    TQ2_0,
    /// 32-element, 4-bit microscaling FP (`GGML_TYPE_MXFP4 = 39`).
    MXFP4,
}

impl GgufType {
    /// Parses a `u32` `ggml_type` discriminant into a [`GgufType`].
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if the value does not match
    /// any known `ggml_type`. Reserved or removed discriminants (4, 5, 31–33,
    /// 36–38) also produce this error.
    fn from_u32(value: u32) -> crate::Result<Self> {
        let ty = match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            other => {
                return Err(AnamnesisError::Unsupported {
                    format: "GGUF".into(),
                    detail: format!("unknown ggml_type discriminant {other}"),
                });
            }
        };
        Ok(ty)
    }

    /// Number of elements per storage block.
    ///
    /// Unquantised scalar types return `1`. Legacy quantised types return
    /// `32`. K-quants and most `IQ*`/`TQ*` types return `256`. `IQ4_NL` and
    /// `MXFP4` return `32`.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => 256,
        }
    }

    /// Number of bytes per storage block, or `None` for types whose block
    /// layout is not yet hard-coded in this crate.
    ///
    /// Returns `Some` for the scalar types (`F32`, `F16`, `BF16`, `F64`,
    /// `I8`–`I64`), the legacy block-wise quantised types (`Q4_0`, `Q4_1`,
    /// `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`), the K-quant super-blocks
    /// (`Q2_K`–`Q8_K`), the two non-linear 4-bit `IQ*` variants (`IQ4_NL`
    /// at 18 bytes, `IQ4_XS` at 136 bytes), and the three 2-bit `IQ*`
    /// variants (`IQ2_XXS` at 66 bytes, `IQ2_XS` at 74 bytes, `IQ2_S` at
    /// 82 bytes). Returns `None` for the remaining `IQ*`, `TQ*`, and
    /// `MXFP4` types — those block layouts will be tabulated when
    /// dequantisation support lands in later Phase 4.5 commits.
    // `Q4_0` and `IQ4_NL` happen to both be 18 bytes (same `ggml_half` +
    // 16 nibble-packed bytes), and other pairs share byte counts too; keeping
    // the arms separate documents the distinct block-format semantics instead
    // of collapsing them into pattern lists.
    #[allow(clippy::match_same_arms)]
    #[must_use]
    pub const fn type_size(self) -> Option<usize> {
        match self {
            Self::I8 => Some(1),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::F32 | Self::I32 => Some(4),
            Self::F64 | Self::I64 => Some(8),
            // Legacy block-wise quants (32-element blocks).
            Self::Q4_0 => Some(18),
            Self::Q4_1 => Some(20),
            Self::Q5_0 => Some(22),
            Self::Q5_1 => Some(24),
            Self::Q8_0 => Some(34),
            Self::Q8_1 => Some(36),
            // K-quants (256-element super-blocks).
            Self::Q2_K => Some(84),
            Self::Q3_K => Some(110),
            Self::Q4_K => Some(144),
            Self::Q5_K => Some(176),
            Self::Q6_K => Some(210),
            Self::Q8_K => Some(292),
            // Non-linear 4-bit IQ variants (share the `kvalues_iq4nl` codebook).
            // IQ4_NL: d (f16, 2 B) + qs (4-bit packed, 16 B) = 18 B per 32-element block.
            // IQ4_XS: d (f16, 2 B) + scales_h (u16, 2 B) + scales_l (4 B) + qs (128 B)
            //         = 136 B per 256-element super-block.
            Self::IQ4_NL => Some(18),
            Self::IQ4_XS => Some(136),
            // 2-bit IQ super-quants (256-element super-blocks). All three share
            // the `ksigns_iq2xs` / `kmask_iq2xs` sign tables; the grid tables
            // differ in size (256 / 512 / 1024 entries, each an 8-byte lattice
            // codebook vector).
            // IQ2_XXS: d (f16, 2 B) + qs (u16[32], 64 B)                    = 66 B.
            // IQ2_XS:  d (f16, 2 B) + qs (u16[32], 64 B) + scales (8 B)     = 74 B.
            // IQ2_S:   d (f16, 2 B) + qs (u8[64], 64 B) + qh (8 B) + scales (8 B) = 82 B.
            Self::IQ2_XXS => Some(66),
            Self::IQ2_XS => Some(74),
            Self::IQ2_S => Some(82),
            // Remaining IQ*, TQ*, MXFP4 — byte sizes are defined by ggml struct
            // layouts that this crate has not yet audited. Deferred to later
            // Phase 4.5 commits.
            Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0
            | Self::MXFP4 => None,
        }
    }

    /// Returns `true` if this type is a quantised block format (as opposed
    /// to a scalar float or integer type).
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    /// Dense `0..GGUF_TYPE_COUNT` index used by [`ParsedGguf::inspect`]'s
    /// dtype-dedup bitmap. The value is an internal implementation detail —
    /// callers should never depend on a specific mapping.
    const fn inspect_index(self) -> usize {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 4,
            Self::Q5_1 => 5,
            Self::Q8_0 => 6,
            Self::Q8_1 => 7,
            Self::Q2_K => 8,
            Self::Q3_K => 9,
            Self::Q4_K => 10,
            Self::Q5_K => 11,
            Self::Q6_K => 12,
            Self::Q8_K => 13,
            Self::IQ2_XXS => 14,
            Self::IQ2_XS => 15,
            Self::IQ3_XXS => 16,
            Self::IQ1_S => 17,
            Self::IQ4_NL => 18,
            Self::IQ3_S => 19,
            Self::IQ2_S => 20,
            Self::IQ4_XS => 21,
            Self::I8 => 22,
            Self::I16 => 23,
            Self::I32 => 24,
            Self::I64 => 25,
            Self::F64 => 26,
            Self::IQ1_M => 27,
            Self::BF16 => 28,
            Self::TQ1_0 => 29,
            Self::TQ2_0 => 30,
            Self::MXFP4 => 31,
        }
    }

    /// Computes the byte size of a contiguous tensor of this type containing
    /// `n_elements` elements.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if this type's `type_size` is
    /// not yet known to the parser (see [`type_size`](Self::type_size)).
    ///
    /// Returns [`AnamnesisError::Parse`] if `n_elements` is not a multiple of
    /// the block size, or if the multiplication overflows `u64`.
    pub fn byte_size_for_n_elements(self, n_elements: u64) -> crate::Result<u64> {
        let type_size = self
            .type_size()
            .ok_or_else(|| AnamnesisError::Unsupported {
                format: "GGUF".into(),
                detail: format!("byte size not hard-coded for ggml_type {self}"),
            })?;
        // CAST: usize → u64, `block_size()` returns at most 256, always fits
        #[allow(clippy::as_conversions)]
        let block_size = self.block_size() as u64;
        // CAST: usize → u64, `type_size()` returns at most 292, always fits
        #[allow(clippy::as_conversions)]
        let type_size_u64 = type_size as u64;
        if !n_elements.is_multiple_of(block_size) {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor: element count {n_elements} not a multiple of block size \
                     {block_size} for type {self}"
                ),
            });
        }
        let n_blocks = n_elements / block_size;
        n_blocks
            .checked_mul(type_size_u64)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor: byte-size overflow ({n_blocks} blocks × {type_size_u64} bytes)"
                ),
            })
    }
}

impl fmt::Display for GgufType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::IQ1_M => "IQ1_M",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// GgufMetadataValue
// ---------------------------------------------------------------------------

/// A value stored in the `GGUF` metadata key-value table.
///
/// Mirrors the 13 `gguf_metadata_value_type` variants defined by the spec.
/// [`Self::Array`] is a boxed [`GgufMetadataArray`] — the array's elements
/// are stored natively for their type (e.g. `Vec<f32>`) rather than as a
/// `Vec<GgufMetadataValue>`. This eliminates the ~8× enum-discriminant
/// bloat on homogeneous numeric arrays and, as a side effect, shrinks
/// `GgufMetadataValue` itself from 32 bytes to 24 bytes because the
/// largest variant is now `String` rather than the old `Vec<Self>`.
///
/// Arrays may nest (e.g. a tokenizer merges list is an array of arrays of
/// strings); the parser refuses to recurse beyond four levels deep.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GgufMetadataValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer (stored little-endian in the file).
    U16(u16),
    /// Signed 16-bit integer (stored little-endian).
    I16(i16),
    /// Unsigned 32-bit integer (stored little-endian).
    U32(u32),
    /// Signed 32-bit integer (stored little-endian).
    I32(i32),
    /// 32-bit IEEE 754 single-precision (stored little-endian).
    F32(f32),
    /// Boolean — encoded in the file as a single byte (0 or 1).
    Bool(bool),
    /// UTF-8 string — encoded as `u64` length followed by raw bytes.
    String(String),
    /// Homogeneous array of a single inner type, boxed to keep
    /// `GgufMetadataValue` small (24 bytes on 64-bit).
    Array(Box<GgufMetadataArray>),
    /// Unsigned 64-bit integer (stored little-endian).
    U64(u64),
    /// Signed 64-bit integer (stored little-endian).
    I64(i64),
    /// 64-bit IEEE 754 double-precision (stored little-endian).
    F64(f64),
}

/// Homogeneous `GGUF` metadata array, stored natively for its element type.
///
/// The parser dispatches on the array's `inner_type` when reading and
/// builds a correctly-typed `Vec<T>` directly from the byte stream. For a
/// 16 M-element `f32` array this consumes ~64 MB of heap instead of the
/// ~488 MB a `Vec<GgufMetadataValue>` would require.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GgufMetadataArray {
    /// Array of unsigned 8-bit integers.
    U8(Vec<u8>),
    /// Array of signed 8-bit integers.
    I8(Vec<i8>),
    /// Array of unsigned 16-bit integers.
    U16(Vec<u16>),
    /// Array of signed 16-bit integers.
    I16(Vec<i16>),
    /// Array of unsigned 32-bit integers.
    U32(Vec<u32>),
    /// Array of signed 32-bit integers.
    I32(Vec<i32>),
    /// Array of 32-bit IEEE 754 floats.
    F32(Vec<f32>),
    /// Array of booleans.
    Bool(Vec<bool>),
    /// Array of UTF-8 strings.
    String(Vec<String>),
    /// Array of (typed) sub-arrays. Each sub-array self-describes its own
    /// inner type, so different elements may have different `GgufMetadataArray`
    /// variants.
    Array(Vec<GgufMetadataArray>),
    /// Array of unsigned 64-bit integers.
    U64(Vec<u64>),
    /// Array of signed 64-bit integers.
    I64(Vec<i64>),
    /// Array of 64-bit IEEE 754 floats.
    F64(Vec<f64>),
}

// Compile-time verification of the size invariants the parser relies on:
//
// * `GgufMetadataValue` stays at 24 bytes because boxing the `Array`
//   variant makes `String` (24 B) the largest payload instead of the old
//   unboxed `Vec<GgufMetadataValue>`. That 25 % shrink applies to every
//   metadata value in the `HashMap`, not just arrays.
//
// * `GgufMetadataArray` stays at 32 bytes because every `Vec<T>` variant
//   is 24 B plus an 8-byte (aligned) discriminant.
//
// If either number drifts, the DoS-guard memory math in the parser's
// module comments is stale and needs to be re-audited.
const _: () = {
    assert!(
        std::mem::size_of::<GgufMetadataValue>() == 24,
        "GgufMetadataValue must be 24 bytes (Array must be Box<GgufMetadataArray>)"
    );
    assert!(
        std::mem::size_of::<GgufMetadataArray>() == 32,
        "GgufMetadataArray must be 32 bytes (largest variant Vec<T> = 24 + 8-byte tag)"
    );
};

impl GgufMetadataValue {
    /// Returns the inner string if the value is `String`, otherwise `None`.
    #[must_use]
    pub fn as_string(&self) -> Option<&str> {
        if let Self::String(s) = self {
            // BORROW: explicit `.as_str()` on the owned `String` instead of
            // relying on `Deref<Target = str>` coercion through `Some(...)`
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Returns the inner `u32` if the value is `U32`, otherwise `None`.
    #[must_use]
    pub const fn as_u32(&self) -> Option<u32> {
        if let Self::U32(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the inner `u64` if the value is `U64`, otherwise `None`.
    #[must_use]
    pub const fn as_u64(&self) -> Option<u64> {
        if let Self::U64(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the inner `bool` if the value is `Bool`, otherwise `None`.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Returns the inner typed array if the value is `Array`, otherwise
    /// `None`.
    #[must_use]
    pub fn as_array(&self) -> Option<&GgufMetadataArray> {
        if let Self::Array(v) = self {
            Some(v.as_ref())
        } else {
            None
        }
    }
}

impl GgufMetadataArray {
    /// Number of elements in the array, regardless of inner type.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Array(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Returns `true` if the array contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the inner slice if this is a `U8` array, otherwise `None`.
    #[must_use]
    pub fn as_u8_slice(&self) -> Option<&[u8]> {
        if let Self::U8(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `I8` array, otherwise `None`.
    #[must_use]
    pub fn as_i8_slice(&self) -> Option<&[i8]> {
        if let Self::I8(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `U16` array, otherwise `None`.
    #[must_use]
    pub fn as_u16_slice(&self) -> Option<&[u16]> {
        if let Self::U16(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `I16` array, otherwise `None`.
    #[must_use]
    pub fn as_i16_slice(&self) -> Option<&[i16]> {
        if let Self::I16(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `U32` array, otherwise `None`.
    #[must_use]
    pub fn as_u32_slice(&self) -> Option<&[u32]> {
        if let Self::U32(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `I32` array, otherwise `None`.
    #[must_use]
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if let Self::I32(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `F32` array, otherwise `None`.
    #[must_use]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if let Self::F32(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `Bool` array, otherwise `None`.
    #[must_use]
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        if let Self::Bool(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `String` array, otherwise `None`.
    #[must_use]
    pub fn as_string_slice(&self) -> Option<&[String]> {
        if let Self::String(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `Array` of sub-arrays,
    /// otherwise `None`.
    #[must_use]
    pub fn as_nested_slice(&self) -> Option<&[GgufMetadataArray]> {
        if let Self::Array(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is a `U64` array, otherwise `None`.
    #[must_use]
    pub fn as_u64_slice(&self) -> Option<&[u64]> {
        if let Self::U64(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `I64` array, otherwise `None`.
    #[must_use]
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        if let Self::I64(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }

    /// Returns the inner slice if this is an `F64` array, otherwise `None`.
    #[must_use]
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        if let Self::F64(v) = self {
            // BORROW: explicit `.as_slice()` avoids relying on `Deref` coercion
            Some(v.as_slice())
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// GgufTensorInfo
// ---------------------------------------------------------------------------

/// Metadata for a single tensor in a `GGUF` file.
///
/// Produced during [`parse_gguf`]. `data_offset` is the **absolute** byte
/// offset inside the memory-mapped file (not the relative offset stored in
/// the `gguf_tensor_info_t` on disk — the parser has already added the
/// tensor-data section start).
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name (e.g., `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Tensor dimensions, **most-significant-first** (same order as
    /// `ggml_tensor::ne`). A row-major `[rows, cols]` matrix is stored as
    /// `shape = [cols, rows]` — consumers that expect NumPy-style ordering
    /// must reverse this before use.
    pub shape: Vec<usize>,
    /// Element / block data type.
    pub dtype: GgufType,
    /// Absolute byte offset of the tensor data inside the memory-mapped
    /// file. Equal to `tensor_data_section_start + relative_offset` where
    /// `relative_offset` is the `u64` stored in the file.
    pub data_offset: u64,
    /// Total byte length of the tensor data, or `None` when
    /// [`GgufType::type_size`] is not yet tabulated for this dtype.
    pub byte_len: Option<u64>,
}

// ---------------------------------------------------------------------------
// GgufTensor
// ---------------------------------------------------------------------------

/// A tensor view into a parsed `GGUF` file.
///
/// Returned by [`ParsedGguf::tensors`]. `name` and `shape` borrow directly
/// from the owning [`ParsedGguf`] — iterating all tensors allocates
/// nothing. `data` is `Cow::Borrowed` with a zero-copy slice of the
/// memory-mapped file for every supported dtype; `Cow::Owned` is reserved
/// for future big-endian support.
#[derive(Debug, Clone)]
pub struct GgufTensor<'a> {
    /// Tensor name (e.g., `"blk.0.attn_q.weight"`).
    pub name: &'a str,
    /// Tensor dimensions, most-significant-first (see
    /// [`GgufTensorInfo::shape`]).
    pub shape: &'a [usize],
    /// Element / block data type.
    pub dtype: GgufType,
    /// Raw bytes in on-disk (little-endian) order. Length equals
    /// `byte_size_for_n_elements(product(shape))`.
    pub data: Cow<'a, [u8]>,
}

// ---------------------------------------------------------------------------
// GgufInspectInfo
// ---------------------------------------------------------------------------

/// Summary information about a parsed `GGUF` file.
///
/// Produced by [`ParsedGguf::inspect`]. No I/O — derived from metadata.
#[derive(Debug, Clone)]
#[must_use]
pub struct GgufInspectInfo {
    /// `GGUF` version read from the header (currently 2 or 3).
    pub version: u32,
    /// Value of the `general.architecture` metadata key, if present.
    pub architecture: Option<String>,
    /// Number of tensors in the file.
    pub tensor_count: usize,
    /// Total byte length of all tensor data whose dtype has a known
    /// `type_size`. Tensors with an unknown dtype are excluded.
    pub total_bytes: u64,
    /// Number of tensors whose dtype has no known byte size (excluded from
    /// `total_bytes`).
    pub unknown_size_tensors: usize,
    /// Distinct dtypes found, in order of first occurrence.
    pub dtypes: Vec<GgufType>,
    /// Effective alignment read from `general.alignment`, or the default of
    /// 32 bytes if the metadata key is absent.
    pub alignment: u32,
}

impl fmt::Display for GgufInspectInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // All labels land at column 13 so successive lines line up visually.
        // `"Format:      "` is 13 chars, `"Arch:        "` is 13 chars, etc.
        write!(f, "Format:      GGUF v{}", self.version)?;
        if let Some(arch) = self.architecture.as_deref() {
            write!(f, "\nArch:        {arch}")?;
        }
        write!(f, "\nTensors:     {}", self.tensor_count)?;
        write!(
            f,
            "\nTotal size:  {}",
            crate::inspect::format_bytes(self.total_bytes)
        )?;
        if self.unknown_size_tensors > 0 {
            write!(
                f,
                " (+{} tensors with dtype of unknown size)",
                self.unknown_size_tensors
            )?;
        }
        let dtype_list: String = self
            .dtypes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "\nDtypes:      {dtype_list}")?;
        write!(f, "\nAlignment:   {} bytes", self.alignment)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ParsedGguf
// ---------------------------------------------------------------------------

/// A parsed `GGUF` file — owns the memory-mapped data and provides
/// zero-copy tensor views.
///
/// Created by [`parse_gguf`]. Call [`tensors`](Self::tensors) to obtain
/// [`GgufTensor`] views borrowed directly from the mapped file.
#[derive(Debug)]
pub struct ParsedGguf {
    /// Memory-mapped file.
    mmap: memmap2::Mmap,
    /// `GGUF` version read from the header.
    version: u32,
    /// Effective tensor-data alignment in bytes.
    alignment: u32,
    /// Metadata key-value pairs in insertion order (backed by `HashMap`).
    metadata: HashMap<String, GgufMetadataValue>,
    /// Per-tensor metadata with absolute byte offsets.
    tensor_infos: Vec<GgufTensorInfo>,
}

impl ParsedGguf {
    /// Returns the `GGUF` format version read from the header.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the effective tensor-data alignment.
    #[must_use]
    pub const fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Returns the number of tensors in the file.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tensor_infos.len()
    }

    /// Returns `true` if the file contains no tensors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tensor_infos.is_empty()
    }

    /// Returns the parsed metadata key-value table.
    #[must_use]
    pub const fn metadata(&self) -> &HashMap<String, GgufMetadataValue> {
        &self.metadata
    }

    /// Returns lightweight per-tensor metadata without slicing the mmap.
    ///
    /// Use this for display paths and type inventories where the raw bytes
    /// are not needed.
    #[must_use]
    pub fn tensor_info(&self) -> &[GgufTensorInfo] {
        &self.tensor_infos
    }

    /// Returns an iterator of tensor views borrowing directly from the
    /// mmap and from `self`.
    ///
    /// For every tensor whose dtype has a known byte size (scalars, legacy
    /// `Q*_0`/`Q*_1`, K-quants), `data` is a zero-copy `Cow::Borrowed` slice
    /// into the mapped file. `name` and `shape` are `&'a str` / `&'a [usize]`
    /// borrowed from the internal `Vec<GgufTensorInfo>` — **no per-tensor
    /// heap allocation**. Tensors whose dtype is not yet tabulated
    /// (`IQ*`, `TQ*`, `MXFP4`) are skipped silently; they are still listed
    /// in [`tensor_info`](Self::tensor_info).
    ///
    /// Callers that want random access can materialise the iterator with
    /// `.collect::<Vec<_>>()`.
    ///
    /// # Memory
    ///
    /// Zero heap allocation per invocation. `GgufTensor::data` is
    /// `Cow::Borrowed` into the mmap, and `GgufTensor::{name, shape}` are
    /// slice references into `self`. Peak memory is just the mmap itself
    /// (unchanged across `tensors()` calls) plus whatever the caller
    /// chooses to collect.
    pub fn tensors(&self) -> impl Iterator<Item = GgufTensor<'_>> + '_ {
        self.tensor_infos.iter().filter_map(|info| {
            let byte_len_u64 = info.byte_len?;
            // `usize::try_from` and `checked_add` are defensive:
            // `data_offset` and `byte_len` were pre-validated against
            // `raw.len()` in `parse_gguf`, so on 64-bit targets every
            // `?` below is dead. On 32-bit targets with a hypothetical
            // >4 GB mmap (which `memmap2` cannot produce) the
            // fallthrough silently skips the tensor.
            let start = usize::try_from(info.data_offset).ok()?;
            let byte_len = usize::try_from(byte_len_u64).ok()?;
            let end = start.checked_add(byte_len)?;
            let slice = self.mmap.get(start..end)?;
            Some(GgufTensor {
                name: info.name.as_str(),
                shape: info.shape.as_slice(),
                dtype: info.dtype,
                data: Cow::Borrowed(slice),
            })
        })
    }

    /// Returns inspection info derived from the parsed metadata. No I/O.
    pub fn inspect(&self) -> GgufInspectInfo {
        let mut total_bytes: u64 = 0;
        let mut unknown_size_tensors: usize = 0;
        // O(1) per-tensor dtype dedup via a fixed-size bitmap keyed on the
        // dense `GgufType::inspect_index` — drops the hot loop from
        // O(n × d) to O(n). `dtypes` still records first-occurrence order.
        let mut seen = [false; GGUF_TYPE_COUNT];
        let mut dtypes: Vec<GgufType> = Vec::new();
        for info in &self.tensor_infos {
            if let Some(byte_len) = info.byte_len {
                total_bytes = total_bytes.saturating_add(byte_len);
            } else {
                unknown_size_tensors = unknown_size_tensors.saturating_add(1);
            }
            let idx = info.dtype.inspect_index();
            // INDEX: `inspect_index` is defined to return a value in
            // `0..GGUF_TYPE_COUNT`, matching the bitmap's length exactly
            #[allow(clippy::indexing_slicing)]
            if !seen[idx] {
                #[allow(clippy::indexing_slicing)]
                {
                    seen[idx] = true;
                }
                dtypes.push(info.dtype);
            }
        }
        let architecture = self
            .metadata
            .get("general.architecture")
            .and_then(GgufMetadataValue::as_string)
            // BORROW: `.to_owned()` converts `&str` to an owned `String`
            // that outlives the `&self` borrow of the metadata map.
            .map(str::to_owned);
        GgufInspectInfo {
            version: self.version,
            architecture,
            tensor_count: self.tensor_infos.len(),
            total_bytes,
            unknown_size_tensors,
            dtypes,
            alignment: self.alignment,
        }
    }

    /// Dequantises a single tensor from the memory-mapped file to `BF16`
    /// bytes.
    ///
    /// Convenience method that slices the internal mmap using `info`'s
    /// offset and byte length, infers the element count from `info.shape`,
    /// and delegates to
    /// [`dequantize_gguf_to_bf16`](crate::remember::gguf::dequantize_gguf_to_bf16).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] if `info.byte_len` is `None`
    /// (the dtype's block layout is not yet tabulated — the remaining
    /// `IQ3_*` / `IQ1_*` / `TQ*` / `MXFP4` types), or if the dtype is a
    /// recognised but not-yet-implemented quantisation type.
    ///
    /// Returns [`AnamnesisError::Parse`] if the element count overflows
    /// `usize`, the mmap slice is out of bounds, or the underlying
    /// dequantisation kernel encounters a data/shape mismatch.
    ///
    /// # Memory
    ///
    /// Allocates a single `Vec<u8>` of length `n_elements * 2` for the
    /// `BF16` output. The input data is read directly from the mmap — no
    /// input copy. Peak heap is the output buffer (O(`n_elements`)).
    pub fn dequantize_tensor(&self, info: &GgufTensorInfo) -> crate::Result<Vec<u8>> {
        let byte_len_u64 = info.byte_len.ok_or_else(|| AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!(
                "byte size not known for dtype {} — dequantisation not yet supported",
                info.dtype
            ),
        })?;
        let start = usize::try_from(info.data_offset).map_err(|_| AnamnesisError::Parse {
            reason: format!(
                "tensor `{}`: data_offset {} exceeds usize",
                info.name, info.data_offset
            ),
        })?;
        let byte_len = usize::try_from(byte_len_u64).map_err(|_| AnamnesisError::Parse {
            reason: format!(
                "tensor `{}`: byte_len {byte_len_u64} exceeds usize",
                info.name
            ),
        })?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "tensor `{}`: data_offset + byte_len overflows usize",
                    info.name
                ),
            })?;
        let data = self
            .mmap
            .get(start..end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "tensor `{}`: byte range {start}..{end} exceeds mmap length {}",
                    info.name,
                    self.mmap.len()
                ),
            })?;
        let n_elements: usize = info
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("tensor `{}`: element count overflows usize", info.name),
            })?;
        crate::remember::gguf::dequantize_gguf_to_bf16(data, info.dtype, n_elements)
    }
}

// ---------------------------------------------------------------------------
// Cursor — minimal little-endian reader over a byte slice
// ---------------------------------------------------------------------------

/// Minimal forward-only cursor over a byte slice, with bounds-checked
/// little-endian primitive readers.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Reads exactly `n` bytes, advancing the cursor.
    fn read_bytes(&mut self, n: usize) -> crate::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF: cursor overflow at pos {} + {n}", self.pos),
            })?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "GGUF: unexpected EOF at pos {} (wanted {n} bytes, have {})",
                    self.pos,
                    self.buf.len().saturating_sub(self.pos)
                ),
            })?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> crate::Result<u8> {
        let bytes = self.read_bytes(1)?;
        // INDEX: read_bytes(1) returned a slice of length exactly 1
        #[allow(clippy::indexing_slicing)]
        Ok(bytes[0])
    }

    fn read_i8(&mut self) -> crate::Result<i8> {
        // CAST: u8 → i8, reinterpret bit pattern — signed/unsigned wrap is intended
        #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
        Ok(self.read_u8()? as i8)
    }

    fn read_u16_le(&mut self) -> crate::Result<u16> {
        let bytes = self.read_bytes(2)?;
        let mut arr = [0u8; 2];
        arr.copy_from_slice(bytes);
        Ok(u16::from_le_bytes(arr))
    }

    fn read_i16_le(&mut self) -> crate::Result<i16> {
        let bytes = self.read_bytes(2)?;
        let mut arr = [0u8; 2];
        arr.copy_from_slice(bytes);
        Ok(i16::from_le_bytes(arr))
    }

    fn read_u32_le(&mut self) -> crate::Result<u32> {
        let bytes = self.read_bytes(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(arr))
    }

    fn read_i32_le(&mut self) -> crate::Result<i32> {
        let bytes = self.read_bytes(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        Ok(i32::from_le_bytes(arr))
    }

    fn read_u64_le(&mut self) -> crate::Result<u64> {
        let bytes = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(arr))
    }

    fn read_i64_le(&mut self) -> crate::Result<i64> {
        let bytes = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(i64::from_le_bytes(arr))
    }

    fn read_f32_le(&mut self) -> crate::Result<f32> {
        let bytes = self.read_bytes(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        Ok(f32::from_le_bytes(arr))
    }

    fn read_f64_le(&mut self) -> crate::Result<f64> {
        let bytes = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(f64::from_le_bytes(arr))
    }

    fn read_bool(&mut self) -> crate::Result<bool> {
        let b = self.read_u8()?;
        match b {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(AnamnesisError::Parse {
                reason: format!("GGUF metadata: invalid bool byte {other} (expected 0 or 1)"),
            }),
        }
    }

    /// Reads a `gguf_string_t` (`u64` length prefix + raw bytes, interpreted
    /// as UTF-8).
    ///
    /// Validation runs on the borrowed mmap slice **before** any allocation,
    /// so a rejected non-UTF-8 string of up to `max_len` bytes costs zero
    /// heap allocation on the error path.
    fn read_string(&mut self, max_len: u64) -> crate::Result<String> {
        let len = self.read_u64_le()?;
        if len > max_len {
            return Err(AnamnesisError::Parse {
                reason: format!("GGUF: string length {len} exceeds cap {max_len}"),
            });
        }
        let len_usz = usize::try_from(len).map_err(|_| AnamnesisError::Parse {
            reason: format!("GGUF: string length {len} overflows usize"),
        })?;
        let bytes = self.read_bytes(len_usz)?;
        let valid = std::str::from_utf8(bytes).map_err(|e| AnamnesisError::Parse {
            reason: format!("GGUF: string is not valid UTF-8: {e}"),
        })?;
        // BORROW: `.to_owned()` copies the validated borrowed slice into an
        // owned `String`; the allocation only happens on the success path.
        Ok(valid.to_owned())
    }
}

// ---------------------------------------------------------------------------
// Metadata value reader
// ---------------------------------------------------------------------------

/// Reads a single metadata value of the given `value_type` discriminant.
///
/// For `ARRAY` (`value_type` 9), dispatches into [`read_typed_array`]
/// which builds a natively-typed `Vec<T>` instead of the old 8×-bloated
/// `Vec<GgufMetadataValue>`.
fn read_metadata_value(
    cursor: &mut Cursor<'_>,
    value_type: u32,
) -> crate::Result<GgufMetadataValue> {
    match value_type {
        0 => Ok(GgufMetadataValue::U8(cursor.read_u8()?)),
        1 => Ok(GgufMetadataValue::I8(cursor.read_i8()?)),
        2 => Ok(GgufMetadataValue::U16(cursor.read_u16_le()?)),
        3 => Ok(GgufMetadataValue::I16(cursor.read_i16_le()?)),
        4 => Ok(GgufMetadataValue::U32(cursor.read_u32_le()?)),
        5 => Ok(GgufMetadataValue::I32(cursor.read_i32_le()?)),
        6 => Ok(GgufMetadataValue::F32(cursor.read_f32_le()?)),
        7 => Ok(GgufMetadataValue::Bool(cursor.read_bool()?)),
        8 => Ok(GgufMetadataValue::String(
            cursor.read_string(MAX_STRING_LEN)?,
        )),
        9 => {
            let inner_type = cursor.read_u32_le()?;
            let len = read_array_len(cursor)?;
            // Initial depth is 0: this call builds the outer array (nesting
            // level 0). Recursive calls increment the depth so that the
            // `depth >= MAX_ARRAY_DEPTH` check inside `read_typed_array`
            // supports `MAX_ARRAY_DEPTH` total nested levels (depths
            // `0..MAX_ARRAY_DEPTH`).
            let array = read_typed_array(cursor, inner_type, len, 0)?;
            Ok(GgufMetadataValue::Array(Box::new(array)))
        }
        10 => Ok(GgufMetadataValue::U64(cursor.read_u64_le()?)),
        11 => Ok(GgufMetadataValue::I64(cursor.read_i64_le()?)),
        12 => Ok(GgufMetadataValue::F64(cursor.read_f64_le()?)),
        other => Err(AnamnesisError::Parse {
            reason: format!("GGUF metadata: unknown value type {other}"),
        }),
    }
}

/// Reads and validates a `GGUF` array length prefix (`u64` from the file).
///
/// Enforces `MAX_ARRAY_LEN` and converts to `usize`.
fn read_array_len(cursor: &mut Cursor<'_>) -> crate::Result<usize> {
    let len = cursor.read_u64_le()?;
    if len > MAX_ARRAY_LEN {
        return Err(AnamnesisError::Parse {
            reason: format!("GGUF metadata: array length {len} exceeds cap {MAX_ARRAY_LEN}"),
        });
    }
    usize::try_from(len).map_err(|_| AnamnesisError::Parse {
        reason: format!("GGUF metadata: array length {len} overflows usize"),
    })
}

/// Reads `len` homogeneous elements of type `inner_type` into a typed
/// [`GgufMetadataArray`].
///
/// `depth` is the current array-nesting level for adversarial-depth
/// protection: `depth = 0` is the outermost array built directly from a
/// metadata key-value pair, and nested arrays increment it. Recursion is
/// rejected when `depth >= MAX_ARRAY_DEPTH`, so the parser supports
/// exactly `MAX_ARRAY_DEPTH` total levels of nesting (depths
/// `0..MAX_ARRAY_DEPTH`).
///
/// Every typed `Vec::with_capacity` call is clamped to
/// `PREALLOC_SOFT_CAP` so an adversarial 20-byte array header claiming
/// `MAX_ARRAY_LEN` elements cannot force ~488 MB of eager allocation;
/// the vector grows geometrically from there.
fn read_typed_array(
    cursor: &mut Cursor<'_>,
    inner_type: u32,
    len: usize,
    depth: u32,
) -> crate::Result<GgufMetadataArray> {
    let cap = len.min(PREALLOC_SOFT_CAP);
    match inner_type {
        0 => {
            let mut v: Vec<u8> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_u8()?);
            }
            Ok(GgufMetadataArray::U8(v))
        }
        1 => {
            let mut v: Vec<i8> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_i8()?);
            }
            Ok(GgufMetadataArray::I8(v))
        }
        2 => {
            let mut v: Vec<u16> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_u16_le()?);
            }
            Ok(GgufMetadataArray::U16(v))
        }
        3 => {
            let mut v: Vec<i16> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_i16_le()?);
            }
            Ok(GgufMetadataArray::I16(v))
        }
        4 => {
            let mut v: Vec<u32> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_u32_le()?);
            }
            Ok(GgufMetadataArray::U32(v))
        }
        5 => {
            let mut v: Vec<i32> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_i32_le()?);
            }
            Ok(GgufMetadataArray::I32(v))
        }
        6 => {
            let mut v: Vec<f32> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_f32_le()?);
            }
            Ok(GgufMetadataArray::F32(v))
        }
        7 => {
            let mut v: Vec<bool> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_bool()?);
            }
            Ok(GgufMetadataArray::Bool(v))
        }
        8 => {
            let mut v: Vec<String> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_string(MAX_STRING_LEN)?);
            }
            Ok(GgufMetadataArray::String(v))
        }
        9 => {
            // Nested array: each element is itself a typed array. Check
            // the recursion depth before reading anything so we fail fast
            // on adversarial nesting.
            if depth >= MAX_ARRAY_DEPTH {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "GGUF metadata: array nesting exceeds depth cap {MAX_ARRAY_DEPTH}"
                    ),
                });
            }
            let mut v: Vec<GgufMetadataArray> = Vec::with_capacity(cap);
            for _ in 0..len {
                let sub_inner = cursor.read_u32_le()?;
                let sub_len = read_array_len(cursor)?;
                v.push(read_typed_array(cursor, sub_inner, sub_len, depth + 1)?);
            }
            Ok(GgufMetadataArray::Array(v))
        }
        10 => {
            let mut v: Vec<u64> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_u64_le()?);
            }
            Ok(GgufMetadataArray::U64(v))
        }
        11 => {
            let mut v: Vec<i64> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_i64_le()?);
            }
            Ok(GgufMetadataArray::I64(v))
        }
        12 => {
            let mut v: Vec<f64> = Vec::with_capacity(cap);
            for _ in 0..len {
                v.push(cursor.read_f64_le()?);
            }
            Ok(GgufMetadataArray::F64(v))
        }
        other => Err(AnamnesisError::Parse {
            reason: format!("GGUF metadata: unknown array inner type {other}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parses a `GGUF` file and returns a [`ParsedGguf`] handle owning the
/// memory-mapped data.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the file cannot be opened or mapped.
///
/// Returns [`AnamnesisError::Parse`] if the magic bytes are missing, the
/// header fields are truncated or out of range, the metadata table contains
/// an invalid value type, a tensor info entry is malformed, or a tensor's
/// resolved byte range falls outside the mapped file.
///
/// Returns [`AnamnesisError::Unsupported`] for `GGUF` v1 files (which use
/// `u32` string lengths instead of `u64`), big-endian `GGUF` files (v3+
/// feature, not yet implemented), legacy pre-`GGUF` formats (`GGML`, `GGJT`,
/// `GGMF`), and tensor dtypes whose `ggml_type` discriminant is not
/// recognised.
///
/// # Memory
///
/// Memory-maps the file with `memmap2::MmapOptions::populate()` to prefault
/// pages. Tensor data is **not** copied during parsing —
/// [`ParsedGguf::tensors`] returns `Cow::Borrowed` slices of the mmap.
/// Peak heap is `O(n_tensors + n_metadata_kv)` (a few dozen bytes per
/// tensor info record plus the metadata map). The mmap is released when
/// the returned `ParsedGguf` is dropped.
#[allow(unsafe_code)]
pub fn parse_gguf(path: impl AsRef<Path>) -> crate::Result<ParsedGguf> {
    let file = std::fs::File::open(path.as_ref())?;
    // SAFETY: memmap2::Mmap requires `unsafe` because the OS could modify
    // the mapped region if another process writes to the underlying file
    // concurrently. Tensor files are read-only artefacts in practice — the
    // same assumption every other anamnesis format parser (pth, safetensors
    // via the `safetensors` crate) relies on.
    let raw =
        unsafe { memmap2::MmapOptions::new().populate().map(&file) }.map_err(AnamnesisError::Io)?;

    // Magic check. Detect legacy GGML/GGJT/GGMF formats and byte-swapped
    // big-endian GGUF files for clearer error messages.
    let magic_bytes = raw.get(..4).ok_or_else(|| AnamnesisError::Parse {
        reason: "GGUF: file shorter than 4 bytes (no magic)".into(),
    })?;
    if magic_bytes != GGUF_MAGIC {
        let mut arr = [0u8; 4];
        arr.copy_from_slice(magic_bytes);
        let as_le = u32::from_le_bytes(arr);
        if as_le == GGUF_MAGIC_BE_U32 {
            return Err(AnamnesisError::Unsupported {
                format: "GGUF".into(),
                detail: "big-endian GGUF files are not yet supported".into(),
            });
        }
        let legacy_name: Option<&'static str> = match magic_bytes {
            b"GGML" => Some("GGML"),
            b"GGJT" => Some("GGJT"),
            b"GGMF" => Some("GGMF"),
            _ => None,
        };
        if let Some(name) = legacy_name {
            return Err(AnamnesisError::Unsupported {
                format: "GGUF".into(),
                detail: format!(
                    "legacy `{name}` format predates GGUF; re-convert with `llama.cpp` to GGUF"
                ),
            });
        }
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF: invalid magic (expected `GGUF`/{GGUF_MAGIC_LE_U32:#010x}, got {as_le:#010x})"
            ),
        });
    }

    // Read the fixed-size header. Cursor starts after the magic.
    let mut cursor = Cursor::new(&raw);
    cursor.pos = 4;
    let version = cursor.read_u32_le()?;
    if version == 1 {
        return Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: "GGUF v1 uses u32 string/array lengths and is not supported; \
                     re-save with a modern `llama.cpp` to produce v2 or v3"
                .into(),
        });
    }
    if version != 2 && version != 3 {
        return Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("unsupported GGUF version {version} (expected 2 or 3)"),
        });
    }
    let tensor_count = cursor.read_u64_le()?;
    let kv_count = cursor.read_u64_le()?;
    if tensor_count > MAX_TENSOR_COUNT {
        return Err(AnamnesisError::Parse {
            reason: format!("GGUF: tensor count {tensor_count} exceeds cap {MAX_TENSOR_COUNT}"),
        });
    }
    if kv_count > MAX_KV_COUNT {
        return Err(AnamnesisError::Parse {
            reason: format!("GGUF: metadata kv count {kv_count} exceeds cap {MAX_KV_COUNT}"),
        });
    }
    let tensor_count_usz = usize::try_from(tensor_count).map_err(|_| AnamnesisError::Parse {
        reason: format!("GGUF: tensor count {tensor_count} overflows usize"),
    })?;
    let kv_count_usz = usize::try_from(kv_count).map_err(|_| AnamnesisError::Parse {
        reason: format!("GGUF: metadata kv count {kv_count} overflows usize"),
    })?;

    // Read metadata key-value pairs. Cap the pre-allocation at
    // `PREALLOC_SOFT_CAP` so an adversarial header claiming a million
    // entries cannot force ~114 MB of eager heap allocation; the HashMap
    // grows geometrically from there on legitimate large inputs.
    let mut metadata: HashMap<String, GgufMetadataValue> =
        HashMap::with_capacity(kv_count_usz.min(PREALLOC_SOFT_CAP));
    for _ in 0..kv_count_usz {
        let key = cursor.read_string(u64::from(u16::MAX))?;
        let value_type = cursor.read_u32_le()?;
        let value = read_metadata_value(&mut cursor, value_type)?;
        metadata.insert(key, value);
    }

    // Resolve alignment (honour `general.alignment` if present).
    let alignment = match metadata.get("general.alignment") {
        Some(GgufMetadataValue::U32(v)) if *v != 0 => *v,
        Some(GgufMetadataValue::U32(_)) => {
            return Err(AnamnesisError::Parse {
                reason: "GGUF: general.alignment is zero".into(),
            });
        }
        Some(other) => {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF: general.alignment has wrong type (expected UINT32, got {})",
                    metadata_type_name(other)
                ),
            });
        }
        None => DEFAULT_ALIGNMENT,
    };
    let alignment_u64 = u64::from(alignment);

    // Read tensor info entries directly into `tensor_infos`. Each entry is:
    //   name (gguf_string_t) | n_dimensions (u32) | dimensions[n_dims] (u64)
    //   | type (u32) | offset (u64)
    // Offsets are relative to the start of the tensor_data section, which
    // we don't know until after the whole table has been read. Store the
    // raw relative offset in `data_offset` for now; a patch sweep below
    // rewrites it to the absolute offset once `data_section_start` is known.
    //
    // Cap the pre-allocation at `PREALLOC_SOFT_CAP` for the same reason as
    // the metadata map above — trust-the-header DoS guard.
    let mut tensor_infos: Vec<GgufTensorInfo> =
        Vec::with_capacity(tensor_count_usz.min(PREALLOC_SOFT_CAP));
    for _ in 0..tensor_count_usz {
        tensor_infos.push(read_tensor_info_relative(&mut cursor)?);
    }

    // CAST: usize → u64, `cursor.pos` is bounded by `raw.len()`, which
    // always fits in `u64` on every supported target (64-bit and 32-bit).
    #[allow(clippy::as_conversions)]
    let tensor_info_end = cursor.pos as u64;
    // CAST: usize → u64, same rationale as `tensor_info_end`
    #[allow(clippy::as_conversions)]
    let file_len_u64 = raw.len() as u64;

    // The tensor_data section begins at the next alignment boundary after
    // the tensor-info table. A file with zero tensors has no data section at
    // all, so skip the alignment and bounds check entirely — a 24-byte
    // header-only GGUF is legitimately well-formed.
    let data_section_start = if tensor_infos.is_empty() {
        tensor_info_end
    } else {
        let start = align_up(tensor_info_end, alignment_u64)?;
        if start > file_len_u64 {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF: tensor data section start {start} exceeds file size {file_len_u64}"
                ),
            });
        }
        start
    };

    // Patch sweep: rewrite the temporary relative offsets into absolute
    // offsets and run the bounds checks that couldn't happen at read time.
    for info in &mut tensor_infos {
        let relative_offset = info.data_offset;
        // The GGUF spec mandates that every tensor's offset field is a
        // multiple of `general.alignment`. `data_section_start` is itself
        // aligned (via `align_up` above), so checking the relative offset
        // is equivalent to checking the absolute offset and catches
        // adversarial files that would hand out unaligned byte slices to
        // SIMD dequant kernels downstream.
        if relative_offset % alignment_u64 != 0 {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor `{}`: relative offset {relative_offset} is not a multiple of alignment {alignment_u64}",
                    info.name
                ),
            });
        }
        let absolute = data_section_start
            .checked_add(relative_offset)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor `{}`: absolute offset overflow ({} + {})",
                    info.name, data_section_start, relative_offset
                ),
            })?;
        // Sanity-check that the tensor at least starts inside the file,
        // even when its dtype's `type_size` is not yet known. This catches
        // adversarial `IQ*`/`TQ*`/`MXFP4` files that would otherwise hand
        // a nonsense `data_offset` out through `tensor_info()`.
        if absolute > file_len_u64 {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor `{}`: data_offset {absolute} exceeds file size {file_len_u64}",
                    info.name
                ),
            });
        }
        if let Some(len) = info.byte_len {
            let end = absolute
                .checked_add(len)
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!("GGUF tensor `{}`: end offset overflow", info.name),
                })?;
            if end > file_len_u64 {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "GGUF tensor `{}`: data range [{absolute}..{end}] exceeds file size {file_len_u64}",
                        info.name
                    ),
                });
            }
        }
        info.data_offset = absolute;
    }

    Ok(ParsedGguf {
        mmap: raw,
        version,
        alignment,
        metadata,
        tensor_infos,
    })
}

/// Reads one `gguf_tensor_info_t` from the cursor and returns a
/// [`GgufTensorInfo`] whose `data_offset` holds the **relative** offset as
/// stored in the file. A patch sweep in [`parse_gguf`] rewrites it to the
/// absolute mmap offset once `data_section_start` is known.
///
/// All the cheap per-entry validation (dimension count, zero dimension,
/// element-count overflow, `byte_size_for_n_elements`, dimension-to-`usize`
/// conversion) happens here so that the patch sweep only needs to do offset
/// arithmetic and the two bounds checks that depend on the data section
/// start.
fn read_tensor_info_relative(cursor: &mut Cursor<'_>) -> crate::Result<GgufTensorInfo> {
    // GGUF tensor names: the spec caps them at 64 bytes, but some encoders
    // produce longer names in practice. Accept up to `MAX_TENSOR_NAME_LEN`.
    let name = cursor.read_string(MAX_TENSOR_NAME_LEN)?;
    let n_dims = cursor.read_u32_le()?;
    if n_dims == 0 {
        return Err(AnamnesisError::Parse {
            reason: format!("GGUF tensor `{name}`: n_dimensions is zero"),
        });
    }
    if n_dims > MAX_TENSOR_DIMS {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "GGUF tensor `{name}`: n_dimensions {n_dims} exceeds cap {MAX_TENSOR_DIMS}"
            ),
        });
    }
    let n_dims_usz = usize::try_from(n_dims).map_err(|_| AnamnesisError::Parse {
        reason: format!("GGUF tensor `{name}`: n_dimensions {n_dims} overflows usize"),
    })?;
    // Read the `n_dims` dimensions and convert them to `usize` at the same
    // time so we never keep both a `Vec<u64>` and a `Vec<usize>` alive.
    let mut shape_usz: Vec<usize> = Vec::with_capacity(n_dims_usz);
    // Track the element-count product as we go so that we can call
    // `byte_size_for_n_elements` below without re-iterating the shape.
    let mut n_elements: u64 = 1;
    for _ in 0..n_dims {
        let d = cursor.read_u64_le()?;
        if d == 0 {
            return Err(AnamnesisError::Parse {
                reason: format!("GGUF tensor `{name}`: zero-sized dimension"),
            });
        }
        n_elements = n_elements
            .checked_mul(d)
            .ok_or_else(|| AnamnesisError::Parse {
                reason: format!("GGUF tensor `{name}`: element count overflow"),
            })?;
        if n_elements > MAX_TENSOR_ELEMENTS {
            return Err(AnamnesisError::Parse {
                reason: format!(
                    "GGUF tensor `{name}`: element count {n_elements} exceeds cap {MAX_TENSOR_ELEMENTS}"
                ),
            });
        }
        let d_usz = usize::try_from(d).map_err(|_| AnamnesisError::Parse {
            reason: format!("GGUF tensor `{name}`: dimension {d} overflows usize"),
        })?;
        shape_usz.push(d_usz);
    }
    let dtype = GgufType::from_u32(cursor.read_u32_le()?)?;
    let relative_offset = cursor.read_u64_le()?;
    let byte_len = if dtype.type_size().is_some() {
        Some(dtype.byte_size_for_n_elements(n_elements)?)
    } else {
        None
    };
    Ok(GgufTensorInfo {
        name,
        shape: shape_usz,
        dtype,
        // Temporarily holds the relative offset; patched to absolute in the
        // caller once `data_section_start` is known.
        data_offset: relative_offset,
        byte_len,
    })
}

/// Returns the canonical name of a metadata value type for error messages.
const fn metadata_type_name(value: &GgufMetadataValue) -> &'static str {
    match value {
        GgufMetadataValue::U8(_) => "UINT8",
        GgufMetadataValue::I8(_) => "INT8",
        GgufMetadataValue::U16(_) => "UINT16",
        GgufMetadataValue::I16(_) => "INT16",
        GgufMetadataValue::U32(_) => "UINT32",
        GgufMetadataValue::I32(_) => "INT32",
        GgufMetadataValue::F32(_) => "FLOAT32",
        GgufMetadataValue::Bool(_) => "BOOL",
        GgufMetadataValue::String(_) => "STRING",
        GgufMetadataValue::Array(_) => "ARRAY",
        GgufMetadataValue::U64(_) => "UINT64",
        GgufMetadataValue::I64(_) => "INT64",
        GgufMetadataValue::F64(_) => "FLOAT64",
    }
}

/// Rounds `offset` up to the next multiple of `alignment`.
///
/// `alignment` must be non-zero; the caller guarantees this by substituting
/// `DEFAULT_ALIGNMENT` whenever the metadata key is absent or zero.
fn align_up(offset: u64, alignment: u64) -> crate::Result<u64> {
    if alignment == 0 {
        return Err(AnamnesisError::Parse {
            reason: "GGUF: general.alignment must be non-zero".into(),
        });
    }
    let rem = offset % alignment;
    if rem == 0 {
        return Ok(offset);
    }
    // `alignment - rem` is strictly less than `alignment`, so the add
    // overflows only if `offset` is already within `alignment` of `u64::MAX`.
    let padding = alignment - rem;
    offset
        .checked_add(padding)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "GGUF: alignment padding overflow (offset {offset}, alignment {alignment})"
            ),
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::wildcard_enum_match_arm,
    clippy::manual_is_multiple_of
)]
mod tests {
    use super::*;
    use std::io::Write;

    // -----------------------------------------------------------------
    // Fixture builder
    // -----------------------------------------------------------------

    /// In-memory builder for synthetic GGUF byte streams. Produces
    /// little-endian v3 files by default; individual fields can be
    /// overridden for negative tests.
    struct GgufBuilder {
        buf: Vec<u8>,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self { buf: Vec::new() }
        }

        fn push_bytes(&mut self, bytes: &[u8]) {
            self.buf.extend_from_slice(bytes);
        }

        fn push_u32(&mut self, v: u32) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        fn push_u64(&mut self, v: u64) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        fn push_string(&mut self, s: &str) {
            self.push_u64(s.len() as u64);
            self.buf.extend_from_slice(s.as_bytes());
        }

        fn push_kv_uint32(&mut self, key: &str, value: u32) {
            self.push_string(key);
            self.push_u32(4); // UINT32
            self.push_u32(value);
        }

        fn push_kv_string(&mut self, key: &str, value: &str) {
            self.push_string(key);
            self.push_u32(8); // STRING
            self.push_string(value);
        }

        fn push_kv_f32_array(&mut self, key: &str, values: &[f32]) {
            self.push_string(key);
            self.push_u32(9); // ARRAY
            self.push_u32(6); // inner type FLOAT32
            self.push_u64(values.len() as u64);
            for v in values {
                self.buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        fn push_tensor_info(
            &mut self,
            name: &str,
            shape: &[u64],
            dtype_disc: u32,
            relative_offset: u64,
        ) {
            self.push_string(name);
            self.push_u32(u32::try_from(shape.len()).expect("shape len fits in u32 for tests"));
            for &d in shape {
                self.push_u64(d);
            }
            self.push_u32(dtype_disc);
            self.push_u64(relative_offset);
        }

        fn pad_to_alignment(&mut self, alignment: usize) {
            while self.buf.len() % alignment != 0 {
                self.buf.push(0);
            }
        }

        fn finish(self) -> Vec<u8> {
            self.buf
        }
    }

    fn build_minimal_gguf() -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3); // version
        b.push_u64(2); // tensor_count
        b.push_u64(3); // kv_count

        // kv pairs
        b.push_kv_string("general.architecture", "test");
        b.push_kv_uint32("general.alignment", 32);
        b.push_kv_f32_array("test.values", &[1.0, 2.0, 3.0]);

        // tensor 0 — F32 [2, 3] → 24 bytes at relative offset 0
        b.push_tensor_info("tensor.a", &[2, 3], 0, 0);
        // tensor 1 — Q4_0 [64] → 2 blocks × 18 bytes = 36 bytes at relative
        //            offset 32 (24 bytes + 8-byte pad to 32)
        b.push_tensor_info("tensor.b", &[64], 2, 32);

        b.pad_to_alignment(32);
        // tensor.a data — 24 bytes of zeros
        b.push_bytes(&[0u8; 24]);
        // pad to next 32-byte boundary
        b.pad_to_alignment(32);
        // tensor.b data — 36 bytes of zeros
        b.push_bytes(&[0u8; 36]);

        b.finish()
    }

    fn write_temp_gguf(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    // -----------------------------------------------------------------
    // Happy-path tests
    // -----------------------------------------------------------------

    #[test]
    fn parse_minimal_gguf_succeeds() {
        let bytes = build_minimal_gguf();
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        assert_eq!(parsed.version(), 3);
        assert_eq!(parsed.alignment(), 32);
        assert_eq!(parsed.len(), 2);
        assert!(!parsed.is_empty());

        let infos = parsed.tensor_info();
        assert_eq!(infos[0].name, "tensor.a");
        assert_eq!(infos[0].shape, vec![2, 3]);
        assert_eq!(infos[0].dtype, GgufType::F32);
        assert_eq!(infos[0].byte_len, Some(24));
        assert_eq!(infos[1].name, "tensor.b");
        assert_eq!(infos[1].shape, vec![64]);
        assert_eq!(infos[1].dtype, GgufType::Q4_0);
        assert_eq!(infos[1].byte_len, Some(36));

        let metadata = parsed.metadata();
        assert_eq!(
            metadata
                .get("general.architecture")
                .and_then(|v| v.as_string()),
            Some("test")
        );
        assert_eq!(
            metadata
                .get("general.alignment")
                .and_then(GgufMetadataValue::as_u32),
            Some(32)
        );
        let arr = metadata
            .get("test.values")
            .and_then(GgufMetadataValue::as_array)
            .unwrap();
        assert_eq!(arr.len(), 3);
        let f32s = arr
            .as_f32_slice()
            .expect("test.values should be an F32 array");
        assert_eq!(f32s, &[1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn tensors_returns_zero_copy_borrowed_slices() {
        let bytes = build_minimal_gguf();
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        let tensors: Vec<_> = parsed.tensors().collect();
        assert_eq!(tensors.len(), 2);
        for t in &tensors {
            assert!(matches!(t.data, Cow::Borrowed(_)));
        }
        assert_eq!(tensors[0].data.len(), 24);
        assert_eq!(tensors[1].data.len(), 36);
        // `name` and `shape` now borrow from the ParsedGguf, not owned.
        assert_eq!(tensors[0].name, "tensor.a");
        assert_eq!(tensors[0].shape, &[2_usize, 3]);
        assert_eq!(tensors[1].name, "tensor.b");
        assert_eq!(tensors[1].shape, &[64_usize]);
    }

    #[test]
    fn inspect_info_reports_expected_fields() {
        let bytes = build_minimal_gguf();
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        let info = parsed.inspect();
        assert_eq!(info.version, 3);
        assert_eq!(info.architecture.as_deref(), Some("test"));
        assert_eq!(info.tensor_count, 2);
        assert_eq!(info.total_bytes, 24 + 36);
        assert_eq!(info.unknown_size_tensors, 0);
        assert_eq!(info.alignment, 32);
        assert_eq!(info.dtypes, vec![GgufType::F32, GgufType::Q4_0]);
        let rendered = info.to_string();
        assert!(rendered.contains("GGUF v3"));
        assert!(rendered.contains("Arch:        test"));
        assert!(rendered.contains("Tensors:     2"));
        assert!(rendered.contains("Dtypes:      F32, Q4_0"));
        assert!(rendered.contains("Alignment:   32 bytes"));
    }

    #[test]
    fn typed_array_f32_uses_native_storage() {
        // Round-trips a 5-element F32 metadata array and asserts the
        // parser materialised it as `GgufMetadataArray::F32(Vec<f32>)` —
        // the fast path that eliminates the 8× enum-discriminant bloat.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(1);
        b.push_kv_f32_array("logits", &[1.5, -2.25, 0.0, 3.5, 7.125]);
        let tmp = write_temp_gguf(&b.finish());
        let parsed = parse_gguf(tmp.path()).unwrap();
        let arr = parsed
            .metadata()
            .get("logits")
            .and_then(GgufMetadataValue::as_array)
            .unwrap();
        assert!(matches!(arr, GgufMetadataArray::F32(_)));
        let slice = arr.as_f32_slice().unwrap();
        assert_eq!(slice, &[1.5f32, -2.25, 0.0, 3.5, 7.125]);
        assert_eq!(arr.len(), 5);
        assert!(!arr.is_empty());
    }

    #[test]
    fn metadata_value_size_is_bounded() {
        // Mirrors the compile-time `const _` assertions near the top of
        // the module so the size invariant shows up in the test suite too.
        assert_eq!(std::mem::size_of::<GgufMetadataValue>(), 24);
        assert_eq!(std::mem::size_of::<GgufMetadataArray>(), 32);
    }

    #[test]
    fn parse_header_only_file_is_accepted() {
        // 24-byte file: magic + version + tensor_count=0 + kv_count=0.
        // No tensor_data section exists, so the alignment-and-bounds check
        // must be skipped instead of rejecting a legitimate empty file.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(0);
        let bytes = b.finish();
        assert_eq!(bytes.len(), 24);
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        assert_eq!(parsed.version(), 3);
        assert_eq!(parsed.alignment(), 32);
        assert_eq!(parsed.len(), 0);
        assert!(parsed.is_empty());
        assert!(parsed.metadata().is_empty());
        assert!(parsed.tensor_info().is_empty());
        assert_eq!(parsed.tensors().count(), 0);
    }

    #[test]
    fn alignment_defaults_to_32_when_metadata_absent() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(1);
        b.push_u64(1);
        b.push_kv_string("general.architecture", "test");
        // Single F32 scalar
        b.push_tensor_info("x", &[1], 0, 0);
        b.pad_to_alignment(32);
        b.push_bytes(&[0u8; 4]);
        let bytes = b.finish();
        let tmp = write_temp_gguf(&bytes);
        let parsed = parse_gguf(tmp.path()).unwrap();
        assert_eq!(parsed.alignment(), 32);
    }

    // -----------------------------------------------------------------
    // Negative / validation tests
    // -----------------------------------------------------------------

    #[test]
    fn reject_file_too_small() {
        let tmp = write_temp_gguf(b"GGU");
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_bad_magic() {
        let tmp = write_temp_gguf(b"XXXX\x00\x00\x00\x00");
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_legacy_ggml_magic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGML");
        bytes.extend_from_slice(&[0u8; 100]);
        let tmp = write_temp_gguf(&bytes);
        let err = parse_gguf(tmp.path()).unwrap_err();
        match err {
            AnamnesisError::Unsupported { format, detail } => {
                assert_eq!(format, "GGUF");
                assert!(detail.contains("GGML"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn reject_v1() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(1); // v1
        b.push_u64(0);
        b.push_u64(0);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Unsupported { .. }));
    }

    #[test]
    fn reject_truncated_file() {
        let bytes = build_minimal_gguf();
        let truncated = &bytes[..bytes.len() - 20];
        let tmp = write_temp_gguf(truncated);
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_tensor_data_out_of_bounds() {
        // 1 tensor, no KV pairs. The tensor claims F32 [1000] = 4000 bytes
        // at relative offset 0, but the file only contains 32 bytes of
        // tensor data, which is far less than that. Parser must reject.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(1);
        b.push_u64(0);
        b.push_tensor_info("huge", &[1000], 0, 0);
        b.pad_to_alignment(32);
        b.push_bytes(&[0u8; 32]);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(reason.contains("exceeds file size"), "got: {reason}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn reject_zero_dimension() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(1);
        b.push_u64(0);
        b.push_tensor_info("zero", &[0], 0, 0);
        b.pad_to_alignment(32);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_unaligned_relative_offset() {
        // The GGUF spec mandates each tensor's offset field is a multiple
        // of `general.alignment`. A well-formed file with `alignment = 32`
        // and a tensor at `relative_offset = 1` must be rejected, because
        // downstream consumers would get unaligned byte slices.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(1);
        b.push_u64(0);
        // F32 [1] — 4 bytes — at relative offset 1 (not a multiple of 32).
        b.push_tensor_info("misaligned", &[1], 0, 1);
        b.pad_to_alignment(32);
        // Enough data so the trailing bounds check cannot mask the
        // alignment check — we need to verify the alignment check is the
        // one that fires, not the "exceeds file size" check.
        b.push_bytes(&[0u8; 64]);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(
                    reason.contains("not a multiple of alignment"),
                    "expected alignment error, got: {reason}"
                );
                assert!(reason.contains("misaligned"), "got: {reason}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn accept_aligned_nonzero_relative_offset() {
        // Regression guard for the alignment check: a legitimate tensor at
        // a non-zero but aligned relative offset (e.g., second tensor in a
        // file, sitting at relative offset 32) must still parse cleanly.
        let bytes = build_minimal_gguf();
        let parsed = parse_gguf(write_temp_gguf(&bytes).path()).unwrap();
        assert_eq!(parsed.len(), 2);
        // tensor.b is at relative offset 32 in the fixture — aligned to 32.
        assert_eq!(parsed.tensor_info()[1].name, "tensor.b");
    }

    #[test]
    fn reject_array_depth_exceeded() {
        // Nest ARRAY values deeper than MAX_ARRAY_DEPTH (4) and expect
        // rejection. The KV value_type=ARRAY is read by `read_metadata_value`,
        // which then calls `read_typed_array` at depth 0 (the outer array).
        // Each (inner_type=9, len=1) pair drives one more level of
        // recursion. Four pairs successfully walk depths 0 → 3 (building the
        // outer array plus 3 sub-arrays). The 5th pair trips the
        // `depth >= MAX_ARRAY_DEPTH` check when the parser tries to enter
        // depth 4.
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(1);
        b.push_string("nested");
        b.push_u32(9); // KV value_type = ARRAY
        for _ in 0..5 {
            b.push_u32(9); // inner_type = ARRAY
            b.push_u64(1); // length = 1
        }
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => {
                assert!(
                    reason.contains("depth cap"),
                    "expected depth-cap error, got: {reason}"
                );
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn reject_bad_bool_byte() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(1);
        b.push_string("weird");
        b.push_u32(7); // BOOL
        b.push_bytes(&[7]); // not 0 or 1
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn reject_zero_alignment() {
        let mut b = GgufBuilder::new();
        b.push_bytes(b"GGUF");
        b.push_u32(3);
        b.push_u64(0);
        b.push_u64(1);
        b.push_kv_uint32("general.alignment", 0);
        let tmp = write_temp_gguf(&b.finish());
        let err = parse_gguf(tmp.path()).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    // -----------------------------------------------------------------
    // GgufType table spot-checks
    // -----------------------------------------------------------------

    #[test]
    fn byte_size_table_spot_checks() {
        assert_eq!(GgufType::F32.block_size(), 1);
        assert_eq!(GgufType::F32.type_size(), Some(4));
        assert_eq!(GgufType::F32.byte_size_for_n_elements(10).unwrap(), 40);

        assert_eq!(GgufType::Q4_0.block_size(), 32);
        assert_eq!(GgufType::Q4_0.type_size(), Some(18));
        assert_eq!(GgufType::Q4_0.byte_size_for_n_elements(64).unwrap(), 36);

        assert_eq!(GgufType::Q4_K.block_size(), 256);
        assert_eq!(GgufType::Q4_K.type_size(), Some(144));
        assert_eq!(GgufType::Q4_K.byte_size_for_n_elements(256).unwrap(), 144);

        assert_eq!(GgufType::Q8_0.block_size(), 32);
        assert_eq!(GgufType::Q8_0.type_size(), Some(34));
        assert_eq!(GgufType::Q8_0.byte_size_for_n_elements(32).unwrap(), 34);

        assert_eq!(GgufType::Q8_K.type_size(), Some(292));
        assert_eq!(GgufType::Q6_K.type_size(), Some(210));

        // Non-linear 4-bit IQ variants landed in Phase 4.5 step 1.
        assert_eq!(GgufType::IQ4_NL.block_size(), 32);
        assert_eq!(GgufType::IQ4_NL.type_size(), Some(18));
        assert_eq!(GgufType::IQ4_NL.byte_size_for_n_elements(64).unwrap(), 36);
        assert_eq!(GgufType::IQ4_XS.block_size(), 256);
        assert_eq!(GgufType::IQ4_XS.type_size(), Some(136));
        assert_eq!(GgufType::IQ4_XS.byte_size_for_n_elements(256).unwrap(), 136);

        // 2-bit IQ super-quants landed in Phase 4.5 step 2.
        assert_eq!(GgufType::IQ2_XXS.block_size(), 256);
        assert_eq!(GgufType::IQ2_XXS.type_size(), Some(66));
        assert_eq!(GgufType::IQ2_XXS.byte_size_for_n_elements(256).unwrap(), 66);
        assert_eq!(GgufType::IQ2_XS.block_size(), 256);
        assert_eq!(GgufType::IQ2_XS.type_size(), Some(74));
        assert_eq!(GgufType::IQ2_XS.byte_size_for_n_elements(256).unwrap(), 74);
        assert_eq!(GgufType::IQ2_S.block_size(), 256);
        assert_eq!(GgufType::IQ2_S.type_size(), Some(82));
        assert_eq!(GgufType::IQ2_S.byte_size_for_n_elements(256).unwrap(), 82);
    }

    #[test]
    fn iq_types_have_unknown_type_size() {
        // IQ4_NL / IQ4_XS (step 1) and IQ2_XXS / IQ2_XS / IQ2_S (step 2) are
        // supported; the remaining IQ* / TQ* / MXFP4 kernels are still deferred.
        assert_eq!(GgufType::IQ3_XXS.type_size(), None);
        assert_eq!(GgufType::IQ3_S.type_size(), None);
        assert_eq!(GgufType::MXFP4.type_size(), None);
        let err = GgufType::IQ3_XXS.byte_size_for_n_elements(256).unwrap_err();
        assert!(matches!(err, AnamnesisError::Unsupported { .. }));
    }

    #[test]
    fn is_quantized_classifies_correctly() {
        assert!(!GgufType::F32.is_quantized());
        assert!(!GgufType::BF16.is_quantized());
        assert!(!GgufType::I32.is_quantized());
        assert!(GgufType::Q4_0.is_quantized());
        assert!(GgufType::Q4_K.is_quantized());
        assert!(GgufType::IQ4_XS.is_quantized());
    }

    #[test]
    fn byte_size_rejects_non_multiple_of_block() {
        // 17 elements of Q4_0 (block size 32) — not a multiple.
        let err = GgufType::Q4_0.byte_size_for_n_elements(17).unwrap_err();
        assert!(matches!(err, AnamnesisError::Parse { .. }));
    }

    #[test]
    fn align_up_behaves() {
        assert_eq!(align_up(0, 32).unwrap(), 0);
        assert_eq!(align_up(1, 32).unwrap(), 32);
        assert_eq!(align_up(32, 32).unwrap(), 32);
        assert_eq!(align_up(33, 32).unwrap(), 64);
        assert_eq!(align_up(100, 16).unwrap(), 112);
    }

    #[test]
    fn gguf_type_display_roundtrip() {
        assert_eq!(GgufType::F32.to_string(), "F32");
        assert_eq!(GgufType::Q4_K.to_string(), "Q4_K");
        assert_eq!(GgufType::IQ4_XS.to_string(), "IQ4_XS");
        assert_eq!(GgufType::BF16.to_string(), "BF16");
    }
}
