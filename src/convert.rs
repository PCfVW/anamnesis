// SPDX-License-Identifier: MIT OR Apache-2.0

//! Format conversion through an in-memory **`BF16` hub**.
//!
//! `convert` normalises any supported input into an in-memory hub — a list of owned
//! tensors carrying `(name, shape, dtype, bytes)` — then writes that hub to the
//! requested [`ConvertTarget`]. Routing every `(input × target)` pair through one
//! hub replaces the per-cell handlers of v0.6.0, which rejected most
//! combinations, and means a new input format costs one *reader* while a new
//! target costs one *writer*.
//!
//! # Dtype policy
//!
//! The hub is a `BF16` *pivot*, not a `BF16` *floor*:
//!
//! - **Quantised** tensors (`FP8` / `GPTQ` / `AWQ` / `BnB` safetensors, quantised
//!   `GGUF` blocks) are dequantised to `BF16` — the only lossless-in-intent
//!   representation the crate produces.
//! - **Scalar** tensors keep their **original dtype** (`F64` / `F32` / `F16` /
//!   `BF16` / `I8`–`I64` / `U8` / `Bool`), so `.pth` → safetensors and
//!   `NPZ`-`F32` → `GGUF` stay bit-for-bit lossless.
//! - `BF16` is materialised for a *scalar* tensor only where the target demands
//!   it — currently just the `BnB-NF4` encoder, whose input contract is `BF16`.
//!
//! Shapes are held **row-major** (the safetensors / `NumPy` convention); the
//! `GGUF` reader and writer reverse them at the boundary, since `GGUF` stores
//! dimensions most-significant-first.
//!
//! # Memory
//!
//! The hub is eager and owned: peak heap is `O(2 × model)` — the same profile as
//! [`ParsedModel::remember_to_bytes`](crate::ParsedModel::remember_to_bytes),
//! since every tensor is materialised before the writer runs. Streaming output
//! is `ROADMAP.md` Phase 10.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{AnamnesisError, Dtype, ParseLimits};

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Target format for [`convert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvertTarget {
    /// `safetensors` (alias `bf16`) — dequantise any quantised input to `BF16`,
    /// passing scalar tensors through in their original dtype.
    Safetensors,
    /// `gguf` — an unquantised (scalar) `GGUF` file. Quantised `GGUF` emit
    /// (`gguf-q4km`, …) needs the Phase 8.5 encode kernels.
    Gguf,
    /// `bnb-nf4` — `BitsAndBytes`-NF4 safetensors: 2-D float weights encoded to
    /// NF4, everything else passed through as `BF16`.
    BnbNf4,
}

impl ConvertTarget {
    /// Parses a CLI `--to` value. Accepted (case-insensitive): `safetensors` /
    /// `bf16`, `gguf`, `bnb-nf4` / `bnb_nf4` / `nf4`.
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Unsupported`] for an unrecognised target.
    pub fn parse(raw: &str) -> crate::Result<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "safetensors" | "bf16" => Ok(Self::Safetensors),
            "gguf" => Ok(Self::Gguf),
            "bnb-nf4" | "bnb_nf4" | "nf4" => Ok(Self::BnbNf4),
            other => Err(AnamnesisError::Unsupported {
                format: other.to_owned(),
                detail: "supported convert targets: `safetensors` (alias `bf16`), \
                         `gguf`, `bnb-nf4`. Quantised GGUF targets need Phase 8.5"
                    .into(),
            }),
        }
    }

    /// The file extension a derived output path gets for this target.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Safetensors | Self::BnbNf4 => "safetensors",
            Self::Gguf => "gguf",
        }
    }

    /// The stem suffix a derived output path gets for this target.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Safetensors => "bf16",
            Self::Gguf => "gguf",
            Self::BnbNf4 => "bnb-nf4",
        }
    }
}

/// Caller-supplied knobs for [`convert`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ConvertOptions {
    /// Resource budget applied to the *input* parse. Defaults to
    /// [`ParseLimits::default`] (unbounded beyond the permanent per-format caps).
    pub limits: ParseLimits,
}

impl ConvertOptions {
    /// Returns options with the default (unbounded) [`ParseLimits`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            limits: ParseLimits::unbounded(),
        }
    }

    /// Sets the resource budget applied to the input parse.
    #[must_use]
    pub const fn with_limits(mut self, limits: ParseLimits) -> Self {
        self.limits = limits;
        self
    }
}

/// What a [`convert`] call produced, for progress reporting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConvertStats {
    /// Total tensors written.
    pub tensors: usize,
    /// Input tensors that were dequantised to `BF16` on the way in.
    pub dequantized: usize,
    /// Tensors the target *quantised* on the way out (the `BnB-NF4` encoder).
    pub quantized: usize,
    /// Tensors written in their incoming dtype.
    pub passthrough: usize,
}

// ---------------------------------------------------------------------------
// The hub
// ---------------------------------------------------------------------------

/// One tensor normalised into the hub: owned bytes in `dtype`, shape row-major.
#[derive(Debug, Clone)]
pub(crate) struct HubTensor {
    /// Tensor name as it will appear in the output.
    pub(crate) name: String,
    /// Row-major (safetensors / `NumPy` order) dimensions.
    pub(crate) shape: Vec<usize>,
    /// Element type of `data`.
    pub(crate) dtype: Dtype,
    /// Raw little-endian bytes, `product(shape) × dtype.byte_size()` long.
    pub(crate) data: Vec<u8>,
}

/// An input normalised for the writers: the tensors plus any source metadata
/// that survives the conversion.
#[derive(Debug, Default)]
pub(crate) struct Hub {
    /// The normalised tensors, in output order.
    pub(crate) tensors: Vec<HubTensor>,
    /// safetensors `__metadata__`, carried only when the *source* was
    /// safetensors (every other reader writes `None`, matching v0.6.0).
    pub(crate) st_metadata: Option<HashMap<String, String>>,
    /// Tensors dequantised while reading.
    pub(crate) dequantized: usize,
    /// `GGUF` key/value metadata carried from a `GGUF` source so a
    /// dequantise-in-place `gguf → gguf` keeps the architecture / tokenizer KV
    /// that makes the output loadable — a re-emitted file with no KV is a bare
    /// tensor container. Empty for every non-`GGUF` source; caller-supplied KV
    /// is merged over it by the writer.
    #[cfg(feature = "gguf")]
    pub(crate) gguf_metadata: HashMap<String, crate::GgufMetadataValue>,
}

// ---------------------------------------------------------------------------
// Format detection (shared by the CLI's parse / inspect / remember paths)
// ---------------------------------------------------------------------------

/// A detected input format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    /// `.safetensors` (or an unrecognised extension that is not `GGUF`).
    Safetensors,
    /// `PyTorch` `.pth` / `.pt` (or a `.bin` with ZIP magic).
    #[cfg(feature = "pth")]
    Pth,
    /// `NumPy` `.npz`.
    #[cfg(feature = "npz")]
    Npz,
    /// `GGUF` (by extension or magic).
    #[cfg(feature = "gguf")]
    Gguf,
}

/// Builds an `Unsupported` error naming the Cargo feature that would enable a
/// detected-but-disabled format.
///
/// Compiled only when at least one of `pth` / `npz` / `gguf` is absent — with all
/// three enabled it has no callers.
#[cfg(not(all(feature = "pth", feature = "npz", feature = "gguf")))]
fn missing_feature_err(format_name: &str, kind: &str, feature_flag: &str) -> AnamnesisError {
    AnamnesisError::Unsupported {
        format: format_name.into(),
        detail: format!(
            "input is {kind} but the `{feature_flag}` Cargo feature is not enabled in this \
             build — rebuild with `cargo install anamnesis --features cli,{feature_flag}` \
             (or `cargo build --features cli,{feature_flag}`) to add support"
        ),
    }
}

/// Returns `true` if the file starts with `magic`. Reads only 4 bytes — does
/// not load the file into memory.
fn has_magic(path: &Path, magic: [u8; 4]) -> bool {
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read as _;
            f.read_exact(&mut buf)
        })
        .is_ok_and(|()| buf == magic)
}

/// Detects the model format from file extension, falling back to magic bytes.
///
/// `.safetensors` → safetensors. `.pth` / `.pt` → `PyTorch`. `.npz` → NPZ.
/// `.gguf` → `GGUF`. `.bin` → ZIP magic (`PyTorch`) then `GGUF` magic. Any other
/// extension → `GGUF` magic, else safetensors.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] when the input matches a format whose
/// Cargo feature is disabled in this build, rather than misrouting it to the
/// safetensors parser.
// `clippy::unnecessary_wraps`: with all of `pth`/`npz`/`gguf` enabled every arm
// is `Ok(_)`; other feature combinations make the wrap load-bearing.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn detect_format(path: &Path) -> crate::Result<Format> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "safetensors" => Ok(Format::Safetensors),
        "pth" | "pt" => {
            #[cfg(feature = "pth")]
            {
                Ok(Format::Pth)
            }
            #[cfg(not(feature = "pth"))]
            {
                Err(missing_feature_err("PyTorch", "a .pth/.pt file", "pth"))
            }
        }
        "npz" => {
            #[cfg(feature = "npz")]
            {
                Ok(Format::Npz)
            }
            #[cfg(not(feature = "npz"))]
            {
                Err(missing_feature_err("NumPy NPZ", "a .npz file", "npz"))
            }
        }
        "gguf" => {
            #[cfg(feature = "gguf")]
            {
                Ok(Format::Gguf)
            }
            #[cfg(not(feature = "gguf"))]
            {
                Err(missing_feature_err("GGUF", "a .gguf file", "gguf"))
            }
        }
        "bin" => {
            if has_magic(path, *b"PK\x03\x04") {
                #[cfg(feature = "pth")]
                {
                    return Ok(Format::Pth);
                }
                #[cfg(not(feature = "pth"))]
                {
                    return Err(missing_feature_err(
                        "PyTorch",
                        "a .bin file with ZIP magic (PyTorch pickle archive)",
                        "pth",
                    ));
                }
            }
            if has_magic(path, *b"GGUF") {
                #[cfg(feature = "gguf")]
                {
                    return Ok(Format::Gguf);
                }
                #[cfg(not(feature = "gguf"))]
                {
                    return Err(missing_feature_err(
                        "GGUF",
                        "a .bin file with GGUF magic",
                        "gguf",
                    ));
                }
            }
            Ok(Format::Safetensors)
        }
        _ => {
            if has_magic(path, *b"GGUF") {
                #[cfg(feature = "gguf")]
                {
                    return Ok(Format::Gguf);
                }
                #[cfg(not(feature = "gguf"))]
                {
                    return Err(missing_feature_err(
                        "GGUF",
                        "a file whose first four bytes are the GGUF magic",
                        "gguf",
                    ));
                }
            }
            Ok(Format::Safetensors)
        }
    }
}

// ---------------------------------------------------------------------------
// Output-path derivation
// ---------------------------------------------------------------------------

/// Quantisation suffixes stripped from an input stem when deriving an output
/// path. Case-sensitive and ordered longest-first so `-GPTQ-Int4` wins over
/// `-gptq`.
const QUANT_SUFFIXES: &[&str] = &[
    "-GPTQ-Int4",
    "-GPTQ-Int8",
    "-gptq-int4",
    "-gptq-int8",
    "-gptq4",
    "-gptq8",
    "-GPTQ",
    "-gptq",
    "_gptq",
    "-AWQ",
    "-awq",
    "_awq",
    "-bnb-4bit",
    "-bnb-int8",
    "-bnb",
    "_bnb",
    "-4bit",
    "-int4",
    "-int8",
    "-fp8",
    "_fp8",
    "-FP8",
];

/// Strips a known quantisation suffix from a file stem, if present.
/// `model-GPTQ-Int4` → `model`; `weights` → `weights`.
///
/// Shared with the CLI's `remember` output-path derivation so both stay on one
/// suffix table.
#[must_use]
pub(crate) fn strip_quant_suffix(stem: &str) -> &str {
    QUANT_SUFFIXES
        .iter()
        .find_map(|qs| stem.strip_suffix(qs))
        .unwrap_or(stem)
}

/// Derives an output path from `input` and `target`: strips a known
/// quantisation suffix from the stem, then appends `-{suffix}.{extension}`.
///
/// `model-fp8.safetensors` + `Safetensors` → `model-bf16.safetensors`;
/// `weights.npz` + `Gguf` → `weights-gguf.gguf`.
#[must_use]
pub fn derive_output_path(input: &Path, target: ConvertTarget) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let new_name = format!(
        "{}-{}.{}",
        strip_quant_suffix(stem),
        target.suffix(),
        target.extension()
    );
    input
        .parent()
        .map_or_else(|| PathBuf::from(&new_name), |p| p.join(&new_name))
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Converts `input` to `target`, writing the result to `output`.
///
/// Every supported `(input format × target)` pair routes through the in-memory
/// hub: the input is parsed and normalised (quantised tensors dequantised to
/// `BF16`, scalar tensors kept in their original dtype), then written to the
/// target. Format detection is automatic: by file extension, falling back to
/// magic bytes for `.bin` and unrecognised extensions.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if the input format's Cargo feature
/// is disabled, if the target is not reachable from this input, or if a tensor's
/// dtype has no counterpart in the target format.
/// Returns [`AnamnesisError::LimitExceeded`] if the input breaches
/// `options.limits` or a permanent per-format cap.
/// Returns [`AnamnesisError::Parse`] on a malformed input, and
/// [`AnamnesisError::Io`] if the input cannot be read or the output written.
///
/// # Memory
///
/// Peak heap is `O(2 × model)`: the whole hub is materialised before the writer
/// runs. See the module docs.
pub fn convert(
    input: &Path,
    target: ConvertTarget,
    output: &Path,
    options: &ConvertOptions,
) -> crate::Result<ConvertStats> {
    let hub = read_hub(input, options)?;
    write_hub(&hub, target, output)
}

/// Parses `input` into the hub, dispatching on the detected format.
fn read_hub(input: &Path, options: &ConvertOptions) -> crate::Result<Hub> {
    match detect_format(input)? {
        Format::Safetensors => read_safetensors(input, &options.limits),
        #[cfg(feature = "pth")]
        Format::Pth => read_pth(input, &options.limits),
        #[cfg(feature = "npz")]
        Format::Npz => read_npz(input, &options.limits),
        #[cfg(feature = "gguf")]
        Format::Gguf => read_gguf(input, &options.limits),
    }
}

/// Writes the hub to `target`.
fn write_hub(hub: &Hub, target: ConvertTarget, output: &Path) -> crate::Result<ConvertStats> {
    match target {
        ConvertTarget::Safetensors => write_safetensors(hub, output),
        #[cfg(feature = "gguf")]
        ConvertTarget::Gguf => write_gguf_target(hub, output),
        #[cfg(not(feature = "gguf"))]
        ConvertTarget::Gguf => Err(AnamnesisError::Unsupported {
            format: "gguf".into(),
            detail: "GGUF emit requires the `gguf` Cargo feature; rebuild with \
                     `--features cli,gguf`"
                .into(),
        }),
        #[cfg(feature = "bnb")]
        ConvertTarget::BnbNf4 => write_bnb_nf4_target(hub, output),
        #[cfg(not(feature = "bnb"))]
        ConvertTarget::BnbNf4 => Err(AnamnesisError::Unsupported {
            format: "bnb-nf4".into(),
            detail: "BnB-NF4 encode requires the `bnb` Cargo feature; rebuild with \
                     `--features cli,bnb`"
                .into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Readers — format → hub
// ---------------------------------------------------------------------------

/// Reads a safetensors file into the hub, dequantising quantised tensors to
/// `BF16` and passing scalar tensors through in their original dtype. Carries
/// the source `__metadata__`.
fn read_safetensors(path: &Path, limits: &ParseLimits) -> crate::Result<Hub> {
    let model = crate::parse_with_limits(path, limits)?;
    let (tensors, dequantized) = model.hub_tensors()?;
    Ok(Hub {
        tensors,
        st_metadata: model.header.metadata.clone(),
        dequantized,
        #[cfg(feature = "gguf")]
        gguf_metadata: HashMap::new(),
    })
}

/// Reads an `NPZ` archive into the hub. Every `NPZ` tensor is full precision, so
/// nothing is dequantised; dtypes are preserved. Tensors are emitted in sorted
/// name order for a deterministic output.
#[cfg(feature = "npz")]
fn read_npz(path: &Path, limits: &ParseLimits) -> crate::Result<Hub> {
    let map = crate::parse_npz_with_limits(path, limits)?;
    let mut names: Vec<&String> = map.keys().collect();
    names.sort();

    let mut tensors = Vec::with_capacity(map.len());
    for name in names {
        let t = map.get(name).ok_or_else(|| AnamnesisError::Parse {
            reason: format!("NPZ tensor `{name}` vanished mid-iteration"),
        })?;
        tensors.push(HubTensor {
            name: name.clone(),
            shape: t.shape.clone(),
            dtype: npz_dtype_to_hub(t.dtype),
            data: t.data.clone(),
        });
    }
    Ok(Hub {
        tensors,
        st_metadata: None,
        dequantized: 0,
        #[cfg(feature = "gguf")]
        gguf_metadata: HashMap::new(),
    })
}

/// Reads a `PyTorch` `.pth` into the hub. Tensor data is already full precision;
/// dtypes are preserved.
#[cfg(feature = "pth")]
fn read_pth(path: &Path, limits: &ParseLimits) -> crate::Result<Hub> {
    let parsed = crate::parse_pth_with_limits(path, limits)?;
    let pth_tensors = parsed.tensors()?;

    let mut tensors = Vec::with_capacity(pth_tensors.len());
    for t in pth_tensors {
        tensors.push(HubTensor {
            name: t.name,
            shape: t.shape,
            dtype: pth_dtype_to_hub(t.dtype)?,
            // BORROW: `into_owned()` copies the (possibly mmap-borrowed) bytes so
            // the hub outlives the `ParsedPth`.
            data: t.data.into_owned(),
        });
    }
    Ok(Hub {
        tensors,
        st_metadata: None,
        dequantized: 0,
        #[cfg(feature = "gguf")]
        gguf_metadata: HashMap::new(),
    })
}

/// Reads a `GGUF` file into the hub, dequantising block-quantised tensors to
/// `BF16` and passing scalar tensors through. `GGUF` shapes are
/// most-significant-first, so they are reversed into the hub's row-major order.
#[cfg(feature = "gguf")]
fn read_gguf(path: &Path, limits: &ParseLimits) -> crate::Result<Hub> {
    let parsed = crate::parse_gguf_with_limits(path, limits)?;

    let mut tensors = Vec::new();
    let mut dequantized = 0usize;
    for tensor in parsed.tensors() {
        let mut shape: Vec<usize> = tensor.shape.to_vec();
        shape.reverse();

        if tensor.dtype.is_quantized() {
            let n_elements = tensor
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| AnamnesisError::Parse {
                    reason: format!(
                        "GGUF tensor `{}` shape {:?} element count overflows usize",
                        tensor.name, tensor.shape
                    ),
                })?;
            let bf16 = crate::dequantize_gguf_to_bf16(&tensor.data, tensor.dtype, n_elements)?;
            tensors.push(HubTensor {
                name: tensor.name.to_owned(),
                shape,
                dtype: Dtype::BF16,
                data: bf16,
            });
            dequantized = dequantized.saturating_add(1);
        } else {
            tensors.push(HubTensor {
                name: tensor.name.to_owned(),
                shape,
                dtype: gguf_type_to_hub(tensor.dtype)?,
                // BORROW: `into_owned()` copies the mmap-borrowed slice so the
                // hub outlives the `ParsedGguf`.
                data: tensor.data.into_owned(),
            });
        }
    }
    Ok(Hub {
        tensors,
        st_metadata: None,
        dequantized,
        // Preserve the source KV so a dequantise-in-place `gguf -> gguf` stays
        // loadable; caller-supplied KV is merged over it by the writer.
        gguf_metadata: parsed.metadata().clone(),
    })
}

// ---------------------------------------------------------------------------
// Writers — hub → format
// ---------------------------------------------------------------------------

/// Writes the hub as a safetensors file, each tensor in its hub dtype.
fn write_safetensors(hub: &Hub, output: &Path) -> crate::Result<ConvertStats> {
    let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        Vec::with_capacity(hub.tensors.len());
    for t in &hub.tensors {
        let st_dtype = t.dtype.to_safetensors_dtype()?;
        let view = safetensors::tensor::TensorView::new(st_dtype, t.shape.clone(), &t.data)
            .map_err(|e| AnamnesisError::Parse {
                reason: format!("failed to create TensorView for `{}`: {e}", t.name),
            })?;
        views.push((t.name.clone(), view));
    }

    safetensors::tensor::serialize_to_file(views, hub.st_metadata.clone(), output).map_err(
        // EXHAUSTIVE: `SafeTensorError` is a foreign type that may gain variants.
        #[allow(clippy::wildcard_enum_match_arm)]
        |e| match e {
            safetensors::SafeTensorError::IoError(io_err) => AnamnesisError::Io(io_err),
            other => AnamnesisError::Parse {
                reason: format!("failed to write safetensors file: {other}"),
            },
        },
    )?;

    Ok(ConvertStats {
        tensors: hub.tensors.len(),
        dequantized: hub.dequantized,
        quantized: 0,
        passthrough: hub.tensors.len(),
    })
}

/// Writes the hub as an unquantised `GGUF` file, reversing shapes back to
/// most-significant-first.
#[cfg(feature = "gguf")]
fn write_gguf_target(hub: &Hub, output: &Path) -> crate::Result<ConvertStats> {
    use crate::{write_gguf, GgufWriteTensor};

    let mut owned: Vec<(String, crate::GgufType, Vec<usize>, &[u8])> =
        Vec::with_capacity(hub.tensors.len());
    for t in &hub.tensors {
        let gguf_dtype = hub_dtype_to_gguf(t.dtype)?;
        let mut msb_first = t.shape.clone();
        msb_first.reverse();
        owned.push((t.name.clone(), gguf_dtype, msb_first, t.data.as_slice()));
    }

    let tensors: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dtype, shape, data)| GgufWriteTensor {
            name: name.as_str(),
            shape: shape.as_slice(),
            dtype: *dtype,
            data,
        })
        .collect();

    write_gguf(output, &tensors, &hub.gguf_metadata)?;

    Ok(ConvertStats {
        tensors: tensors.len(),
        dequantized: hub.dequantized,
        quantized: 0,
        passthrough: tensors.len(),
    })
}

/// Writes the hub as `BitsAndBytes`-NF4 safetensors. The encoder's input
/// contract is `BF16`, so float tensors are converted on the way in; 2-D weights
/// are encoded to NF4 and everything else passes through as `BF16`.
#[cfg(feature = "bnb")]
fn write_bnb_nf4_target(hub: &Hub, output: &Path) -> crate::Result<ConvertStats> {
    use crate::{classify_inputs, write_bnb_nf4_safetensors, BnbWriteInput};

    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(hub.tensors.len());
    for t in &hub.tensors {
        let bf16 = to_bf16_bytes(&t.data, t.dtype, &t.name)?;
        owned.push((t.name.clone(), t.shape.clone(), bf16));
    }

    let inputs: Vec<BnbWriteInput<'_>> = owned
        .iter()
        .map(|(name, shape, bf16)| BnbWriteInput {
            name: name.as_str(),
            shape: shape.as_slice(),
            bf16_data: bf16.as_slice(),
        })
        .collect();

    let stats = classify_inputs(&inputs);
    write_bnb_nf4_safetensors(&inputs, output)?;

    Ok(ConvertStats {
        tensors: inputs.len(),
        dequantized: hub.dequantized,
        quantized: stats.quantized,
        passthrough: stats.passthrough,
    })
}

// ---------------------------------------------------------------------------
// Dtype mapping
// ---------------------------------------------------------------------------

/// Maps an [`NpzDtype`](crate::NpzDtype) to the hub dtype. Total — every `NPZ`
/// dtype has a safetensors counterpart.
#[cfg(feature = "npz")]
const fn npz_dtype_to_hub(dtype: crate::NpzDtype) -> Dtype {
    use crate::NpzDtype;
    match dtype {
        NpzDtype::Bool => Dtype::Bool,
        NpzDtype::U8 => Dtype::U8,
        NpzDtype::I8 => Dtype::I8,
        NpzDtype::U16 => Dtype::U16,
        NpzDtype::I16 => Dtype::I16,
        NpzDtype::U32 => Dtype::U32,
        NpzDtype::I32 => Dtype::I32,
        NpzDtype::U64 => Dtype::U64,
        NpzDtype::I64 => Dtype::I64,
        NpzDtype::F16 => Dtype::F16,
        NpzDtype::BF16 => Dtype::BF16,
        NpzDtype::F32 => Dtype::F32,
        NpzDtype::F64 => Dtype::F64,
    }
}

/// Maps a [`PthDtype`](crate::PthDtype) to the hub dtype. Total — every `.pth`
/// dtype the parser accepts has a safetensors counterpart.
///
/// # Errors
///
/// Currently infallible; returns `Result` so a future `PthDtype` without a
/// counterpart can be rejected without a breaking change.
#[cfg(feature = "pth")]
#[allow(clippy::unnecessary_wraps)]
const fn pth_dtype_to_hub(dtype: crate::PthDtype) -> crate::Result<Dtype> {
    use crate::PthDtype;
    Ok(match dtype {
        PthDtype::F16 => Dtype::F16,
        PthDtype::BF16 => Dtype::BF16,
        PthDtype::F32 => Dtype::F32,
        PthDtype::F64 => Dtype::F64,
        PthDtype::U8 => Dtype::U8,
        PthDtype::I8 => Dtype::I8,
        PthDtype::I16 => Dtype::I16,
        PthDtype::I32 => Dtype::I32,
        PthDtype::I64 => Dtype::I64,
        PthDtype::Bool => Dtype::Bool,
    })
}

/// Maps a **scalar** [`GgufType`](crate::GgufType) to the hub dtype.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] for a quantised or otherwise
/// non-scalar `GGUF` type (callers dequantise those before reaching here).
#[cfg(feature = "gguf")]
fn gguf_type_to_hub(dtype: crate::GgufType) -> crate::Result<Dtype> {
    use crate::GgufType;
    // EXHAUSTIVE: `GgufType` is `#[non_exhaustive]`; the wildcard covers future
    // block types, which are quantised and never reach this mapping.
    #[allow(clippy::wildcard_enum_match_arm)]
    match dtype {
        GgufType::F32 => Ok(Dtype::F32),
        GgufType::F16 => Ok(Dtype::F16),
        GgufType::BF16 => Ok(Dtype::BF16),
        GgufType::F64 => Ok(Dtype::F64),
        GgufType::I8 => Ok(Dtype::I8),
        GgufType::I16 => Ok(Dtype::I16),
        GgufType::I32 => Ok(Dtype::I32),
        GgufType::I64 => Ok(Dtype::I64),
        other => Err(AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("no scalar hub dtype for {other}"),
        }),
    }
}

/// Maps a hub dtype to a scalar [`GgufType`](crate::GgufType) for the `GGUF`
/// writer.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] for dtypes outside the `GGUF` scalar
/// surface (`Bool`, unsigned integers wider than nothing, and `FP8`).
#[cfg(feature = "gguf")]
fn hub_dtype_to_gguf(dtype: Dtype) -> crate::Result<crate::GgufType> {
    use crate::GgufType;
    match dtype {
        Dtype::F32 => Ok(GgufType::F32),
        Dtype::F16 => Ok(GgufType::F16),
        Dtype::BF16 => Ok(GgufType::BF16),
        Dtype::F64 => Ok(GgufType::F64),
        Dtype::I8 => Ok(GgufType::I8),
        Dtype::I16 => Ok(GgufType::I16),
        Dtype::I32 => Ok(GgufType::I32),
        Dtype::I64 => Ok(GgufType::I64),
        Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::Bool
        | Dtype::U8
        | Dtype::U16
        | Dtype::U32
        | Dtype::U64 => Err(AnamnesisError::Unsupported {
            format: "gguf".into(),
            detail: format!(
                "no GGUF dtype counterpart for {dtype} \
                 (Bool/unsigned-integer/FP8 are not in the GGUF scalar surface)"
            ),
        }),
    }
}

/// Converts a float tensor's bytes to `BF16` for the `BnB-NF4` encoder.
/// `BF16` input is returned unchanged; `F32` / `F16` are truncated to the upper
/// 16 bits, matching the crate's `f32_bits_to_bf16_bits` convention.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] for a non-float dtype and
/// [`AnamnesisError::Parse`] if the byte count is not a whole number of
/// elements.
#[cfg(feature = "bnb")]
fn to_bf16_bytes(data: &[u8], dtype: Dtype, name: &str) -> crate::Result<Vec<u8>> {
    match dtype {
        Dtype::BF16 => Ok(data.to_vec()),
        Dtype::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "bnb-nf4 `{name}`: F32 byte count {} is not a multiple of 4",
                        data.len()
                    ),
                });
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(4) {
                // INDEX: `chunks_exact(4)` guarantees exactly 4 bytes per chunk.
                #[allow(clippy::indexing_slicing)]
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let bits = u32::from_le_bytes(arr);
                // CAST: BF16 is the upper 16 bits of an f32 — truncation intended.
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                let bf16 = (bits >> 16) as u16;
                out.extend_from_slice(&bf16.to_le_bytes());
            }
            Ok(out)
        }
        Dtype::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(AnamnesisError::Parse {
                    reason: format!(
                        "bnb-nf4 `{name}`: F16 byte count {} is not a multiple of 2",
                        data.len()
                    ),
                });
            }
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(2) {
                // INDEX: `chunks_exact(2)` guarantees exactly 2 bytes per chunk.
                #[allow(clippy::indexing_slicing)]
                let arr: [u8; 2] = [chunk[0], chunk[1]];
                let bits = half::f16::from_le_bytes(arr).to_f32().to_bits();
                // CAST: BF16 is the upper 16 bits of an f32 — truncation intended.
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                let bf16 = (bits >> 16) as u16;
                out.extend_from_slice(&bf16.to_le_bytes());
            }
            Ok(out)
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
            format: "bnb-nf4".into(),
            detail: format!(
                "tensor `{name}` has dtype {dtype}; only F32/F16/BF16 inputs are \
                 supported for BnB-NF4 conversion"
            ),
        }),
    }
}
