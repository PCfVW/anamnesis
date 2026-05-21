// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command-line interface implementation, shared by the `anamnesis` and
//! `amn` binaries.
//!
//! Feature-gated behind `cli`; pulls in `clap` for argument parsing.
//! The two binary entry points (`src/bin/anamnesis.rs` and
//! `src/bin/amn.rs`) are 5-line wrappers that each delegate to `run`,
//! so the actual CLI code compiles exactly once and links into both
//! binaries instead of being compiled twice as two separate crate
//! roots (which is what the previous shared-`src/bin/main.rs` shape
//! did, producing the Cargo *"file found to be present in multiple
//! build targets"* warning on every invocation).

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use crate::{format_bytes, parse, InspectInfo, TargetDtype};

/// Parse any format, recover any precision.
#[derive(Parser)]
#[command(name = "anamnesis", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse and summarize a model file.
    Parse {
        /// Path to the model file (`.safetensors`, `.pth`, `.pt`, `.bin`, `.gguf`).
        path: PathBuf,
    },
    /// Inspect format, tensor counts, and size estimates.
    #[command(alias = "info")]
    Inspect {
        /// Path to the model file.
        path: PathBuf,
    },
    /// Dequantize (recover precision) or convert to a target format.
    #[command(alias = "dequantize")]
    Remember {
        /// Path to the input model file.
        path: PathBuf,
        /// Target dtype (currently only `bf16`) or `safetensors` for `.pth`/`.gguf` conversion.
        #[arg(long, default_value = "bf16")]
        to: String,
        /// Output file path (derived from input if omitted).
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Convert a model file to a different format.
    ///
    /// Targets available in this build (Phase 6):
    /// - `safetensors` (alias `bf16`) — dequantise any quantised input to a
    ///   BF16 safetensors file (passes through unquantised inputs losslessly).
    /// - `gguf` — write an unquantised GGUF file. Quantised GGUF emit
    ///   (`gguf-q4km`, …) is deferred to Phase 7.5 via the same dispatch.
    /// - `bnb-nf4` — encode the BF16 source into a BitsAndBytes-NF4
    ///   safetensors file (2-D tensors only; biases / norms / embeddings
    ///   pass through unchanged in BF16).
    Convert {
        /// Path to the input model file.
        path: PathBuf,
        /// Target format. Accepted values: `safetensors`/`bf16`, `gguf`,
        /// `bnb-nf4` (case-insensitive).
        #[arg(long)]
        to: String,
        /// Output file path (derived from input if omitted).
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

/// Detected model file format.
enum Format {
    Safetensors,
    #[cfg(feature = "pth")]
    Pth,
    #[cfg(feature = "npz")]
    Npz,
    #[cfg(feature = "gguf")]
    Gguf,
}

/// Builds an `AnamnesisError::Unsupported` explaining that the input
/// matched a format whose Cargo feature is disabled in the current build.
///
/// `format_name` is the user-facing format name (e.g., `"PyTorch"`,
/// `"GGUF"`). `kind` describes how the format was detected (extension or
/// magic bytes). `feature_flag` is the Cargo feature that enables the
/// corresponding parser.
///
/// Compiled only when at least one of `pth`, `npz`, `gguf` is **not**
/// enabled — when the binary is built with all three the helper has no
/// callers and triggers `dead_code`.
#[cfg(not(all(feature = "pth", feature = "npz", feature = "gguf")))]
fn missing_feature_err(format_name: &str, kind: &str, feature_flag: &str) -> crate::AnamnesisError {
    crate::AnamnesisError::Unsupported {
        format: format_name.into(),
        detail: format!(
            "input is {kind} but the `{feature_flag}` Cargo feature is not enabled in this \
             build — rebuild with `cargo install anamnesis --features cli,{feature_flag}` \
             (or `cargo build --features cli,{feature_flag}`) to add support"
        ),
    }
}

/// Detects the model format from file extension and magic bytes.
///
/// `.safetensors` → `Safetensors`. `.pth`/`.pt` → `Pth`. `.npz` → `Npz`.
/// `.gguf` → `Gguf`. `.bin` → check ZIP magic (`PK\x03\x04`) for
/// `PyTorch`, then `GGUF` magic. Unknown extensions try `GGUF` magic
/// before defaulting to `Safetensors`.
///
/// # Errors
///
/// Returns `AnamnesisError::Unsupported` when the input matches a
/// format whose Cargo feature (`pth`, `npz`, or `gguf`) is not enabled
/// in this build. This replaces the previous silent fall-through to
/// `Safetensors`, which produced cryptic downstream errors when users
/// pointed the CLI at a `.pth` / `.npz` / `.gguf` file built without
/// the matching feature.
// `clippy::unnecessary_wraps`: when all three of `pth`, `npz`, `gguf`
// are enabled every branch is `Ok(_)` and clippy can't see that other
// feature combinations make the wrap load-bearing. This allow keeps
// the all-features clippy build clean without weakening detection
// elsewhere.
#[allow(clippy::unnecessary_wraps)]
fn detect_format(path: &std::path::Path) -> crate::Result<Format> {
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
            if has_zip_magic(path) {
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
            if has_gguf_magic(path) {
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
            if has_gguf_magic(path) {
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

/// Returns `true` if the file starts with the ZIP local header magic
/// `PK\x03\x04`. Reads only 4 bytes — does not load the file into memory.
///
/// Always available (no feature gate) because [`detect_format`] needs to
/// detect `.bin`-extension `PyTorch` archives even in builds where the
/// `pth` feature is disabled, so it can return a helpful "feature not
/// enabled" error instead of silently misrouting to the safetensors
/// parser.
fn has_zip_magic(path: &std::path::Path) -> bool {
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            f.read_exact(&mut buf)
        })
        .is_ok_and(|()| buf == *b"PK\x03\x04")
}

/// Returns `true` if the file starts with the `GGUF` magic bytes
/// (`"GGUF"`). Reads only 4 bytes — does not load the file into memory.
///
/// Always available (no feature gate) for the same reason as
/// [`has_zip_magic`]: feature-disabled builds still need to recognise
/// GGUF files and produce a helpful error rather than silently falling
/// through to the safetensors parser.
fn has_gguf_magic(path: &std::path::Path) -> bool {
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            f.read_exact(&mut buf)
        })
        .is_ok_and(|()| buf == *b"GGUF")
}

// ---------------------------------------------------------------------------
// Subcommand runners
// ---------------------------------------------------------------------------

/// Parses CLI arguments and dispatches to the appropriate subcommand
/// runner.
///
/// Entry point shared by the `anamnesis` and `amn` binaries; both thin
/// wrappers under `src/bin/` call this function and translate any
/// returned error into a `process::exit(1)`.
///
/// # Errors
///
/// Propagates any [`crate::AnamnesisError`] returned by the underlying
/// format parsers, dequantisation kernels, or output writers.
pub fn run() -> crate::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { path } => run_parse(&path),
        Commands::Inspect { path } => run_inspect(&path),
        Commands::Remember { path, to, output } => run_remember(&path, &to, output.as_deref()),
        Commands::Convert { path, to, output } => run_convert(&path, &to, output.as_deref()),
    }
}

fn run_parse(path: &std::path::Path) -> crate::Result<()> {
    match detect_format(path)? {
        Format::Safetensors => run_parse_safetensors(path),
        #[cfg(feature = "pth")]
        Format::Pth => run_parse_pth(path),
        #[cfg(feature = "npz")]
        Format::Npz => run_parse_npz(path),
        #[cfg(feature = "gguf")]
        Format::Gguf => run_parse_gguf(path),
    }
}

fn run_parse_safetensors(path: &std::path::Path) -> crate::Result<()> {
    let model = parse(path)?;
    let info = InspectInfo::from(&model.header);
    let total = model.header.tensors.len();

    println!("{total} tensors parsed");

    let quantized = model.header.quantized_count();
    if quantized > 0 {
        println!("  {quantized:>3} quantized   {}", model.header.scheme);
    }

    let scales = model.header.scale_count();
    if scales > 0 {
        let mut dtypes: Vec<String> = Vec::new();
        for entry in model.header.scale_tensors() {
            let s = entry.dtype.to_string();
            if !dtypes.contains(&s) {
                dtypes.push(s);
            }
        }
        let dtype_list = dtypes.join(", ");
        println!("  {scales:>3} scale       {dtype_list}");
    }

    let zeropoints = model.header.zeropoint_count();
    if zeropoints > 0 {
        println!("  {zeropoints:>3} zero-point  I32 (packed)");
    }

    let group_indices = model.header.group_index_count();
    if group_indices > 0 {
        println!("  {group_indices:>3} g_idx       I32 (activation-order)");
    }

    let passthrough = model.header.passthrough_count();
    if passthrough > 0 {
        // Collect passthrough dtype summary.
        let mut dtypes: Vec<String> = Vec::new();
        for entry in model.header.passthrough_tensors() {
            let s = entry.dtype.to_string();
            if !dtypes.contains(&s) {
                dtypes.push(s);
            }
        }
        let dtype_list = dtypes.join(", ");
        println!("  {passthrough:>3} passthrough {dtype_list} (norms, embeddings, lm_head)");
    }

    println!("File: {}", format_bytes(info.current_size));
    Ok(())
}

#[cfg(feature = "pth")]
fn run_parse_pth(path: &std::path::Path) -> crate::Result<()> {
    let parsed = crate::parse_pth(path)?;
    let info = parsed.inspect();
    // Use tensor_info() (metadata only) instead of tensors() — avoids
    // materializing tensor data just for the display path.
    let tensor_info = parsed.tensor_info();

    println!(
        "Parsed {} (PyTorch state_dict)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)")
    );
    println!("  Tensors:    {}", info.tensor_count);
    println!("  Total size: {}", format_bytes(info.total_bytes));
    let dtype_list: String = info
        .dtypes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    println!("  Dtypes:     {dtype_list}");
    let endian = if info.big_endian {
        "big-endian"
    } else {
        "little-endian"
    };
    println!("  Byte order: {endian}");
    println!();

    for t in &tensor_info {
        let shape_str = format!("{:?}", t.shape);
        // CAST: usize → u64, tensor byte lengths fit in u64
        #[allow(clippy::as_conversions)]
        let byte_len = t.byte_len as u64;
        println!(
            "  {:<30} {:<6} {:<15} {}",
            t.name,
            t.dtype,
            shape_str,
            format_bytes(byte_len)
        );
    }
    Ok(())
}

#[cfg(feature = "npz")]
fn run_parse_npz(path: &std::path::Path) -> crate::Result<()> {
    // Use inspect_npz (header-only) instead of parse_npz — avoids loading
    // all tensor data into memory for a display-only operation.
    let info = crate::inspect_npz(path)?;

    println!(
        "Parsed {} (NPZ archive)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)")
    );
    println!("  Tensors:    {}", info.tensors.len());
    println!("  Total size: {}", format_bytes(info.total_bytes));
    println!();

    for t in &info.tensors {
        let shape_str = format!("{:?}", t.shape);
        // CAST: usize → u64, tensor byte lengths fit
        #[allow(clippy::as_conversions)]
        let byte_len = t.byte_len as u64;
        println!(
            "  {:<30} {:<6} {:<15} {}",
            t.name,
            t.dtype,
            shape_str,
            format_bytes(byte_len)
        );
    }
    Ok(())
}

#[cfg(feature = "npz")]
fn run_inspect_npz(path: &std::path::Path) -> crate::Result<()> {
    // Header-only — no tensor data loaded.
    let info = crate::inspect_npz(path)?;
    println!("{info}");
    Ok(())
}

fn run_inspect(path: &std::path::Path) -> crate::Result<()> {
    match detect_format(path)? {
        Format::Safetensors => {
            let model = parse(path)?;
            let info = InspectInfo::from(&model.header);
            println!("{info}");
        }
        #[cfg(feature = "pth")]
        Format::Pth => {
            let parsed = crate::parse_pth(path)?;
            let info = parsed.inspect();
            println!("{info}");
        }
        #[cfg(feature = "npz")]
        Format::Npz => run_inspect_npz(path)?,
        #[cfg(feature = "gguf")]
        Format::Gguf => {
            let parsed = crate::parse_gguf(path)?;
            let info = parsed.inspect();
            println!("{info}");
        }
    }
    Ok(())
}

fn run_remember(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
) -> crate::Result<()> {
    match detect_format(path)? {
        Format::Safetensors => run_remember_safetensors(path, to, output),
        #[cfg(feature = "pth")]
        Format::Pth => {
            let to_lower = to.to_ascii_lowercase();
            if to_lower != "safetensors" && to_lower != "bf16" {
                return Err(crate::AnamnesisError::Unsupported {
                    format: "pth".into(),
                    detail: format!(
                        "unsupported --to value `{to}` for .pth files \
                         (supported: `safetensors`, `bf16` — .pth conversion \
                         always produces safetensors)"
                    ),
                });
            }
            run_remember_pth(path, output)
        }
        #[cfg(feature = "npz")]
        Format::Npz => Err(crate::AnamnesisError::Unsupported {
            format: "NPZ".into(),
            detail: "NPZ tensors are already full-precision; \
                     no dequantization or conversion needed"
                .into(),
        }),
        #[cfg(feature = "gguf")]
        Format::Gguf => {
            let to_lower = to.to_ascii_lowercase();
            if to_lower != "safetensors" && to_lower != "bf16" {
                return Err(crate::AnamnesisError::Unsupported {
                    format: "GGUF".into(),
                    detail: format!(
                        "unsupported --to value `{to}` for .gguf files \
                         (supported: `safetensors`, `bf16`)"
                    ),
                });
            }
            run_remember_gguf(path, output)
        }
    }
}

fn run_remember_safetensors(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
) -> crate::Result<()> {
    let target: TargetDtype = to.parse()?;

    let model = parse(path)?;
    let info = InspectInfo::from(&model.header);

    let total = model.header.tensors.len();
    let quantized = model.header.quantized_count();
    println!("Parsing...  {total} tensors, {}", model.header.scheme);

    let output_path = match output {
        Some(p) => p.to_owned(),
        None => derive_output_path(path, target),
    };

    #[cfg(feature = "indicatif")]
    {
        use indicatif::{ProgressBar, ProgressStyle};

        // CAST: usize → u64, tensor count fits in u64
        #[allow(clippy::as_conversions)]
        let pb = ProgressBar::new(quantized as u64);
        let style = ProgressStyle::with_template("Recalling... {pos} tensors [{bar:20}] {elapsed}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=> ");
        pb.set_style(style);
        model.remember_with_progress(&output_path, target, || pb.inc(1))?;
        pb.finish();
        println!();
    }

    #[cfg(not(feature = "indicatif"))]
    {
        println!("Recalling... {quantized} tensors");
        model.remember(&output_path, target)?;
    }

    println!(
        "Output: {} ({})",
        output_path.display(),
        format_bytes(info.dequantized_size),
    );
    Ok(())
}

#[cfg(feature = "pth")]
fn run_remember_pth(path: &std::path::Path, output: Option<&std::path::Path>) -> crate::Result<()> {
    let parsed = crate::parse_pth(path)?;
    let info = parsed.inspect();

    let output_path = if let Some(p) = output {
        p.to_owned()
    } else {
        // Replace extension: model.pth → model.safetensors
        let mut out = path.to_owned();
        out.set_extension("safetensors");
        out
    };

    println!(
        "Converting {} → {}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );
    println!(
        "  {} tensors, {}",
        info.tensor_count,
        format_bytes(info.total_bytes)
    );

    parsed.to_safetensors(&output_path)?;
    println!("  Done.");
    Ok(())
}

#[cfg(feature = "gguf")]
fn run_parse_gguf(path: &std::path::Path) -> crate::Result<()> {
    let parsed = crate::parse_gguf(path)?;
    let info = parsed.inspect();
    let tensor_info = parsed.tensor_info();

    println!(
        "Parsed {} (GGUF v{})",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)"),
        info.version
    );
    if let Some(arch) = info.architecture.as_deref() {
        println!("  Arch:       {arch}");
    }
    println!("  Tensors:    {}", info.tensor_count);
    println!("  Total size: {}", format_bytes(info.total_bytes));
    let dtype_list: String = info
        .dtypes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    println!("  Dtypes:     {dtype_list}");
    println!("  Alignment:  {} bytes", info.alignment);
    println!();

    for t in tensor_info {
        let shape_str = format!("{:?}", t.shape);
        let byte_len_str = t.byte_len.map_or_else(|| "?".into(), format_bytes);
        println!(
            "  {:<40} {:<8} {:<15} {}",
            t.name, t.dtype, shape_str, byte_len_str
        );
    }
    Ok(())
}

#[cfg(feature = "gguf")]
fn run_remember_gguf(
    path: &std::path::Path,
    output: Option<&std::path::Path>,
) -> crate::Result<()> {
    let parsed = crate::parse_gguf(path)?;
    let info = parsed.inspect();

    let output_path = if let Some(p) = output {
        p.to_owned()
    } else {
        let mut out = path.to_owned();
        out.set_extension("safetensors");
        out
    };

    println!(
        "Converting {} → {}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );
    println!("  {} tensors", info.tensor_count);

    // Dequantize quantized tensors to BF16; pass through non-quantized
    // tensors (F32, F16, BF16, integer types) with their original dtype.
    // Collect owned data because TensorView borrows data and all views
    // must be alive simultaneously for serialize_to_file.
    let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>, safetensors::Dtype)> =
        Vec::with_capacity(info.tensor_count);
    let mut dequantized_count: usize = 0;

    for tensor in parsed.tensors() {
        // GGUF shape is most-significant-first; safetensors expects
        // row-major (NumPy-style). Reverse the dimensions.
        let mut shape: Vec<usize> = tensor.shape.to_vec();
        shape.reverse();

        if tensor.dtype.is_quantized() {
            let n_elements: usize = tensor
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| crate::AnamnesisError::Parse {
                    reason: format!(
                        "GGUF tensor `{}` shape {:?} element count overflows usize",
                        tensor.name, tensor.shape
                    ),
                })?;
            let bf16_data = crate::dequantize_gguf_to_bf16(&tensor.data, tensor.dtype, n_elements)?;
            tensor_data.push((
                tensor.name.to_owned(),
                bf16_data,
                shape,
                safetensors::Dtype::BF16,
            ));
            dequantized_count += 1;
        } else {
            let st_dtype = gguf_type_to_safetensors_dtype(tensor.dtype)?;
            // BORROW: `.into_owned()` copies borrowed mmap bytes into an
            // owned Vec so the data outlives the parsed borrow.
            tensor_data.push((
                tensor.name.to_owned(),
                tensor.data.into_owned(),
                shape,
                st_dtype,
            ));
        }
    }

    println!(
        "  {} dequantized to BF16, {} passed through",
        dequantized_count,
        tensor_data.len() - dequantized_count
    );

    // Build TensorView list and serialize to file.
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        tensor_data
            .iter()
            .map(|(name, data, shape, dtype)| {
                let view = safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                    .map_err(|e| crate::AnamnesisError::Parse {
                        reason: format!("failed to create TensorView for `{name}`: {e}"),
                    })?;
                Ok((name.clone(), view))
            })
            .collect::<crate::Result<Vec<_>>>()?;

    safetensors::tensor::serialize_to_file(views, &None, output_path.as_ref()).map_err(
        // EXHAUSTIVE: SafeTensorError is a foreign type that may gain variants
        #[allow(clippy::wildcard_enum_match_arm)]
        |e| match e {
            safetensors::SafeTensorError::IoError(io_err) => crate::AnamnesisError::Io(io_err),
            other => crate::AnamnesisError::Parse {
                reason: format!("failed to write safetensors file: {other}"),
            },
        },
    )?;

    println!("  Output: {}", output_path.display());
    Ok(())
}

/// Maps a non-quantized [`GgufType`](crate::GgufType) to the corresponding
/// `safetensors::Dtype`.
#[cfg(feature = "gguf")]
fn gguf_type_to_safetensors_dtype(dtype: crate::GgufType) -> crate::Result<safetensors::Dtype> {
    // EXHAUSTIVE: GgufType is a foreign #[non_exhaustive] enum — new
    // variants may be added. The wildcard covers future types.
    #[allow(clippy::wildcard_enum_match_arm)]
    match dtype {
        crate::GgufType::F32 => Ok(safetensors::Dtype::F32),
        crate::GgufType::F16 => Ok(safetensors::Dtype::F16),
        crate::GgufType::BF16 => Ok(safetensors::Dtype::BF16),
        crate::GgufType::F64 => Ok(safetensors::Dtype::F64),
        crate::GgufType::I8 => Ok(safetensors::Dtype::I8),
        crate::GgufType::I16 => Ok(safetensors::Dtype::I16),
        crate::GgufType::I32 => Ok(safetensors::Dtype::I32),
        crate::GgufType::I64 => Ok(safetensors::Dtype::I64),
        other => Err(crate::AnamnesisError::Unsupported {
            format: "GGUF".into(),
            detail: format!("no safetensors equivalent for {other}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// `convert` subcommand
// ---------------------------------------------------------------------------

/// Parsed `--to` value for the [`Commands::Convert`] dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConvertTarget {
    /// `safetensors` (alias `bf16`) — dequantise to BF16 safetensors or
    /// passthrough non-quantised inputs to safetensors losslessly.
    Safetensors,
    /// `gguf` — write an unquantised GGUF file. Phase 6 scalar passthrough
    /// only; `gguf-q4km` etc. land in Phase 7.5.
    Gguf,
    /// `bnb-nf4` — encode BF16 source to BitsAndBytes-NF4 safetensors.
    BnbNf4,
}

impl ConvertTarget {
    fn parse(raw: &str) -> crate::Result<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "safetensors" | "bf16" => Ok(Self::Safetensors),
            "gguf" => Ok(Self::Gguf),
            "bnb-nf4" | "bnb_nf4" | "nf4" => Ok(Self::BnbNf4),
            other => Err(crate::AnamnesisError::Unsupported {
                format: other.to_owned(),
                detail: "supported convert targets: `safetensors` (alias `bf16`), \
                         `gguf`, `bnb-nf4`. Quantised GGUF targets land in Phase 7.5."
                    .into(),
            }),
        }
    }

    const fn extension(self) -> &'static str {
        match self {
            Self::Safetensors | Self::BnbNf4 => "safetensors",
            Self::Gguf => "gguf",
        }
    }

    const fn suffix(self) -> &'static str {
        match self {
            Self::Safetensors => "bf16",
            Self::Gguf => "gguf",
            Self::BnbNf4 => "bnb-nf4",
        }
    }
}

/// Derives an output path for `convert` from the input path and target.
///
/// Strips known quantisation suffixes (same table as
/// [`derive_output_path`]) and appends `-{suffix}.{ext}` from
/// [`ConvertTarget`].
fn derive_convert_output_path(input: &std::path::Path, target: ConvertTarget) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let clean_stem = QUANT_SUFFIXES
        .iter()
        .find_map(|qs| stem.strip_suffix(qs))
        .unwrap_or(stem);
    let new_name = format!("{clean_stem}-{}.{}", target.suffix(), target.extension());
    input
        .parent()
        .map_or_else(|| PathBuf::from(&new_name), |p| p.join(&new_name))
}

#[cfg(any(feature = "pth", feature = "npz", feature = "gguf"))]
fn unsupported_combination(input_label: &str, target_label: &str) -> crate::AnamnesisError {
    crate::AnamnesisError::Unsupported {
        format: format!("{input_label}->{target_label}"),
        detail: format!(
            "convert {input_label} -> {target_label} is not yet implemented \
             (not part of the v0.6.0 Phase 6 conversion matrix)"
        ),
    }
}

fn run_convert(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
) -> crate::Result<()> {
    let target = ConvertTarget::parse(to)?;
    let output_path =
        output.map_or_else(|| derive_convert_output_path(path, target), Path::to_owned);

    let fmt = detect_format(path)?;
    match (fmt, target) {
        // safetensors -> safetensors (BF16): reuse existing remember path
        (Format::Safetensors, ConvertTarget::Safetensors) => {
            run_remember_safetensors(path, "bf16", Some(&output_path))
        }
        // safetensors -> GGUF (requires gguf feature)
        #[cfg(feature = "gguf")]
        (Format::Safetensors, ConvertTarget::Gguf) => {
            run_convert_safetensors_to_gguf(path, &output_path)
        }
        #[cfg(not(feature = "gguf"))]
        (Format::Safetensors, ConvertTarget::Gguf) => Err(crate::AnamnesisError::Unsupported {
            format: "safetensors->gguf".into(),
            detail: "GGUF emit requires the `gguf` Cargo feature; rebuild with \
                     `--features cli,gguf`"
                .into(),
        }),
        // safetensors -> BnB-NF4 (requires bnb feature)
        #[cfg(feature = "bnb")]
        (Format::Safetensors, ConvertTarget::BnbNf4) => {
            run_convert_safetensors_to_bnb_nf4(path, &output_path)
        }
        #[cfg(not(feature = "bnb"))]
        (Format::Safetensors, ConvertTarget::BnbNf4) => Err(crate::AnamnesisError::Unsupported {
            format: "safetensors->bnb-nf4".into(),
            detail: "BnB-NF4 encode requires the `bnb` Cargo feature; rebuild with \
                     `--features cli,bnb`"
                .into(),
        }),
        #[cfg(feature = "pth")]
        (Format::Pth, ConvertTarget::Safetensors) => run_remember_pth(path, Some(&output_path)),
        #[cfg(all(feature = "pth", feature = "gguf"))]
        (Format::Pth, ConvertTarget::Gguf) => run_convert_pth_to_gguf(path, &output_path),
        #[cfg(all(feature = "pth", not(feature = "gguf")))]
        (Format::Pth, ConvertTarget::Gguf) => Err(crate::AnamnesisError::Unsupported {
            format: "pth->gguf".into(),
            detail: "GGUF emit requires the `gguf` Cargo feature".into(),
        }),
        #[cfg(feature = "pth")]
        (Format::Pth, ConvertTarget::BnbNf4) => Err(unsupported_combination("pth", "bnb-nf4")),
        #[cfg(feature = "npz")]
        (Format::Npz, ConvertTarget::Safetensors) => {
            run_convert_npz_to_safetensors(path, &output_path)
        }
        #[cfg(all(feature = "npz", feature = "gguf"))]
        (Format::Npz, ConvertTarget::Gguf) => run_convert_npz_to_gguf(path, &output_path),
        #[cfg(all(feature = "npz", not(feature = "gguf")))]
        (Format::Npz, ConvertTarget::Gguf) => Err(crate::AnamnesisError::Unsupported {
            format: "npz->gguf".into(),
            detail: "GGUF emit requires the `gguf` Cargo feature".into(),
        }),
        #[cfg(feature = "npz")]
        (Format::Npz, ConvertTarget::BnbNf4) => Err(unsupported_combination("npz", "bnb-nf4")),
        #[cfg(feature = "gguf")]
        (Format::Gguf, ConvertTarget::Safetensors) => run_remember_gguf(path, Some(&output_path)),
        #[cfg(feature = "gguf")]
        (Format::Gguf, ConvertTarget::Gguf) => Err(unsupported_combination("gguf", "gguf")),
        #[cfg(feature = "gguf")]
        (Format::Gguf, ConvertTarget::BnbNf4) => Err(unsupported_combination("gguf", "bnb-nf4")),
    }
}

#[cfg(feature = "gguf")]
fn run_convert_safetensors_to_gguf(
    path: &std::path::Path,
    output: &std::path::Path,
) -> crate::Result<()> {
    use crate::parse::gguf::GgufType;
    use crate::parse::gguf_write::{write_gguf, GgufWriteTensor};

    let model = parse(path)?;
    if model.header.scheme != crate::QuantScheme::Unquantized {
        return Err(crate::AnamnesisError::Unsupported {
            format: "safetensors->gguf".into(),
            detail: format!(
                "input is quantised ({}); dequantise to BF16 first \
                 via `amn remember --to bf16 -o tmp.safetensors`",
                model.header.scheme
            ),
        });
    }

    println!(
        "Converting {} -> {} (safetensors -> GGUF)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );

    // Collect (name, GgufType, shape_msb_first, owned_bytes) so the TensorViews stay alive.
    let mut owned: Vec<(String, GgufType, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(model.header.tensors.len());
    for entry in &model.header.tensors {
        let gguf_dtype = safetensors_dtype_to_gguf(entry.dtype)?;
        let data_offset = model.header.header_size + 8;
        let abs_start = data_offset
            .checked_add(entry.data_offsets.0)
            .ok_or_else(|| crate::AnamnesisError::Parse {
                reason: format!("safetensors->gguf `{}`: tensor offset overflow", entry.name),
            })?;
        let abs_end = data_offset
            .checked_add(entry.data_offsets.1)
            .ok_or_else(|| crate::AnamnesisError::Parse {
                reason: format!("safetensors->gguf `{}`: tensor end overflow", entry.name),
            })?;
        // We must own the data so the parsed model can drop while the
        // GgufWriteTensor list lives.
        let bytes = std::fs::read(path).map_err(crate::AnamnesisError::Io)?;
        let slice = bytes
            .get(abs_start..abs_end)
            .ok_or_else(|| crate::AnamnesisError::Parse {
                reason: format!(
                    "safetensors->gguf `{}`: tensor data offsets {abs_start}..{abs_end} \
                     out of bounds (file size {})",
                    entry.name,
                    bytes.len()
                ),
            })?;
        let owned_bytes = slice.to_vec();
        let mut msb_first_shape = entry.shape.clone();
        msb_first_shape.reverse();
        owned.push((entry.name.clone(), gguf_dtype, msb_first_shape, owned_bytes));
    }

    let tensors: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dtype, shape, data)| GgufWriteTensor {
            name: name.as_str(),
            shape: shape.as_slice(),
            dtype: *dtype,
            data: data.as_slice(),
        })
        .collect();
    write_gguf(output, &tensors, &std::collections::HashMap::new())?;
    println!("  Wrote {} tensors -> {}", tensors.len(), output.display());
    Ok(())
}

/// Maps an anamnesis safetensors [`crate::Dtype`] to a non-quantised
/// [`crate::GgufType`] for the GGUF writer. Errors on FP8 (quantised) and
/// any future dtype that lacks a `GgufType` counterpart.
#[cfg(feature = "gguf")]
fn safetensors_dtype_to_gguf(dtype: crate::Dtype) -> crate::Result<crate::GgufType> {
    use crate::Dtype;
    match dtype {
        Dtype::F32 => Ok(crate::GgufType::F32),
        Dtype::F16 => Ok(crate::GgufType::F16),
        Dtype::BF16 => Ok(crate::GgufType::BF16),
        Dtype::F64 => Ok(crate::GgufType::F64),
        Dtype::I8 => Ok(crate::GgufType::I8),
        Dtype::I16 => Ok(crate::GgufType::I16),
        Dtype::I32 => Ok(crate::GgufType::I32),
        Dtype::I64 => Ok(crate::GgufType::I64),
        Dtype::F8E4M3
        | Dtype::F8E5M2
        | Dtype::Bool
        | Dtype::U8
        | Dtype::U16
        | Dtype::U32
        | Dtype::U64 => Err(crate::AnamnesisError::Unsupported {
            format: "safetensors->gguf".into(),
            detail: format!(
                "no GGUF dtype counterpart for safetensors {dtype} \
                 (Bool/unsigned-integer/FP8 not in the GGUF scalar surface)"
            ),
        }),
    }
}

#[cfg(feature = "bnb")]
fn run_convert_safetensors_to_bnb_nf4(
    path: &std::path::Path,
    output: &std::path::Path,
) -> crate::Result<()> {
    use crate::lethe::bnb_writer::{classify_inputs, write_bnb_nf4_safetensors, BnbWriteInput};

    let model = parse(path)?;
    if model.header.scheme != crate::QuantScheme::Unquantized {
        return Err(crate::AnamnesisError::Unsupported {
            format: "safetensors->bnb-nf4".into(),
            detail: format!(
                "input is already quantised ({}); dequantise to BF16 first \
                 via `amn remember --to bf16 -o tmp.safetensors`",
                model.header.scheme
            ),
        });
    }

    println!(
        "Converting {} -> {} (safetensors -> BnB-NF4)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );

    let raw_bytes = std::fs::read(path).map_err(crate::AnamnesisError::Io)?;
    let data_offset = model.header.header_size + 8;

    // Resolve each tensor's bytes; convert to BF16 if it's an F16/F32 float.
    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(model.header.tensors.len());
    for entry in &model.header.tensors {
        let abs_start = data_offset
            .checked_add(entry.data_offsets.0)
            .ok_or_else(|| crate::AnamnesisError::Parse {
                reason: format!("safetensors->bnb-nf4 `{}`: offset overflow", entry.name),
            })?;
        let abs_end = data_offset
            .checked_add(entry.data_offsets.1)
            .ok_or_else(|| crate::AnamnesisError::Parse {
                reason: format!("safetensors->bnb-nf4 `{}`: end overflow", entry.name),
            })?;
        let slice =
            raw_bytes
                .get(abs_start..abs_end)
                .ok_or_else(|| crate::AnamnesisError::Parse {
                    reason: format!(
                        "safetensors->bnb-nf4 `{}`: tensor bytes out of file bounds",
                        entry.name
                    ),
                })?;
        let bf16 = float_to_bf16_bytes(slice, entry.dtype, &entry.name)?;
        owned.push((entry.name.clone(), entry.shape.clone(), bf16));
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
    println!(
        "  {} quantised to NF4, {} passed through as BF16",
        stats.quantized, stats.passthrough
    );
    write_bnb_nf4_safetensors(&inputs, output)?;
    println!("  Wrote -> {}", output.display());
    Ok(())
}

/// Converts a raw float byte slice (`F32`/`F16`/`BF16`) to a `BF16` byte
/// vector. Truncation from `F32` follows the same upper-16-bits convention
/// used by [`remember::fp8::f32_bits_to_bf16_bits`](crate::remember::fp8).
#[cfg(feature = "bnb")]
fn float_to_bf16_bytes(data: &[u8], dtype: crate::Dtype, name: &str) -> crate::Result<Vec<u8>> {
    use crate::Dtype;
    match dtype {
        Dtype::BF16 => Ok(data.to_vec()),
        Dtype::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(crate::AnamnesisError::Parse {
                    reason: format!(
                        "safetensors->bnb-nf4 `{name}`: F32 byte count {} not a multiple of 4",
                        data.len()
                    ),
                });
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(4) {
                // INDEX: chunks_exact(4) guarantees exactly 4 bytes per chunk
                #[allow(clippy::indexing_slicing)]
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let bits = u32::from_le_bytes(arr);
                // BITWISE: BF16 = upper 16 bits of f32 (truncate)
                // CAST: u32 -> u16 truncation is intentional (bf16 is upper 16 bits)
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                let bf16 = (bits >> 16) as u16;
                out.extend_from_slice(&bf16.to_le_bytes());
            }
            Ok(out)
        }
        Dtype::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(crate::AnamnesisError::Parse {
                    reason: format!(
                        "safetensors->bnb-nf4 `{name}`: F16 byte count {} not a multiple of 2",
                        data.len()
                    ),
                });
            }
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(2) {
                // INDEX: chunks_exact(2) guarantees exactly 2 bytes per chunk
                #[allow(clippy::indexing_slicing)]
                let arr: [u8; 2] = [chunk[0], chunk[1]];
                let f = half::f16::from_le_bytes(arr).to_f32();
                let bits = f.to_bits();
                // CAST: u32 -> u16 truncation is intentional (bf16 is upper 16 bits)
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
        | Dtype::I64 => Err(crate::AnamnesisError::Unsupported {
            format: "safetensors->bnb-nf4".into(),
            detail: format!(
                "tensor `{name}` has dtype {dtype}; only F32/F16/BF16 inputs \
                 are supported for BnB-NF4 conversion in this build"
            ),
        }),
    }
}

#[cfg(all(feature = "pth", feature = "gguf"))]
fn run_convert_pth_to_gguf(path: &std::path::Path, output: &std::path::Path) -> crate::Result<()> {
    use crate::parse::gguf::GgufType;
    use crate::parse::gguf_write::{write_gguf, GgufWriteTensor};

    let parsed = crate::parse_pth(path)?;
    let pth_tensors = parsed.tensors()?;

    println!(
        "Converting {} -> {} (pth -> GGUF)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );

    let mut owned: Vec<(String, GgufType, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(pth_tensors.len());
    for t in &pth_tensors {
        let gguf_dtype = pth_dtype_to_gguf(t.dtype, &t.name)?;
        let mut msb_first = t.shape.clone();
        msb_first.reverse();
        owned.push((t.name.clone(), gguf_dtype, msb_first, t.data.to_vec()));
    }
    let tensors: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dtype, shape, data)| GgufWriteTensor {
            name: name.as_str(),
            shape: shape.as_slice(),
            dtype: *dtype,
            data: data.as_slice(),
        })
        .collect();
    write_gguf(output, &tensors, &std::collections::HashMap::new())?;
    println!("  Wrote {} tensors -> {}", tensors.len(), output.display());
    Ok(())
}

#[cfg(all(feature = "pth", feature = "gguf"))]
fn pth_dtype_to_gguf(dtype: crate::PthDtype, name: &str) -> crate::Result<crate::GgufType> {
    use crate::PthDtype;
    match dtype {
        PthDtype::F32 => Ok(crate::GgufType::F32),
        PthDtype::F16 => Ok(crate::GgufType::F16),
        PthDtype::BF16 => Ok(crate::GgufType::BF16),
        PthDtype::F64 => Ok(crate::GgufType::F64),
        PthDtype::I8 => Ok(crate::GgufType::I8),
        PthDtype::I16 => Ok(crate::GgufType::I16),
        PthDtype::I32 => Ok(crate::GgufType::I32),
        PthDtype::I64 => Ok(crate::GgufType::I64),
        PthDtype::U8 | PthDtype::Bool => Err(crate::AnamnesisError::Unsupported {
            format: "pth->gguf".into(),
            detail: format!(
                "tensor `{name}` has dtype {dtype} which has no GGUF scalar counterpart"
            ),
        }),
    }
}

#[cfg(feature = "npz")]
fn run_convert_npz_to_safetensors(
    path: &std::path::Path,
    output: &std::path::Path,
) -> crate::Result<()> {
    let map = crate::parse_npz(path)?;
    println!(
        "Converting {} -> {} (NPZ -> safetensors)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );
    crate::npz_to_safetensors(&map, output)?;
    println!("  Wrote {} tensors -> {}", map.len(), output.display());
    Ok(())
}

#[cfg(all(feature = "npz", feature = "gguf"))]
fn run_convert_npz_to_gguf(path: &std::path::Path, output: &std::path::Path) -> crate::Result<()> {
    use crate::parse::gguf::GgufType;
    use crate::parse::gguf_write::{write_gguf, GgufWriteTensor};
    use crate::parse::npz::NpzDtype;

    let map = crate::parse_npz(path)?;
    println!(
        "Converting {} -> {} (NPZ -> GGUF)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );

    let mut names: Vec<&String> = map.keys().collect();
    names.sort();
    let mut owned: Vec<(String, GgufType, Vec<usize>, Vec<u8>)> = Vec::with_capacity(map.len());
    for name in &names {
        let t = map.get(*name).ok_or_else(|| crate::AnamnesisError::Parse {
            reason: format!("NPZ tensor `{name}` vanished mid-iteration"),
        })?;
        let gguf_dtype = match t.dtype {
            NpzDtype::F32 => GgufType::F32,
            NpzDtype::F16 => GgufType::F16,
            NpzDtype::BF16 => GgufType::BF16,
            NpzDtype::F64 => GgufType::F64,
            NpzDtype::I8 => GgufType::I8,
            NpzDtype::I16 => GgufType::I16,
            NpzDtype::I32 => GgufType::I32,
            NpzDtype::I64 => GgufType::I64,
            NpzDtype::Bool | NpzDtype::U8 | NpzDtype::U16 | NpzDtype::U32 | NpzDtype::U64 => {
                return Err(crate::AnamnesisError::Unsupported {
                    format: "npz->gguf".into(),
                    detail: format!(
                        "NPZ tensor `{name}` has dtype {} which has no GGUF scalar counterpart",
                        t.dtype
                    ),
                });
            }
        };
        let mut msb_first = t.shape.clone();
        msb_first.reverse();
        owned.push(((*name).clone(), gguf_dtype, msb_first, t.data.clone()));
    }
    let tensors: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dtype, shape, data)| GgufWriteTensor {
            name: name.as_str(),
            shape: shape.as_slice(),
            dtype: *dtype,
            data: data.as_slice(),
        })
        .collect();
    write_gguf(output, &tensors, &std::collections::HashMap::new())?;
    println!("  Wrote {} tensors -> {}", tensors.len(), output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Output path derivation
// ---------------------------------------------------------------------------

/// Known quantization suffixes stripped from input filenames when deriving output paths.
///
/// Case-sensitive: common conventions use lowercase, uppercase, or mixed-case.
/// Ordered longest-first so that e.g. `-GPTQ-Int4` is tried before `-gptq`.
const QUANT_SUFFIXES: &[&str] = &[
    // GPTQ
    "-GPTQ-Int4",
    "-GPTQ-Int8",
    "-gptq-int4",
    "-gptq-int8",
    "-gptq4",
    "-gptq8",
    "-GPTQ",
    "-gptq",
    "_gptq",
    // AWQ
    "-AWQ",
    "-awq",
    "_awq",
    // BitsAndBytes
    "-bnb-4bit",
    "-bnb-int8",
    "-bnb",
    "_bnb",
    "-4bit",
    "-int4",
    "-int8",
    // FP8
    "-fp8",
    "_fp8",
    "-FP8",
];

/// Derive an output path from the input path and target dtype.
///
/// `model-fp8.safetensors`  → `model-bf16.safetensors`
/// `model-GPTQ-Int4.safetensors` → `model-bf16.safetensors`
/// `weights.safetensors`    → `weights-bf16.safetensors`
fn derive_output_path(input: &std::path::Path, target: TargetDtype) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let suffix = target.to_string().to_lowercase();

    // Strip known quantization suffixes before appending target.
    let clean_stem = QUANT_SUFFIXES
        .iter()
        .find_map(|qs| stem.strip_suffix(qs))
        .unwrap_or(stem);

    let new_name = format!("{clean_stem}-{suffix}.safetensors");
    input
        .parent()
        .map_or_else(|| PathBuf::from(&new_name), |p| p.join(&new_name))
}
