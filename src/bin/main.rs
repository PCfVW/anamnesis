// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

use anamnesis::{format_bytes, parse, InspectInfo, TargetDtype};

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
        /// Path to the model file (`.safetensors`, `.pth`, `.pt`, `.bin`).
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
        /// Target dtype (currently only `bf16`) or `safetensors` for `.pth` conversion.
        #[arg(long, default_value = "bf16")]
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
}

/// Detects the model format from file extension and magic bytes.
///
/// `.safetensors` → `Safetensors`. `.pth`/`.pt` → `Pth`. `.npz` → `Npz`.
/// `.bin` → check ZIP magic (`PK\x03\x04`) to distinguish `PyTorch` ZIP
/// from safetensors. Unknown extensions default to `Safetensors`.
fn detect_format(path: &std::path::Path) -> Format {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "safetensors" => Format::Safetensors,
        #[cfg(feature = "pth")]
        "pth" | "pt" => Format::Pth,
        #[cfg(feature = "npz")]
        "npz" => Format::Npz,
        #[cfg(feature = "pth")]
        "bin" => {
            if has_zip_magic(path) {
                Format::Pth
            } else {
                Format::Safetensors
            }
        }
        _ => Format::Safetensors,
    }
}

/// Returns `true` if the file starts with the ZIP local header magic `PK\x03\x04`.
///
/// Reads only 4 bytes — does not load the file into memory.
#[cfg(feature = "pth")]
fn has_zip_magic(path: &std::path::Path) -> bool {
    let mut buf = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            f.read_exact(&mut buf)
        })
        .map(|()| buf == *b"PK\x03\x04")
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Subcommand runners
// ---------------------------------------------------------------------------

fn run() -> anamnesis::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { path } => run_parse(&path),
        Commands::Inspect { path } => run_inspect(&path),
        Commands::Remember { path, to, output } => run_remember(&path, &to, output.as_deref()),
    }
}

fn run_parse(path: &std::path::Path) -> anamnesis::Result<()> {
    match detect_format(path) {
        Format::Safetensors => run_parse_safetensors(path),
        #[cfg(feature = "pth")]
        Format::Pth => run_parse_pth(path),
        #[cfg(feature = "npz")]
        Format::Npz => run_parse_npz(path),
    }
}

fn run_parse_safetensors(path: &std::path::Path) -> anamnesis::Result<()> {
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
fn run_parse_pth(path: &std::path::Path) -> anamnesis::Result<()> {
    let parsed = anamnesis::parse_pth(path)?;
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
fn run_parse_npz(path: &std::path::Path) -> anamnesis::Result<()> {
    let tensors = anamnesis::parse_npz(path)?;

    println!(
        "Parsed {} (NPZ archive)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)")
    );
    println!("  Tensors: {}", tensors.len());

    // Sort by name for stable output.
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    // CAST: usize → u64, tensor byte lengths fit in u64
    #[allow(clippy::as_conversions)]
    let total_bytes: u64 = tensors.values().map(|t| t.data.len() as u64).sum();
    println!("  Total size: {}", format_bytes(total_bytes));
    println!();

    for name in &names {
        if let Some(t) = tensors.get(*name) {
            let shape_str = format!("{:?}", t.shape);
            // CAST: usize → u64, tensor byte lengths fit
            #[allow(clippy::as_conversions)]
            let byte_len = t.data.len() as u64;
            println!(
                "  {:<30} {:<6} {:<15} {}",
                name,
                t.dtype,
                shape_str,
                format_bytes(byte_len)
            );
        }
    }
    Ok(())
}

#[cfg(feature = "npz")]
fn run_inspect_npz(path: &std::path::Path) -> anamnesis::Result<()> {
    let tensors = anamnesis::parse_npz(path)?;

    // CAST: usize → u64, tensor byte lengths fit in u64
    #[allow(clippy::as_conversions)]
    let total_bytes: u64 = tensors.values().map(|t| t.data.len() as u64).sum();

    let mut dtypes: Vec<String> = Vec::new();
    for t in tensors.values() {
        let s = t.dtype.to_string();
        if !dtypes.contains(&s) {
            dtypes.push(s);
        }
    }

    println!("Format:      NPZ archive");
    println!("Tensors:     {}", tensors.len());
    println!("Total size:  {}", format_bytes(total_bytes));
    println!("Dtypes:      {}", dtypes.join(", "));
    Ok(())
}

fn run_inspect(path: &std::path::Path) -> anamnesis::Result<()> {
    match detect_format(path) {
        Format::Safetensors => {
            let model = parse(path)?;
            let info = InspectInfo::from(&model.header);
            println!("{info}");
        }
        #[cfg(feature = "pth")]
        Format::Pth => {
            let parsed = anamnesis::parse_pth(path)?;
            let info = parsed.inspect();
            println!("{info}");
        }
        #[cfg(feature = "npz")]
        Format::Npz => run_inspect_npz(path)?,
    }
    Ok(())
}

fn run_remember(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
) -> anamnesis::Result<()> {
    match detect_format(path) {
        Format::Safetensors => run_remember_safetensors(path, to, output),
        #[cfg(feature = "pth")]
        Format::Pth => {
            let to_lower = to.to_ascii_lowercase();
            if to_lower != "safetensors" && to_lower != "bf16" {
                return Err(anamnesis::AnamnesisError::Unsupported {
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
        Format::Npz => Err(anamnesis::AnamnesisError::Unsupported {
            format: "NPZ".into(),
            detail: "NPZ tensors are already full-precision; \
                     no dequantization or conversion needed"
                .into(),
        }),
    }
}

fn run_remember_safetensors(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
) -> anamnesis::Result<()> {
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
fn run_remember_pth(
    path: &std::path::Path,
    output: Option<&std::path::Path>,
) -> anamnesis::Result<()> {
    let parsed = anamnesis::parse_pth(path)?;
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

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
