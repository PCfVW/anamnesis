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
    /// Parse and summarize a safetensors file.
    Parse {
        /// Path to the safetensors file.
        path: PathBuf,
    },
    /// Inspect format, tensor counts, and size estimates.
    #[command(alias = "info")]
    Inspect {
        /// Path to the safetensors file.
        path: PathBuf,
    },
    /// Dequantize (recover precision) to a target dtype.
    #[command(alias = "dequantize")]
    Remember {
        /// Path to the input safetensors file.
        path: PathBuf,
        /// Target dtype (currently only `bf16`).
        #[arg(long, default_value = "bf16")]
        to: String,
        /// Output file path (derived from input if omitted).
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
}

fn run() -> anamnesis::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { path } => run_parse(&path),
        Commands::Inspect { path } => run_inspect(&path),
        Commands::Remember { path, to, output } => run_remember(&path, &to, output.as_deref()),
    }
}

fn run_parse(path: &std::path::Path) -> anamnesis::Result<()> {
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
        println!("  {scales:>3} scale       F32");
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

fn run_inspect(path: &std::path::Path) -> anamnesis::Result<()> {
    let model = parse(path)?;
    let info = InspectInfo::from(&model.header);
    println!("{info}");
    Ok(())
}

fn run_remember(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
) -> anamnesis::Result<()> {
    let target = match to {
        "bf16" => TargetDtype::BF16,
        other => {
            return Err(anamnesis::AnamnesisError::Unsupported {
                format: other.to_owned(),
                detail: "supported target dtypes: bf16".to_owned(),
            });
        }
    };

    let model = parse(path)?;
    let info = InspectInfo::from(&model.header);

    let total = model.header.tensors.len();
    let quantized = model.header.quantized_count();
    println!("Parsing...  {total} tensors, {}", model.header.scheme);

    let output_path = match output {
        Some(p) => p.to_owned(),
        None => derive_output_path(path, target),
    };

    println!("Recalling... {quantized} tensors");
    model.remember(&output_path, target)?;

    println!(
        "Output: {} ({})",
        output_path.display(),
        format_bytes(info.dequantized_size),
    );
    Ok(())
}

/// Derive an output path from the input path and target dtype.
///
/// `model-fp8.safetensors` → `model-bf16.safetensors`
/// `weights.safetensors`   → `weights-bf16.safetensors`
fn derive_output_path(input: &std::path::Path, target: TargetDtype) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let suffix = match target {
        TargetDtype::BF16 => "bf16",
        _ => "unknown",
    };

    // Strip known quantization suffixes before appending target.
    let clean_stem = stem
        .strip_suffix("-fp8")
        .or_else(|| stem.strip_suffix("_fp8"))
        .or_else(|| stem.strip_suffix("-FP8"))
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
