// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI smoke tests for `amn convert` — Phase 6 step 2.
//!
//! Spawns the built binary via `std::process::Command` to verify argument
//! parsing, dispatch routing, and end-to-end conversion for each
//! v0.6.0-available target. Pairs with the in-process library-level
//! round-trip suite at `tests/cross_validation_convert.rs` (step 3), which
//! exercises the same conversion paths without the CLI hop.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::same_item_push
)]

use std::process::Command;

/// Locates the built `amn`/`anamnesis` binary. Same shape as
/// `tests/cli.rs::binary_path`.
fn binary_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe()
        .expect("cannot determine test executable path")
        .parent()
        .expect("no parent directory")
        .parent()
        .expect("no grandparent directory")
        .to_path_buf();
    path.push(if cfg!(windows) { "amn.exe" } else { "amn" });
    assert!(
        path.exists(),
        "amn binary not found at {}. Run `cargo build --features cli,npz,pth,gguf,bnb` \
         before `cargo test`.",
        path.display()
    );
    path
}

// ---------------------------------------------------------------------------
// Synthetic fixture builders
// ---------------------------------------------------------------------------

/// Builds a tiny BF16 safetensors file in-memory.
fn build_safetensors_bf16(tensors: &[(&str, &[usize], &[u8])]) -> Vec<u8> {
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, data)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                shape.to_vec(),
                data,
            )
            .unwrap();
            (*name, view)
        })
        .collect();
    safetensors::tensor::serialize(views, None).unwrap()
}

/// Builds a tiny F32 NPZ archive in-memory.
#[cfg(feature = "npz")]
fn build_npz_f32(tensors: &[(&str, &[usize], &[u8])]) -> Vec<u8> {
    use std::io::Write;
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::<u8>::new()));
    let options: zip::write::SimpleFileOptions =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, shape, data) in tensors {
        let entry = format!("{name}.npy");
        zip.start_file(&entry, options).unwrap();
        // Write minimal NPY v1.0 header: magic + version + header_len + dict + padding.
        let shape_str: Vec<String> = shape.iter().map(usize::to_string).collect();
        let shape_tuple = if shape.len() == 1 {
            format!("({},)", shape_str[0])
        } else {
            format!("({})", shape_str.join(", "))
        };
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_tuple}, }}");
        // Header layout: 10 bytes (magic+version+u16 len) + dict + padding + newline,
        // padded to 64-byte boundary.
        let mut header = dict.into_bytes();
        let header_total = 10 + header.len() + 1;
        let pad = (64 - header_total % 64) % 64;
        for _ in 0..pad {
            header.push(b' ');
        }
        header.push(b'\n');
        let header_len_u16 = u16::try_from(header.len()).unwrap();
        let mut entry_bytes: Vec<u8> = Vec::with_capacity(10 + header.len() + data.len());
        entry_bytes.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0]);
        entry_bytes.extend_from_slice(&header_len_u16.to_le_bytes());
        entry_bytes.extend_from_slice(&header);
        entry_bytes.extend_from_slice(data);
        zip.write_all(&entry_bytes).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

fn write_temp(bytes: &[u8], ext: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join(format!("fixture.{ext}"));
    std::fs::write(&path, bytes).unwrap();
    (dir, path)
}

#[cfg(feature = "bnb")]
fn bf16_bytes_from_f32_iter<I: IntoIterator<Item = f32>>(values: I) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        // BF16 = upper 16 bits of f32 (truncate).
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "npz")]
fn convert_npz_to_safetensors_smokes() {
    let f32_data: Vec<u8> = (0..4u32)
        .flat_map(|i| f32::from(i as u16).to_le_bytes())
        .collect();
    let other: Vec<u8> = (0..6u32)
        .flat_map(|i| (f32::from(i as u16) * 0.5).to_le_bytes())
        .collect();
    let npz_bytes = build_npz_f32(&[("w", &[2, 2], &f32_data), ("b", &[6], &other)]);
    let (_dir, in_path) = write_temp(&npz_bytes, "npz");
    let out_path = in_path.with_extension("safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "safetensors",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert failed\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(out_path.exists(), "output file missing");

    // Re-parse the safetensors output and confirm both tensors landed.
    let bytes = std::fs::read(&out_path).unwrap();
    let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut names: Vec<&str> = parsed.names();
    names.sort_unstable();
    assert_eq!(names, vec!["b", "w"]);
}

#[test]
#[cfg(feature = "gguf")]
fn convert_safetensors_bf16_to_gguf_smokes() {
    // 8 BF16 elements arranged as [2, 4]
    let bf16: Vec<u8> = (0..8u32)
        .flat_map(|i| {
            let v = i as f32;
            // CAST: u32 -> u16 truncation intentional
            let bits = (v.to_bits() >> 16) as u16;
            bits.to_le_bytes()
        })
        .collect();
    let st_bytes = build_safetensors_bf16(&[("w", &[2, 4], &bf16)]);
    let (_dir, in_path) = write_temp(&st_bytes, "safetensors");
    let out_path = in_path.with_file_name("out.gguf");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "gguf",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    // Re-parse the GGUF output, expect 1 BF16 tensor.
    let parsed = anamnesis::parse_gguf(&out_path).unwrap();
    let info = parsed.inspect();
    assert_eq!(info.tensor_count, 1);
    let collected: Vec<_> = parsed.tensors().collect();
    assert_eq!(collected[0].name, "w");
    assert_eq!(collected[0].dtype, anamnesis::GgufType::BF16);
    // Safetensors shape [2, 4] is row-major, GGUF shape should be reversed [4, 2].
    assert_eq!(collected[0].shape, &[4, 2]);
    assert_eq!(collected[0].data.as_ref(), bf16.as_slice());
}

#[test]
#[cfg(feature = "bnb")]
fn convert_safetensors_bf16_to_bnb_nf4_smokes() {
    // 64 BF16 elements arranged as [64, 1] — exactly one NF4 block of 64.
    let bf16 = bf16_bytes_from_f32_iter((0..64).map(|i| (i as f32 - 31.5) / 32.0));
    let st_bytes = build_safetensors_bf16(&[("linear", &[64, 1], &bf16)]);
    let (_dir, in_path) = write_temp(&st_bytes, "safetensors");
    let out_path = in_path.with_file_name("out-bnb-nf4.safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "bnb-nf4",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    // Re-parse — the safetensors should be classified as Bnb4.
    let model = anamnesis::parse(&out_path).unwrap();
    assert_eq!(model.header.scheme, anamnesis::QuantScheme::Bnb4);
    let names: Vec<&str> = model
        .header
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(names.contains(&"linear.weight"));
    assert!(names.contains(&"linear.weight.absmax"));
    assert!(names.contains(&"linear.weight.quant_map"));
    assert!(names.contains(&"linear.weight.quant_state.bitsandbytes__nf4"));
}

#[test]
#[cfg(all(feature = "npz", feature = "bnb"))]
fn convert_npz_to_bnb_nf4_smokes() {
    // 64 F32 elements as [64, 1] — exactly one NF4 block. Before Phase 6.14 this
    // pair returned `Unsupported` (the NF4 target accepted only a plain-BF16
    // safetensors source); the BF16 hub now routes NPZ -> BF16 -> NF4 encoder.
    let f32_data: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32 - 31.5) / 32.0).to_le_bytes())
        .collect();
    let npz_bytes = build_npz_f32(&[("linear", &[64, 1], &f32_data)]);
    let (_dir, in_path) = write_temp(&npz_bytes, "npz");
    let out_path = in_path.with_file_name("out-npz-bnb-nf4.safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "bnb-nf4",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert npz -> bnb-nf4 failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    let model = anamnesis::parse(&out_path).unwrap();
    assert_eq!(model.header.scheme, anamnesis::QuantScheme::Bnb4);
    let names: Vec<&str> = model
        .header
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(names.contains(&"linear.weight"));
    assert!(names.contains(&"linear.weight.absmax"));
}

/// `.pth` → `bnb-nf4` (Phase 6.14 Step 2) — previously `Unsupported`; the BF16
/// hub now routes it.
///
/// This fixture pins the **passthrough** half of the encoder contract: its three
/// F32 tensors are `[2, 1]`, `[2, 2]`, `[2, 2]` (2–4 elements), all far below the
/// 64-element NF4 block, so none is eligible and every tensor is written through
/// as `BF16` with no NF4 companions. The hub still performs the `F32` → `BF16`
/// conversion the encoder's input contract requires. The *quantising* half of the
/// contract is covered by the NPZ and GGUF cases above, which use 64-element
/// tensors.
#[test]
#[cfg(all(feature = "pth", feature = "bnb"))]
fn convert_pth_to_bnb_nf4_passes_ineligible_tensors_through() {
    let in_path = std::path::Path::new("tests/fixtures/pth_reference/algzoo_rnn_small.pth");
    assert!(in_path.exists(), "committed .pth fixture missing");
    let dir = tempfile::tempdir().expect("tempdir");
    let out_path = dir.path().join("out-pth-bnb-nf4.safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "bnb-nf4",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert pth -> bnb-nf4 failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    let model = anamnesis::parse(&out_path).unwrap();
    let names: Vec<&str> = model
        .header
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(names.len(), 3, "all three tensors survive: {names:?}");
    assert!(names.contains(&"rnn.weight_ih_l0"));
    assert!(names.contains(&"rnn.weight_hh_l0"));
    assert!(names.contains(&"linear.weight"));
    // Nothing was eligible, so no NF4 companion tensors were emitted.
    assert!(
        !names.iter().any(|n| n.ends_with(".absmax")),
        "no tensor is NF4-eligible, so none should gain companions: {names:?}"
    );
    // The hub converted F32 -> BF16 on the way to the encoder.
    assert!(
        model
            .header
            .tensors
            .iter()
            .all(|t| t.dtype == anamnesis::Dtype::BF16),
        "passthrough tensors should be BF16"
    );
}

/// `gguf` → `bnb-nf4` (Phase 6.14 Step 2): the hub dequantises/normalises the
/// GGUF source to BF16, then the NF4 encoder runs. GGUF stores dimensions
/// most-significant-first, so `[1, 64]` on disk is `[64, 1]` row-major — exactly
/// one NF4 block.
#[test]
#[cfg(all(feature = "gguf", feature = "bnb"))]
fn convert_gguf_to_bnb_nf4_smokes() {
    use std::collections::HashMap;

    let f32_bytes: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32 - 31.5) / 32.0).to_le_bytes())
        .collect();
    let shape = [1usize, 64];
    let tensors = [anamnesis::GgufWriteTensor {
        name: "linear",
        shape: &shape,
        dtype: anamnesis::GgufType::F32,
        data: &f32_bytes,
    }];

    let dir = tempfile::tempdir().expect("tempdir");
    let in_path = dir.path().join("in.gguf");
    anamnesis::write_gguf(&in_path, &tensors, &HashMap::new()).expect("write gguf fixture");
    let out_path = dir.path().join("out-gguf-bnb-nf4.safetensors");

    let output = Command::new(binary_path())
        .args([
            "convert",
            in_path.to_str().unwrap(),
            "--to",
            "bnb-nf4",
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run amn convert");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "amn convert gguf -> bnb-nf4 failed\nstdout: {stdout}\nstderr: {stderr}"
    );

    let model = anamnesis::parse(&out_path).unwrap();
    assert_eq!(model.header.scheme, anamnesis::QuantScheme::Bnb4);
    let names: Vec<&str> = model
        .header
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(names.contains(&"linear.weight"));
    assert!(names.contains(&"linear.weight.absmax"));
}

#[test]
fn convert_unknown_target_errors_cleanly() {
    let bf16: Vec<u8> = vec![0u8; 2];
    let st_bytes = build_safetensors_bf16(&[("w", &[1], &bf16)]);
    let (_dir, in_path) = write_temp(&st_bytes, "safetensors");

    let output = Command::new(binary_path())
        .args(["convert", in_path.to_str().unwrap(), "--to", "frobnicate"])
        .output()
        .expect("run amn convert");
    assert!(!output.status.success(), "should reject unknown target");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("supported convert targets"),
        "expected target-list error, got stderr: {stderr}"
    );
}
