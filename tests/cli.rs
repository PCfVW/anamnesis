// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI integration tests for the `anamnesis` / `amn` binary.
//!
//! These tests build and invoke the binary via `std::process::Command` to
//! verify argument parsing, subcommand routing, and output format. They
//! complement the library-level tests in `cross_validation.rs`.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::process::Command;

/// Path to the built binary (cargo sets this via the test harness).
fn binary_path() -> std::path::PathBuf {
    // `cargo test` builds binaries into target/debug/
    let mut path = std::env::current_exe()
        .expect("cannot determine test executable path")
        .parent()
        .expect("no parent directory")
        .parent()
        .expect("no grandparent directory")
        .to_path_buf();
    path.push(if cfg!(windows) {
        "anamnesis.exe"
    } else {
        "anamnesis"
    });
    path
}

/// Build a minimal safetensors file in a temp directory for testing.
///
/// Contains:
/// - 1 FP8 weight tensor (2×2, all 1.0 in E4M3)
/// - 1 F32 scale tensor (scalar [1], value 2.0)
/// - 1 BF16 passthrough tensor (norm, 1 element)
fn create_test_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("test-fp8.safetensors");

    let fp8_data = vec![0x38u8; 4]; // 2×2 of 1.0 in E4M3
    let scale_data = 2.0_f32.to_le_bytes().to_vec();
    let norm_data = vec![0x80, 0x3F]; // BF16 1.0

    let mut header_map = serde_json::Map::new();

    let mut w_info = serde_json::Map::new();
    w_info.insert("dtype".into(), "F8_E4M3".into());
    w_info.insert("shape".into(), serde_json::json!([2, 2]));
    w_info.insert("data_offsets".into(), serde_json::json!([0, 4]));
    header_map.insert("layer.weight".into(), w_info.into());

    let mut s_info = serde_json::Map::new();
    s_info.insert("dtype".into(), "F32".into());
    s_info.insert("shape".into(), serde_json::json!([1]));
    s_info.insert("data_offsets".into(), serde_json::json!([4, 8]));
    header_map.insert("layer.weight_scale".into(), s_info.into());

    let mut n_info = serde_json::Map::new();
    n_info.insert("dtype".into(), "BF16".into());
    n_info.insert("shape".into(), serde_json::json!([1]));
    n_info.insert("data_offsets".into(), serde_json::json!([8, 10]));
    header_map.insert("norm.weight".into(), n_info.into());

    let header_json = serde_json::to_string(&header_map).unwrap();
    let header_bytes = header_json.as_bytes();

    // CAST: usize → u64, header length fits in u64
    #[allow(clippy::as_conversions)]
    let header_len = header_bytes.len() as u64;
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&header_len.to_le_bytes());
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.extend_from_slice(&fp8_data);
    file_bytes.extend_from_slice(&scale_data);
    file_bytes.extend_from_slice(&norm_data);

    std::fs::write(&path, &file_bytes).unwrap();
    (dir, path)
}

// ---------------------------------------------------------------------------
// Parse subcommand
// ---------------------------------------------------------------------------

#[test]
fn cli_parse_subcommand() {
    let (_dir, fixture) = create_test_fixture();

    let output = Command::new(binary_path())
        .args(["parse", fixture.to_str().unwrap()])
        .output()
        .expect("failed to run binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "parse failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("3 tensors parsed"), "stdout: {stdout}");
    assert!(stdout.contains("quantized"), "stdout: {stdout}");
    assert!(stdout.contains("passthrough"), "stdout: {stdout}");
}

// ---------------------------------------------------------------------------
// Inspect subcommand
// ---------------------------------------------------------------------------

#[test]
fn cli_inspect_subcommand() {
    let (_dir, fixture) = create_test_fixture();

    let output = Command::new(binary_path())
        .args(["inspect", fixture.to_str().unwrap()])
        .output()
        .expect("failed to run binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "inspect failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("Format:"), "stdout: {stdout}");
    assert!(stdout.contains("FP8"), "stdout: {stdout}");
    assert!(stdout.contains("Passthrough:"), "stdout: {stdout}");
}

#[test]
fn cli_info_alias() {
    let (_dir, fixture) = create_test_fixture();

    let output = Command::new(binary_path())
        .args(["info", fixture.to_str().unwrap()])
        .output()
        .expect("failed to run binary");

    assert!(
        output.status.success(),
        "info alias failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// Remember subcommand
// ---------------------------------------------------------------------------

#[test]
fn cli_remember_subcommand() {
    let (dir, fixture) = create_test_fixture();
    let output_path = dir.path().join("test-bf16.safetensors");

    let output = Command::new(binary_path())
        .args([
            "remember",
            fixture.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "remember failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("Parsing..."), "stdout: {stdout}");
    assert!(stdout.contains("Output:"), "stdout: {stdout}");
    assert!(output_path.exists(), "output file not created");
}

#[test]
fn cli_dequantize_alias() {
    let (dir, fixture) = create_test_fixture();
    let output_path = dir.path().join("test-bf16.safetensors");

    let output = Command::new(binary_path())
        .args([
            "dequantize",
            fixture.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run binary");

    assert!(
        output.status.success(),
        "dequantize alias failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists(), "output file not created");
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn cli_nonexistent_file() {
    let output = Command::new(binary_path())
        .args(["parse", "/tmp/nonexistent_anamnesis_cli_test.safetensors"])
        .output()
        .expect("failed to run binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error:"), "stderr: {stderr}");
}

#[test]
fn cli_unsupported_target_dtype() {
    let (_dir, fixture) = create_test_fixture();

    let output = Command::new(binary_path())
        .args(["remember", fixture.to_str().unwrap(), "--to", "int8"])
        .output()
        .expect("failed to run binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error:"), "stderr: {stderr}");
}

#[test]
fn cli_no_subcommand_shows_help() {
    let output = Command::new(binary_path())
        .output()
        .expect("failed to run binary");

    // clap exits with error when no subcommand is given
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Usage") || stderr.contains("usage"),
        "stderr: {stderr}"
    );
}

#[test]
fn cli_version_flag() {
    let output = Command::new(binary_path())
        .args(["--version"])
        .output()
        .expect("failed to run binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("anamnesis"), "stdout: {stdout}");
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "stdout: {stdout}"
    );
}
