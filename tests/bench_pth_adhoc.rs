// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc `.pth` parsing benchmark + phase profiling on torchvision models.
//!
//! Run: `cargo test --release --features pth --test bench_pth_adhoc -- --nocapture`

#![cfg(feature = "pth")]
#![allow(
    clippy::unwrap_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::wildcard_enum_match_arm,
    unsafe_code
)]

use std::path::PathBuf;
use std::time::Instant;

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pth_benchmark")
}

fn bench_file(label: &str, filename: &str, iters: u64) {
    let path = bench_dir().join(filename);
    if !path.exists() {
        println!("  SKIP {label} (not found)");
        return;
    }
    let size_mb = std::fs::metadata(&path).unwrap().len() as f64 / 1024.0 / 1024.0;

    // Warmup
    let parsed = anamnesis::parse_pth(&path).unwrap();
    let tensors = parsed.tensors().unwrap();
    let n_tensors = tensors.len();
    let n_params: usize = tensors
        .iter()
        .map(|t| t.shape.iter().copied().product::<usize>())
        .sum();
    drop(tensors);
    drop(parsed);

    // Benchmark parse + tensors() together (the full pipeline).
    let start = Instant::now();
    for _ in 0..iters {
        let p = anamnesis::parse_pth(&path).unwrap();
        let t = p.tensors().unwrap();
        std::hint::black_box(&t);
    }
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    let throughput = size_mb / (ms / 1000.0);

    println!(
        "  {label}: {ms:.1} ms  ({n_tensors} tensors, {n_params} params, {throughput:.0} MB/s)"
    );
}

/// Profile individual phases using mmap (matching what `parse_pth` does).
fn profile_phases(label: &str, filename: &str, iters: u64) {
    let path = bench_dir().join(filename);
    if !path.exists() {
        println!("  SKIP {label} profile (not found)");
        return;
    }
    let size_mb = std::fs::metadata(&path).unwrap().len() as f64 / 1024.0 / 1024.0;

    // Phase 1: mmap only
    let start = Instant::now();
    for _ in 0..iters {
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        std::hint::black_box(&mmap);
    }
    let mmap_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Phase 2: mmap + ZipArchive::new
    let start = Instant::now();
    for _ in 0..iters {
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        let cursor = std::io::Cursor::new(&mmap[..]);
        let archive = zip::ZipArchive::new(cursor).unwrap();
        std::hint::black_box(&archive);
    }
    let zip_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Phase 3: full parse_pth
    let start = Instant::now();
    for _ in 0..iters {
        let _ = anamnesis::parse_pth(&path).unwrap();
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Phase 4: memcpy baseline (simulates the unavoidable tensor copy cost)
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        let copy = mmap[..].to_vec();
        std::hint::black_box(&copy);
    }
    let memcpy_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Phase 5: fs::read for comparison
    let start = Instant::now();
    for _ in 0..iters {
        let raw = std::fs::read(&path).unwrap();
        std::hint::black_box(&raw);
    }
    let fsread_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let rest_ms = total_ms - zip_ms;

    println!("  {label} ({size_mb:.0} MB) phase breakdown:");
    println!("    mmap:             {mmap_ms:6.1} ms");
    println!(
        "    + ZipArchive:     {zip_ms:6.1} ms  (zip dir = {:.1} ms)",
        zip_ms - mmap_ms
    );
    println!("    full parse_pth:   {total_ms:6.1} ms  (pickle+index+copy = {rest_ms:.1} ms)");
    println!("    memcpy baseline:  {memcpy_ms:6.1} ms  (mmap[..].to_vec)");
    println!("    fs::read (ref):   {fsread_ms:6.1} ms");
    println!();
}

#[test]
fn bench_pth_resnet18() {
    println!();
    bench_file("resnet18 (45 MB)", "resnet18.pth", 20);
}

#[test]
fn bench_pth_resnet50() {
    println!();
    bench_file("resnet50 (98 MB)", "resnet50.pth", 10);
}

#[test]
fn bench_pth_vit_b_16() {
    println!();
    bench_file("vit_b_16 (330 MB)", "vit_b_16.pth", 5);
}

#[test]
fn profile_pth_phases() {
    println!("\n  === Phase Profiling (mmap path) ===\n");
    profile_phases("resnet18", "resnet18.pth", 20);
    profile_phases("resnet50", "resnet50.pth", 10);
    profile_phases("vit_b_16", "vit_b_16.pth", 5);
}
