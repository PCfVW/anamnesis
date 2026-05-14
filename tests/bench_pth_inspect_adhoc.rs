// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ad-hoc benchmark comparing `parse_pth(path).inspect()` (mmap-backed)
//! against `inspect_pth_from_reader(File::open(path)?)` (reader-generic
//! over a `std::fs::File`) on the in-tree `AlgZoo` `.pth` fixtures.
//!
//! Not part of CI — `#[ignore]`-gated so it does not run on every
//! `cargo test`. Run with:
//!
//! ```text
//! cargo test --release --features pth --test bench_pth_inspect_adhoc \
//!     bench_pth_inspect_paths -- --nocapture --ignored
//! ```
//!
//! With `target-cpu=native` for the perf-claim measurement protocol
//! (see `CLAUDE.md` § Performance Changes):
//!
//! ```text
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo test --release --features pth --test bench_pth_inspect_adhoc \
//!     bench_pth_inspect_paths -- --nocapture --ignored
//! ```
//!
//! Prints best-of-5 median per file with min/max range, plus the median
//! reader/mmap ratio across all files. Used to validate the
//! `inspect_pth_from_reader` rustdoc's parity claim (reader path at
//! parity with the mmap-backed `parse_pth(path).inspect()`).

#![cfg(feature = "pth")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::indexing_slicing
)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anamnesis::{inspect_pth_from_reader, parse_pth};

/// Number of timed iterations per file per substrate.
///
/// 5 is the project's standard sample size for perf-claim measurements
/// (see `CLAUDE.md` § Performance Changes — best-of-5 median).
const ITERATIONS: usize = 5;

/// One additional warm-up iteration before timing begins, to amortise
/// the cold-cache cost of the first read on a freshly-opened file.
const WARMUP_ITERATIONS: usize = 1;

/// Returns `(min, median, max)` of a slice of `Duration`s. Sorts the input.
fn min_median_max(samples: &mut [Duration]) -> (Duration, Duration, Duration) {
    samples.sort_unstable();
    let lo = samples[0];
    let hi = samples[samples.len() - 1];
    let mid = samples[samples.len() / 2];
    (lo, mid, hi)
}

/// Times one closure `ITERATIONS` times after warming the file cache.
fn time_loop<T, F: FnMut() -> T>(mut f: F) -> Vec<Duration> {
    for _ in 0..WARMUP_ITERATIONS {
        let _ = f();
    }
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let t = Instant::now();
        let _ = f();
        samples.push(t.elapsed());
    }
    samples
}

#[test]
#[ignore = "ad-hoc perf benchmark; run on demand with --release --ignored"]
fn bench_pth_inspect_paths() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("pth_reference");

    let fixtures = [
        "algzoo_rnn_small.pth",
        "algzoo_transformer_small.pth",
        "algzoo_rnn_blog.pth",
    ];

    eprintln!(
        "\nPTH inspect benchmark — {} files, best-of-{} median (warmup={})",
        fixtures.len(),
        ITERATIONS,
        WARMUP_ITERATIONS,
    );
    eprintln!(
        "{:<40}  {:>7}  {:>21}  {:>21}  {:>7}",
        "file", "size", "mmap min/med/max (\u{b5}s)", "reader min/med/max (\u{b5}s)", "ratio"
    );
    eprintln!("{}", "=".repeat(112));

    let mut ratios: Vec<f64> = Vec::with_capacity(fixtures.len());

    for name in fixtures {
        let path = fixture_dir.join(name);
        let file_size = std::fs::metadata(&path).ok().map_or(0, |m| m.len());

        // mmap-backed path: parse_pth(path).inspect()
        let mut mmap_samples = time_loop(|| parse_pth(&path).expect("parse_pth").inspect());
        let (mmap_lo, mmap_mid, mmap_hi) = min_median_max(&mut mmap_samples);

        // Reader-generic path: inspect_pth_from_reader(File::open(path)?)
        let mut reader_samples = time_loop(|| {
            let f = std::fs::File::open(&path).expect("open");
            inspect_pth_from_reader(f).expect("inspect_pth_from_reader")
        });
        let (reader_lo, reader_mid, reader_hi) = min_median_max(&mut reader_samples);

        let ratio = reader_mid.as_secs_f64() / mmap_mid.as_secs_f64();
        ratios.push(ratio);

        eprintln!(
            "{name:<40}  {size_kib:>5.1} KiB  {mmap_lo_us:>6.1}/{mmap_mid_us:>6.1}/{mmap_hi_us:>6.1}  {reader_lo_us:>6.1}/{reader_mid_us:>6.1}/{reader_hi_us:>6.1}  {ratio:>6.2}\u{d7}",
            size_kib = file_size as f64 / 1024.0,
            mmap_lo_us = mmap_lo.as_secs_f64() * 1e6,
            mmap_mid_us = mmap_mid.as_secs_f64() * 1e6,
            mmap_hi_us = mmap_hi.as_secs_f64() * 1e6,
            reader_lo_us = reader_lo.as_secs_f64() * 1e6,
            reader_mid_us = reader_mid.as_secs_f64() * 1e6,
            reader_hi_us = reader_hi.as_secs_f64() * 1e6,
        );
    }

    eprintln!("{}", "=".repeat(112));
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ratios.len();
    let r_min = ratios[0];
    let r_med = ratios[n / 2];
    let r_max = ratios[n - 1];
    let r_mean = ratios.iter().sum::<f64>() / n as f64;
    eprintln!(
        "reader / mmap ratio across {n} files: min={r_min:.2}\u{d7}  median={r_med:.2}\u{d7}  mean={r_mean:.2}\u{d7}  max={r_max:.2}\u{d7}",
    );
}
