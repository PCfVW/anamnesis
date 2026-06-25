// SPDX-License-Identifier: MIT OR Apache-2.0

//! Panic-freedom invariant (Phase 6.13 Step 3): **no public parse/inspect entry
//! point may panic or abort on any input** — a malformed or hostile artefact is
//! always a clean `Ok`/`Err`, never an unwinding panic (which under the shipped
//! `panic = "abort"` profile would be an uncatchable process kill, and which the
//! Phase 8 bindings must be able to surface as a catchable `PanicException`).
//!
//! Each entry point is exercised over a battery of adversarial inputs (synthetic
//! malformed shapes + truncations and bit-flips of the committed fixtures) inside
//! [`std::panic::catch_unwind`], asserting it never unwinds. This runs under the
//! default (debug) test profile, so debug-only integer-overflow panics are in
//! scope. The `cargo fuzz` harness extends this to coverage-guided exploration;
//! this test pins the contract in stable CI.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::wildcard_enum_match_arm
)]

use std::panic::{catch_unwind, AssertUnwindSafe};

/// Runs `f` and asserts it returns normally (`Ok`/`Err`) rather than unwinding.
/// The produced value is irrelevant — only "did it panic?" matters.
fn assert_no_panic<T>(label: &str, f: impl FnOnce() -> T) {
    let outcome = catch_unwind(AssertUnwindSafe(f));
    assert!(outcome.is_ok(), "PANICKED on input `{label}`");
}

/// Committed (always-present) fixtures, truncated and bit-flipped below to build
/// realistic "almost-valid" adversarial inputs.
const FIXTURES: &[&str] = &[
    "tests/fixtures/safetensors_reference/fp8.safetensors",
    "tests/fixtures/safetensors_reference/gptq.safetensors",
    "tests/fixtures/safetensors_reference/awq.safetensors",
    "tests/fixtures/safetensors_reference/bnb_nf4.safetensors",
    "tests/fixtures/pth_reference/algzoo_rnn_small.pth",
    "tests/fixtures/npz_reference/gemma_scope_small.npz",
];

/// The adversarial input battery: `(label, bytes)`.
fn adversarial_inputs() -> Vec<(String, Vec<u8>)> {
    let mut inputs: Vec<(String, Vec<u8>)> = vec![
        ("empty".to_owned(), Vec::new()),
        ("one-zero".to_owned(), vec![0u8]),
        ("zeros-64".to_owned(), vec![0u8; 64]),
        ("ones-64".to_owned(), vec![0xFFu8; 64]),
        // 8 bytes of 0xFF: a safetensors `u64` header length of `u64::MAX`.
        ("u64-max-prefix".to_owned(), vec![0xFFu8; 8]),
        ("u64-max-prefix+junk".to_owned(), {
            let mut b = vec![0xFFu8; 8];
            b.extend_from_slice(b"junk-after-an-absurd-length");
            b
        }),
        // Small declared length + a partial JSON header.
        ("small-len+partial-json".to_owned(), {
            let mut b = 2u64.to_le_bytes().to_vec();
            b.extend_from_slice(b"{");
            b
        }),
        // ZIP local-file magic + junk (drives the `.npz` / `.pth` container).
        ("zip-magic+junk".to_owned(), {
            let mut b = b"PK\x03\x04".to_vec();
            b.extend_from_slice(&[0xABu8; 64]);
            b
        }),
        // GGUF magic + junk.
        ("gguf-magic+junk".to_owned(), {
            let mut b = b"GGUF".to_vec();
            b.extend_from_slice(&[0xCDu8; 64]);
            b
        }),
        // Raw pickle protocol bytes (legacy `.pth` detection path).
        ("pickle-proto".to_owned(), vec![0x80, 0x02, b'}', b'.']),
        // Deeply repetitive bytes — stress recursion / nesting guards.
        ("open-brackets-8k".to_owned(), vec![b'['; 8192]),
        ("pickle-mark-flood".to_owned(), vec![0x28u8; 8192]), // `(` = MARK
        ("repeating-2byte-100k".to_owned(), {
            b"\x80\x02".iter().copied().cycle().take(100_000).collect()
        }),
    ];

    // A couple of deterministic pseudo-random blobs (no RNG dependency).
    for seed in [0x9E37_79B9u32, 0x1234_5678u32] {
        let blob: Vec<u8> = (0u32..512)
            .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 13) as u8)
            .collect();
        inputs.push((format!("prng-{seed:08x}"), blob));
    }

    // Truncations and single-byte flips of each committed fixture: "almost
    // valid" inputs are the ones most likely to slip past a length check into a
    // downstream panic.
    for path in FIXTURES {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let len = bytes.len();
        let name = path.rsplit('/').next().unwrap_or(path);
        for cut in [1usize, 4, 8, 16, len / 4, len / 2, len.saturating_sub(1)] {
            let cut = cut.min(len);
            inputs.push((format!("{name}@trunc{cut}"), bytes[..cut].to_vec()));
        }
        for pos in [0usize, len / 2, len.saturating_sub(1)] {
            if pos < len {
                let mut flipped = bytes.clone();
                flipped[pos] ^= 0xFF;
                inputs.push((format!("{name}@flip{pos}"), flipped));
            }
        }
    }

    inputs
}

// ---------------------------------------------------------------------------
// safetensors (always-on)
// ---------------------------------------------------------------------------

#[test]
fn safetensors_entry_points_never_panic() {
    use anamnesis::{
        parse_bytes, parse_from_reader, parse_safetensors_header_from_reader, ParseLimits,
    };
    use std::io::Cursor;

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("safetensors parse_bytes / {label}"), || {
            parse_bytes(bytes.clone())
        });
        assert_no_panic(&format!("safetensors parse_from_reader / {label}"), || {
            parse_from_reader(Cursor::new(bytes.clone()))
        });
        // A tightened budget exercises the limit-checking arithmetic too.
        let tight = ParseLimits::default().with_max_single_alloc(16);
        assert_no_panic(&format!("safetensors parse_bytes(tight) / {label}"), || {
            anamnesis::parse_bytes_with_limits(bytes.clone(), &tight)
        });
        assert_no_panic(&format!("safetensors header_from_reader / {label}"), || {
            parse_safetensors_header_from_reader(Cursor::new(bytes.clone()))
        });
    }
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

#[cfg(feature = "gguf")]
#[test]
fn gguf_entry_points_never_panic() {
    use anamnesis::{inspect_gguf_from_reader, parse_gguf_bytes, parse_gguf_from_reader};
    use std::io::Cursor;

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("gguf parse_gguf_bytes / {label}"), || {
            parse_gguf_bytes(bytes.clone())
        });
        assert_no_panic(&format!("gguf parse_gguf_from_reader / {label}"), || {
            parse_gguf_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(&format!("gguf inspect_from_reader / {label}"), || {
            inspect_gguf_from_reader(Cursor::new(bytes.clone()))
        });
    }
}

// ---------------------------------------------------------------------------
// PyTorch .pth
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
#[test]
fn pth_entry_points_never_panic() {
    use anamnesis::{inspect_pth_from_reader, parse_pth_bytes, parse_pth_from_reader};
    use std::io::Cursor;

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("pth parse_pth_bytes / {label}"), || {
            parse_pth_bytes(bytes.clone())
        });
        assert_no_panic(&format!("pth parse_pth_from_reader / {label}"), || {
            parse_pth_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(&format!("pth inspect_from_reader / {label}"), || {
            inspect_pth_from_reader(Cursor::new(bytes.clone()))
        });
    }
}

// ---------------------------------------------------------------------------
// NPZ
// ---------------------------------------------------------------------------

#[cfg(feature = "npz")]
#[test]
fn npz_entry_points_never_panic() {
    use anamnesis::{inspect_npz_from_reader, parse_npz_with_limits, ParseLimits};
    use std::io::Cursor;

    // `parse_npz*` is path-based; reuse one temp file, overwritten per input.
    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path().to_path_buf();

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("npz inspect_from_reader / {label}"), || {
            inspect_npz_from_reader(Cursor::new(bytes.clone()))
        });

        std::fs::write(&path, &bytes).expect("write temp npz");
        assert_no_panic(&format!("npz parse_npz_with_limits / {label}"), || {
            parse_npz_with_limits(&path, &ParseLimits::default())
        });
    }
}
