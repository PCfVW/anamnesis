// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity + safety tests for the copy-based (`no-mmap`) parse paths added in
//! Phase 6.13 Step 1.
//!
//! Each format is parsed three ways — path/mmap, `parse_*_bytes` (owned), and
//! `parse_*_from_reader` (owned) — and the results must be identical, proving
//! the `Backing` plumbing reads the same bytes regardless of origin. Plus:
//! malformed bytes yield a clean `Err` (never a panic), and a tightened
//! `ParseLimits` rejects an oversized read.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::wildcard_enum_match_arm
)]

use std::fs;

use anamnesis::ParseLimits;

// ---------------------------------------------------------------------------
// safetensors (always-on)
// ---------------------------------------------------------------------------

const ST_FP8: &str = "tests/fixtures/safetensors_reference/fp8.safetensors";

#[test]
fn safetensors_owned_paths_match_mmap() {
    use anamnesis::{parse, parse_bytes, parse_from_reader, TargetDtype};

    let bytes = fs::read(ST_FP8).expect("read fp8 fixture");

    let m_path = parse(ST_FP8).expect("path parse");
    let m_bytes = parse_bytes(bytes).expect("bytes parse");
    let m_reader = parse_from_reader(fs::File::open(ST_FP8).expect("open")).expect("reader parse");

    // Header parity.
    let header = format!("{:?}", m_path.inspect());
    assert_eq!(
        header,
        format!("{:?}", m_bytes.inspect()),
        "bytes inspect differs"
    );
    assert_eq!(
        header,
        format!("{:?}", m_reader.inspect()),
        "reader inspect differs"
    );

    // Full tensor-data parity, through dequantisation to BF16.
    let d_path = m_path
        .remember_to_bytes(TargetDtype::BF16)
        .expect("remember path");
    let d_bytes = m_bytes
        .remember_to_bytes(TargetDtype::BF16)
        .expect("remember bytes");
    let d_reader = m_reader
        .remember_to_bytes(TargetDtype::BF16)
        .expect("remember reader");
    assert_eq!(d_path, d_bytes, "bytes-path dequant differs from mmap");
    assert_eq!(d_path, d_reader, "reader-path dequant differs from mmap");
    assert!(!d_path.is_empty());
}

#[test]
fn safetensors_malformed_bytes_is_clean_err() {
    use anamnesis::parse_bytes;
    // The first 8 bytes decode to an absurd header length → rejected by the cap,
    // not a panic.
    assert!(parse_bytes(b"not a real safetensors file, only bytes".to_vec()).is_err());
    // Empty input: header length prefix cannot be read.
    assert!(parse_bytes(Vec::new()).is_err());
}

#[test]
fn safetensors_reader_respects_max_single_alloc() {
    use anamnesis::parse_from_reader_with_limits;
    // The fixture is 96 bytes; an 8-byte ceiling must reject the read.
    let limits = ParseLimits::default().with_max_single_alloc(8);
    let Err(err) = parse_from_reader_with_limits(fs::File::open(ST_FP8).expect("open"), &limits)
    else {
        panic!("oversized read should be rejected, not OOM");
    };
    assert!(
        matches!(err, anamnesis::AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
        "expected LimitExceeded(max_single_alloc_bytes), got: {err}"
    );
}

// ---------------------------------------------------------------------------
// PyTorch .pth
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
mod pth {
    use super::{fs, ParseLimits};
    use anamnesis::{parse_pth, parse_pth_bytes, parse_pth_from_reader, ParsedPth};

    const PTH: &str = "tests/fixtures/pth_reference/algzoo_rnn_small.pth";

    /// `(name, data)` for every tensor, sorted — the parity fingerprint.
    fn fingerprint(p: &ParsedPth) -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<(String, Vec<u8>)> = p
            .tensors()
            .expect("tensors")
            .into_iter()
            .map(|t| (t.name, t.data.into_owned()))
            .collect();
        v.sort();
        v
    }

    #[test]
    fn pth_owned_paths_match_mmap() {
        let bytes = fs::read(PTH).expect("read pth fixture");

        let p_path = parse_pth(PTH).expect("path parse");
        let p_bytes = parse_pth_bytes(bytes).expect("bytes parse");
        let p_reader =
            parse_pth_from_reader(fs::File::open(PTH).expect("open")).expect("reader parse");

        let header = format!("{:?}", p_path.inspect());
        assert_eq!(header, format!("{:?}", p_bytes.inspect()));
        assert_eq!(header, format!("{:?}", p_reader.inspect()));

        let fp = fingerprint(&p_path);
        assert!(!fp.is_empty(), "fixture should have tensors");
        assert_eq!(
            fp,
            fingerprint(&p_bytes),
            "bytes-path tensors differ from mmap"
        );
        assert_eq!(
            fp,
            fingerprint(&p_reader),
            "reader-path tensors differ from mmap"
        );
    }

    #[test]
    fn pth_malformed_bytes_is_clean_err() {
        assert!(parse_pth_bytes(b"garbage, not a zip archive".to_vec()).is_err());
        assert!(parse_pth_bytes(Vec::new()).is_err());
    }

    #[test]
    fn pth_reader_respects_max_single_alloc() {
        use anamnesis::parse_pth_from_reader_with_limits;
        let limits = ParseLimits::default().with_max_single_alloc(8);
        let Err(err) =
            parse_pth_from_reader_with_limits(fs::File::open(PTH).expect("open"), &limits)
        else {
            panic!("oversized read should be rejected");
        };
        assert!(
            matches!(err, anamnesis::AnamnesisError::LimitExceeded { limit, .. } if limit == "max_single_alloc_bytes"),
            "expected LimitExceeded(max_single_alloc_bytes), got: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// GGUF (fixture is local-only / not committed → guarded by `exists()`)
// ---------------------------------------------------------------------------

#[cfg(feature = "gguf")]
mod gguf {
    use super::fs;
    use std::path::Path;

    use anamnesis::{parse_gguf, parse_gguf_bytes, parse_gguf_from_reader, ParsedGguf};

    const GGUF: &str = "tests/fixtures/gguf_reference/models/SmolLM2-135M-Instruct-Q4_K_M.gguf";

    /// `(name, data)` for every tensor, sorted — the parity fingerprint.
    fn fingerprint(p: &ParsedGguf) -> Vec<(String, Vec<u8>)> {
        let mut v: Vec<(String, Vec<u8>)> = p
            .tensors()
            .map(|t| (t.name.to_string(), t.data.into_owned()))
            .collect();
        v.sort();
        v
    }

    #[test]
    fn gguf_owned_paths_match_mmap() {
        if !Path::new(GGUF).exists() {
            eprintln!("skipping gguf parity: fixture not present ({GGUF})");
            return;
        }
        let bytes = fs::read(GGUF).expect("read gguf fixture");

        let p_path = parse_gguf(GGUF).expect("path parse");
        let p_bytes = parse_gguf_bytes(bytes).expect("bytes parse");
        let p_reader =
            parse_gguf_from_reader(fs::File::open(GGUF).expect("open")).expect("reader parse");

        let header = format!("{:?}", p_path.inspect());
        assert_eq!(header, format!("{:?}", p_bytes.inspect()));
        assert_eq!(header, format!("{:?}", p_reader.inspect()));

        let fp = fingerprint(&p_path);
        assert!(!fp.is_empty(), "fixture should have tensors");
        assert_eq!(
            fp,
            fingerprint(&p_bytes),
            "bytes-path tensors differ from mmap"
        );
        assert_eq!(
            fp,
            fingerprint(&p_reader),
            "reader-path tensors differ from mmap"
        );
    }

    #[test]
    fn gguf_malformed_bytes_is_clean_err() {
        assert!(parse_gguf_bytes(b"XXXX not a gguf file".to_vec()).is_err());
        assert!(parse_gguf_bytes(Vec::new()).is_err());
    }
}
