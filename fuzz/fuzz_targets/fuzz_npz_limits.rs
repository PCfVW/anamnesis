// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: `parse_npz_with_limits` over arbitrary bytes **under a
//! `ParseLimits` derived from the input**, so the fuzzer co-explores
//! `(malformed file × tightened limits)`. NPZ exercises all four axes — the
//! single-allocation cap, the cumulative-byte `Budget` (`checked_add`), the
//! item-count gate, and the decompression-ratio cap (`checked_mul`). The
//! limit-enforcement branches must never panic/OOM — `Ok` or clean `Err`.

use std::io::Write;

use libfuzzer_sys::fuzz_target;

/// Derives four `ParseLimits` axes from the first 8 bytes (four little-endian
/// `u16`s); `0xFFFF` maps to unbounded so the fuzzer can also reach the
/// accept path. Small values keep the reject branches hot.
fn derive_limits(prefix: &[u8]) -> anamnesis::ParseLimits {
    let axis = |i: usize| -> u64 {
        let lo = prefix.get(i * 2).copied().unwrap_or(0xFF);
        let hi = prefix.get(i * 2 + 1).copied().unwrap_or(0xFF);
        match u16::from_le_bytes([lo, hi]) {
            u16::MAX => u64::MAX,
            v => u64::from(v),
        }
    };
    anamnesis::ParseLimits::default()
        .with_max_single_alloc(axis(0))
        .with_max_total_bytes(axis(1))
        .with_max_item_count(axis(2))
        .with_max_decompression_ratio(axis(3))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let (prefix, body) = data.split_at(8);
    let limits = derive_limits(prefix);

    let mut f = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if f.write_all(body).and_then(|()| f.flush()).is_err() {
        return;
    }
    let _ = anamnesis::parse_npz_with_limits(f.path(), &limits);
});
