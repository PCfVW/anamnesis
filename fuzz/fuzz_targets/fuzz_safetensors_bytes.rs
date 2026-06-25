// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: the copy-based safetensors full-parse entry point
//! [`parse_bytes`] (Phase 6.13 Step 1) — the recommended path for untrusted
//! input. Parses a safetensors header over an owned `Vec<u8>` (no mmap), the
//! exact path the Phase 8 Python bindings route uploads through. Must never
//! panic / abort / OOM on arbitrary bytes — only `Ok` or a clean `Err`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::parse_bytes(data.to_vec());
});
