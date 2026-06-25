// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: the copy-based GGUF full-parse entry point
//! [`parse_gguf_bytes`] (Phase 6.13 Step 1) — the recommended path for
//! untrusted input. Drives the full header + metadata-KV + tensor-info parse
//! over an owned `Vec<u8>` (no mmap), reaching deeper than the inspect-only
//! targets. Must never panic / abort / OOM — only `Ok` or a clean `Err`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::parse_gguf_bytes(data.to_vec());
});
