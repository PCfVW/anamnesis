// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: GGUF header + metadata inspection over arbitrary bytes.
//! Exercises the metadata KV reader, the typed-array reader, and the
//! variable-length `read_bytes` path (incl. the `ensure_remaining`
//! reject-before-allocate guard). Must never panic/OOM — `Ok` or clean `Err`.

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::inspect_gguf_from_reader(Cursor::new(data));
});
