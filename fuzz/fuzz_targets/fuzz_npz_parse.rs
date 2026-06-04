// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: full NPZ parse over arbitrary bytes. Unlike `fuzz_npz` (which
//! inspects headers only), this materialises the input to a temp file and
//! calls `parse_npz`, exercising the **data-extraction path** — `read_array_data`
//! and the `entry.size()` cross-check (Phase 6.7 Step 1). Must never
//! panic/OOM — `Ok` or clean `Err`.

use std::io::Write;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut f = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if f.write_all(data).and_then(|()| f.flush()).is_err() {
        return;
    }
    let _ = anamnesis::parse_npz(f.path());
});
