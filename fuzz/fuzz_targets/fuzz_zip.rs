// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: the vendored read-only ZIP central-directory reader
//! (`src/parse/zip.rs`, Phase 6.12) over arbitrary bytes.
//!
//! The reader is crate-private, so this target drives it through every public
//! entry point that routes container parsing through it: `parse_pth` (mmap /
//! `SliceSource` substrate), `parse_npz` (reader substrate, STORED + DEFLATE),
//! and `inspect_pth_from_reader` (reader substrate). The index-level
//! *differential* check against the `zip` crate lives in the in-crate unit
//! tests (`differential_random_archives` + the hand-built fixtures), which can
//! reach the private reader; here the contract is the robustness floor shared
//! by every parser: arbitrary bytes must yield `Ok` or a clean `Err` — never a
//! panic, OOM, or stack overflow.

use std::io::{Cursor, Write};

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Reader substrate (no temp file): the lightest path over the container.
    let _ = anamnesis::inspect_pth_from_reader(Cursor::new(data));

    // mmap + path substrates: materialise once, exercise both format readers.
    let mut f = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if f.write_all(data).and_then(|()| f.flush()).is_err() {
        return;
    }

    // `.pth` mmap path → vendored reader → pickle VM + tensor extraction.
    if let Ok(parsed) = anamnesis::parse_pth(f.path()) {
        let _ = parsed.tensors();
    }

    // `.npz` reader path → vendored reader → STORED / DEFLATE entry inflate.
    let _ = anamnesis::parse_npz(f.path());
});
