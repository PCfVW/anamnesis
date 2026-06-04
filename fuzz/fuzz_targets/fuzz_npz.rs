// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: NPZ header inspection over arbitrary bytes. Exercises the ZIP
//! central-directory walk and the NPY header parser (incl. the
//! `NPY_MAX_HEADER_BYTES` guard). Must never panic/OOM — `Ok` or clean `Err`.

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::inspect_npz_from_reader(Cursor::new(data));
});
