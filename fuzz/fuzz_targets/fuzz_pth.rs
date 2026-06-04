// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: PyTorch `.pth` inspection over arbitrary bytes. The highest-
//! value target: it drives the ZIP walk **and the pickle VM** (opcode
//! interpreter, `GLOBAL` allowlist, memo/mark stacks, recursive structure,
//! `_rebuild_tensor_v2` arg parsing) — the one deep state machine in the
//! crate. Must never panic/OOM/recurse-to-overflow — `Ok` or clean `Err`.

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::inspect_pth_from_reader(Cursor::new(data));
});
