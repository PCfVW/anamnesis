// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: the copy-based PyTorch `.pth` full-parse entry point
//! [`parse_pth_bytes`] (Phase 6.13 Step 1) — the recommended path for untrusted
//! input. Drives the ZIP container walk **and the pickle VM** over owned bytes
//! (no mmap), the deepest state machine in the crate, then extracts tensor
//! metadata. Must never panic / abort / OOM / recurse-to-overflow — only `Ok`
//! or a clean `Err`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::parse_pth_bytes(data.to_vec());
});
