// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: full PyTorch `.pth` parse over arbitrary bytes. Unlike
//! `fuzz_pth` (reader/inspect), this materialises the input to a temp file and
//! calls `parse_pth`, exercising the **mmap path and tensor extraction**
//! (`build_entry_index`, the `MAX_PKL_SIZE` mmap guard, stride/offset
//! `copy_to_contiguous`) on top of the pickle VM. Must never panic/OOM —
//! `Ok` or clean `Err`.

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
    if let Ok(parsed) = anamnesis::parse_pth(f.path()) {
        // Resolve the tensor views (storage offsets, strides, byteswap) rather
        // than stopping at the parsed pickle metadata.
        let _ = parsed.tensors();
    }
});
