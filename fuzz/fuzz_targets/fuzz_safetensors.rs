// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target: the safetensors header parser must never panic, over-read, or
//! OOM on arbitrary bytes — it must return `Ok` or a clean `AnamnesisError`.
//! libFuzzer treats any panic/abort/OOM as a crash.

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = anamnesis::parse_safetensors_header_from_reader(Cursor::new(data));
});
