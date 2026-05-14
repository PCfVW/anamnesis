// SPDX-License-Identifier: MIT OR Apache-2.0

//! `amn` binary entry point — short alias for `anamnesis`. The actual CLI
//! implementation lives in [`anamnesis::cli`] so it compiles exactly once
//! and links into both the `anamnesis` and `amn` binaries.

fn main() {
    if let Err(e) = anamnesis::cli::run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
