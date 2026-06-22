// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw-byte backing store shared by every parsed-model type.
//!
//! A `Parsed*` type (`ParsedModel`, `ParsedGguf`, `ParsedPth`) holds its source
//! bytes behind a `Backing`, which is either:
//!
//! - `Backing::Mmap` — a memory-mapped view of the file, used by the path-based
//!   `parse*` entry points. The OS pages bytes in lazily (no heap), the
//!   **trusted-local-file fast path**. A truncated or concurrently-written file
//!   can fault the mapping (`SIGBUS`), so this variant is *not* for untrusted
//!   input.
//! - `Backing::Owned` — an owned `Vec<u8>` read fully into the heap, used by the
//!   copy-based `parse_*_bytes` / `parse_*_from_reader` entry points. No mmap, so
//!   a truncated or hostile source yields a clean `Err`, never a `SIGBUS` — the
//!   **recommended path for untrusted input**.
//!
//! Both variants `Deref` to `[u8]`, so the structure parsers and tensor-data
//! accessors read the bytes identically regardless of origin.

use std::ops::Deref;

/// Byte storage behind a parsed model — a memory map or an owned copy.
///
/// Crate-internal: callers never see a `Backing`; the `Parsed*` types expose
/// their bytes through `&[u8]` / `Cow<[u8]>` accessors instead.
#[derive(Debug)]
pub(crate) enum Backing {
    /// Memory-mapped file bytes (lazy paging, no heap). Constructed only on the
    /// trusted path-based `parse*` entry points; can fault if the file is
    /// truncated or written concurrently.
    Mmap(memmap2::Mmap),
    /// Owned bytes on the heap, read fully up front. No mmap → cannot fault on a
    /// truncated/concurrent source. Used by the copy-based untrusted-input path.
    Owned(Vec<u8>),
}

impl Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Owned(v) => v,
        }
    }
}
