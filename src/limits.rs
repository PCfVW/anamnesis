// SPDX-License-Identifier: MIT OR Apache-2.0

//! Caller-configurable resource limits for the parsers.
//!
//! Every anamnesis parser already enforces a permanent, per-format constant
//! cap on each header-declared allocation (`NPZ_MAX_ARRAY_BYTES`,
//! `MAX_PKL_SIZE`, the `GGUF` `MAX_*` family, `MAX_SAFETENSORS_HEADER_BYTES`,
//! …). Those caps are tuned for server / GPU hosts and cannot be relaxed.
//! [`ParseLimits`] layers a *second*, caller-supplied ceiling on top of them:
//! a memory-constrained edge board or a per-slot `MLaaS` worker passes a
//! [`ParseLimits`] tightened to its own budget, and the parser rejects an
//! over-budget declaration fail-fast with [`AnamnesisError::Parse`](crate::AnamnesisError::Parse)
//! **before** it allocates.
//!
//! The ceiling is **tighten-only**. The effective limit at any allocation site
//! is `min(format_constant, parse_limit)`, so a caller can make the parser
//! stricter but never weaker than the built-in floor. [`ParseLimits::default`]
//! is *unbounded* on every axis ([`u64::MAX`] sentinel), so the default leaves
//! only the per-format constants in force — behaviour identical to a build with
//! no [`ParseLimits`] at all.

/// Caller-supplied resource budget threaded through the parser entry points.
///
/// Construct with [`ParseLimits::default`] (unbounded — today's behaviour) and
/// tighten individual axes with the `with_*` builders:
///
/// ```
/// use anamnesis::ParseLimits;
///
/// // Reject any single allocation over 256 MiB and any file declaring
/// // more than 4096 tensors / arrays / KV entries.
/// let limits = ParseLimits::default()
///     .with_max_single_alloc(256 * 1024 * 1024)
///     .with_max_item_count(4096);
/// ```
///
/// Pass it to the `parse_*_with_limits` entry points. Each axis is enforced
/// *before* the corresponding allocation, so an over-budget file returns an
/// error without first committing the memory.
// Deliberately not `Copy`: `ParseLimits` is a configuration/budget value passed
// by `&ParseLimits` everywhere (and borrowed by the `.pth` pickle VM), so a
// `Copy` derive would trip clippy's `trivially_copy_pass_by_ref` on the 16-byte
// struct. `Clone` covers the rare by-value need.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseLimits {
    /// Upper bound, in bytes, on any single header-declared buffer a parser
    /// allocates eagerly: an `NPZ` array and `NPY` header, a `.pth` `data.pkl`
    /// entry and each pickle string/bytes payload, a `GGUF` variable-length
    /// (string) read, and the safetensors header. [`u64::MAX`] means unbounded
    /// — only the per-format constant cap applies.
    ///
    /// Not covered (by design): incrementally-grown `GGUF` metadata arrays
    /// (bounded by the `MAX_ARRAY_LEN` element cap; their *aggregate* bytes are
    /// the `max_total_bytes` axis added in Phase 6.8 Step 2) and memory-mapped
    /// tensor bodies (no heap allocation at parse time).
    max_single_alloc_bytes: u64,

    /// Upper bound on the total number of declared items in a file — `GGUF`
    /// tensors and metadata KV entries, or `NPZ` archive entries. [`u64::MAX`]
    /// means unbounded — only the per-format constant cap applies.
    max_item_count: u64,
}

impl ParseLimits {
    /// Returns an unbounded budget: every axis is [`u64::MAX`], so only the
    /// permanent per-format constant caps apply. Identical to
    /// [`ParseLimits::default`], but usable in `const` context.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            max_single_alloc_bytes: u64::MAX,
            max_item_count: u64::MAX,
        }
    }

    /// Sets the maximum single-allocation budget, in bytes, and returns the
    /// updated value. See [`ParseLimits::max_single_alloc_bytes`].
    #[must_use]
    pub const fn with_max_single_alloc(mut self, bytes: u64) -> Self {
        self.max_single_alloc_bytes = bytes;
        self
    }

    /// Sets the maximum declared-item-count budget and returns the updated
    /// value. See [`ParseLimits::max_item_count`].
    #[must_use]
    pub const fn with_max_item_count(mut self, count: u64) -> Self {
        self.max_item_count = count;
        self
    }

    /// Returns the maximum single-allocation budget, in bytes ([`u64::MAX`] if
    /// unbounded).
    #[must_use]
    pub const fn max_single_alloc_bytes(&self) -> u64 {
        self.max_single_alloc_bytes
    }

    /// Returns the maximum declared-item-count budget ([`u64::MAX`] if
    /// unbounded).
    #[must_use]
    pub const fn max_item_count(&self) -> u64 {
        self.max_item_count
    }

    /// Rejects a single allocation of `requested` bytes if it exceeds the
    /// caller's [`ParseLimits::max_single_alloc_bytes`] budget. Called at every
    /// site that allocates a header-declared buffer, immediately after the
    /// permanent per-format constant check.
    ///
    /// `context` names the offending region for the error message (e.g.
    /// `` "NPZ array `weight`" ``).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`](crate::AnamnesisError::Parse) if
    /// `requested` exceeds the configured maximum single allocation.
    pub(crate) fn check_alloc(&self, requested: u64, context: &str) -> crate::Result<()> {
        if requested > self.max_single_alloc_bytes {
            return Err(crate::AnamnesisError::Parse {
                reason: format!(
                    "requested allocation {requested} bytes exceeds caller \
                     ParseLimits max_single_alloc {} ({context})",
                    self.max_single_alloc_bytes
                ),
            });
        }
        Ok(())
    }

    /// Rejects a declared item count if it exceeds the caller's
    /// [`ParseLimits::max_item_count`] budget. Called at every site that reads
    /// a file-declared count of tensors / arrays / KV entries, immediately
    /// after the permanent per-format constant check.
    ///
    /// `context` names the counted population for the error message (e.g.
    /// `"GGUF tensor count"`).
    ///
    /// # Errors
    ///
    /// Returns [`AnamnesisError::Parse`](crate::AnamnesisError::Parse) if
    /// `count` exceeds the configured maximum item count.
    // Only the `npz` and `gguf` parse paths read a file-declared item count;
    // with neither feature enabled this helper has no caller, which is correct
    // (not dead code in the public sense). `check_alloc` always has a caller
    // via the always-on safetensors path, so it needs no such guard.
    #[cfg_attr(not(any(feature = "npz", feature = "gguf")), allow(dead_code))]
    pub(crate) fn check_item_count(&self, count: u64, context: &str) -> crate::Result<()> {
        if count > self.max_item_count {
            return Err(crate::AnamnesisError::Parse {
                reason: format!(
                    "declared item count {count} exceeds caller ParseLimits \
                     max_item_count {} ({context})",
                    self.max_item_count
                ),
            });
        }
        Ok(())
    }
}

impl Default for ParseLimits {
    /// Returns an unbounded budget — every axis [`u64::MAX`] — so the default
    /// leaves only the permanent per-format constant caps in force. Parsing
    /// with `ParseLimits::default()` is byte-for-byte identical to parsing
    /// through the limit-free entry points.
    fn default() -> Self {
        Self::unbounded()
    }
}

#[cfg(test)]
mod tests {
    use super::ParseLimits;

    #[test]
    fn default_is_unbounded() {
        let limits = ParseLimits::default();
        assert_eq!(limits.max_single_alloc_bytes(), u64::MAX);
        assert_eq!(limits.max_item_count(), u64::MAX);
        assert_eq!(limits, ParseLimits::unbounded());
    }

    #[test]
    fn builders_set_only_their_axis() {
        let limits = ParseLimits::default().with_max_single_alloc(1024);
        assert_eq!(limits.max_single_alloc_bytes(), 1024);
        assert_eq!(limits.max_item_count(), u64::MAX);

        let limits = ParseLimits::default().with_max_item_count(8);
        assert_eq!(limits.max_item_count(), 8);
        assert_eq!(limits.max_single_alloc_bytes(), u64::MAX);
    }

    #[test]
    fn check_alloc_boundary() {
        let limits = ParseLimits::default().with_max_single_alloc(1024);
        // At the cap passes; one over fails.
        assert!(limits.check_alloc(1024, "ctx").is_ok());
        assert!(limits.check_alloc(1025, "ctx").is_err());
        // Unbounded never fires.
        assert!(ParseLimits::default().check_alloc(u64::MAX, "ctx").is_ok());
    }

    #[test]
    fn check_item_count_boundary() {
        let limits = ParseLimits::default().with_max_item_count(4);
        assert!(limits.check_item_count(4, "ctx").is_ok());
        assert!(limits.check_item_count(5, "ctx").is_err());
        assert!(ParseLimits::default()
            .check_item_count(u64::MAX, "ctx")
            .is_ok());
    }
}
