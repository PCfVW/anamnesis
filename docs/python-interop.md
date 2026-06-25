# Python interop — design notes for the PyO3 bindings

<!-- Last updated: 2026-06-25, anamnesis v0.6.7 (Phase 6.13) -->

This is the contract the [Phase 8](../ROADMAP.md#phase-8-python-bindings-pyo3)
PyO3 bindings (`pip install anamnesis`) implement. It is written **before** the
bindings so the core can honour each guarantee and the API shape is frozen rather
than retrofitted. The Rust core is hardened library-side (every guarantee below
benefits Rust consumers too); this file records the Python-facing consequences.

## Panic safety & the `unwind` requirement (Phase 6.13 Step 3)

**Guarantee.** No public parse/inspect entry point panics or aborts on *any*
input. A malformed, truncated, or hostile artefact is always a clean
`Result::Err` (`AnamnesisError`), never an unwinding panic and never a `SIGBUS`
(the copy-based `parse_bytes` / `parse_*_from_reader` paths from Step 1 use no
mmap). This is pinned by `tests/no_panic.rs` (a `catch_unwind` battery over
adversarial inputs across every entry point, run in debug so integer-overflow
panics are in scope) and the coverage-guided `cargo fuzz` harness (including the
owned-path `fuzz_*_bytes` targets).

**Why the binding must build `panic = "unwind"`.** PyO3 wraps each `#[pyfunction]`
in a panic boundary that converts an unwinding Rust panic into a Python
`pyo3_runtime.PanicException` — *but only while panics unwind*. Under
`panic = "abort"` a panic is an immediate, uncatchable process kill: one hostile
upload would take down a multi-tenant worker, voiding the "never a dead worker"
pledge.

anamnesis's shipped library/CLI builds set `panic = "abort"` deliberately — that
is the correct fail-closed posture for a standalone parser (a *reachable* panic,
were one to exist, should kill the process rather than unwind through C FFI or an
inference loop). To reconcile the two, the cdylib is built with a dedicated
profile:

```toml
# Cargo.toml
[profile.release]          # CLI / library
panic = "abort"

[profile.python]           # the PyO3 cdylib — maturin `--profile python`
inherits = "release"       # identical codegen to the shipped wheel
panic = "unwind"           # so PyO3 yields a catchable PanicException
```

The Phase 8 maturin build selects `[profile.python]`; the contract is guarded in
stable CI today by `tests/panic_profile.rs` (asserts release = abort **and**
python = unwind), so it cannot silently regress before the cdylib exists.

**Net for a Python host.** Because (a) the core never panics on untrusted input
and (b) the wheel unwinds, the binding can wrap a hostile upload in
`try/except AnamnesisError` and map it to an HTTP response — a `LimitExceeded`
→ *413*, a `Parse` → *400*, a `DisallowedGlobal` → a flagged security event — and
even an *unexpected* panic (a bug, not an input) surfaces as a catchable
`PanicException` instead of killing the worker. See the error → exception map on
`AnamnesisError` (and the README "Parsing untrusted input" section).

<!-- Phase 6.13 Step 4 will add: ## NumPy / BF16 data-ownership contract -->
