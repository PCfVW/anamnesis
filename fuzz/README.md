# Fuzzing anamnesis

Coverage-guided fuzz harness for the four parser entry points, built on
[`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) + libFuzzer. Each
target feeds **arbitrary attacker bytes** to a parser and asserts the only
acceptable outcomes are `Ok(_)` or a clean `AnamnesisError` — libFuzzer treats
any panic, abort, OOB, or OOM as a crash.

This complements (does not replace) the Phase 6.6 / 6.7 security hardening and
the in-tree regression tests: the caps and guards are the *fix*, the regression
tests *pin* them, and fuzzing *searches* for inputs we didn't think of.

## Targets

| Target | Entry point | What it exercises |
|---|---|---|
| `fuzz_safetensors` | `parse_safetensors_header_from_reader` | safetensors header parse + cap |
| `fuzz_npz` | `inspect_npz_from_reader` | ZIP central-directory walk + NPY header parser (`NPY_MAX_HEADER_BYTES`) |
| `fuzz_gguf` | `inspect_gguf_from_reader` | metadata KV / typed-array readers + `read_bytes`/`ensure_remaining` |
| `fuzz_pth` | `inspect_pth_from_reader` | **ZIP walk + the pickle VM** (opcodes, `GLOBAL` allowlist, memo/mark stacks, recursion) — the highest-value target |

## Prerequisites — Linux / macOS / WSL (not Windows-MSVC)

libFuzzer needs an LLVM-backed nightly and a C linker; it does **not** build on
Windows-MSVC. On Windows, run this under **WSL** (Ubuntu) — where this harness
was authored and verified:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
# a C linker (gcc/cc) must be present; clang is not required
```

## Build & run

```bash
# compile-check every target
cargo +nightly fuzz build

# run a target (Ctrl-C to stop, or bound it)
cargo +nightly fuzz run fuzz_pth -- -max_total_time=60 -rss_limit_mb=2048
```

### Seeding the corpus (strongly recommended for the ZIP-based targets)

Random bytes almost never form a valid ZIP, so `fuzz_npz` / `fuzz_pth` reach
the interesting code (the pickle VM!) far faster when seeded from the real
fixtures already in the repo:

```bash
mkdir -p fuzz/corpus/fuzz_pth
cp tests/fixtures/pth_reference/algzoo_*.pth fuzz/corpus/fuzz_pth/
cargo +nightly fuzz run fuzz_pth fuzz/corpus/fuzz_pth -- -max_total_time=300

mkdir -p fuzz/corpus/fuzz_npz
cp tests/fixtures/npz_reference/*.npz fuzz/corpus/fuzz_npz/

mkdir -p fuzz/corpus/fuzz_safetensors
cp tests/fixtures/safetensors_reference/*.safetensors fuzz/corpus/fuzz_safetensors/
```

(The AlgZoo `.pth` fixtures are small — ideal seeds. The torchvision
`resnet*/vit_*` fixtures are large and slow the loop; don't seed with them.)

A crash writes the minimal reproducer to `fuzz/artifacts/<target>/`; replay it
with `cargo +nightly fuzz run <target> fuzz/artifacts/<target>/<file>` and turn
it into an in-tree regression fixture.

## What is and isn't committed

`Cargo.toml`, `fuzz_targets/`, this README, and `.gitignore` are tracked. The
generated `corpus/`, `artifacts/`, `target/`, and `coverage/` directories are
git-ignored. The whole `fuzz/` tree is excluded from the published crate via
`exclude = ["/fuzz"]` in the root `Cargo.toml`, so ordinary users never build
it and it never ships to crates.io.

## Status

Authored and run under WSL2 Ubuntu (nightly `rustc` + `cargo-fuzz` 0.13,
libFuzzer). Smoke campaigns (Phase 6.7): each target ~20 s unseeded
(~0.2–0.8 M runs each), plus a 62 s seeded `fuzz_pth` campaign (213 k runs,
coverage 1412 / features 5372, corpus 675 inputs, RSS steady ~489 MB) — **zero
crashes** across all four. Not yet wired into CI; a scheduled Linux fuzz job is
a candidate follow-up (it needs nightly + `cargo-fuzz` install in the runner,
so it is intentionally kept out of the stable-only push/PR matrix).
