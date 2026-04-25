# Claude Code Instructions

## Coding Conventions

Always apply the rules in `CONVENTIONS.md` to all code changes. Every annotation pattern, doc-comment rule, and style rule in that file is mandatory.

Every `.rs` file must start with `// SPDX-License-Identifier: MIT OR Apache-2.0` as its first line.

## Pre-commit Checks

Before every commit, run and fix any issues from:
1. `cargo build --features cli` (ensures CLI binary is current before integration tests)
2. `cargo fmt`
3. `cargo clippy --all-targets --all-features -- -D warnings`
4. `cargo test`
4. Update `CHANGELOG.md` — add a bullet under the `[Unreleased]` section for any user-visible change (new feature, fix, breaking change). Follow [Keep a Changelog](https://keepachangelog.com/) categories: Added, Changed, Fixed, Removed.

## Release Checklist

Before tagging a release (`v*`), complete these steps in order:
1. Bump `version` in `Cargo.toml` to match the tag (e.g., `"0.4.0"` for `v0.4.0`)
2. Run `cargo check` to update `Cargo.lock`
3. Rename `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`
4. **Dry-run the publish workflow locally** — runs the same gauntlet `.github/workflows/publish.yml` runs, plus `cargo publish --dry-run`. The dry-run catches packaging issues that the regular CI does not exercise: missing `Cargo.toml` metadata (`license`, `description`, `repository`, `readme`, `keywords`, `categories`), files referenced by `include`/`exclude` that don't exist, the 10 MiB published-tarball cap, or version-already-on-registry conflicts. Every step must succeed before committing the version bump:
   ```powershell
   cargo fmt --check;
   cargo clippy --all-targets -- -D warnings;
   cargo clippy --all-targets --all-features -- -D warnings;
   cargo test --all-features;
   $env:RUSTDOCFLAGS = "-D warnings";
   cargo doc --all-features --no-deps;
   $env:RUSTDOCFLAGS = $null;
   cargo publish --dry-run --allow-dirty
   ```
   `--allow-dirty` is required because step 4 runs **before** the bump commit (steps 1–3 leave `Cargo.toml`/`Cargo.lock`/`CHANGELOG.md` uncommitted by design). The real publish workflow runs against a tagged commit and never sees a dirty tree. If `cargo publish --dry-run` flags issues, fix them in-place before creating the bump commit.
5. Commit as `bump version to vX.Y.Z, update changelog date`
6. Push the commit, wait for CI to go GREEN
7. `git tag vX.Y.Z && git push origin vX.Y.Z`
8. Wait for the publish workflow to go GREEN

**Never tag before bumping `Cargo.toml`** — `cargo publish` will reject the crate if the version in the registry already exists.

## Shell Environment

The user runs PowerShell on Windows. Use PowerShell syntax for all suggested commands:
- Use `$env:VAR="value";` instead of `VAR=value` for environment variables
- Use semicolons to chain commands, not `&&`
- Use forward slashes in paths when running Rust/cargo commands
