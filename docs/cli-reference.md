# CLI reference

<!-- Last updated: 2026-06-25, anamnesis v0.6.8 -->

Every subcommand, flag, and output shape for the `anamnesis` / `amn` CLI. The
[README](../README.md) has the quick tour; this is the complete reference.

## Install

```sh
cargo install anamnesis --features cli,pth,gguf
```

Installs two binaries: **`anamnesis`** and **`amn`** (a short alias — identical
behaviour). Add features for the formats and capabilities you need:

| Feature | Enables |
|---|---|
| `cli` | the CLI itself (pulls in `clap`) — **required** for the binaries |
| `pth` | PyTorch `.pth` / `.pt` parsing & conversion |
| `gguf` | GGUF parsing, dequantization & writing |
| `npz` | NumPy `.npz` parsing & conversion |
| `gptq` / `awq` / `bnb` | GPTQ / AWQ / BitsAndBytes dequantization (and BnB-NF4 encode for `bnb`) |
| `ollama` | the `ollama:` URL scheme (see below) |
| `indicatif` | a progress bar during `remember` |

`amn --version` and `amn --help` (and `amn <command> --help`) are always
available.

## Commands at a glance

| Command | Description |
|---|---|
| `amn parse <file>` | Parse and summarize a model file (`.safetensors`, `.pth`/`.pt`, `.npz`, `.gguf`, `.bin`) |
| `amn inspect <file>` *(alias `info`)* | Show format, tensor counts, size estimates, dtypes, byte order |
| `amn remember <file>` *(alias `dequantize`)* | Dequantize to BF16 (safetensors) or convert `.pth`/`.gguf` → `.safetensors` |
| `amn convert <file> --to <target>` | Convert any input to `safetensors` / `gguf` / `bnb-nf4` through one dispatch |

Format detection is automatic (see [Format detection](#format-detection)).

---

## `amn parse <file>`

Parse the file and print a per-format summary, including a per-tensor table.

| Argument | Description |
|---|---|
| `<file>` | Path to the model file, or an `ollama:` URL |

Example:

```
$ amn parse model.pth
Parsed model.pth (PyTorch state_dict)
  Tensors:    3
  Total size: 1.7 KB
  Dtypes:     F32
  Byte order: little-endian

  rnn.weight_ih_l0               F32 [16, 1]         64 B
  rnn.weight_hh_l0               F32 [16, 16]        1.0 KB
  linear.weight                  F32 [10, 16]        640 B
```

For safetensors, the summary breaks tensors down by role (quantized / scale /
zero-point / `g_idx` / passthrough) and the detected quantization scheme.

## `amn inspect <file>` (alias `info`)

Header-only summary — format, tensor count, total size, dtypes, byte order /
alignment — without materialising tensor data.

| Argument | Description |
|---|---|
| `<file>` | Path to the model file, or an `ollama:` URL |

```
$ amn inspect weights.npz
Format:      NPZ archive
Tensors:     5
Total size:  160 B
Dtypes:      F32
```

`amn info` is an exact alias.

## `amn remember <file>` (alias `dequantize`)

Recover precision: dequantize a quantized safetensors to BF16, or convert a
`.pth` / `.gguf` to safetensors (dequantizing any quantized GGUF tensors to BF16,
passing scalar tensors through).

| Flag | Default | Description |
|---|---|---|
| `--to <value>` | `bf16` | Target. Only `bf16` / `safetensors` are accepted for `.pth` / `.gguf` inputs (they always produce safetensors). |
| `--output`, `-o <path>` | *(derived)* | Output path; derived from the input if omitted (see [Output paths](#output-path-derivation)). |

```
$ amn remember model.pth
Converting model.pth → model.safetensors
  3 tensors, 1.7 KB
  Done.
```

`amn dequantize` is an exact alias. An `.npz` input is rejected with a clear
`Unsupported` error (NPZ tensors are already full precision).

## `amn convert <file> --to <target>`

Convert any supported input to a different format through a single dispatch
(Phase 6, v0.6.0).

| Flag | Description |
|---|---|
| `--to <target>` | **Required.** One of `safetensors` (alias `bf16`), `gguf`, `bnb-nf4` (aliases `bnb_nf4` / `nf4`). Case-insensitive. |
| `--output`, `-o <path>` | Output path; derived from the input if omitted. |

### Conversion matrix (v0.6.0)

| Input ↓ \ Target → | `safetensors` / `bf16` | `gguf` | `bnb-nf4` |
|---|---|---|---|
| **safetensors** | ✅ dequant or lossless passthrough | ✅ scalar passthrough¹ | ✅ encode NF4² |
| **`.pth`** | ✅ | ✅¹ | ❌ `Unsupported` |
| **`.npz`** | ✅ | ✅¹ | ❌ `Unsupported` |
| **`.gguf`** | ✅ dequant to BF16 | ❌ `Unsupported`³ | ❌ `Unsupported` |

¹ `gguf` target requires the `gguf` feature; writes an **unquantised** GGUF
(quantised GGUF emit — `gguf-q4km`, … — is deferred to Phase 6.14 / 7.5 through
the same dispatch). safetensors → gguf rejects a *quantised* safetensors input
(dequantise to BF16 first).
² `bnb-nf4` requires the `bnb` feature; 2-D F32/F16/BF16 tensors are encoded to
NF4, biases / norms / embeddings pass through as BF16. Rejects a quantised input.
³ `gguf → gguf` (dequantise-in-place) lands in **Phase 6.14**.

Combinations marked ❌ return a clear `AnamnesisError::Unsupported` rather than
silently falling through. A combination whose Cargo feature is disabled returns
an `Unsupported` error naming the feature to rebuild with.

```
$ amn convert model.npz --to safetensors
Converting model.npz -> model-bf16.safetensors (NPZ -> safetensors)
  Wrote 5 tensors -> model-bf16.safetensors
```

---

## Output path derivation

When `--output` / `-o` is omitted, the output path is derived from the input:
a known quantization suffix is stripped from the stem, then `-{target}.{ext}` is
appended.

- `remember`: → `<stem>-bf16.safetensors`
- `convert`: → `<stem>-{bf16|gguf|bnb-nf4}.{safetensors|gguf}`

Stripped suffixes (case-sensitive, longest-first) include `-fp8`, `-GPTQ-Int4`,
`-gptq`, `-AWQ`, `-awq`, `-bnb-4bit`, `-bnb-int8`, `-4bit`, `-int8`, … — e.g.
`model-GPTQ-Int4.safetensors` → `model-bf16.safetensors`,
`weights-fp8.safetensors --to gguf` → `weights-gguf.gguf`.

## `ollama:` URL scheme

Build with the `ollama` feature (`cargo install anamnesis --features cli,gguf,ollama`)
and **every** subcommand accepts an `ollama:` URL in place of a file path:

```
$ amn inspect ollama:llama3.2:1b
Format:      GGUF v3
Arch:        llama
Tensors:     147
Total size:  1.22 GB
Dtypes:      F32, Q8_0
Alignment:   32 bytes
```

- `ollama:<model>:<tag>` — resolves the manifest at
  `~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>` to its
  model-layer GGUF blob under `~/.ollama/models/blobs/sha256-<hash>`.
- `ollama:<model>` — same, with the tag defaulting to `latest`.

Pure path arithmetic plus a single JSON read — no `ollama` CLI shell-out, no Go
interop. Honours the `OLLAMA_MODELS` environment variable for non-default cache
locations. Without the `ollama` feature, an `ollama:` input returns a clear
`Unsupported` error.

## Format detection

Detection is automatic, by extension then magic bytes:

- `.safetensors` → safetensors
- `.pth` / `.pt` → PyTorch pickle (`pth` feature)
- `.npz` → NumPy NPZ (`npz` feature)
- `.gguf` → GGUF (`gguf` feature)
- `.bin` → probed for ZIP magic (`PK\x03\x04` → PyTorch) then GGUF magic, else safetensors
- any other extension → probed for GGUF magic, else safetensors

If the input matches a format whose Cargo feature is **not** enabled, the CLI
returns an `Unsupported` error naming the feature to rebuild with (e.g. a `.gguf`
file in a build without `gguf`) — never a cryptic downstream failure from
misrouting to the safetensors parser.

## Exit codes

`0` on success; `1` on any error (the underlying `AnamnesisError` message is
printed to stderr). For the error *kinds* and how a host can branch on them, see
the [README error taxonomy](../README.md#parsing-untrusted-input).
