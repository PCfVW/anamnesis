# GGUF `IQ*` / `TQ*` / `MXFP4` — fixture-sourcing reference

One-page reference for finding HuggingFace GGUF files that ship each of the `IQ*`, `TQ*`, and `MXFP4` block types, for use in anamnesis cross-validation fixtures. Distilled from a sourcing investigation on 2026-04-22 after Phase 4.5 steps 1–2 landed.

Phase 4.5 step 7 (`cross-validation extension`) requires a 65 536-element fixture per block type, extracted from a real model tensor where possible and synthesised via the Python `gguf` package where not. This document tracks where each type comes from, which downloads we've already committed, and which we still owe.

## Contents

- [Block-layout summary](#block-layout-summary)
- [Model sourcing matrix](#model-sourcing-matrix)
- [Gotchas discovered along the way](#gotchas-discovered-along-the-way)
- [Remote-header probe recipe](#remote-header-probe-recipe)
- [Python `gguf` quantize coverage](#python-gguf-quantize-coverage)
- [Canonical sources](#canonical-sources)

---

## Block-layout summary

All sizes from `ggml-common.h` at commit cut 2026-04-22. `QK_K = 256`.

| `ggml_type` | Disc | Block | `type_size` | Notes |
|---|---:|---:|---:|---|
| `IQ4_NL` | 20 | 32 | 18 | shipped (`Phase 4.5 step 1`) |
| `IQ4_XS` | 23 | 256 | 136 | shipped (`Phase 4.5 step 1`) |
| `IQ2_XXS` | 16 | 256 | 66 | shipped (`Phase 4.5 step 2`) |
| `IQ2_XS` | 17 | 256 | 74 | shipped (`Phase 4.5 step 2`) |
| `IQ2_S` | 22 | 256 | 82 | shipped (`Phase 4.5 step 2`) |
| `IQ3_XXS` | 18 | 256 | 98 | shipped (`Phase 4.5 step 3`) |
| `IQ3_S` | 21 | 256 | 110 | shipped (`Phase 4.5 step 3`) |
| `IQ1_S` | 19 | 256 | **TBD** (~50 B) | **step 4, pending** |
| `IQ1_M` | 29 | 256 | **TBD** (~56 B) | **step 4, pending** |
| `TQ1_0` | 34 | 256 | 54 | **step 5, pending** (confirmed via `gguf.GGML_QUANT_SIZES`) |
| `TQ2_0` | 35 | 256 | 66 | **step 5, pending** (confirmed via `gguf.GGML_QUANT_SIZES`) |
| `MXFP4` | 39 | 32 | 17 | **step 6, pending** (confirmed via `gguf.GGML_QUANT_SIZES`) |

The shipped types' byte sizes can be verified in [`src/parse/gguf.rs::type_size()`](../../src/parse/gguf.rs). Pending types return `None` from `type_size()` today.

---

## Model sourcing matrix

Download sizes and tensor counts from remote-header probes performed 2026-04-22. Tensor thresholds are `≥ 65 536 elements` (the slice size Phase 4 fixtures use).

### Shipped kernels (reference — fixtures already committed)

| Kernel | Fixture | HF source | Download |
|---|---|---|---|
| `Q4_0` / `Q4_1` / `Q8_0` | `smollm2_q4_0.bin` / `_q4_1.bin` / `_q8_0.bin` | `bartowski/SmolLM2-135M-Instruct-GGUF` | already local |
| `Q5_0` / `Q2_K` | `tinyllama_q5_0.bin` / `_q2_k.bin` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | already local |
| `Q5_1` / `Q3_K`–`Q6_K` | `smollm2_*.bin` | `bartowski/SmolLM2-135M-Instruct-GGUF` (various `Q*_K_*.gguf`) | already local |
| `IQ4_NL` | `smollm2_iq4_nl.bin` | `SmolLM2-135M-Instruct-Q2_K.gguf` (mix contains IQ4_NL) | already local |
| `IQ4_XS` | `smollm2_iq4_xs.bin` | `SmolLM2-135M-Instruct-IQ4_XS.gguf` | already local |
| `IQ2_XXS` | `mistral_7b_iq2_xxs.bin` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ2_XXS.gguf` | **1.86 GB** (one-off) |
| `IQ2_XS` | `mistral_7b_iq2_xs.bin` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ2_XS.gguf` | **2.05 GB** (one-off) |
| `IQ2_S` | `qwen25_iq2_s.bin` | `bartowski/Qwen2.5-0.5B-Instruct-GGUF` / `...-IQ2_M.gguf` | already local (313 MB) |
| `IQ3_XXS` | `mistral_7b_iq3_xxs.bin` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ3_XXS.gguf` | **2.64 GB** (one-off) |
| `IQ3_S` | `mistral_7b_iq3_s.bin` | same `...-IQ3_XXS.gguf` file (ships 33 `IQ3_S` secondary tensors) | already local (via `IQ3_XXS`) |

### Pending kernels (Phase 4.5 steps 4–6)

| Kernel | Source strategy | HF source | New download |
|---|---|---|---:|
| `IQ1_S` | real model | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ1_S.gguf` — **156 `IQ1_S` tensors ≥ 65 K elem** | **1.50 GB** |
| `IQ1_M` | real model | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ1_M.gguf` — **156 `IQ1_M` tensors ≥ 65 K elem** | **1.64 GB** |
| `TQ1_0` | **synthetic** via `gguf.quants.quantize()` | `np.random.randn(65536) × 0.1`, f32 → `TQ1_0` → dequant reference | **0 GB** |
| `TQ2_0` | **synthetic** via `gguf.quants.quantize()` | same | **0 GB** |
| `MXFP4` | **synthetic** via `gguf.quants.quantize()` | same | **0 GB** |

**Total new download to finish Phase 4.5:** **~3.14 GB** (only Mistral-7B `IQ1_S` + `IQ1_M`; step 3's `IQ3_XXS.gguf` is now also local).

### Alternative real-model sources (for future cross-checking)

| Kernel | Alternative | Size | Status |
|---|---|---:|---|
| `TQ2_0` | `gianni-cor/bitnet_b1_58-large-TQ2_0/bitnet_b1_58-large-TQ2_0.gguf` | 207 MiB | real-model cross-check option — unverified |
| `TQ1_0` | `BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0_gguf_chunked` | chunked, ~1.5 GB | real-model cross-check option — chunked upload, unverified |
| `MXFP4` | `ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf` | 11.28 GiB | **72 `MXFP4` tensors confirmed** — too large to justify if synthetic path works |

The synthetic path is sufficient for bit-exact cross-validation (the ggml reference dequant is deterministic). Real-model sources are only needed if we ever want to validate against a byte-identical real-world byte stream — not a current requirement.

---

## Gotchas discovered along the way

Things that would have wasted hours if hit blind.

### 1. `Mistral-7B-Instruct-v0.3-IQ2_S.gguf` does **not** ship `IQ2_S` tensors

Despite the filename, remote-header probe reveals it actually ships **`IQ2_XS` (156) + `IQ3_S` (37) + `Q4_K` (32) + `Q5_K` (1) + `F32` (65)** — **zero `IQ2_S` tensors**. Confirmed 2026-04-22.

The filename advertises the **intended** base quant of the mix recipe, not the only quant actually used. For step 2 we sourced `IQ2_S` from `Qwen2.5-0.5B-Instruct-IQ2_M.gguf` instead (which does ship 21 real `IQ2_S` tensors).

Silver lining: the same file is a free source of `IQ3_S` tensors for step 3, so the 2.16 GB download already on disk is reusable.

### 2. Small models' `IQ2_M.gguf` mixes dropped `IQ2_XXS` and `IQ2_XS` entirely

For all of `bartowski/SmolLM2-135M-Instruct-GGUF`, `bartowski/Qwen2.5-0.5B-Instruct-GGUF`, `bartowski/Qwen2.5-1.5B-Instruct-GGUF`, `bartowski/Phi-3.5-mini-instruct-GGUF` — their `IQ2_M.gguf` mix contains `IQ2_S` + `IQ3_S` + `Q4_K` / `Q5_K` only, **no `IQ2_XXS` or `IQ2_XS`** anywhere. The `IQ2_M` base quant was revised in newer `llama.cpp` to skip the less-common 2-bit variants for small models.

Verified on 2026-04-22 on all four files. Mistral-7B-v0.3 is the smallest repo shipping **pure** `IQ2_XXS.gguf` and `IQ2_XS.gguf` as separate files.

### 3. `microsoft/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` uses `ggml_type 36`, not `TQ1_0`/`TQ2_0`

Type 36 is in the reserved/removed range (34 = `TQ1_0`, 35 = `TQ2_0`, 39 = `MXFP4`; 36–38 reserved). Microsoft's BitNet toolchain ships its own non-standard `I2_S` quant that anamnesis **cannot** dequantise without explicit support for type 36. Out of scope for Phase 4.5.

The community uploads (`gianni-cor/bitnet_b1_58-large-TQ2_0`, `BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0_gguf_chunked`) use the standard `TQ1_0` / `TQ2_0` types and are the correct real-model sources if ever needed.

### 4. Python `gguf.quants.quantize()` coverage is uneven

Confirmed 2026-04-22 on `gguf==0.17+`:

| Type | `quantize()` | `dequantize()` | Notes |
|---|:-:|:-:|---|
| `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` | ✅ | ✅ | legacy block quants |
| `Q2_K`–`Q8_K` | ✅ | ✅ | K-quants |
| `IQ4_NL`, `IQ4_XS` | ✅ | ✅ | `ggml` hand-coded |
| `IQ2_XXS`, `IQ2_XS`, `IQ2_S` | ❌ `NotImplementedError` | ✅ | **real-model source required** |
| `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M` | ❌ `NotImplementedError` | ✅ | **real-model source required** |
| `TQ1_0`, `TQ2_0`, `MXFP4` | ✅ | ✅ | **synthetic fixtures viable** |

This asymmetry drives the "synthetic vs real-model" column in the sourcing matrix above.

---

## Remote-header probe recipe

`hf-fm inspect` is `.safetensors`-only as of `hf-fetch-model v0.9.7`. For GGUF, we built a one-off Python probe that fetches only the first ~6 MB of a remote GGUF via HTTP `Range` request and parses the tensor-info table manually. This enumerates tensor types and counts without pulling the (multi-GB) tensor data.

Pattern:
```python
import struct, urllib.request
from collections import Counter

HEADER_BYTES = 6 * 1024 * 1024  # bump to 24 MB for files with huge metadata KV sections (e.g. gpt-oss, BitNet)

def probe(repo, filename):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES-1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        buf = r.read()
    assert buf[:4] == b"GGUF"
    # parse magic + version + n_tensors + n_kv, skip each KV pair by type,
    # then walk the tensor info table counting ggml_type discriminants.
    # See /tmp/probe_gguf_manual.py in the 2026-04-22 session for the full code.
```

The script worked for all Mistral-7B IQ variants, the IQ3_M/IQ3_XS small-model variants, and Qwen2.5-*-IQ3_XS at 6 MB; BitNet and gpt-oss needed 24 MB due to long metadata KV sections.

**Future:** extending `hf-fm inspect` to accept GGUF files (using the same Range-request trick) would make this a one-line CLI call. The format is well-defined and the magic is unambiguous. Flagged as a natural follow-up for the hf-fetch-model crate.

---

## Python `gguf` quantize coverage

Script that confirmed the coverage matrix above, kept inline for reproducibility:

```python
import numpy as np
from gguf import quantize, dequantize, GGMLQuantizationType, GGML_QUANT_SIZES

np.random.seed(42)
x = np.random.randn(65536).astype(np.float32) * 0.1
for name, tt in [
    ('MXFP4', GGMLQuantizationType.MXFP4),
    ('TQ1_0', GGMLQuantizationType.TQ1_0),
    ('TQ2_0', GGMLQuantizationType.TQ2_0),
    ('IQ3_XXS', GGMLQuantizationType.IQ3_XXS),
    ('IQ3_S',   GGMLQuantizationType.IQ3_S),
    ('IQ1_S',   GGMLQuantizationType.IQ1_S),
    ('IQ1_M',   GGMLQuantizationType.IQ1_M),
]:
    bs, ts = GGML_QUANT_SIZES[tt]
    try:
        raw = quantize(x, tt)
        y = dequantize(raw, tt)
        print(f'{name}: OK, block={bs}, type_size={ts}, raw={raw.nbytes} bytes')
    except NotImplementedError:
        print(f'{name}: dequant only (real-model source required)')
```

---

## Canonical sources

- **Block layouts**: `ggml-org/llama.cpp/ggml/src/ggml-common.h` — `block_iq*`, `block_tq*`, `block_mxfp4` structs with `_Static_assert` byte counts.
- **Dequant reference**: `ggml-org/llama.cpp/ggml/src/ggml-quants.c` — `dequantize_row_iq*`, `dequantize_row_tq*`, `dequantize_row_mxfp4` scalar functions (the Python `gguf` package's `dequantize()` mirrors these).
- **Python reference**: `gguf` PyPI package, `gguf/quants.py` — the `Q*Class.quantize_blocks` / `dequantize_blocks` methods.
- **Probe script** (session-local, 2026-04-22): `/tmp/probe_gguf_manual.py`, `/tmp/probe_all_remaining.py`.

Update this document whenever a new Phase 4.5 step lands or a new source surfaces.
