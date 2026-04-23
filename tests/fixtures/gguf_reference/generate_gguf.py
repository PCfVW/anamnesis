#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate GGUF dequantization reference fixtures from real models.

Reads real GGUF model files (downloaded via ``hf-fm``), extracts a small
slice of quantized tensor data, dequantizes with the ``gguf`` Python
package (the official reference implementation from ``ggml-org``), and
writes compact binary fixture files that the Rust cross-validation test
can compare against.

The ``gguf`` package's ``dequantize`` function mirrors
``ggml-quants.c``'s ``dequantize_row_*`` reference implementations — the
same code path that ``llama.cpp`` uses — so matching it is equivalent to
matching ``llama.cpp`` output.

Fixture binary format (all little-endian):
  4 bytes: ggml_type discriminant (u32, e.g. 2 = Q4_0)
  4 bytes: n_elements (u32)
  4 bytes: raw_data_len (u32) — bytes of quantized block data
  4 bytes: golden_len (u32) — bytes of BF16 output (= n_elements × 2)
  [raw_data_len bytes]: raw quantized block data
  [golden_len bytes]: expected BF16 output

BF16 conversion uses round-to-nearest-even, identical to anamnesis's
``f32_bits_to_bf16_bits``:

    lsb = (bits >> 16) & 1
    bf16 = (bits + 0x7FFF + lsb) >> 16

Models (download via ``hf-fm download-file --flat --output-dir models/``):
  bartowski/SmolLM2-135M-Instruct-GGUF — Q4_0, Q4_1, Q8_0, Q2_K–Q6_K, IQ4_NL, IQ4_XS
  TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF — Q5_0
  bartowski/Qwen2.5-0.5B-Instruct-GGUF — IQ2_S (via the IQ2_M mix file)
  bartowski/Mistral-7B-Instruct-v0.3-GGUF — IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S
    (IQ3_S via the IQ3_XXS.gguf file's 33 IQ3_S secondary tensors — one
     download, two fixtures)

  NOTE: Mistral-7B-Instruct-v0.3-IQ2_S.gguf is *misleadingly named* — it actually
  ships IQ2_XS + IQ3_S tensors, NO IQ2_S. Use the Qwen2.5-0.5B IQ2_M file for IQ2_S
  instead. Verified by remote-header probe 2026-04-22.

Usage:
  pip install gguf numpy
  python generate_gguf.py
"""

import struct
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType, GGML_QUANT_SIZES, dequantize

# Number of elements to extract per fixture.  65 536 = 256 × 256, same
# slice size the FP8/GPTQ/AWQ/BnB fixtures use.  For legacy quants
# (block_size=32) that's 2 048 blocks; for K-quants (block_size=256)
# that's 256 super-blocks.
SLICE_ELEMENTS = 65_536

MODELS_DIR = Path(__file__).parent / "models"

# Each entry: (fixture_name, gguf_filename, target GGMLQuantizationType)
FIXTURES = [
    # ---- Legacy quants (block_size=32) ----
    # bartowski SmolLM2
    ("smollm2_q4_0",  "SmolLM2-135M-Instruct-Q4_0.gguf",   GGMLQuantizationType.Q4_0),
    ("smollm2_q4_1",  "SmolLM2-135M-Instruct-Q4_0.gguf",   GGMLQuantizationType.Q4_1),
    ("smollm2_q8_0",  "SmolLM2-135M-Instruct-Q8_0.gguf",   GGMLQuantizationType.Q8_0),
    # TheBloke TinyLlama — Q5_0 and Q5_1 not shipped by bartowski
    ("tinyllama_q5_0", "tinyllama-1.1b-chat-v1.0.Q5_0.gguf", GGMLQuantizationType.Q5_0),
    ("smollm2_q5_1",  "SmolLM2-135M-Instruct-Q3_K_M.gguf", GGMLQuantizationType.Q5_1),
    # IQ4_NL (non-linear 4-bit, 32-element block, shares kvalues_iq4nl codebook
    # with IQ4_XS). SmolLM2's Q2_K file mixes Q3_K with IQ4_NL for high-impact
    # layers, so no extra model download is needed.
    ("smollm2_iq4_nl", "SmolLM2-135M-Instruct-Q2_K.gguf",   GGMLQuantizationType.IQ4_NL),
    # ---- K-quants (block_size=256) ----
    # TinyLlama Q2_K file — SmolLM2's "Q2_K" file uses Q3_K+IQ4_NL, not Q2_K
    ("tinyllama_q2_k", "tinyllama-1.1b-chat-v1.0.Q2_K.gguf", GGMLQuantizationType.Q2_K),
    # SmolLM2's "Q2_K" file ships Q3_K tensors for higher-impact layers
    ("smollm2_q3_k",  "SmolLM2-135M-Instruct-Q2_K.gguf",   GGMLQuantizationType.Q3_K),
    ("smollm2_q4_k",  "SmolLM2-135M-Instruct-Q4_K_M.gguf", GGMLQuantizationType.Q4_K),
    ("smollm2_q5_k",  "SmolLM2-135M-Instruct-Q5_K_M.gguf", GGMLQuantizationType.Q5_K),
    ("smollm2_q6_k",  "SmolLM2-135M-Instruct-Q6_K.gguf",   GGMLQuantizationType.Q6_K),
    # IQ4_XS (non-linear 4-bit super-block, 256 elements, 136 bytes). Most
    # widely used member of the IQ* family on HuggingFace as of 2025-2026.
    ("smollm2_iq4_xs", "SmolLM2-135M-Instruct-IQ4_XS.gguf", GGMLQuantizationType.IQ4_XS),
    # IQ2 variants (2-bit super-quants, lattice codebooks).  Small models'
    # "IQ2_M" mix quantisers mostly dropped IQ2_XXS/IQ2_XS in favour of
    # IQ2_S for critical tensors, so IQ2_XXS and IQ2_XS fixtures come from
    # Mistral-7B-v0.3 (smallest repo shipping pure IQ2_XXS/IQ2_XS files).
    ("mistral_7b_iq2_xxs", "Mistral-7B-Instruct-v0.3-IQ2_XXS.gguf", GGMLQuantizationType.IQ2_XXS),
    ("mistral_7b_iq2_xs",  "Mistral-7B-Instruct-v0.3-IQ2_XS.gguf",  GGMLQuantizationType.IQ2_XS),
    # IQ2_S from Qwen2.5-0.5B's IQ2_M mix (21 IQ2_S tensors inside).
    ("qwen25_iq2_s", "Qwen2.5-0.5B-Instruct-IQ2_M.gguf", GGMLQuantizationType.IQ2_S),
    # ---- IQ3 variants (3-bit super-quants) ----
    # Mistral-7B-v0.3-IQ3_XXS.gguf primary type is IQ3_XXS (96 tensors); it
    # ALSO ships 33 IQ3_S tensors as the high-precision secondary — so a
    # single file covers both IQ3 fixtures with no extra download.
    ("mistral_7b_iq3_xxs", "Mistral-7B-Instruct-v0.3-IQ3_XXS.gguf", GGMLQuantizationType.IQ3_XXS),
    ("mistral_7b_iq3_s",   "Mistral-7B-Instruct-v0.3-IQ3_XXS.gguf", GGMLQuantizationType.IQ3_S),
    # Q8_1, Q8_K: not shipped by any real model — unit tests cover these
]


def f32_array_to_bf16_bytes(f32_arr: np.ndarray) -> bytes:
    """Convert an f32 numpy array to BF16 bytes (round-to-nearest-even).

    Matches anamnesis's ``f32_bits_to_bf16_bits`` exactly:
        lsb = (bits >> 16) & 1
        bf16 = (bits + 0x7FFF + lsb) >> 16
    """
    bits = f32_arr.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    bf16_bits = ((bits + rounding_bias) >> 16).astype(np.uint16)
    return bf16_bits.tobytes()


def generate_fixture(name: str, gguf_path: Path, target_type: GGMLQuantizationType) -> None:
    """Extract one tensor slice and write the binary fixture."""
    reader = GGUFReader(str(gguf_path))

    # Find the first tensor of the target quant type with enough elements.
    tensor = None
    for t in reader.tensors:
        if t.tensor_type == target_type and t.n_elements >= SLICE_ELEMENTS:
            tensor = t
            break

    if tensor is None:
        print(f"  SKIP {name}: no {target_type.name} tensor with >= {SLICE_ELEMENTS} elements")
        return

    block_size, type_size = GGML_QUANT_SIZES[target_type]
    n_blocks = SLICE_ELEMENTS // block_size
    raw_byte_len = n_blocks * type_size

    # Flatten and extract the first `raw_byte_len` bytes.
    raw_flat = tensor.data.reshape(-1)[:raw_byte_len]
    raw_bytes = raw_flat.tobytes()
    assert len(raw_bytes) == raw_byte_len

    # Dequantize with the gguf reference implementation.
    best_us = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        f32 = dequantize(np.frombuffer(raw_bytes, dtype=np.uint8), target_type)
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)

    assert f32.shape == (SLICE_ELEMENTS,), f"expected {SLICE_ELEMENTS}, got {f32.shape}"
    assert f32.dtype == np.float32

    # Convert f32 → BF16 using round-to-nearest-even.
    golden_bytes = f32_array_to_bf16_bytes(f32)
    golden_len = SLICE_ELEMENTS * 2
    assert len(golden_bytes) == golden_len

    # Write fixture.
    output_path = Path(__file__).parent / f"{name}.bin"
    disc = target_type.value
    with open(output_path, "wb") as out:
        out.write(struct.pack("<I", disc))
        out.write(struct.pack("<I", SLICE_ELEMENTS))
        out.write(struct.pack("<I", raw_byte_len))
        out.write(struct.pack("<I", golden_len))
        out.write(raw_bytes)
        out.write(golden_bytes)

    print(
        f"  {name}: tensor={tensor.name}, {target_type.name} (disc={disc}), "
        f"{SLICE_ELEMENTS} elements, "
        f"raw={raw_byte_len} B, golden={golden_len} B, "
        f"fixture={output_path.stat().st_size} B, "
        f"gguf dequant={best_us:.1f} µs (best of 5)"
    )


if __name__ == "__main__":
    print("Generating GGUF cross-validation fixtures...")
    print(f"Models directory: {MODELS_DIR}")
    for name, filename, target_type in FIXTURES:
        gguf_path = MODELS_DIR / filename
        if not gguf_path.exists():
            print(f"\n  SKIP {name}: model not found at {gguf_path}")
            print(f"       Download: hf-fm download-file <repo> \"{filename}\" "
                  f"--flat --output-dir \"{MODELS_DIR}\"")
            continue
        generate_fixture(name, gguf_path, target_type)
    print("\nDone.")
