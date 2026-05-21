# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Ollama-distributed GGUF dequantization reference fixtures.

This is the Phase 6.5 "Real-world cross-validation — Ollama track" fixture
generator. Mirrors ``tests/fixtures/gguf_reference/generate_gguf.py`` but
sources the GGUF blob from the local Ollama model cache (resolving the
manifest JSON → blob hash → path), rather than a fresh ``hf-fm`` download.

The point: anamnesis's GGUF dequant has been validated byte-exact against
``gguf-py`` reference on bartowski/TheBloke quantisations. Extending the
validation to a real Ollama-distributed blob proves the same kernels work
on the dominant local-LLM distribution channel.

The output fixture file format is **byte-identical** to
``tests/fixtures/gguf_reference/*.bin`` so the test side can reuse the
same parser (16-byte header + raw + golden BF16):

  4 bytes: ggml_type discriminant (u32)
  4 bytes: n_elements (u32)
  4 bytes: raw_data_len (u32)
  4 bytes: golden_len (u32)
  [raw_data_len bytes]: raw quantized block data
  [golden_len bytes]: expected BF16 output

Usage::

    pip install gguf numpy
    python generate_ollama_fixture.py

Requires the source model to be present in the local Ollama cache. The
generator does not call ``ollama pull`` — pre-populate the cache via a
manual ``ollama pull llama3.2:1b`` (or any other model added to
``FIXTURES`` below).

Why not just ``ollama pull`` from the script: the same script must run
in environments where ``ollama`` is not installed (CI fixture refresh
on a different machine, contributors with only the GGUF blob copied
out-of-band, …). Resolving the manifest path explicitly keeps the
script ``ollama``-binary-free; the only requirement is that the blob
exists where Ollama would have put it.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType, GGML_QUANT_SIZES, dequantize


SLICE_ELEMENTS = 65_536
"""Number of elements per fixture slice. Same value as
``tests/fixtures/gguf_reference/`` so the cross-validation tests on
both sides agree on the slice size without per-fixture configuration."""


def ollama_root() -> Path:
    """Return the local Ollama models directory.

    Honours the ``OLLAMA_MODELS`` environment variable (Ollama's standard
    override), otherwise falls back to the default location for the
    current platform.
    """
    explicit = os.environ.get("OLLAMA_MODELS")
    if explicit:
        return Path(explicit)
    return Path.home() / ".ollama" / "models"


def resolve_ollama_blob(model_name: str, tag: str) -> Path:
    """Resolve ``<model_name>:<tag>`` to the on-disk GGUF blob path.

    Steps:
    1. Read ``<ollama_root>/manifests/registry.ollama.ai/library/<model_name>/<tag>``
       (a single-line JSON manifest).
    2. Find the layer with ``mediaType == "application/vnd.ollama.image.model"``.
    3. Convert the layer's ``digest`` (``sha256:<hex>``) into the blob path
       ``<ollama_root>/blobs/sha256-<hex>``.

    Raises ``FileNotFoundError`` if the manifest or blob is missing
    (typically means the model has not been ``ollama pull``-ed yet).
    """
    root = ollama_root()
    manifest_path = root / "manifests" / "registry.ollama.ai" / "library" / model_name / tag
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ollama manifest not found at {manifest_path}; run `ollama pull {model_name}:{tag}` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_layer = next(
        (layer for layer in manifest["layers"]
         if layer.get("mediaType") == "application/vnd.ollama.image.model"),
        None,
    )
    if model_layer is None:
        raise RuntimeError(
            f"manifest {manifest_path} has no application/vnd.ollama.image.model layer"
        )
    digest = model_layer["digest"]
    if not digest.startswith("sha256:"):
        raise RuntimeError(f"unexpected digest format {digest!r}")
    blob_hash = digest[len("sha256:") :]
    blob_path = root / "blobs" / f"sha256-{blob_hash}"
    if not blob_path.exists():
        raise FileNotFoundError(
            f"Ollama blob not found at {blob_path}; the manifest references it but the blob is absent"
        )
    return blob_path


# Each entry: (fixture_name, model_name, tag, target tensor name, target quant type).
#
# Picked tensor name MUST exist in the GGUF and be of the named quant type with
# at least SLICE_ELEMENTS elements. `blk.0.attn_q.weight` is the standard
# attention-query weight for the first transformer block in any Llama-arch
# model — structurally stable across releases.
FIXTURES = [
    (
        "llama3_2_1b_q8_0",
        "llama3.2",
        "1b",
        "blk.0.attn_q.weight",
        GGMLQuantizationType.Q8_0,
    ),
]


def f32_array_to_bf16_bytes(f32_arr: np.ndarray) -> bytes:
    """Convert f32 -> BF16 bytes (round-to-nearest-even).

    Matches anamnesis's ``f32_bits_to_bf16_bits`` exactly. Same helper
    as ``tests/fixtures/gguf_reference/generate_gguf.py``.
    """
    bits = f32_arr.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    bf16_bits = ((bits + rounding_bias) >> 16).astype(np.uint16)
    return bf16_bits.tobytes()


def find_tensor(reader: GGUFReader, name: str, target_type: GGMLQuantizationType):
    """Locate the named tensor in the GGUF, verifying its quant type."""
    for tensor in reader.tensors:
        if tensor.name == name:
            if tensor.tensor_type != target_type:
                raise RuntimeError(
                    f"tensor `{name}` has type {tensor.tensor_type.name}, "
                    f"expected {target_type.name}"
                )
            if tensor.n_elements < SLICE_ELEMENTS:
                raise RuntimeError(
                    f"tensor `{name}` has only {tensor.n_elements} elements, "
                    f"need >= {SLICE_ELEMENTS}"
                )
            return tensor
    raise RuntimeError(f"tensor `{name}` not found in GGUF")


def generate_fixture(
    fixture_name: str,
    model_name: str,
    tag: str,
    tensor_name: str,
    target_type: GGMLQuantizationType,
) -> None:
    """Extract one tensor slice and write the binary fixture."""
    blob_path = resolve_ollama_blob(model_name, tag)
    print(f"  source: {model_name}:{tag} -> {blob_path.name} ({blob_path.stat().st_size:,} B)")

    reader = GGUFReader(str(blob_path))
    tensor = find_tensor(reader, tensor_name, target_type)

    block_size, type_size = GGML_QUANT_SIZES[target_type]
    n_blocks = SLICE_ELEMENTS // block_size
    raw_byte_len = n_blocks * type_size

    raw_flat = tensor.data.reshape(-1)[:raw_byte_len]
    raw_bytes = raw_flat.tobytes()
    assert len(raw_bytes) == raw_byte_len, (
        f"sliced {len(raw_bytes)} bytes, expected {raw_byte_len}"
    )

    # Dequantize with the gguf reference implementation, best-of-5 timing.
    best_us = float("inf")
    f32: Optional[np.ndarray] = None
    for _ in range(5):
        t0 = time.perf_counter()
        f32 = dequantize(np.frombuffer(raw_bytes, dtype=np.uint8), target_type)
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)
    assert f32 is not None
    assert f32.shape == (SLICE_ELEMENTS,), f"expected {SLICE_ELEMENTS}, got {f32.shape}"
    assert f32.dtype == np.float32

    golden_bytes = f32_array_to_bf16_bytes(f32)
    golden_len = SLICE_ELEMENTS * 2
    assert len(golden_bytes) == golden_len

    output_path = Path(__file__).parent / f"{fixture_name}.bin"
    disc = target_type.value
    with open(output_path, "wb") as out:
        out.write(struct.pack("<I", disc))
        out.write(struct.pack("<I", SLICE_ELEMENTS))
        out.write(struct.pack("<I", raw_byte_len))
        out.write(struct.pack("<I", golden_len))
        out.write(raw_bytes)
        out.write(golden_bytes)

    print(
        f"  fixture: {fixture_name}.bin "
        f"(tensor={tensor.name}, {target_type.name} disc={disc}, "
        f"{SLICE_ELEMENTS} elements, raw={raw_byte_len} B, golden={golden_len} B, "
        f"total={output_path.stat().st_size} B, gguf dequant={best_us:.1f} us best-of-5)"
    )


def main() -> int:
    print(f"Ollama root: {ollama_root()}")
    for fixture_name, model_name, tag, tensor_name, target_type in FIXTURES:
        try:
            generate_fixture(fixture_name, model_name, tag, tensor_name, target_type)
        except FileNotFoundError as e:
            print(f"  SKIP {fixture_name}: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - we want to log and continue
            print(f"  FAIL {fixture_name}: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
