#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate NPZ cross-validation fixtures from Gemma Scope SAE weights.

Reads the real params.npz (302 MB, 5 F32 arrays) and produces:

1. gemma_scope_small.npz  — tiny NPZ with first 8 elements of each array
2. gemma_scope_reference.bin — binary metadata + expected raw bytes

Binary fixture format (all little-endian):
  4 bytes: num_arrays (u32)
  For each array (sorted by name):
    4 bytes: name_len (u32)
    [name_len bytes]: name (UTF-8)
    4 bytes: num_dims (u32)
    [num_dims * 8 bytes]: original shape (u64 per dim)
    4 bytes: dtype_id (u32) — matches NpzDtype variant order
    4 bytes: data_len (u32)
    [data_len bytes]: raw LE data (first 8 elements as f32)

Usage:
  python generate_npz.py
"""

import struct
from pathlib import Path

import numpy as np


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_DIR = HF_CACHE / "models--google--gemma-scope-2b-pt-res"

# NpzDtype variant IDs (must match Rust enum order)
DTYPE_IDS = {
    "bool": 0,   # Bool
    "uint8": 1,  # U8
    "int8": 2,   # I8
    "uint16": 3, # U16
    "int16": 4,  # I16
    "uint32": 5, # U32
    "int32": 6,  # I32
    "uint64": 7, # U64
    "int64": 8,  # I64
    "float16": 9,  # F16
    # 10 = BF16 (no numpy native dtype)
    "float32": 11, # F32
    "float64": 12, # F64
}

SPOT_ELEMENTS = 8  # first N elements extracted per array


def find_params_npz() -> Path:
    """Locate params.npz in the HuggingFace cache."""
    snapshots = MODEL_DIR / "snapshots"
    snapshot = next(snapshots.iterdir())
    npz_path = snapshot / "layer_0" / "width_16k" / "average_l0_105" / "params.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"params.npz not found at {npz_path}")
    return npz_path


def main():
    out_dir = Path(__file__).parent
    npz_path = find_params_npz()
    print(f"Reading {npz_path} ({npz_path.stat().st_size / 1e6:.0f} MB)")

    archive = np.load(npz_path)
    names = sorted(archive.files)
    print(f"Arrays: {names}")

    # --- Build small NPZ with truncated arrays ---
    small_arrays = {}
    for name in names:
        arr = archive[name]
        flat = arr.flatten()[:SPOT_ELEMENTS]
        small_arrays[name] = flat
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"first {SPOT_ELEMENTS} = {flat.tolist()}")

    small_path = out_dir / "gemma_scope_small.npz"
    np.savez(small_path, **small_arrays)
    print(f"Wrote {small_path} ({small_path.stat().st_size} bytes)")

    # --- Build binary reference fixture ---
    buf = bytearray()
    buf += struct.pack("<I", len(names))  # num_arrays

    for name in names:
        arr = archive[name]
        flat = arr.flatten()[:SPOT_ELEMENTS].astype(np.float32)
        raw_bytes = flat.tobytes()  # already LE on x86

        name_bytes = name.encode("utf-8")
        buf += struct.pack("<I", len(name_bytes))
        buf += name_bytes

        buf += struct.pack("<I", len(arr.shape))
        for dim in arr.shape:
            buf += struct.pack("<Q", dim)  # u64

        dtype_id = DTYPE_IDS.get(arr.dtype.name)
        if dtype_id is None:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")
        buf += struct.pack("<I", dtype_id)

        buf += struct.pack("<I", len(raw_bytes))
        buf += raw_bytes

    ref_path = out_dir / "gemma_scope_reference.bin"
    ref_path.write_bytes(buf)
    print(f"Wrote {ref_path} ({len(buf)} bytes)")


if __name__ == "__main__":
    main()
