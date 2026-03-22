#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate GPTQ dequantization reference fixtures from real models.

Extracts a small slice of GPTQ tensors (qweight, scales, qzeros, g_idx)
from a real safetensors model, dequantizes with the standard GPTQ formula
matching AutoGPTQ/GPTQModel, and writes a binary fixture file that the
Rust cross-validation test can compare against.

Fixture binary format (all little-endian):
  4 bytes: bits (u32) — 4 or 8
  4 bytes: group_size (u32)
  4 bytes: in_features (u32) — unpacked input dimension
  4 bytes: out_features (u32)
  4 bytes: scale_dtype (u32) — 0=F32, 1=BF16, 2=F16
  4 bytes: has_g_idx (u32) — 0 or 1
  4 bytes: qweight_len (u32) — bytes of packed qweight data
  4 bytes: scales_len (u32) — bytes of scale data
  4 bytes: qzeros_len (u32) — bytes of packed qzeros data
  4 bytes: g_idx_len (u32) — bytes of g_idx data (0 if absent)
  4 bytes: expected_len (u32) — bytes of expected BF16 output
  [qweight_len bytes]: raw packed qweight data
  [scales_len bytes]: raw scale data
  [qzeros_len bytes]: raw packed qzeros data
  [g_idx_len bytes]: raw g_idx data (empty if has_g_idx=0)
  [expected_len bytes]: expected BF16 output from PyTorch

Usage:
  python generate_gptq.py
"""

import json
import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

SCALE_DTYPE_MAP = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
SCALE_DTYPE_NAMES = {0: "F32", 1: "BF16", 2: "F16"}


def find_snapshot(model_dir: Path) -> Path:
    """Find the snapshot directory for a cached model."""
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"No snapshots directory in {model_dir}")
    dirs = list(snapshots.iterdir())
    if not dirs:
        raise FileNotFoundError(f"No snapshot found in {snapshots}")
    return dirs[0]


def load_quantize_config(snapshot: Path) -> dict:
    """Load GPTQ quantization config from various sources."""
    # Try quantize_config.json first
    qc_path = snapshot / "quantize_config.json"
    if qc_path.exists():
        with open(qc_path) as f:
            return json.load(f)

    # Try config.json → quantization_config
    cfg_path = snapshot / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
            if "quantization_config" in cfg:
                return cfg["quantization_config"]

    raise FileNotFoundError(f"No quantize config found in {snapshot}")


# Model definitions
MODELS = [
    # --- 4-bit models ---
    {
        "name": "falcon3_1b_int4",
        "model_dir": HF_CACHE / "models--tiiuae--Falcon3-1B-Instruct-GPTQ-Int4",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "llama_3_2_1b_int4",
        "model_dir": HF_CACHE / "models--shuyuej--Llama-3.2-1B-Instruct-GPTQ",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    # --- 8-bit models ---
    {
        "name": "falcon3_1b_int8",
        "model_dir": HF_CACHE / "models--tiiuae--Falcon3-1B-Instruct-GPTQ-Int8",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "llama_3_2_1b_gptqmodel_int8",
        "model_dir": HF_CACHE / "models--iproskurina--Llama-3.2-1B-gptqmodel-8bit",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
]


def dequant_gptq_pytorch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor | None,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Dequantize GPTQ to BF16 using the standard AutoGPTQ formula.

    Formula: dequant = (qw - (qz + 1)) * scale
    """
    pack_factor = 32 // bits
    maxq = (1 << bits) - 1

    # Unpack qweight: [packed_rows, out_features] → [in_features, out_features]
    weight = torch.zeros(in_features, out_features, dtype=torch.int32)
    for pos in range(pack_factor):
        shift = bits * pos
        start = pos
        # Each packed_row contributes one value per position
        unpacked = (qweight >> shift) & maxq
        weight[pos::pack_factor, :] = unpacked

    # Unpack qzeros: [num_groups, packed_cols] → [num_groups, out_features]
    num_groups = qzeros.shape[0]
    packed_cols = qzeros.shape[1]
    zeros = torch.zeros(num_groups, out_features, dtype=torch.int32)
    for pos in range(pack_factor):
        shift = bits * pos
        unpacked = (qzeros >> shift) & maxq
        zeros[:, pos::pack_factor] = unpacked
    # Standard GPTQ +1 offset
    zeros = zeros + 1

    # Build group assignment
    if g_idx is not None:
        g = g_idx.long()
    else:
        g = torch.arange(in_features, dtype=torch.long) // group_size

    # Dequantize
    scales_f32 = scales.float()
    weight_f32 = weight.float()
    zeros_f32 = zeros.float()

    result = (weight_f32 - zeros_f32[g, :]) * scales_f32[g, :]
    return result.to(torch.bfloat16)


def generate_fixture(model_info: dict) -> None:
    name = model_info["name"]
    model_dir = model_info["model_dir"]
    layer_base = model_info["layer_base"]

    if not model_dir.exists():
        print(f"\nSKIP {name}: model not found at {model_dir}")
        return

    snapshot = find_snapshot(model_dir)
    print(f"\n{name}:")
    print(f"  Snapshot: {snapshot}")

    # Load config
    config = load_quantize_config(snapshot)
    bits = config.get("bits", 4)
    group_size = config.get("group_size", 128)
    print(f"  Config: bits={bits}, group_size={group_size}")

    # Find safetensors file
    st_files = list(snapshot.glob("*.safetensors"))
    if not st_files:
        print(f"  ERROR: no safetensors files found")
        return

    # Load tensors
    qweight_name = f"{layer_base}.qweight"
    scales_name = f"{layer_base}.scales"
    qzeros_name = f"{layer_base}.qzeros"
    g_idx_name = f"{layer_base}.g_idx"

    with safe_open(str(st_files[0]), framework="pt") as f:
        qweight = f.get_tensor(qweight_name)
        scales = f.get_tensor(scales_name)
        qzeros = f.get_tensor(qzeros_name)
        g_idx = f.get_tensor(g_idx_name) if g_idx_name in f.keys() else None

    pack_factor = 32 // bits
    packed_rows = qweight.shape[0]
    out_features_full = qweight.shape[1]
    in_features_full = packed_rows * pack_factor

    print(f"  qweight: {list(qweight.shape)}, dtype={qweight.dtype}")
    print(f"  scales:  {list(scales.shape)}, dtype={scales.dtype}")
    print(f"  qzeros:  {list(qzeros.shape)}, dtype={qzeros.dtype}")
    print(f"  g_idx:   {'present' if g_idx is not None else 'absent'}")
    print(f"  in_features={in_features_full}, out_features={out_features_full}")

    # Extract a 256×256 slice (or smaller if model is smaller)
    slice_in = min(256, in_features_full)
    slice_out = min(256, out_features_full)
    # Ensure slice_in is a multiple of group_size and pack_factor
    slice_in = (slice_in // group_size) * group_size
    if slice_in == 0:
        slice_in = group_size
    # Ensure slice_out is a multiple of pack_factor
    slice_out = (slice_out // pack_factor) * pack_factor
    if slice_out == 0:
        slice_out = pack_factor

    packed_slice_rows = slice_in // pack_factor
    packed_slice_cols = slice_out // pack_factor
    num_groups = slice_in // group_size

    print(f"  Slice: {slice_in}×{slice_out} (packed: {packed_slice_rows}×{slice_out})")

    # Extract slices
    qw_slice = qweight[:packed_slice_rows, :slice_out].contiguous()
    sc_slice = scales[:num_groups, :slice_out].contiguous()
    qz_slice = qzeros[:num_groups, :packed_slice_cols].contiguous()
    gi_slice = g_idx[:slice_in].contiguous() if g_idx is not None else None

    # For g_idx, remap to local group indices (0..num_groups-1)
    if gi_slice is not None:
        # The g_idx values may reference groups beyond our slice.
        # For the slice, we need them to be in [0, num_groups).
        # Since we're taking the first slice_in features, and group_size
        # divides slice_in, the sequential g_idx would be [0,0,...,1,1,...].
        # For act-order models, we just use the original g_idx values
        # but clamp to our group range.
        gi_max = gi_slice.max().item()
        if gi_max >= num_groups:
            # Remap: use sequential assignment for the slice
            print(f"  Note: remapping g_idx (max={gi_max} >= num_groups={num_groups})")
            gi_slice = torch.arange(slice_in, dtype=torch.int32) // group_size

    # Get raw bytes
    qw_bytes = qw_slice.numpy().tobytes()
    sc_bytes = sc_slice.view(torch.uint8).reshape(-1).cpu().numpy().tobytes() if scales.dtype != torch.float32 else sc_slice.numpy().tobytes()
    qz_bytes = qz_slice.numpy().tobytes()
    gi_bytes = gi_slice.numpy().tobytes() if gi_slice is not None else b""

    scale_dtype_id = SCALE_DTYPE_MAP.get(scales.dtype)
    if scale_dtype_id is None:
        print(f"  ERROR: unexpected scale dtype {scales.dtype}")
        return

    print(f"  Scale dtype: {SCALE_DTYPE_NAMES[scale_dtype_id]}")

    # Dequantize with PyTorch and measure time (best of 5 runs to exclude JIT warmup)
    best_us = float("inf")
    for run in range(5):
        t0 = time.perf_counter()
        result_bf16 = dequant_gptq_pytorch(
            qw_slice, sc_slice, qz_slice, gi_slice,
            bits, group_size, slice_in, slice_out,
        )
        t1 = time.perf_counter()
        run_us = (t1 - t0) * 1e6
        best_us = min(best_us, run_us)

    expected_bytes = result_bf16.view(torch.uint8).reshape(-1).numpy().tobytes()
    print(f"  PyTorch dequant: {best_us:.1f} µs (best of 5, {slice_in}×{slice_out} = {slice_in*slice_out} elements)")
    print(f"  Output shape: {list(result_bf16.shape)}")

    # Write fixture
    output_path = Path(__file__).parent / f"{name}.bin"
    has_g_idx = 1 if gi_slice is not None else 0

    with open(output_path, "wb") as out:
        out.write(struct.pack("<I", bits))
        out.write(struct.pack("<I", group_size))
        out.write(struct.pack("<I", slice_in))
        out.write(struct.pack("<I", slice_out))
        out.write(struct.pack("<I", scale_dtype_id))
        out.write(struct.pack("<I", has_g_idx))
        out.write(struct.pack("<I", len(qw_bytes)))
        out.write(struct.pack("<I", len(sc_bytes)))
        out.write(struct.pack("<I", len(qz_bytes)))
        out.write(struct.pack("<I", len(gi_bytes)))
        out.write(struct.pack("<I", len(expected_bytes)))
        out.write(qw_bytes)
        out.write(sc_bytes)
        out.write(qz_bytes)
        out.write(gi_bytes)
        out.write(expected_bytes)

    print(f"  Written: {output_path.name} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    print("Generating GPTQ cross-validation fixtures...")
    for model in MODELS:
        generate_fixture(model)
    print("\nDone.")
