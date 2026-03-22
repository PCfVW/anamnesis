#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate AWQ dequantization reference fixtures from real models.

AWQ packs along out_features (columns), unlike GPTQ which packs along
in_features (rows). The dequantization formula is:
  dequant = (qw - qz) * scale   (NO +1 offset on qzeros)

Fixture binary format (all little-endian):
  4 bytes: bits (u32) — 4 or 8
  4 bytes: group_size (u32)
  4 bytes: in_features (u32)
  4 bytes: out_features (u32)
  4 bytes: scale_dtype (u32) — 0=F32, 1=BF16, 2=F16
  4 bytes: qweight_len (u32)
  4 bytes: scales_len (u32)
  4 bytes: qzeros_len (u32)
  4 bytes: expected_len (u32)
  [qweight_len bytes]: raw packed qweight data
  [scales_len bytes]: raw scale data
  [qzeros_len bytes]: raw packed qzeros data
  [expected_len bytes]: expected BF16 output from PyTorch

Usage:
  python generate_awq.py
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
    snapshots = model_dir / "snapshots"
    dirs = list(snapshots.iterdir())
    return dirs[0]


def load_quantize_config(snapshot: Path) -> dict:
    for cfg_name in ["quantize_config.json", "config.json"]:
        cfg_path = snapshot / cfg_name
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
                if cfg_name == "config.json":
                    cfg = cfg.get("quantization_config", {})
                if cfg:
                    return cfg
    return {}


MODELS = [
    {
        "name": "llama_3_2_1b_awq",
        "model_dir": HF_CACHE / "models--casperhansen--llama-3.2-1b-instruct-awq",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "falcon3_1b_awq",
        "model_dir": HF_CACHE / "models--tiiuae--Falcon3-1B-Instruct-AWQ",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
]


def dequant_awq_pytorch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Dequantize AWQ to BF16 using the standard AutoAWQ formula.

    AWQ packs along out_features (columns).
    Formula: dequant = (qw - qz) * scale   (NO +1 offset)
    """
    pack_factor = 32 // bits
    maxq = (1 << bits) - 1

    # Unpack qweight: [in_features, packed_cols] → [in_features, out_features]
    weight = torch.zeros(in_features, out_features, dtype=torch.int32)
    for pos in range(pack_factor):
        shift = bits * pos
        unpacked = (qweight >> shift) & maxq
        weight[:, pos::pack_factor] = unpacked

    # Unpack qzeros: [num_groups, packed_cols] → [num_groups, out_features]
    num_groups = qzeros.shape[0]
    zeros = torch.zeros(num_groups, out_features, dtype=torch.int32)
    for pos in range(pack_factor):
        shift = bits * pos
        unpacked = (qzeros >> shift) & maxq
        zeros[:, pos::pack_factor] = unpacked
    # AWQ: NO +1 offset (unlike GPTQ)

    # Build group assignment (always sequential for AWQ)
    g = torch.arange(in_features, dtype=torch.long) // group_size

    # Dequantize
    result = (weight.float() - zeros[g, :].float()) * scales[g, :].float()
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

    config = load_quantize_config(snapshot)
    bits = config.get("bits", config.get("w_bit", 4))
    group_size = config.get("group_size", config.get("q_group_size", 128))
    print(f"  Config: bits={bits}, group_size={group_size}")

    st_files = list(snapshot.glob("*.safetensors"))
    if not st_files:
        print("  ERROR: no safetensors files found")
        return

    with safe_open(str(st_files[0]), framework="pt") as f:
        qweight = f.get_tensor(f"{layer_base}.qweight")
        scales = f.get_tensor(f"{layer_base}.scales")
        qzeros = f.get_tensor(f"{layer_base}.qzeros")

    pack_factor = 32 // bits
    in_features_full = qweight.shape[0]
    out_features_full = qweight.shape[1] * pack_factor

    print(f"  qweight: {list(qweight.shape)}, dtype={qweight.dtype}")
    print(f"  scales:  {list(scales.shape)}, dtype={scales.dtype}")
    print(f"  qzeros:  {list(qzeros.shape)}, dtype={qzeros.dtype}")
    print(f"  in_features={in_features_full}, out_features={out_features_full}")

    # Extract a 256×256 slice
    slice_in = min(256, in_features_full)
    slice_in = (slice_in // group_size) * group_size
    if slice_in == 0:
        slice_in = group_size
    slice_out = min(256, out_features_full)
    slice_out = (slice_out // pack_factor) * pack_factor
    if slice_out == 0:
        slice_out = pack_factor

    packed_slice_cols = slice_out // pack_factor
    num_groups = slice_in // group_size

    print(f"  Slice: {slice_in}×{slice_out} (packed cols: {packed_slice_cols})")

    qw_slice = qweight[:slice_in, :packed_slice_cols].contiguous()
    sc_slice = scales[:num_groups, :slice_out].contiguous()
    qz_slice = qzeros[:num_groups, :packed_slice_cols].contiguous()

    qw_bytes = qw_slice.numpy().tobytes()
    sc_bytes = sc_slice.view(torch.uint8).reshape(-1).cpu().numpy().tobytes() if scales.dtype != torch.float32 else sc_slice.numpy().tobytes()
    qz_bytes = qz_slice.numpy().tobytes()

    scale_dtype_id = SCALE_DTYPE_MAP.get(scales.dtype)
    if scale_dtype_id is None:
        print(f"  ERROR: unexpected scale dtype {scales.dtype}")
        return

    print(f"  Scale dtype: {SCALE_DTYPE_NAMES[scale_dtype_id]}")

    # Dequantize with PyTorch (best of 5)
    best_us = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        result_bf16 = dequant_awq_pytorch(
            qw_slice, sc_slice, qz_slice,
            bits, group_size, slice_in, slice_out,
        )
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)

    expected_bytes = result_bf16.view(torch.uint8).reshape(-1).numpy().tobytes()
    print(f"  PyTorch dequant: {best_us:.1f} µs (best of 5, {slice_in}×{slice_out} = {slice_in*slice_out} elements)")

    # Write fixture
    output_path = Path(__file__).parent / f"{name}.bin"
    with open(output_path, "wb") as out:
        out.write(struct.pack("<I", bits))
        out.write(struct.pack("<I", group_size))
        out.write(struct.pack("<I", slice_in))
        out.write(struct.pack("<I", slice_out))
        out.write(struct.pack("<I", scale_dtype_id))
        out.write(struct.pack("<I", len(qw_bytes)))
        out.write(struct.pack("<I", len(sc_bytes)))
        out.write(struct.pack("<I", len(qz_bytes)))
        out.write(struct.pack("<I", len(expected_bytes)))
        out.write(qw_bytes)
        out.write(sc_bytes)
        out.write(qz_bytes)
        out.write(expected_bytes)

    print(f"  Written: {output_path.name} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    print("Generating AWQ cross-validation fixtures...")
    for model in MODELS:
        generate_fixture(model)
    print("\nDone.")
