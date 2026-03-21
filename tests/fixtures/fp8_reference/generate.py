#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate FP8 dequantization reference fixtures from real models.

Extracts a small slice of FP8 weight data + scale factor from a real
safetensors model, dequantizes with PyTorch, and writes a binary fixture
file that the Rust cross-validation test can compare against.

Fixture binary format (all little-endian):
  4 bytes: scheme (0 = fine-grained, 1 = per-tensor)
  4 bytes: scale_dtype (0 = F32, 1 = BF16)
  4 bytes: rows (u32)
  4 bytes: cols (u32)
  4 bytes: weight_len (u32) — bytes of FP8 weight data
  4 bytes: scale_len (u32) — bytes of scale data
  4 bytes: expected_len (u32) — bytes of expected BF16 output
  [weight_len bytes]: raw FP8 weight data
  [scale_len bytes]: raw scale data (F32 or BF16 LE)
  [expected_len bytes]: expected BF16 output from PyTorch

Usage:
  python generate.py
"""

import struct
from pathlib import Path

import torch
from safetensors import safe_open


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

MODELS = [
    {
        "name": "exaone_fine_grained",
        "path": HF_CACHE
        / "models--LGAI-EXAONE--EXAONE-4.0-1.2B-FP8"
        / "snapshots"
        / "7a81ddc21b34a54777679de80f2e159c588be076"
        / "model.safetensors",
        "weight_name": "model.layers.0.mlp.down_proj.weight",
        "scale_name": "model.layers.0.mlp.down_proj.weight_scale_inv",
        "scheme": 0,  # fine-grained
        # Extract a 256x256 slice (2 full 128-blocks in each dimension)
        "rows": 256,
        "cols": 256,
    },
    {
        "name": "llama_per_tensor",
        "path": HF_CACHE
        / "models--RedHatAI--Llama-3.2-1B-Instruct-FP8"
        / "snapshots"
        / "fb49430ca7e61099fa1ff30f12fecb290a8ebb65"
        / "model.safetensors",
        "weight_name": "model.layers.0.mlp.down_proj.weight",
        "scale_name": "model.layers.0.mlp.down_proj.weight_scale",
        "scheme": 1,  # per-tensor
        # Extract a 256x256 slice
        "rows": 256,
        "cols": 256,
    },
]


def generate_fixture(model_info: dict) -> None:
    name = model_info["name"]
    path = model_info["path"]
    rows = model_info["rows"]
    cols = model_info["cols"]
    scheme = model_info["scheme"]

    print(f"Generating {name}...")
    print(f"  Model: {path}")

    with safe_open(str(path), framework="pt") as f:
        weight_tensor = f.get_tensor(model_info["weight_name"])
        scale_tensor = f.get_tensor(model_info["scale_name"])

    print(f"  Weight: {weight_tensor.shape}, dtype={weight_tensor.dtype}")
    print(f"  Scale:  {scale_tensor.shape}, dtype={scale_tensor.dtype}")

    # Extract a slice of the weight
    weight_slice = weight_tensor[:rows, :cols].contiguous()
    weight_bytes = weight_slice.view(torch.uint8).numpy().tobytes()

    # Get scale data
    if scheme == 0:
        # Fine-grained: extract matching block-scale slice
        scale_rows = (rows + 127) // 128
        scale_cols = (cols + 127) // 128
        scale_slice = scale_tensor[:scale_rows, :scale_cols].contiguous()
    else:
        # Per-tensor: scalar scale
        scale_slice = scale_tensor

    # Determine scale dtype
    if scale_slice.dtype == torch.float32:
        scale_dtype_id = 0
        scale_bytes = scale_slice.cpu().numpy().tobytes()
    elif scale_slice.dtype == torch.bfloat16:
        scale_dtype_id = 1
        # BF16 doesn't have numpy support; convert via view
        scale_bytes = scale_slice.view(torch.uint8).reshape(-1).numpy().tobytes()
        if len(scale_bytes) == 0:
            # Scalar BF16: read 2 bytes from the raw tensor
            scale_bytes = scale_slice.to(torch.float32).numpy().tobytes()
            # Re-encode as BF16
            scale_f32 = struct.unpack("<f", scale_bytes)[0]
            scale_bf16 = struct.pack("<H", struct.unpack("<I", struct.pack("<f", scale_f32))[0] >> 16)
            scale_bytes = scale_bf16
    else:
        raise ValueError(f"Unexpected scale dtype: {scale_slice.dtype}")

    print(f"  Scale dtype: {'F32' if scale_dtype_id == 0 else 'BF16'}")
    print(f"  Slice: {rows}x{cols} weight, {len(scale_bytes)} scale bytes")

    # Dequantize with PyTorch
    weight_fp8 = weight_slice.to(torch.float8_e4m3fn)
    weight_f32 = weight_fp8.to(torch.float32)

    if scheme == 0:
        # Fine-grained: apply block-wise scale
        scale_f32 = scale_slice.to(torch.float32)
        block_size = 128
        result = torch.zeros(rows, cols, dtype=torch.float32)
        for br in range(scale_rows):
            for bc in range(scale_cols):
                r_start = br * block_size
                r_end = min(r_start + block_size, rows)
                c_start = bc * block_size
                c_end = min(c_start + block_size, cols)
                result[r_start:r_end, c_start:c_end] = (
                    weight_f32[r_start:r_end, c_start:c_end] * scale_f32[br, bc]
                )
    else:
        # Per-tensor: single scale
        scale_f32 = scale_slice.to(torch.float32)
        result = weight_f32 * scale_f32

    # Convert to BF16
    result_bf16 = result.to(torch.bfloat16)
    expected_bytes = result_bf16.view(torch.uint8).reshape(-1).numpy().tobytes()

    print(f"  Expected output: {len(expected_bytes)} bytes")

    # Write fixture
    output_path = Path(__file__).parent / f"{name}.bin"
    with open(output_path, "wb") as out:
        out.write(struct.pack("<I", scheme))
        out.write(struct.pack("<I", scale_dtype_id))
        out.write(struct.pack("<I", rows))
        out.write(struct.pack("<I", cols))
        out.write(struct.pack("<I", len(weight_bytes)))
        out.write(struct.pack("<I", len(scale_bytes)))
        out.write(struct.pack("<I", len(expected_bytes)))
        out.write(weight_bytes)
        out.write(scale_bytes)
        out.write(expected_bytes)

    print(f"  Written: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    for model in MODELS:
        if not model["path"].exists():
            print(f"SKIP {model['name']}: model not found at {model['path']}")
            continue
        generate_fixture(model)
    print("Done.")
