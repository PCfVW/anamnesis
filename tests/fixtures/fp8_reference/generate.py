#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate FP8 dequantization reference fixtures from real models.

Extracts a small slice of FP8 weight data + scale factor from a real
safetensors model, dequantizes with PyTorch, and writes a binary fixture
file that the Rust cross-validation test can compare against.

Fixture binary format (all little-endian):
  4 bytes: scheme (0 = fine-grained, 1 = per-tensor, 2 = per-channel)
  4 bytes: scale_dtype (0 = F32, 1 = BF16, 2 = F16)
  4 bytes: rows (u32)
  4 bytes: cols (u32)
  4 bytes: weight_len (u32) — bytes of FP8 weight data
  4 bytes: scale_len (u32) — bytes of scale data
  4 bytes: expected_len (u32) — bytes of expected BF16 output
  [weight_len bytes]: raw FP8 weight data
  [scale_len bytes]: raw scale data
  [expected_len bytes]: expected BF16 output from PyTorch

Usage:
  python generate.py
"""

import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# scheme: 0 = fine-grained, 1 = per-tensor, 2 = per-channel
MODELS = [
    # --- Fine-grained (3 models) ---
    {
        "name": "exaone_fine_grained",
        "path": HF_CACHE
        / "models--LGAI-EXAONE--EXAONE-4.0-1.2B-FP8"
        / "snapshots"
        / "7a81ddc21b34a54777679de80f2e159c588be076"
        / "model.safetensors",
        "weight_name": "model.layers.0.mlp.down_proj.weight",
        "scale_name": "model.layers.0.mlp.down_proj.weight_scale_inv",
        "scheme": 0,
        "rows": 256,
        "cols": 256,
    },
    {
        "name": "qwen3_1_7b_fine_grained",
        "path": HF_CACHE
        / "models--Qwen--Qwen3-1.7B-FP8"
        / "snapshots"
        / "1641e6c1b620b7ed7e8711b443990429a23b1b99"
        / "model.safetensors",
        "weight_name": "model.layers.0.mlp.down_proj.weight",
        "scale_name": "model.layers.0.mlp.down_proj.weight_scale_inv",
        "scheme": 0,
        "rows": 256,
        "cols": 256,
    },
    {
        "name": "qwen3_4b_fine_grained_f16",
        "path": HF_CACHE
        / "models--Qwen--Qwen3-4B-Instruct-2507-FP8"
        / "snapshots"
        / "8591804019c8b22094c3b5b4454e0edc05dffc98"
        / "model.safetensors",
        "weight_name": "model.layers.0.mlp.down_proj.weight",
        "scale_name": "model.layers.0.mlp.down_proj.weight_scale_inv",
        "scheme": 0,
        "rows": 256,
        "cols": 256,
    },
    # --- Per-tensor (3 models) ---
    {
        "name": "ministral_per_tensor",
        "path": HF_CACHE
        / "models--mistralai--Ministral-3-3B-Instruct-2512"
        / "snapshots"
        / "cfcb068fa7c44114cf77a462357c6cdcd2c304b4"
        / "model.safetensors",
        "weight_name": "language_model.model.layers.0.mlp.down_proj.weight",
        "scale_name": "language_model.model.layers.0.mlp.down_proj.weight_scale_inv",
        "scheme": 1,
        "rows": 256,
        "cols": 256,
    },
    {
        "name": "llama_static_per_tensor",
        "path": HF_CACHE
        / "models--RedHatAI--Llama-3.2-1B-Instruct-FP8"
        / "snapshots"
        / "fb49430ca7e61099fa1ff30f12fecb290a8ebb65"
        / "model.safetensors",
        "weight_name": "model.layers.0.mlp.down_proj.weight",
        "scale_name": "model.layers.0.mlp.down_proj.weight_scale",
        "scheme": 1,
        "rows": 256,
        "cols": 256,
    },
    {
        "name": "nvidia_llama_per_tensor",
        "path": HF_CACHE
        / "models--nvidia--Llama-3.1-8B-Instruct-FP8"
        / "snapshots"
        / "42d9515ebd69eea3a87351d079c671c3c5ff0a31"
        / "model-00001-of-00002.safetensors",
        "weight_name": "model.layers.0.mlp.down_proj.weight",
        "scale_name": "model.layers.0.mlp.down_proj.weight_scale",
        "scheme": 1,
        "rows": 256,
        "cols": 256,
    },
    # --- Per-channel (1 model) ---
    {
        "name": "llama_dynamic_per_channel",
        "path": HF_CACHE
        / "models--RedHatAI--Llama-3.2-1B-Instruct-FP8-dynamic"
        / "snapshots"
        / "e23d444f8d7da0a3e556cae44a7d3c46f127e642"
        / "model.safetensors",
        "weight_name": "model.layers.0.mlp.down_proj.weight",
        "scale_name": "model.layers.0.mlp.down_proj.weight_scale",
        "scheme": 2,
        "rows": 256,
        "cols": 256,
    },
]

SCALE_DTYPE_MAP = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
SCALE_DTYPE_NAMES = {0: "F32", 1: "BF16", 2: "F16"}


def scale_to_bytes(tensor: torch.Tensor) -> tuple[bytes, int]:
    """Convert a scale tensor to raw LE bytes and a dtype id."""
    dtype_id = SCALE_DTYPE_MAP.get(tensor.dtype)
    if dtype_id is None:
        raise ValueError(f"Unexpected scale dtype: {tensor.dtype}")

    if tensor.dtype == torch.float32:
        return tensor.cpu().numpy().tobytes(), dtype_id

    # BF16 / F16: ensure at least 1D before viewing as bytes
    t = tensor.contiguous()
    if t.dim() == 0:
        t = t.unsqueeze(0)
    raw = t.view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
    return raw, dtype_id


def dequant_pytorch(
    weight_slice: torch.Tensor,
    scale_slice: torch.Tensor,
    scheme: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Dequantize FP8 to BF16 using PyTorch, matching anamnesis logic."""
    weight_f32 = weight_slice.to(torch.float32)
    scale_f32 = scale_slice.to(torch.float32)

    if scheme == 0:
        # Fine-grained: block-wise scale
        block_size = 128
        scale_rows = (rows + 127) // 128
        scale_cols = (cols + 127) // 128
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
    elif scheme == 1:
        # Per-tensor: single scale
        result = weight_f32 * scale_f32
    elif scheme == 2:
        # Per-channel: one scale per row, broadcast across columns
        result = weight_f32 * scale_f32  # scale_f32 shape [rows, 1] broadcasts
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return result.to(torch.bfloat16)


def generate_fixture(model_info: dict) -> None:
    name = model_info["name"]
    path = model_info["path"]
    rows = model_info["rows"]
    cols = model_info["cols"]
    scheme = model_info["scheme"]

    print(f"\n{name}:")
    print(f"  Model: {path.name}")

    with safe_open(str(path), framework="pt") as f:
        weight_tensor = f.get_tensor(model_info["weight_name"])
        scale_tensor = f.get_tensor(model_info["scale_name"])

    print(f"  Weight: {list(weight_tensor.shape)}, dtype={weight_tensor.dtype}")
    print(f"  Scale:  {list(scale_tensor.shape)}, dtype={scale_tensor.dtype}")

    # Extract a slice of the weight
    weight_slice = weight_tensor[:rows, :cols].contiguous()
    weight_bytes = weight_slice.view(torch.uint8).numpy().tobytes()

    # Get scale data
    if scheme == 0:
        # Fine-grained: extract matching block-scale slice
        sr = (rows + 127) // 128
        sc = (cols + 127) // 128
        scale_slice = scale_tensor[:sr, :sc].contiguous()
    elif scheme == 2:
        # Per-channel: extract rows matching the weight slice
        scale_slice = scale_tensor[:rows].contiguous()
    else:
        # Per-tensor: scalar scale
        scale_slice = scale_tensor

    scale_bytes, scale_dtype_id = scale_to_bytes(scale_slice)
    print(f"  Scale dtype: {SCALE_DTYPE_NAMES[scale_dtype_id]}, {len(scale_bytes)} bytes")

    # Dequantize with PyTorch and measure time (best of 5 runs to exclude JIT warmup)
    best_us = float("inf")
    for _run in range(5):
        t0 = time.perf_counter()
        result_bf16 = dequant_pytorch(weight_slice, scale_slice, scheme, rows, cols)
        t1 = time.perf_counter()
        run_us = (t1 - t0) * 1e6
        best_us = min(best_us, run_us)

    expected_bytes = result_bf16.view(torch.uint8).reshape(-1).numpy().tobytes()
    print(f"  PyTorch dequant: {best_us:.1f} µs (best of 5, {rows}x{cols} = {rows*cols} elements)")

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

    print(f"  Written: {output_path.name} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    print("Generating FP8 cross-validation fixtures...")
    for model in MODELS:
        if not model["path"].exists():
            print(f"\nSKIP {model['name']}: model not found at {model['path']}")
            continue
        generate_fixture(model)
    print("\nDone.")
