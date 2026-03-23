#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate BitsAndBytes dequantization reference fixtures from real models.

Extracts a small block-aligned slice of BnB tensors (weight, absmax, quant_map,
and optionally nested_absmax/nested_quant_map or SCB), dequantizes with the
bitsandbytes Python formula, and writes a binary fixture file for Rust
cross-validation.

Fixture binary format — NF4/FP4 (all little-endian):
  4 bytes: format_id (u32) — 0=NF4/FP4, 1=INT8, 2=NF4/FP4 double-quant
  4 bytes: total_elements (u32) — number of dequantized elements
  4 bytes: block_size (u32) — elements per absmax block (typically 64)
  4 bytes: weight_len (u32) — bytes of packed U8 weight data
  4 bytes: absmax_len (u32) — bytes of absmax data (F32 or U8 for double-quant)
  4 bytes: quant_map_len (u32) — bytes of quant_map data (F32[16] = 64 bytes)
  4 bytes: nested_absmax_len (u32) — bytes of nested absmax (0 if not double-quant)
  4 bytes: nested_quant_map_len (u32) — bytes of nested quant_map (0 if not double-quant)
  4 bytes: expected_len (u32) — bytes of expected BF16 output
  [weight_len bytes]: raw U8 weight data
  [absmax_len bytes]: raw absmax data
  [quant_map_len bytes]: raw quant_map data
  [nested_absmax_len bytes]: raw nested absmax (empty if not double-quant)
  [nested_quant_map_len bytes]: raw nested quant_map (empty if not double-quant)
  [expected_len bytes]: expected BF16 output

Fixture binary format — INT8:
  4 bytes: format_id (u32) — 1
  4 bytes: out_features (u32)
  4 bytes: in_features (u32)
  4 bytes: weight_len (u32) — bytes of I8 weight data
  4 bytes: scb_len (u32) — bytes of SCB data (F32)
  4 bytes: expected_len (u32) — bytes of expected BF16 output
  [weight_len bytes]: raw I8 weight data
  [scb_len bytes]: raw SCB data
  [expected_len bytes]: expected BF16 output

Usage:
  python generate_bnb.py
"""

import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def find_snapshot(model_dir: Path) -> Path:
    """Find the snapshot directory for a cached model."""
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"No snapshots directory in {model_dir}")
    dirs = list(snapshots.iterdir())
    if not dirs:
        raise FileNotFoundError(f"No snapshot found in {snapshots}")
    return dirs[0]


# ---- NF4/FP4 dequantization (matching bitsandbytes) ----

def dequant_bnb4_pytorch(
    weight_u8: torch.Tensor,
    absmax_f32: torch.Tensor,
    quant_map: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize NF4/FP4 to BF16 using the bitsandbytes formula.

    weight_u8: flat U8 tensor (2 nibbles per byte)
    absmax_f32: F32 tensor, one per block
    quant_map: F32[16] lookup table
    """
    # Unpack nibbles: low first, high second
    weight_bytes = weight_u8.to(torch.uint8)
    low_nibbles = weight_bytes & 0x0F
    high_nibbles = weight_bytes >> 4

    # Interleave: [low0, high0, low1, high1, ...]
    total_elements = weight_bytes.numel() * 2
    unpacked = torch.zeros(total_elements, dtype=torch.long)
    unpacked[0::2] = low_nibbles.long()
    unpacked[1::2] = high_nibbles.long()

    # Lookup
    values = quant_map[unpacked]

    # Scale by block absmax
    num_blocks = total_elements // block_size
    values = values.reshape(num_blocks, block_size)
    absmax_expanded = absmax_f32[:num_blocks].unsqueeze(1)
    result = values * absmax_expanded
    return result.reshape(-1).to(torch.bfloat16)


def dequant_bnb4_double_quant_pytorch(
    weight_u8: torch.Tensor,
    absmax_u8: torch.Tensor,
    quant_map: torch.Tensor,
    nested_absmax_f32: torch.Tensor,
    nested_quant_map: torch.Tensor,
    block_size: int,
    nested_block_size: int,
) -> torch.Tensor:
    """Dequantize double-quant NF4/FP4: first recover absmax, then dequant."""
    # Step 1: dequant nested absmax (U8 → F32)
    num_blocks = absmax_u8.numel()
    absmax_indices = absmax_u8.long()
    absmax_values = nested_quant_map[absmax_indices]

    # Scale by nested absmax
    import math
    num_nested = math.ceil(num_blocks / nested_block_size)
    recovered = torch.zeros(num_blocks, dtype=torch.float32)
    for i in range(num_blocks):
        nb_idx = i // nested_block_size
        recovered[i] = absmax_values[i] * nested_absmax_f32[nb_idx]

    # Step 2: dequant weights using recovered absmax
    return dequant_bnb4_pytorch(weight_u8, recovered, quant_map, block_size)


# ---- INT8 dequantization ----

def dequant_bnb_int8_pytorch(
    weight_i8: torch.Tensor,
    scb: torch.Tensor,
) -> torch.Tensor:
    """Dequantize BnB INT8 to BF16: weight_i8 * SCB[row] / 127.0"""
    scale = scb / 127.0  # [out_features]
    weight_f32 = weight_i8.float()
    result = weight_f32 * scale.unsqueeze(1)  # broadcast scale across columns
    return result.to(torch.bfloat16)


# ---- Model definitions ----

MODELS = [
    {
        "name": "llama_1b_nf4",
        "model_dir": HF_CACHE / "models--medmekk--Llama-3.2-1B-Instruct-bnb-nf4",
        "format": "bnb4",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "llama_1b_nf4_double_quant",
        "model_dir": HF_CACHE / "models--medmekk--Llama-3.2-1B-Instruct-bnb-nf4-double-quant",
        "format": "bnb4_dq",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "llama_1b_fp4",
        "model_dir": HF_CACHE / "models--HF-Quantization--Llama-3.2-1B-BNB-FP4",
        "format": "bnb4",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "llama_1b_int8",
        "model_dir": HF_CACHE / "models--HF-Quantization--Llama-3.2-1B-BNB-INT8",
        "format": "int8",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
]


def generate_bnb4_fixture(model_info: dict) -> None:
    """Generate fixture for NF4/FP4 (plain or double-quant)."""
    name = model_info["name"]
    model_dir = model_info["model_dir"]
    layer_base = model_info["layer_base"]
    is_double_quant = model_info["format"] == "bnb4_dq"

    if not model_dir.exists():
        print(f"\nSKIP {name}: model not found at {model_dir}")
        return

    snapshot = find_snapshot(model_dir)
    print(f"\n{name}:")
    print(f"  Snapshot: {snapshot}")

    st_files = list(snapshot.glob("*.safetensors"))
    if not st_files:
        print(f"  ERROR: no safetensors files found")
        return

    # Tensor names
    weight_name = f"{layer_base}.weight"
    absmax_name = f"{layer_base}.weight.absmax"
    quant_map_name = f"{layer_base}.weight.quant_map"

    with safe_open(str(st_files[0]), framework="pt") as f:
        weight_raw = f.get_tensor(weight_name)
        absmax_raw = f.get_tensor(absmax_name)
        quant_map = f.get_tensor(quant_map_name)

        if is_double_quant:
            nested_absmax_name = f"{layer_base}.weight.nested_absmax"
            nested_quant_map_name = f"{layer_base}.weight.nested_quant_map"
            nested_absmax = f.get_tensor(nested_absmax_name)
            nested_quant_map = f.get_tensor(nested_quant_map_name)

    total_elements_full = weight_raw.numel() * 2
    num_blocks_full = absmax_raw.numel()
    block_size = total_elements_full // num_blocks_full if num_blocks_full > 0 else 64

    print(f"  weight:    {list(weight_raw.shape)}, dtype={weight_raw.dtype}")
    print(f"  absmax:    {list(absmax_raw.shape)}, dtype={absmax_raw.dtype}")
    print(f"  quant_map: {list(quant_map.shape)}, dtype={quant_map.dtype}")
    print(f"  block_size={block_size}, total_elements={total_elements_full}")
    if is_double_quant:
        print(f"  nested_absmax:    {list(nested_absmax.shape)}, dtype={nested_absmax.dtype}")
        print(f"  nested_quant_map: {list(nested_quant_map.shape)}, dtype={nested_quant_map.dtype}")

    # Extract a slice: 4096 elements = 2048 bytes = 64 blocks (for block_size=64)
    slice_elements = min(4096, total_elements_full)
    slice_blocks = slice_elements // block_size
    slice_bytes = slice_elements // 2

    weight_slice = weight_raw.reshape(-1)[:slice_bytes].contiguous()

    if is_double_quant:
        absmax_slice = absmax_raw.reshape(-1)[:slice_blocks].contiguous()
        # Nested block size: infer from absmax count / nested_absmax count
        nested_block_size = max(1, num_blocks_full // max(1, nested_absmax.numel()))
        import math
        nested_needed = math.ceil(slice_blocks / nested_block_size)
        nested_absmax_slice = nested_absmax.reshape(-1)[:nested_needed].contiguous()
        nested_quant_map_slice = nested_quant_map.contiguous()
    else:
        absmax_slice = absmax_raw.reshape(-1)[:slice_blocks].contiguous()

    print(f"  Slice: {slice_elements} elements ({slice_bytes} bytes, {slice_blocks} blocks)")

    # Dequantize and time it
    best_us = float("inf")
    for run in range(5):
        t0 = time.perf_counter()
        if is_double_quant:
            result_bf16 = dequant_bnb4_double_quant_pytorch(
                weight_slice, absmax_slice, quant_map,
                nested_absmax_slice, nested_quant_map_slice,
                block_size, nested_block_size,
            )
        else:
            result_bf16 = dequant_bnb4_pytorch(
                weight_slice, absmax_slice.float(), quant_map, block_size,
            )
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)

    expected_bytes = result_bf16.view(torch.uint8).reshape(-1).numpy().tobytes()
    print(f"  PyTorch dequant: {best_us:.1f} µs (best of 5, {slice_elements} elements)")

    # Get raw bytes
    w_bytes = weight_slice.numpy().tobytes()
    qm_bytes = quant_map.numpy().tobytes()

    if is_double_quant:
        format_id = 2
        a_bytes = absmax_slice.numpy().tobytes()  # U8
        na_bytes = nested_absmax_slice.numpy().tobytes()
        nqm_bytes = nested_quant_map_slice.numpy().tobytes()
    else:
        format_id = 0
        a_bytes = absmax_slice.numpy().tobytes()  # F32
        na_bytes = b""
        nqm_bytes = b""

    # Write fixture
    output_path = Path(__file__).parent / f"{name}.bin"
    with open(output_path, "wb") as out:
        out.write(struct.pack("<I", format_id))
        out.write(struct.pack("<I", slice_elements))
        out.write(struct.pack("<I", block_size))
        out.write(struct.pack("<I", len(w_bytes)))
        out.write(struct.pack("<I", len(a_bytes)))
        out.write(struct.pack("<I", len(qm_bytes)))
        out.write(struct.pack("<I", len(na_bytes)))
        out.write(struct.pack("<I", len(nqm_bytes)))
        out.write(struct.pack("<I", len(expected_bytes)))
        out.write(w_bytes)
        out.write(a_bytes)
        out.write(qm_bytes)
        out.write(na_bytes)
        out.write(nqm_bytes)
        out.write(expected_bytes)

    print(f"  Written: {output_path.name} ({output_path.stat().st_size} bytes)")


def generate_int8_fixture(model_info: dict) -> None:
    """Generate fixture for INT8 (LLM.int8())."""
    name = model_info["name"]
    model_dir = model_info["model_dir"]
    layer_base = model_info["layer_base"]

    if not model_dir.exists():
        print(f"\nSKIP {name}: model not found at {model_dir}")
        return

    snapshot = find_snapshot(model_dir)
    print(f"\n{name}:")
    print(f"  Snapshot: {snapshot}")

    st_files = list(snapshot.glob("*.safetensors"))
    if not st_files:
        print(f"  ERROR: no safetensors files found")
        return

    weight_name = f"{layer_base}.weight"
    scb_name = f"{layer_base}.SCB"

    with safe_open(str(st_files[0]), framework="pt") as f:
        weight_raw = f.get_tensor(weight_name)
        scb_raw = f.get_tensor(scb_name)

    out_features_full = weight_raw.shape[0]
    in_features_full = weight_raw.shape[1]

    print(f"  weight: {list(weight_raw.shape)}, dtype={weight_raw.dtype}")
    print(f"  SCB:    {list(scb_raw.shape)}, dtype={scb_raw.dtype}")

    # Extract a 256×256 slice
    slice_out = min(256, out_features_full)
    slice_in = min(256, in_features_full)

    weight_slice = weight_raw[:slice_out, :slice_in].contiguous()
    scb_slice = scb_raw[:slice_out].contiguous()

    print(f"  Slice: {slice_out}×{slice_in} = {slice_out * slice_in} elements")

    # Dequantize and time it
    best_us = float("inf")
    for run in range(5):
        t0 = time.perf_counter()
        result_bf16 = dequant_bnb_int8_pytorch(weight_slice, scb_slice)
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)

    expected_bytes = result_bf16.view(torch.uint8).reshape(-1).numpy().tobytes()
    print(f"  PyTorch dequant: {best_us:.1f} µs (best of 5, {slice_out}×{slice_in} = {slice_out * slice_in} elements)")

    # Write the weight as unsigned bytes (I8 → U8 reinterpret)
    w_bytes = weight_slice.view(torch.uint8).reshape(-1).numpy().tobytes()
    scb_bytes = scb_slice.numpy().tobytes()

    output_path = Path(__file__).parent / f"{name}.bin"
    with open(output_path, "wb") as out:
        out.write(struct.pack("<I", 1))  # format_id = INT8
        out.write(struct.pack("<I", slice_out))
        out.write(struct.pack("<I", slice_in))
        out.write(struct.pack("<I", len(w_bytes)))
        out.write(struct.pack("<I", len(scb_bytes)))
        out.write(struct.pack("<I", len(expected_bytes)))
        out.write(w_bytes)
        out.write(scb_bytes)
        out.write(expected_bytes)

    print(f"  Written: {output_path.name} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    print("Generating BitsAndBytes cross-validation fixtures...")
    for model in MODELS:
        if model["format"] in ("bnb4", "bnb4_dq"):
            generate_bnb4_fixture(model)
        elif model["format"] == "int8":
            generate_int8_fixture(model)
    print("\nDone.")
