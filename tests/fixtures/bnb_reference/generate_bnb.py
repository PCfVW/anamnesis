#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate BitsAndBytes dequantization reference fixtures from real models.

Extracts a small block-aligned slice of BnB tensors (weight, absmax, quant_map,
and optionally nested_absmax/nested_quant_map or SCB), dequantizes with the
**real bitsandbytes library** (`bitsandbytes.functional.dequantize_4bit` /
`int8_vectorwise_dequant`), and writes a binary fixture file for Rust
cross-validation.

External-reference discipline (the v0.6.4 rule)
-----------------------------------------------
The "expected" output MUST come from the canonical library's own code — never
from a hand-rolled reimplementation of the formula. A previous version of this
script reimplemented the dequant in plain PyTorch and hard-coded a WRONG
nibble order (low-nibble-first; bitsandbytes is high-nibble-first), which let
an element-permuting decode bug ship green at 0 ULP. See
docs/dogfooding-feedbacks/bnb-nibble-order-and-circular-fixture-validation.md.

Generation runs the CUDA kernel (`dequantize_4bit` dispatches to the GPU when
the tensors live there). This is deliberate:

- The CUDA kernel is what the ecosystem's inference actually runs, and it
  computes `code[nibble] * absmax` in f32 with a single rounding at the output
  store — measured bit-identical (0/4096 mismatches across NF4, NF4-DQ, and
  FP4 fixtures) to anamnesis' f32-multiply + round-to-nearest-even BF16 path.
- The bitsandbytes CPU kernel (0.49) computes in the target dtype instead and
  double-rounds, diverging from the CUDA kernel by 1 ULP on ~19 % of elements.
  Anchoring to it would force max_ulp=1 and hide real regressions.

Only *generation* needs the GPU; the committed `.bin` fixture and the Rust
cross-validation tests remain 100 % CPU.

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
  4 bytes: nested_offset (f32) — double-quant absmax offset (0.0 if not double-quant)
  [weight_len bytes]: raw U8 weight data
  [absmax_len bytes]: raw absmax data
  [quant_map_len bytes]: raw quant_map data
  [nested_absmax_len bytes]: raw nested absmax (empty if not double-quant)
  [nested_quant_map_len bytes]: raw nested quant_map (empty if not double-quant)
  [expected_len bytes]: expected BF16 output

The `nested_offset` field is new in v0.6.4: real bitsandbytes double-quant
recovers `absmax = dequantize_blockwise(absmax_u8, state2) + offset`, where
`offset` (the mean of the original absmax values) is stored in the
`quant_state` JSON blob as `nested_offset`. The previous hand-rolled
reference omitted it — circularly hiding the same omission in the Rust
decoder.

Fixture binary format — INT8 (unchanged):
  4 bytes: format_id (u32) — 1
  4 bytes: out_features (u32)
  4 bytes: in_features (u32)
  4 bytes: weight_len (u32) — bytes of I8 weight data
  4 bytes: scb_len (u32) — bytes of SCB data (F32)
  4 bytes: expected_len (u32) — bytes of expected BF16 output
  [weight_len bytes]: raw I8 weight data
  [scb_len bytes]: raw SCB data
  [expected_len bytes]: expected BF16 output

Sidecar timing file (PyTorch quantize baseline for encode cross-validation):
Each fixture also emits `<name>.timing.json` with:
  {
    "pytorch_quantize_ns": <best-of-N ns>,
    "pytorch_quantize_iters": N,
    "operation": "bitsandbytes.functional.quantize_4bit (cuda)"
  }
The Rust encode cross-validation tests read this sidecar (if present) to
print a side-by-side runtime comparison. Sidecar absence is non-fatal.
The timing now measures the REAL bitsandbytes quantize kernel on the GPU
(the previous version timed a hand-rolled CPU reimplementation), so the
numbers are an inference-ecosystem baseline, not a CPU-vs-CPU comparison.

Usage:
  python generate_bnb.py
"""

import json
import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open

import bitsandbytes
from bitsandbytes.functional import (
    QuantState,
    dequantize_4bit,
    int8_vectorwise_dequant,
    quantize_4bit,
)


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def find_snapshot(model_dir: Path) -> Path:
    """Find the snapshot directory for a cached model."""
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"No snapshots directory in {model_dir}")
    dirs = list(snapshots.iterdir())
    if not dirs:
        raise FileNotFoundError(f"No snapshot found in {snapshots}")
    return dirs[0]


# ---- Sidecar writer ----

def write_timing_sidecar(name: str, op: str, ns_iters: list[int]) -> None:
    """Write `<name>.timing.json` next to the .bin fixture.

    `ns_iters` is the list of per-iteration durations in nanoseconds;
    we record the best-of-N.
    """
    output_path = Path(__file__).parent / f"{name}.timing.json"
    best_ns = min(ns_iters)
    payload = {
        "pytorch_quantize_ns": int(best_ns),
        "pytorch_quantize_iters": len(ns_iters),
        "operation": op,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  Sidecar:  {output_path.name} (best={best_ns / 1000:.1f} µs)")


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
    # Cross-architecture plain-FP4 fixture (Qwen3). Proves the sign-of-zero
    # preservation rule generalises to a different architecture and HF org.
    {
        "name": "qwen3_mcqa_fp4",
        "model_dir": HF_CACHE / "models--ema1234--qwen_mcqa_bnb_fp4",
        "format": "bnb4",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    # Cross-architecture double-quant NF4 fixtures (Qwen2.5 and Phi-3.5) —
    # the ecosystem-realistic test set for the NF4 path: every non-Llama
    # BnB-NF4 model checked uses double-quant (bitsandbytes' default).
    {
        "name": "qwen2_5_1_5b_nf4_dq",
        "model_dir": HF_CACHE / "models--unsloth--Qwen2.5-1.5B-Instruct-bnb-4bit",
        "format": "bnb4_dq",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "phi3_5_mini_nf4_dq",
        "model_dir": HF_CACHE / "models--unsloth--Phi-3.5-mini-instruct-bnb-4bit",
        "format": "bnb4_dq",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
]


def load_quant_state_blob(f, layer_base: str) -> dict:
    """Read and parse the `quant_state.bitsandbytes__nf4/__fp4` JSON blob."""
    qs_names = [
        n for n in f.keys() if n.startswith(f"{layer_base}.weight.quant_state.bitsandbytes__")
    ]
    if not qs_names:
        raise KeyError(f"no quant_state tensor under {layer_base}")
    blob = f.get_tensor(qs_names[0])
    meta = json.loads(bytes(blob.numpy().tobytes()).decode("utf-8"))
    meta["_quant_type"] = qs_names[0].rsplit("__", 1)[1]  # "nf4" or "fp4"
    return meta


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
        print("  ERROR: no safetensors files found")
        return

    weight_name = f"{layer_base}.weight"
    absmax_name = f"{layer_base}.weight.absmax"
    quant_map_name = f"{layer_base}.weight.quant_map"

    with safe_open(str(st_files[0]), framework="pt") as f:
        weight_raw = f.get_tensor(weight_name)
        absmax_raw = f.get_tensor(absmax_name)
        quant_map = f.get_tensor(quant_map_name)
        qs_meta = load_quant_state_blob(f, layer_base)

        if is_double_quant:
            nested_absmax = f.get_tensor(f"{layer_base}.weight.nested_absmax")
            nested_quant_map = f.get_tensor(f"{layer_base}.weight.nested_quant_map")

    quant_type = qs_meta["_quant_type"]
    block_size = int(qs_meta["blocksize"])
    out_dtype = TORCH_DTYPES[qs_meta["dtype"]]
    total_elements_full = weight_raw.numel() * 2

    print(f"  weight:    {list(weight_raw.shape)}, dtype={weight_raw.dtype}")
    print(f"  absmax:    {list(absmax_raw.shape)}, dtype={absmax_raw.dtype}")
    print(f"  quant_map: {list(quant_map.shape)}, dtype={quant_map.dtype}")
    print(f"  quant_state: type={quant_type}, blocksize={block_size}, dtype={qs_meta['dtype']}")

    nested_offset = 0.0
    if is_double_quant:
        nested_block_size = int(qs_meta["nested_blocksize"])
        nested_offset = float(qs_meta["nested_offset"])
        print(
            f"  nested: blocksize={nested_block_size}, offset={nested_offset:.9f}, "
            f"absmax={list(nested_absmax.shape)}, quant_map={list(nested_quant_map.shape)}"
        )

    # Extract a slice: 4096 elements = 2048 bytes = 64 blocks (for block_size=64)
    slice_elements = min(4096, total_elements_full)
    slice_blocks = slice_elements // block_size
    slice_bytes = slice_elements // 2

    weight_slice = weight_raw.reshape(-1)[:slice_bytes].contiguous()
    absmax_slice = absmax_raw.reshape(-1)[:slice_blocks].contiguous()

    if is_double_quant:
        nested_needed = -(-slice_blocks // nested_block_size)  # ceil div
        nested_absmax_slice = nested_absmax.reshape(-1)[:nested_needed].contiguous()
        nested_quant_map_slice = nested_quant_map.contiguous()

    print(f"  Slice: {slice_elements} elements ({slice_bytes} bytes, {slice_blocks} blocks)")

    # ---- Build the QuantState for the REAL bitsandbytes dequant (CUDA) ----
    if is_double_quant:
        state2 = QuantState(
            absmax=nested_absmax_slice.float().cuda(),
            code=nested_quant_map_slice.float().cuda(),
            blocksize=nested_block_size,
            dtype=torch.float32,
        )
        qstate = QuantState(
            absmax=absmax_slice.cuda(),
            shape=(slice_elements, 1),
            code=quant_map.float().cuda(),
            blocksize=block_size,
            quant_type=quant_type,
            dtype=out_dtype,
            offset=torch.tensor(nested_offset, dtype=torch.float32).cuda(),
            state2=state2,
        )
    else:
        qstate = QuantState(
            absmax=absmax_slice.float().cuda(),
            shape=(slice_elements, 1),
            code=quant_map.float().cuda(),
            blocksize=block_size,
            quant_type=quant_type,
            dtype=out_dtype,
        )

    weight_cuda = weight_slice.reshape(-1, 1).cuda()

    # Dequantize with the canonical kernel and time it (best of 5)
    best_us = float("inf")
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = dequantize_4bit(weight_cuda, qstate, quant_type=quant_type)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)

    # Expected is BF16: a no-op for bf16-dtype quant states; for f16-dtype
    # states (e.g. the FP4 fixtures) this is the canonical f16 output
    # converted once to BF16 — measured bit-identical to anamnesis'
    # direct f32→BF16 path on every fixture (0 mismatches).
    result_bf16 = result.reshape(-1).to(torch.bfloat16).cpu()
    expected_bytes = result_bf16.view(torch.uint8).numpy().tobytes()
    print(
        f"  bitsandbytes dequantize_4bit (cuda): {best_us:.1f} µs "
        f"(best of 5, {slice_elements} elements)"
    )

    # ---- Time the REAL quantize kernel for the encode-side sidecar ----
    quantize_iters_ns: list[int] = []
    src = result.reshape(-1, 1)
    for _ in range(5):
        torch.cuda.synchronize()
        q0 = time.perf_counter_ns()
        _packed, _st = quantize_4bit(
            src,
            blocksize=block_size,
            compress_statistics=is_double_quant,
            quant_type=quant_type,
        )
        torch.cuda.synchronize()
        q1 = time.perf_counter_ns()
        quantize_iters_ns.append(q1 - q0)
    write_timing_sidecar(
        name,
        f"bitsandbytes.functional.quantize_4bit (cuda, compress_statistics={is_double_quant})",
        quantize_iters_ns,
    )

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
        out.write(struct.pack("<f", nested_offset))
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
        print("  ERROR: no safetensors files found")
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

    # Dequantize with the canonical function (CUDA) and time it (best of 5).
    # `int8_vectorwise_dequant` computes `A * stats * (1/127)` in f32 — the
    # named LLM.int8() weight dequant in bitsandbytes. Measured bit-identical
    # to anamnesis' `w × (SCB/127)` after BF16 rounding (0/65536 mismatches).
    w_cuda = weight_slice.cuda()
    scb_cuda = scb_slice.cuda()
    best_us = float("inf")
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result_f32 = int8_vectorwise_dequant(w_cuda, scb_cuda)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)

    result_bf16 = result_f32.to(torch.bfloat16).cpu()
    expected_bytes = result_bf16.view(torch.uint8).reshape(-1).numpy().tobytes()
    print(
        f"  bitsandbytes int8_vectorwise_dequant (cuda): {best_us:.1f} µs "
        f"(best of 5, {slice_out}×{slice_in} = {slice_out * slice_in} elements)"
    )

    # ---- Time the inverse (quantize) for the encode-side sidecar ----
    # int8_vectorwise_quant is the canonical row-wise int8 quantizer.
    from bitsandbytes.functional import int8_vectorwise_quant

    src = result_bf16.to(torch.float16).cuda()
    quantize_iters_ns: list[int] = []
    for _ in range(5):
        torch.cuda.synchronize()
        q0 = time.perf_counter_ns()
        _q, _stats, _outliers = int8_vectorwise_quant(src)
        torch.cuda.synchronize()
        q1 = time.perf_counter_ns()
        quantize_iters_ns.append(q1 - q0)
    write_timing_sidecar(
        name, "bitsandbytes.functional.int8_vectorwise_quant (cuda)", quantize_iters_ns
    )

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
    print(f"Generating BitsAndBytes cross-validation fixtures (bitsandbytes {bitsandbytes.__version__})...")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for fixture GENERATION: the anchor is the canonical "
            "CUDA dequant kernel (single f32 rounding). The committed fixtures and "
            "the Rust cross-validation tests remain CPU-only."
        )
    for model in MODELS:
        if model["format"] in ("bnb4", "bnb4_dq"):
            generate_bnb4_fixture(model)
        elif model["format"] == "int8":
            generate_int8_fixture(model)
    print("\nDone.")
