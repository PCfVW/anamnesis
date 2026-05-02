#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate safetensors header cross-validation fixtures.

Builds 4 small ``.safetensors`` files (one per quantization scheme
anamnesis detects) and, for each, a sibling ``.expected.json`` reference
recording exactly what the upstream HuggingFace ``safetensors`` Python
library reports about that file's header. The Rust integration tests in
``tests/cross_validation_safetensors.rs`` cross-validate anamnesis output
against these Python-sourced references.

Schemes covered:
  - fp8.safetensors        — fine-grained FP8 (E4M3) with ``_scale_inv``
                             companion + BF16 passthrough norm
  - gptq.safetensors       — GPTQ INT4 with ``.qweight``/``.scales``/
                             ``.qzeros``/``.g_idx`` companions and the
                             ``gptq_bits``/``gptq_group_size`` file metadata
                             keys ``AutoGPTQ`` writes
  - awq.safetensors        — AWQ INT4 with ``.qweight``/``.scales``/
                             ``.qzeros`` companions (no ``.g_idx``)
  - bnb_nf4.safetensors    — BitsAndBytes NF4 with ``.weight``/
                             ``.weight.absmax``/``.weight.quant_map`` plus
                             a JSON ``.weight.quant_state.bitsandbytes__nf4``
                             companion encoding the original ``[8, 8]``
                             tensor shape

For each fixture, ``record_reference()`` walks the raw 8-byte length prefix
plus JSON header (per the safetensors format specification), then
cross-checks the parse against ``safetensors.safe_open`` from the upstream
Python library before serialising the result as ``<fixture>.expected.json``.
This gives the Rust test a triple-checked oracle: spec-derived JSON parse
== upstream library view == anamnesis output.

Usage:
  python generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file as save_numpy
from safetensors.torch import save_file as save_torch


OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------


def build_fp8_fixture(path: Path) -> None:
    """Fine-grained FP8: weight [8, 8] + 2D scale_inv [2, 2] + BF16 norm."""
    weight = torch.zeros((8, 8), dtype=torch.float8_e4m3fn)
    scale_inv = torch.ones((2, 2), dtype=torch.float32)
    norm = torch.ones(8, dtype=torch.bfloat16)

    save_torch(
        {
            "model.layers.0.weight": weight,
            "model.layers.0.weight_scale_inv": scale_inv,
            "model.norm.weight": norm,
        },
        str(path),
    )


def build_gptq_fixture(path: Path) -> None:
    """GPTQ INT4: qweight [8, 32] (in_features=64, out_features=32, bits=4,
    pack_factor=8, group_size=32 → num_groups=2) plus scales/qzeros/g_idx
    companions and AutoGPTQ-style metadata."""
    in_features, out_features = 64, 32
    bits, group_size = 4, 32
    pack_factor = 32 // bits  # 8
    num_groups = in_features // group_size  # 2

    qweight = np.zeros(
        (in_features // pack_factor, out_features), dtype=np.int32
    )
    scales = np.ones((num_groups, out_features), dtype=np.float16)
    qzeros = np.zeros(
        (num_groups, out_features // pack_factor), dtype=np.int32
    )
    g_idx = np.repeat(np.arange(num_groups, dtype=np.int32), group_size)
    norm = np.ones(out_features, dtype=np.float32).astype(np.float16)

    save_numpy(
        {
            "model.layers.0.qweight": qweight,
            "model.layers.0.scales": scales,
            "model.layers.0.qzeros": qzeros,
            "model.layers.0.g_idx": g_idx,
            "model.norm.weight": norm,
        },
        str(path),
        metadata={"gptq_bits": "4", "gptq_group_size": "32"},
    )


def build_awq_fixture(path: Path) -> None:
    """AWQ INT4: qweight packs along columns, so qweight shape is
    [in_features, out_features / pack_factor]. No g_idx (sequential
    groups). in_features=64, out_features=32, bits=4, group_size=32 →
    qweight [64, 4], scales [2, 32], qzeros [2, 4]."""
    in_features, out_features = 64, 32
    bits, group_size = 4, 32
    pack_factor = 32 // bits  # 8
    num_groups = in_features // group_size  # 2

    qweight = np.zeros(
        (in_features, out_features // pack_factor), dtype=np.int32
    )
    scales = np.ones((num_groups, out_features), dtype=np.float16)
    qzeros = np.zeros(
        (num_groups, out_features // pack_factor), dtype=np.int32
    )
    norm = np.ones(out_features, dtype=np.float32).astype(np.float16)

    save_numpy(
        {
            "model.layers.0.qweight": qweight,
            "model.layers.0.scales": scales,
            "model.layers.0.qzeros": qzeros,
            "model.norm.weight": norm,
        },
        str(path),
    )


def build_bnb_nf4_fixture(path: Path) -> None:
    """BnB NF4: U8 weight stored as [N, 1] (2 NF4 values per byte) plus
    F32 absmax + F32 quant_map (16 entries) + JSON quant_state. Logical
    shape [8, 8] = 64 elements; block_size 64 = 1 block; weight stored
    as [32, 1] U8 bytes."""
    logical_shape = [8, 8]
    total_elements = logical_shape[0] * logical_shape[1]  # 64
    block_size = 64
    num_blocks = total_elements // block_size  # 1
    packed_bytes = total_elements // 2  # 32

    weight = np.zeros((packed_bytes, 1), dtype=np.uint8)
    absmax = np.array([1.0] * num_blocks, dtype=np.float32)
    quant_map = np.zeros(16, dtype=np.float32)

    quant_state_payload = {
        "shape": logical_shape,
        "blocksize": block_size,
        "quant_type": "nf4",
        "dtype": "bfloat16",
        "nested_blocksize": 0,
        "nested_offset": 0.0,
    }
    quant_state_bytes = np.frombuffer(
        json.dumps(quant_state_payload).encode("utf-8"), dtype=np.uint8
    ).copy()

    norm = np.ones(8, dtype=np.float32).astype(np.float16)

    save_numpy(
        {
            "model.layers.0.weight": weight,
            "model.layers.0.weight.absmax": absmax,
            "model.layers.0.weight.quant_map": quant_map,
            "model.layers.0.weight.quant_state.bitsandbytes__nf4": quant_state_bytes,
            "model.norm.weight": norm,
        },
        str(path),
    )


# ---------------------------------------------------------------------------
# Reference recorder
# ---------------------------------------------------------------------------


def record_reference(safetensors_path: Path, scheme: str) -> dict:
    """Build the canonical reference for one safetensors fixture.

    Sources the metadata two ways and asserts they agree:

    1. Spec-derived: read the 8-byte LE length prefix, parse the JSON
       header per the safetensors format spec.
    2. Library view: ``safetensors.safe_open`` reports the same names,
       dtypes, shapes, and file metadata.

    Then serialise the spec-derived view as the JSON reference. This is
    exactly the metadata anamnesis must reproduce.
    """
    with open(safetensors_path, "rb") as fh:
        prefix = fh.read(8)
        header_size = int.from_bytes(prefix, "little")
        header_bytes = fh.read(header_size)
    header_json = json.loads(header_bytes)

    metadata_section = header_json.pop("__metadata__", None)

    tensors_from_json = sorted(
        (
            {
                "name": name,
                "dtype": info["dtype"],
                "shape": list(info["shape"]),
                "data_offsets": list(info["data_offsets"]),
            }
            for name, info in header_json.items()
        ),
        key=lambda t: t["name"],
    )

    # Cross-check against the upstream Python library. ``safe_open`` does
    # not surface ``data_offsets`` directly, so we only validate what it
    # does surface (names, dtypes, shapes, file metadata).
    with safe_open(str(safetensors_path), framework="numpy") as f:
        lib_metadata = f.metadata() or {}
        lib_names = sorted(f.keys())
        assert lib_names == [t["name"] for t in tensors_from_json], (
            f"name mismatch between header JSON and safe_open: "
            f"{lib_names} vs {[t['name'] for t in tensors_from_json]}"
        )
        for tensor_view, ref in zip(
            (f.get_slice(name) for name in lib_names), tensors_from_json
        ):
            lib_dtype = tensor_view.get_dtype()
            lib_shape = list(tensor_view.get_shape())
            assert lib_dtype == ref["dtype"], (
                f"dtype mismatch for {ref['name']}: {lib_dtype} vs {ref['dtype']}"
            )
            assert lib_shape == ref["shape"], (
                f"shape mismatch for {ref['name']}: {lib_shape} vs {ref['shape']}"
            )

    expected_metadata = metadata_section or {}
    assert lib_metadata == expected_metadata, (
        f"file metadata mismatch: {lib_metadata} vs {expected_metadata}"
    )

    return {
        "scheme": scheme,
        "header_size": header_size,
        "metadata": expected_metadata,
        "tensors": tensors_from_json,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    fixtures = [
        ("fp8", build_fp8_fixture),
        ("gptq", build_gptq_fixture),
        ("awq", build_awq_fixture),
        ("bnb_nf4", build_bnb_nf4_fixture),
    ]

    for scheme, builder in fixtures:
        st_path = OUT_DIR / f"{scheme}.safetensors"
        ref_path = OUT_DIR / f"{scheme}.expected.json"
        builder(st_path)
        reference = record_reference(st_path, scheme)
        ref_path.write_text(json.dumps(reference, indent=2) + "\n")
        print(
            f"  {scheme}: {st_path.stat().st_size} B fixture, "
            f"header_size={reference['header_size']}, "
            f"{len(reference['tensors'])} tensors"
        )


if __name__ == "__main__":
    main()
