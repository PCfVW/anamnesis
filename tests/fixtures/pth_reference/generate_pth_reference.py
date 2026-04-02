"""Generate reference data for AlgZoo .pth cross-validation.

For each .pth fixture, loads with torch.load() and writes a JSON manifest
containing tensor names, shapes, dtypes, and raw F32 bytes as hex strings.
The Rust test reads these manifests to verify byte-exact parsing.

Usage:
    python tests/fixtures/pth_reference/generate_pth_reference.py

Requirements:
    torch >= 2.0
"""

import json
import os
import struct
import sys

import torch

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

FIXTURES = [
    "algzoo_rnn_small.pth",        # 10 params, 3 tensors (RNN, newer PyTorch format)
    "algzoo_transformer_small.pth", # 50 params, 7 tensors (Transformer, newer format)
    "algzoo_rnn_blog.pth",          # 432 params, 3 tensors (RNN, older PyTorch format)
]


def dtype_name(t: torch.Tensor) -> str:
    """Map torch dtype to anamnesis PthDtype name."""
    return {
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.uint8: "U8",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.bool: "BOOL",
    }[t.dtype]


def tensor_to_bytes_hex(t: torch.Tensor) -> str:
    """Convert a tensor to contiguous row-major bytes as a hex string."""
    # .contiguous() ensures row-major layout, .cpu() ensures host memory
    data = t.detach().cpu().contiguous().numpy().tobytes()
    return data.hex()


def process_fixture(filename: str) -> None:
    """Load a .pth file and write a .json reference manifest."""
    path = os.path.join(FIXTURE_DIR, filename)
    if not os.path.exists(path):
        print(f"  SKIP {filename} (not found)", file=sys.stderr)
        return

    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    tensors = []
    for name, tensor in state_dict.items():
        tensors.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": dtype_name(tensor),
            "data_hex": tensor_to_bytes_hex(tensor),
        })

    manifest = {
        "source": filename,
        "torch_version": torch.__version__,
        "num_tensors": len(tensors),
        "tensors": tensors,
    }

    out_path = os.path.join(FIXTURE_DIR, filename.replace(".pth", "_reference.json"))
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_params = sum(t.numel() for t in state_dict.values())
    print(f"  {filename} -> {os.path.basename(out_path)}: "
          f"{len(tensors)} tensors, {total_params} params")


def main():
    print(f"Generating .pth reference data (torch {torch.__version__})")
    for fixture in FIXTURES:
        process_fixture(fixture)
    print("Done.")


if __name__ == "__main__":
    main()
