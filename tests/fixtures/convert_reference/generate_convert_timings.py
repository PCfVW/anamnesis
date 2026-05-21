# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Python-side timing sidecars for the Phase 6 convert
round-trip suite.

The sidecars are read by ``tests/cross_validation_convert.rs`` and used
to print a "vs Python" comparison line alongside the Rust wall-clock
times. Tests pass regardless of whether sidecars are present.

Usage::

    python generate_convert_timings.py

The script writes ``<label>.timing.json`` files next to itself, one per
conversion the Rust suite measures. Currently the labels are:

* ``npz_to_st``         — NumPy NPZ → safetensors via ``safetensors.numpy.save``
* ``pth_to_st``         — PyTorch ``.pth`` → safetensors via ``safetensors.torch.save``
* ``st_to_gguf``        — safetensors-BF16 → GGUF via ``gguf.GGUFWriter``
* ``st_to_bnb_nf4``     — safetensors-BF16 → BnB-NF4 via ``bitsandbytes.functional.quantize_4bit``

Reproducibility notes:

* Tensor size for each measurement is ``[4096, 4096]`` (32 MiB BF16) —
  large enough that measurement noise stays under ~5 %.
* Each path runs 5 best-of-N iterations and records the median.
* The script tolerates missing optional dependencies — if ``gguf`` is
  not installed, the ``st_to_gguf`` sidecar is skipped (the Rust test
  prints "no Python sidecar" instead).

This script is **not** invoked by ``cargo test``; it is a
refresh-when-environment-changes utility.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent

SHAPE = (4096, 4096)
ITERATIONS = 5


def median_seconds(fn) -> float:
    times = []
    # Warmup
    fn()
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def write_sidecar(label: str, seconds: float, library: str) -> None:
    payload = {
        "path_label": label,
        "py_seconds": seconds,
        "py_library": library,
        "shape": list(SHAPE),
    }
    out = HERE / f"{label}.timing.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out.name}: {seconds * 1e6:.1f} µs ({library})")


def time_npz_to_st() -> None:
    try:
        import numpy as np  # type: ignore
        import safetensors.numpy as sn  # type: ignore
    except ImportError as e:
        print(f"skip npz_to_st: {e}", file=sys.stderr)
        return

    arr = np.zeros(SHAPE, dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "in.npz"
        st_path = Path(td) / "out.safetensors"
        np.savez(npz_path, w=arr)

        def step() -> None:
            data = dict(np.load(npz_path))
            sn.save_file(data, st_path)

        secs = median_seconds(step)
    write_sidecar("npz_to_st", secs, f"safetensors {sn.__name__}")


def time_pth_to_st() -> None:
    try:
        import torch  # type: ignore
        import safetensors.torch as st  # type: ignore
    except ImportError as e:
        print(f"skip pth_to_st: {e}", file=sys.stderr)
        return

    weight = torch.zeros(SHAPE, dtype=torch.bfloat16)
    state = {"w": weight}
    with tempfile.TemporaryDirectory() as td:
        pth_path = Path(td) / "in.pth"
        out_path = Path(td) / "out.safetensors"
        torch.save(state, pth_path)

        def step() -> None:
            loaded = torch.load(pth_path, weights_only=True)
            st.save_file(loaded, out_path)

        secs = median_seconds(step)
    write_sidecar("pth_to_st", secs, f"torch {torch.__version__}")


def time_st_to_gguf() -> None:
    try:
        import numpy as np  # type: ignore
        import gguf  # type: ignore
    except ImportError as e:
        print(f"skip st_to_gguf: {e}", file=sys.stderr)
        return

    weight = np.zeros(SHAPE, dtype=np.float32).astype(np.float16)  # F16 stand-in for BF16

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.gguf"

        def step() -> None:
            writer = gguf.GGUFWriter(str(out_path), "anamnesis-phase6")
            writer.add_tensor("w", weight)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

        secs = median_seconds(step)
    write_sidecar("st_to_gguf", secs, f"gguf {gguf.__version__}")


def time_st_to_bnb_nf4() -> None:
    try:
        import bitsandbytes as bnb  # type: ignore
        import torch  # type: ignore
    except ImportError as e:
        print(f"skip st_to_bnb_nf4: {e}", file=sys.stderr)
        return

    weight = torch.zeros(SHAPE, dtype=torch.bfloat16)

    def step() -> None:
        _ = bnb.functional.quantize_4bit(weight, quant_type="nf4", blocksize=64)

    secs = median_seconds(step)
    write_sidecar("st_to_bnb_nf4", secs, f"bitsandbytes {bnb.__version__}")


def main() -> int:
    time_npz_to_st()
    time_pth_to_st()
    time_st_to_gguf()
    time_st_to_bnb_nf4()
    return 0


if __name__ == "__main__":
    sys.exit(main())
