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
    # `gguf` does not expose `__version__` consistently across releases;
    # the importlib.metadata lookup is the canonical fallback.
    try:
        import importlib.metadata as ilm

        gguf_version = ilm.version("gguf")
    except Exception:  # pragma: no cover
        gguf_version = "unknown"
    write_sidecar("st_to_gguf", secs, f"gguf {gguf_version}")


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


# ---------------------------------------------------------------------------
# PyTorch-CPU equivalents for the two non-PyTorch reference paths above.
# These do not replace the canonical numpy / gguf-py measurements; they sit
# alongside them so the Rust report can show both "vs ecosystem default"
# and "vs PyTorch-CPU" ratios — the latter mattering because anamnesis has
# historically benchmarked against PyTorch-CPU on its other conversion
# paths (FP8 / GPTQ / AWQ / BnB / GGUF dequant).
# ---------------------------------------------------------------------------


def time_npz_to_st_torch() -> None:
    """PyTorch-CPU equivalent of `npz_to_st`: load NPZ via NumPy, wrap in
    a `torch.Tensor` on CPU, then write via `safetensors.torch.save_file`.

    The non-PyTorch baseline (`npz_to_st`, above) uses
    `safetensors.numpy.save_file`, which talks to NumPy arrays directly.
    This variant routes the bytes through a `torch.from_numpy` so the
    cost is comparable to "how Python practitioners actually do it"
    when they need a `torch.Tensor` on the other side.
    """
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import safetensors.torch as st  # type: ignore
    except ImportError as e:
        print(f"skip npz_to_st_torch: {e}", file=sys.stderr)
        return

    arr = np.zeros(SHAPE, dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "in.npz"
        out_path = Path(td) / "out.safetensors"
        np.savez(npz_path, w=arr)

        def step() -> None:
            data = dict(np.load(npz_path))
            tensors = {k: torch.from_numpy(v) for k, v in data.items()}
            st.save_file(tensors, out_path)

        secs = median_seconds(step)
    write_sidecar("npz_to_st_torch", secs, f"torch {torch.__version__}")


def time_st_to_gguf_torch() -> None:
    """PyTorch-CPU equivalent of `st_to_gguf`: load a safetensors file
    into a `torch.Tensor` via `safetensors.torch.load_file`, then feed
    the tensor's bytes into `gguf.GGUFWriter.add_tensor` after a
    `.numpy()` round-trip (`gguf-py` requires a NumPy view).

    There is no pure-PyTorch GGUF writer in the ecosystem, so the
    "PyTorch-CPU" qualifier here means "the load side goes through
    torch tensors". The write side still uses `gguf-py` — that is
    inherent: anamnesis's own GGUF writer has no equivalent in PyTorch
    or any other Python library.
    """
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import safetensors.torch as stt  # type: ignore
        import gguf  # type: ignore
    except ImportError as e:
        print(f"skip st_to_gguf_torch: {e}", file=sys.stderr)
        return

    # F16 stand-in for BF16: gguf-py's add_tensor accepts numpy arrays
    # and BF16 has no NumPy dtype, so we use F16 with the same byte
    # width to keep the comparison size-honest.
    tensor = torch.zeros(SHAPE, dtype=torch.float16)
    with tempfile.TemporaryDirectory() as td:
        st_path = Path(td) / "in.safetensors"
        out_path = Path(td) / "out.gguf"
        stt.save_file({"w": tensor}, st_path)

        def step() -> None:
            loaded = stt.load_file(st_path)
            arr = loaded["w"].numpy()
            writer = gguf.GGUFWriter(str(out_path), "anamnesis-phase6")
            writer.add_tensor("w", arr)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

        secs = median_seconds(step)
    try:
        import importlib.metadata as ilm

        gguf_version = ilm.version("gguf")
    except Exception:  # pragma: no cover
        gguf_version = "unknown"
    write_sidecar(
        "st_to_gguf_torch",
        secs,
        f"torch {torch.__version__} + gguf {gguf_version}",
    )


def main() -> int:
    time_npz_to_st()
    time_pth_to_st()
    time_st_to_gguf()
    time_st_to_bnb_nf4()
    # PyTorch-CPU equivalents for the two non-PyTorch baselines.
    time_npz_to_st_torch()
    time_st_to_gguf_torch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
