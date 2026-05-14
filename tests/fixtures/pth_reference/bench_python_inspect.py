"""Time PyTorch's `torch.load(weights_only=True)` metadata extraction on
the AlgZoo `.pth` fixtures, for cross-language comparison with the Rust
`inspect_pth_from_reader` and `parse_pth(path).inspect()` paths.

Note: PyTorch has no separate "inspect-only" primitive — the closest
equivalent to anamnesis's `PthInspectInfo` is to call `torch.load(...,
weights_only=True)` (which fully reads every tensor) and then iterate
the resulting `state_dict` to collect counts / shapes / dtypes. The Rust
crate's reader-generic path reads only the ZIP central directory +
`data.pkl`, so the speedup measured here is a lower bound — Rust's
path scales sub-linearly in tensor-data size while `torch.load` scales
linearly.

Usage:
    python tests/fixtures/pth_reference/bench_python_inspect.py

Requirements:
    torch >= 2.0
"""

import os
import statistics
import sys
import time
import warnings

import torch

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

FIXTURES = [
    "algzoo_rnn_small.pth",
    "algzoo_transformer_small.pth",
    "algzoo_rnn_blog.pth",
]

# Match the Rust bench harness: 1 warm-up + 5 timed iterations,
# best-of-5 median (and min/max range).
WARMUP = 1
ITERATIONS = 5


def time_load(path: str) -> list[float]:
    """Time torch.load(weights_only=True) + summary extraction.

    Returns a list of elapsed seconds (one per timed iteration).
    The summary extraction mirrors `PthInspectInfo`: tensor_count,
    total_bytes, distinct dtypes in first-occurrence order.
    """
    samples: list[float] = []
    for _ in range(WARMUP):
        _ = torch.load(path, map_location="cpu", weights_only=True)
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        sd = torch.load(path, map_location="cpu", weights_only=True)
        # Summary work that PthInspectInfo also does.
        seen_dtypes: list[torch.dtype] = []
        total_bytes = 0
        tensor_count = 0
        for t in sd.values():
            tensor_count += 1
            total_bytes += t.numel() * t.element_size()
            if t.dtype not in seen_dtypes:
                seen_dtypes.append(t.dtype)
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)
        # Suppress the unused-locals warning while still touching the
        # values so the optimiser can't elide the summary loop.
        del seen_dtypes, total_bytes, tensor_count
    return samples


def fmt_us(s: float) -> str:
    return f"{s * 1e6:6.1f}"


def main() -> int:
    print(f"\nPython `torch.load(weights_only=True)` benchmark "
          f"(torch {torch.__version__})")
    print(f"  {ITERATIONS} timed iterations, {WARMUP} warm-up\n")
    print(f"{'file':<40}  {'size':>7}  "
          f"{'torch min/med/max (µs)':>26}")
    print("=" * 80)

    medians: list[tuple[str, float]] = []
    for name in FIXTURES:
        path = os.path.join(FIXTURE_DIR, name)
        if not os.path.exists(path):
            print(f"SKIP {name} (not found)", file=sys.stderr)
            continue

        size_b = os.path.getsize(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            samples = sorted(time_load(path))
        lo, mid, hi = samples[0], samples[len(samples) // 2], samples[-1]
        medians.append((name, mid))
        print(f"{name:<40}  {size_b / 1024:5.1f} KiB  "
              f"{fmt_us(lo)}/{fmt_us(mid)}/{fmt_us(hi)}")

    print("=" * 80)
    if medians:
        med_of_meds = statistics.median(m for _, m in medians)
        print(f"median torch.load time across {len(medians)} fixtures: "
              f"{med_of_meds * 1e6:.1f} µs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
