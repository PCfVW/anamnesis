"""Time PyTorch's `torch.load(weights_only=True)` metadata extraction on
`.pth` fixtures, for cross-language comparison with the Rust
`inspect_pth_from_reader` and `parse_pth(path).inspect()` paths.

Note: PyTorch has no separate "inspect-only" primitive — the closest
equivalent to anamnesis's `PthInspectInfo` is to call `torch.load(...,
weights_only=True)` (which fully reads every tensor) and then iterate
the resulting `state_dict` to collect counts / shapes / dtypes. The Rust
crate's reader-generic path reads only the ZIP central directory +
`data.pkl`, so the speedup measured here is a lower bound — Rust's
path scales sub-linearly in tensor-data size while `torch.load` scales
linearly.

Two modes are provided:

- **Default mode** — times the 3 in-tree `AlgZoo` fixtures. Per-file
  output with min / median / max.
- **Sweep mode** — set `ANAMNESIS_ALGZOO_DIR` to an external directory
  (typically `algzoo_weights/`); times every `.pth` file in it,
  per-file output is suppressed, only the global + per-family summary
  table is printed. Mirrors the Rust `bench_pth_inspect_algzoo_sweep`
  test.

Usage (default mode):
    python tests/fixtures/pth_reference/bench_python_inspect.py

Usage (sweep mode):
    $env:ANAMNESIS_ALGZOO_DIR = "C:/Users/Eric JACOPIN/Documents/Data/algzoo_weights"
    python tests/fixtures/pth_reference/bench_python_inspect.py

Requirements:
    torch >= 2.0
"""

import os
import re
import statistics
import sys
import time
import warnings

import torch

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# In-tree fixtures used in default mode.
DEFAULT_FIXTURES = [
    "algzoo_rnn_small.pth",
    "algzoo_transformer_small.pth",
    "algzoo_rnn_blog.pth",
]

ALGZOO_DIR_ENV = "ANAMNESIS_ALGZOO_DIR"

# Match the Rust bench harness: 1 warm-up + 5 timed iterations.
WARMUP = 1
ITERATIONS = 5


def time_load(path: str) -> list[float]:
    """Time torch.load(weights_only=True) + summary extraction.

    Returns a list of elapsed seconds (one per timed iteration). The
    summary extraction mirrors anamnesis's `PthInspectInfo`:
    tensor_count, total_bytes, distinct dtypes in first-occurrence order.
    """
    samples: list[float] = []
    for _ in range(WARMUP):
        _ = torch.load(path, map_location="cpu", weights_only=True)
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        sd = torch.load(path, map_location="cpu", weights_only=True)
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
        # Keep the local bindings live so the optimiser can't elide the
        # summary loop.
        del seen_dtypes, total_bytes, tensor_count
    return samples


def fmt_us(s: float) -> str:
    return f"{s * 1e6:6.1f}"


def family_of(filename: str) -> str:
    """Strip the trailing `_<digits>...` suffix from a stem to get the
    task-family name (e.g., `2nd_argmax_16_10_0_0.pth` → `2nd_argmax`).

    Cannot key on "first digit" because some family names start with a
    digit (e.g., `2nd_argmax`); we cut at the first underscore-digit
    boundary, which separates the task name from its hyperparameters.
    """
    stem = os.path.splitext(filename)[0]
    m = re.search(r"_\d", stem)
    if not m:
        return stem
    return stem[: m.start()]


def distribution(samples: list[float]) -> tuple[float, float, float, float, float, float]:
    """Returns (min, p25, median, p75, mean, max) over the unordered list."""
    s = sorted(samples)
    n = len(s)
    return (s[0], s[n // 4], s[n // 2], s[(n * 3) // 4], statistics.mean(s), s[-1])


def run_default_mode() -> int:
    """Per-file timings over the in-tree AlgZoo fixtures."""
    print(f"\nPython `torch.load(weights_only=True)` benchmark "
          f"(torch {torch.__version__})")
    print(f"  {ITERATIONS} timed iterations, {WARMUP} warm-up\n")
    print(f"{'file':<40}  {'size':>7}  "
          f"{'torch min/med/max (us)':>26}")
    print("=" * 80)

    medians: list[tuple[str, float]] = []
    for name in DEFAULT_FIXTURES:
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
              f"{med_of_meds * 1e6:.1f} us")
    return 0


def run_sweep_mode(sweep_dir: str) -> int:
    """Sweep every `.pth` file under `sweep_dir`, emit aggregate stats."""
    paths = sorted(
        os.path.join(sweep_dir, f)
        for f in os.listdir(sweep_dir)
        if f.endswith(".pth") and os.path.isfile(os.path.join(sweep_dir, f))
    )
    if not paths:
        print(f"SKIP: no .pth files under {sweep_dir}", file=sys.stderr)
        return 0

    print(f"\nPython `torch.load(weights_only=True)` AlgZoo sweep "
          f"(torch {torch.__version__})")
    print(f"  {len(paths)} files in {sweep_dir}")
    print(f"  {ITERATIONS} timed iterations per file, {WARMUP} warm-up")

    all_medians: list[float] = []
    family_buckets: dict[str, list[float]] = {}
    total_bytes = 0
    sweep_start = time.perf_counter()
    last_report = sweep_start

    for i, path in enumerate(paths):
        total_bytes += os.path.getsize(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            samples = time_load(path)
        samples.sort()
        med = samples[len(samples) // 2]
        all_medians.append(med)
        family_buckets.setdefault(family_of(os.path.basename(path)), []).append(med)

        # Lightweight progress indicator every ~5 s of wall clock.
        now = time.perf_counter()
        if now - last_report > 5.0:
            elapsed = now - sweep_start
            done = i + 1
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (len(paths) - done) / rate if rate > 0 else float("inf")
            print(f"  ... {done}/{len(paths)} ({rate:.1f} files/s, ETA {eta:.0f}s)",
                  file=sys.stderr)
            last_report = now

    sweep_elapsed = time.perf_counter() - sweep_start
    print(f"\n  sweep wall-clock: {sweep_elapsed:.1f}s  "
          f"({len(paths) / sweep_elapsed:.1f} files/s)")
    print(f"  total bytes inspected: {total_bytes / (1024 * 1024):.1f} MiB")

    lo, p25, med, p75, mean, hi = distribution(all_medians)
    print(f"\n  {'metric (us)':<14}  {'min':>9}  {'p25':>9}  {'median':>9}"
          f"  {'p75':>9}  {'mean':>9}  {'max':>9}")
    print("  " + "-" * 82)
    print(f"  {'torch.load':<14}  "
          f"{lo * 1e6:>9.1f}  {p25 * 1e6:>9.1f}  {med * 1e6:>9.1f}  "
          f"{p75 * 1e6:>9.1f}  {mean * 1e6:>9.1f}  {hi * 1e6:>9.1f}")

    # Per-family breakdown.
    if len(family_buckets) > 1:
        print(f"\n  {'family':<18}  {'count':>6}  {'torch.load med (us)':>22}")
        print("  " + "-" * 52)
        for family in sorted(family_buckets):
            samples = family_buckets[family]
            med_family = statistics.median(samples)
            print(f"  {family:<18}  {len(samples):>6}  {med_family * 1e6:>20.1f}")

    return 0


def main() -> int:
    sweep_dir = os.environ.get(ALGZOO_DIR_ENV)
    if sweep_dir and os.path.isdir(sweep_dir):
        return run_sweep_mode(sweep_dir)
    if sweep_dir:
        print(f"WARN: {ALGZOO_DIR_ENV} set but {sweep_dir} is not a directory; "
              f"falling back to default mode", file=sys.stderr)
    return run_default_mode()


if __name__ == "__main__":
    raise SystemExit(main())
