"""Benchmark one detector on the CPU, in its own process.

**One model per process, deliberately.** Resident memory is the number that
matters for an edge deployment, and it cannot be measured for two models loaded
into one interpreter: torch's allocator does not return freed pages promptly,
so whichever model loads second appears far cheaper than it is. Running each in
a fresh process and reading peak RSS is the only way to get a figure that means
what it says.

Everything is pinned to the CPU — the device is passed explicitly rather than
left to autodetect, because on this machine autodetect chooses MPS and the
whole point of this run is the CPU number.

Cold start is measured from interpreter start, so it includes importing the
framework as well as reading the weights. That is what a process actually pays
on its first request.

Usage::

    python scripts/benchmark_cpu.py --model yolo26 --imgsz 672
    python scripts/benchmark_cpu.py --model rfdetr --imgsz 480
"""

from __future__ import annotations

import argparse
import json
import os

try:
    import resource
except ImportError:
    resource = None  # Windows — peak_rss_mb() uses ctypes fallback
import statistics
import sys
import time
from pathlib import Path

PROCESS_STARTED = time.perf_counter()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WARMUP = 3
MEASURED = 30


def peak_rss_mb() -> float:
    """Return peak resident set size in MB.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux — a difference that
    silently produces a 1000x error if assumed either way.  On Windows the
    ``resource`` module is unavailable, so peak working set size is read via
    the Win32 ``K32GetProcessMemoryInfo`` API instead.
    """
    if resource is not None:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024
    # Windows fallback
    import ctypes
    import ctypes.wintypes

    # Named for the Win32 struct it binds to, so it can be checked
    # against Microsoft's documentation rather than a local rename.
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):  # noqa: N801
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(pmc), pmc.cb
    )
    return pmc.PeakWorkingSetSize / (1024 * 1024)


def main() -> int:
    """Benchmark one detector and print a JSON result line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["yolo26", "rfdetr"], required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument(
        "--images", default="datasets/columns_all_1280_yolo/test/images"
    )
    args = parser.parse_args()

    # Belt and braces: some stacks probe accelerators at import time.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    import torch

    torch.set_num_threads(os.cpu_count() or 1)

    image_dir = Path(args.images)
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[: WARMUP + MEASURED]

    rss_before = peak_rss_mb()
    load_started = time.perf_counter()

    if args.model == "yolo26":
        from ultralytics import YOLO

        model = YOLO("models/columns/yolo26_nano_seg.pt")
        # Ultralytics loads lazily; force the graph onto the CPU now so the
        # cost lands in cold start rather than in the first measured image.
        model.to("cpu")

        def run(path: Path) -> None:
            model.predict(source=str(path), conf=0.25, imgsz=args.imgsz,
                          device="cpu", verbose=False)
    else:
        from PIL import Image
        from rfdetr import RFDETRSegNano

        model = RFDETRSegNano.from_checkpoint(
            "models/columns/rfdetr_nano_seg.pt",
            resolution=args.imgsz,
            device="cpu",
        )

        def run(path: Path) -> None:
            with Image.open(path) as handle:
                model.predict(handle.convert("RGB"), threshold=0.25)

    load_seconds = time.perf_counter() - load_started
    cold_start_seconds = time.perf_counter() - PROCESS_STARTED
    rss_loaded = peak_rss_mb()

    for path in images[:WARMUP]:
        run(path)

    timings: list[float] = []
    for path in images[WARMUP:]:
        started = time.perf_counter()
        run(path)
        timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    mean = statistics.mean(timings)
    report = {
        "model": args.model,
        "imgsz": args.imgsz,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "images_measured": len(timings),
        "cold_start_s": round(cold_start_seconds, 3),
        "weights_load_s": round(load_seconds, 3),
        "latency_mean_ms": round(mean, 1),
        "latency_median_ms": round(statistics.median(timings), 1),
        "latency_p90_ms": round(timings[int(len(timings) * 0.9)], 1),
        "latency_min_ms": round(timings[0], 1),
        "fps": round(1000.0 / mean, 2),
        "rss_baseline_mb": round(rss_before, 1),
        "rss_after_load_mb": round(rss_loaded, 1),
        "rss_peak_mb": round(peak_rss_mb(), 1),
    }
    print("RESULT " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
