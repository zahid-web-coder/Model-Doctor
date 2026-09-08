"""Benchmark a saved Model Doctor run and persist its compute cost.

**Deliberately separate from evaluation.** mAP is a property of a model on a
dataset and travels between machines; latency and memory are properties of a
model on *one device* and do not. Running them together would invite a single
result row that is half portable, and a reader with no way to tell which half.

**One model per process, also deliberately.** Resident memory cannot be
measured for two models loaded into one interpreter: torch's allocator does not
return freed pages promptly, so whichever loads second looks far cheaper than
it is. A fresh process per measurement is the only way to get a figure that
means what it says.

**Not the same thing as ``inference_ms``.** Every diagnosis run already times
its own forward passes, but that number includes the first image's lazy kernel
compilation and weight paging, and it is whatever device the run happened to
use. This script warms up first, measures a fixed number of images, and records
the device explicitly. The two are never presented as equivalent.

The device is required rather than autodetected, because the entire value of
the number depends on knowing which one produced it.

**A device is a measurement, not a recommendation.** A row here says only that
this model was measured on that device on this host. It does not say the device
is a sensible deployment target, and nothing in this project ranks devices — the
comparison page will not even put two of them in one table. Two figures for one
model mean it was measured twice, not that the faster one is endorsed.

**Detector families bind their device differently, and it matters.** Ultralytics
takes ``device=`` on each ``predict`` call, so the requested device is honoured
per image. RF-DETR binds its device when the model is constructed and its
``predict`` accepts no device at all; passing nothing does not fall back to
``config.DEVICE``, it lets the library choose. On Apple silicon that choice is
MPS regardless of what was asked for, which once produced rows recorded as
``cpu`` while running on MPS — a 588 ms measurement stored as 162 ms.
:meth:`RFDetrDetector.load` now passes the device explicitly; the regression is
covered in ``tests/test_rfdetr_adapter.py``.

**Threading caveat, from a separate codebase.** Concurrent RF-DETR inference on
MPS from multiple threads is reported to deadlock in Metal
(``[mutex.cc:452] RAW: Lock blocking``), and at least one production service
pins RF-DETR to CPU on macOS and serialises every prediction because of it.
This script is unaffected: it is one process, one model, one sequential loop,
and no thread ever issues a second prediction concurrently. No lock is needed
here and none is taken. Anything that later drives these detectors from a
threaded server should not assume the same.

Usage::

    python scripts/benchmark_run.py --run-id 1 --device cpu
    python scripts/benchmark_run.py --run-id 2 --device mps
"""

from __future__ import annotations

import argparse
import os
import platform

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

# Project imports live inside `main`, not here, and that is not a style
# preference. Cold start is measured from interpreter start, so it must include
# importing torch and the detector stack — which is what a process actually
# pays before its first request. Importing at module scope would move that cost
# above `PROCESS_STARTED` and quietly report a cold start that no deployment
# ever experiences.

WARMUP_IMAGES = 3
MEASURED_IMAGES = 30


def _project() -> tuple:
    """Import the project stack, timed as part of cold start.

    Returns the handful of names this script needs, so the import cost lands
    after :data:`PROCESS_STARTED` rather than before it.
    """
    from model_doctor import config
    from model_doctor.app import evaluation, storage
    from model_doctor.app.detectors import build_detector, detect_family
    from model_doctor.app.storage import file_sha256
    from model_doctor.utils.dataset import load_dataset_config
    from model_doctor.utils.logging_utils import get_logger

    return (
        config, evaluation, storage, build_detector, detect_family,
        file_sha256, load_dataset_config, get_logger,
    )


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


def gpu_memory_mb(device: str) -> tuple[float | None, float | None]:
    """Return ``(allocated_mb, driver_mb)`` for the device, or ``(None, None)``.

    Null rather than zero when the device has no such counter: an unmeasured
    figure is not a measurement of nothing.
    """
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return (
            torch.cuda.max_memory_allocated() / (1024 * 1024),
            torch.cuda.max_memory_reserved() / (1024 * 1024),
        )
    if device == "mps" and torch.backends.mps.is_available():
        # `driver_allocated_memory` is process-wide and includes cached blocks,
        # so it is reported beside the allocation figure rather than instead of
        # it — the two answer different questions.
        return (
            torch.mps.current_allocated_memory() / (1024 * 1024),
            torch.mps.driver_allocated_memory() / (1024 * 1024),
        )
    return (None, None)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument(
        "--device",
        required=True,
        help="Device to measure on, e.g. cpu, mps, cuda:0. Required: a "
        "latency without its device cannot be compared with anything.",
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=WARMUP_IMAGES)
    parser.add_argument("--measured", type=int, default=MEASURED_IMAGES)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Benchmark one run on one device and persist the result."""
    args = build_parser().parse_args(argv)

    if args.device == "cpu":
        # Some stacks probe accelerators at import time; pin before torch loads.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    (
        config, evaluation, storage, build_detector, detect_family,
        file_sha256, load_dataset_config, get_logger,
    ) = _project()
    logger = get_logger(__name__)

    db_path = args.db or config.DB_PATH

    # The operating point, not the evaluation sweep: a sweep threshold makes the
    # model emit far more boxes than a deployment ever would, and
    # post-processing that surplus is work no user pays for. Latency should
    # describe the configuration people actually run.
    benchmark_confidence = config.CONFIDENCE_THRESHOLD

    with storage.connect(db_path) as connection:
        run = next(
            (r for r in storage.list_runs(connection) if r.id == args.run_id), None
        )
        if run is None:
            raise SystemExit(f"No run {args.run_id} in {db_path}")

        weights = Path(run.model_path)
        if not weights.is_file():
            raise SystemExit(
                f"Run {run.id} names a checkpoint that is not on this machine:\n"
                f"  {weights}"
            )
        actual = file_sha256(weights)
        if actual != run.model_sha256:
            raise SystemExit(
                f"Checkpoint SHA-256 does not match run {run.id}.\n"
                f"  recorded: {run.model_sha256}\n"
                f"  on disk : {actual}\n"
                "Refusing to attribute a benchmark to weights that have changed."
            )

        dataset = load_dataset_config(Path(run.dataset_yaml))
        images_dir = dataset.splits.get(run.split)
        if images_dir is None:
            raise SystemExit(f"Split '{run.split}' not present in {run.dataset_yaml}")

        images = evaluation.list_images(images_dir)[: args.warmup + args.measured]
        if len(images) < args.warmup + 1:
            raise SystemExit(
                f"Need more than {args.warmup} images to benchmark; "
                f"found {len(images)}."
            )

        import torch

        rss_before = peak_rss_mb()
        load_started = time.perf_counter()

        family = detect_family(weights)
        detector = build_detector(
            family,
            model_path=str(weights),
            confidence=benchmark_confidence,
            image_size=run.image_size,
            device=args.device,
        )
        detector.load()

        load_seconds = time.perf_counter() - load_started
        cold_start_seconds = time.perf_counter() - PROCESS_STARTED
        rss_loaded = peak_rss_mb()

        # Warm-up is excluded: the first calls pay for lazy kernel compilation
        # and weight paging, which is a cost per process, not per image.
        for path in images[: args.warmup]:
            detector.predict_image(path, save_annotated=False)

        timings: list[float] = []
        for path in images[args.warmup :]:
            started = time.perf_counter()
            detector.predict_image(path, save_annotated=False)
            timings.append((time.perf_counter() - started) * 1000.0)

        if not timings:
            raise SystemExit("No images were measured.")

        timings.sort()
        mean_ms = statistics.mean(timings)
        allocated_mb, driver_mb = gpu_memory_mb(args.device)

        measurements = {
            "torch_threads": torch.get_num_threads(),
            "cold_start_s": round(cold_start_seconds, 3),
            "weights_load_s": round(load_seconds, 3),
            "latency_mean_ms": round(mean_ms, 2),
            "latency_median_ms": round(statistics.median(timings), 2),
            "latency_p90_ms": round(timings[int(len(timings) * 0.9)], 2),
            "latency_min_ms": round(timings[0], 2),
            "fps": round(1000.0 / mean_ms, 3),
            "rss_baseline_mb": round(rss_before, 1),
            "rss_after_load_mb": round(rss_loaded, 1),
            "rss_peak_mb": round(peak_rss_mb(), 1),
            "gpu_allocated_mb": (
                round(allocated_mb, 1) if allocated_mb is not None else None
            ),
            "gpu_driver_mb": round(driver_mb, 1) if driver_mb is not None else None,
            "checkpoint_bytes": weights.stat().st_size,
            "host_platform": f"{platform.system()} {platform.machine()}",
        }

        storage.save_benchmark(
            connection,
            run_id=run.id,
            device=args.device,
            image_size=run.image_size,
            warmup_images=args.warmup,
            measured_images=len(timings),
            measurements=measurements,
        )

    logger.info(
        "Run %d on %s: %.1f ms/image (%.2f FPS), peak RSS %.0f MB, %d image(s)",
        args.run_id,
        args.device,
        measurements["latency_mean_ms"],
        measurements["fps"],
        measurements["rss_peak_mb"],
        len(timings),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
