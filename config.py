"""Central configuration for Model Doctor.

Every path, device choice, and tunable constant lives here so that no other
module needs to hardcode them. Modules import ``config`` and read attributes;
they never build paths from string literals.

Design rules enforced by this module:

* **Nothing is hardcoded at the call site.** Class names, class counts, and
  dataset layout are *discovered* at runtime (from the model file and from
  ``data.yaml``), never written down here.
* **Everything is overridable.** Each path can be redirected with an
  environment variable, so the same code runs on a laptop, a CI runner, or a
  container without edits.
* **Importing this module never fails.** It performs no I/O beyond resolving
  paths and never asserts that a file exists. Verification is a separate,
  explicit step (see :mod:`utils.resources`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is derived from this file's location rather than from the
# current working directory. This makes the project runnable from anywhere:
# `python app/inference.py`, `python -m app.inference`, pytest, or Streamlit
# all resolve to the same root.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent


def _path_from_env(env_var: str, default: Path) -> Path:
    """Return the path in ``env_var`` if set, otherwise ``default``.

    Relative values in the environment are resolved against
    :data:`PROJECT_ROOT`, not the current working directory, so behaviour does
    not change depending on where the process was launched from.

    Args:
        env_var: Name of the environment variable to consult.
        default: Fallback path used when the variable is unset or blank.

    Returns:
        An absolute, resolved :class:`~pathlib.Path`.
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


MODELS_DIR: Final[Path] = _path_from_env("MD_MODELS_DIR", PROJECT_ROOT / "models")
DATASETS_DIR: Final[Path] = _path_from_env("MD_DATASETS_DIR", PROJECT_ROOT / "datasets")
RESULTS_DIR: Final[Path] = _path_from_env("MD_RESULTS_DIR", PROJECT_ROOT / "results")
PREDICTIONS_DIR: Final[Path] = _path_from_env(
    "MD_PREDICTIONS_DIR", RESULTS_DIR / "predictions"
)

# Where validation artefacts (confusion matrix, PR curves) are written.
VALIDATION_DIR: Final[Path] = _path_from_env(
    "MD_VALIDATION_DIR", RESULTS_DIR / "validation"
)

# The diagnosis database. Separate from RESULTS_DIR because results are
# disposable render output, whereas this accumulates across runs and is the
# interface another developer builds against.
DB_DIR: Final[Path] = _path_from_env("MD_DB_DIR", PROJECT_ROOT / "db")
DB_PATH: Final[Path] = _path_from_env("MD_DB_PATH", DB_DIR / "model_doctor.db")

# ---------------------------------------------------------------------------
# Model + dataset locations
# ---------------------------------------------------------------------------
# MODEL_PATH is a *preferred* location, not a promise. When it is absent,
# utils.resources.discover_model() falls back to scanning MODELS_DIR for any
# .pt file, so the project works whether the user drops in `best.pt` or
# `yolov8n_run7.pt`. Nothing downstream assumes a particular filename.
MODEL_FILENAME: Final[str] = os.getenv("MD_MODEL_FILENAME", "best.pt")
MODEL_PATH: Final[Path] = _path_from_env("MD_MODEL_PATH", MODELS_DIR / MODEL_FILENAME)

# The dataset descriptor YOLO consumes. Its `names:` field is the authoritative
# source of class names for ground-truth labels — we never retype them.
DATA_YAML_PATH: Final[Path] = _path_from_env(
    "MD_DATA_YAML", DATASETS_DIR / "data.yaml"
)

# Image file extensions we treat as inference inputs. Kept here (not inline in
# a loop) so that adding a format is a one-line config change.
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

# ---------------------------------------------------------------------------
# Inference parameters
# ---------------------------------------------------------------------------
# CONFIDENCE_THRESHOLD is deliberately low-ish. Model Doctor's job is failure
# *analysis*: a detection the model made at 0.30 and got wrong is exactly the
# signal we want to study. A high threshold would hide the very errors we are
# built to surface. Downstream analysis can always filter upward; it cannot
# recover detections that were discarded here.
CONFIDENCE_THRESHOLD: Final[float] = float(os.getenv("MD_CONF", "0.25"))

# IoU threshold used by Non-Maximum Suppression to merge duplicate boxes.
NMS_IOU_THRESHOLD: Final[float] = float(os.getenv("MD_NMS_IOU", "0.45"))

# IoU at or above which a prediction counts as landing on a ground-truth box.
# 0.5 is the community default (COCO mAP50).
MATCH_IOU_THRESHOLD: Final[float] = float(os.getenv("MD_MATCH_IOU", "0.50"))

# IoU floor for the diagnosis engine's second matching pass. A prediction that
# overlaps a ground-truth box by at least this much, but less than
# MATCH_IOU_THRESHOLD, is reported as poor localisation rather than as a false
# positive plus a false negative.
#
# This is a deliberate divergence from how mAP counts errors, and the reason
# the project exists: "found it, outlined it badly" is an explanation, whereas
# "one spurious detection and one miss" describes a single object as two
# unrelated failures. See DECISIONS D-017.
#
# 0.10 is low on purpose. Set it too high and genuinely wild predictions get
# excused as near-misses; too low and unrelated objects pair up. Overlap this
# small still means the model looked in roughly the right place.
LOCALIZATION_IOU_FLOOR: Final[float] = float(os.getenv("MD_LOC_IOU", "0.10"))

# Longest-side size images are letterboxed to before entering the network.
# Must match what the model was trained at, or accuracy silently degrades.
IMAGE_SIZE: Final[int] = int(os.getenv("MD_IMGSZ", "640"))

# Upper bound on detections returned per image. Guards against a pathological
# image producing tens of thousands of boxes and exhausting memory.
MAX_DETECTIONS: Final[int] = int(os.getenv("MD_MAX_DET", "300"))

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
# CLIP encoder used to turn failed regions into comparable vectors. ViT-B-32 is
# the smallest widely-used variant: fast on CPU, ~600MB of weights, and its
# 512-dimensional output is well within what clustering handles without
# dimensionality reduction. Larger variants give marginally better separation
# at several times the download and compute.
CLIP_MODEL: Final[str] = os.getenv("MD_CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED: Final[str] = os.getenv("MD_CLIP_PRETRAINED", "laion2b_s34b_b79k")

# ---------------------------------------------------------------------------
# Root-cause analysis
# ---------------------------------------------------------------------------
# Thresholds for the factors the engine attributes failures to. Every one is a
# judgement about "how bad is bad", so all are configurable rather than baked
# into the detectors — a dataset of night-time photographs needs a different
# darkness threshold than one shot indoors.

# Variance of the Laplacian below which a region is considered blurred. The
# standard sharpness measure: a blurred region has little high-frequency
# content, so its second derivative varies little.
BLUR_VARIANCE_THRESHOLD: Final[float] = float(os.getenv("MD_BLUR_VAR", "100.0"))

# Mean luminance (0-255) below which a region is considered underexposed.
LOW_LIGHT_THRESHOLD: Final[float] = float(os.getenv("MD_LOW_LIGHT", "60.0"))

# Fallback fraction of the image below which an object counts as small. 0.12%
# matches the COCO convention for "small". It is only used when the run's own
# size distribution is unavailable — a fixed cut-off does not transfer between
# datasets. On a dataset photographed close up it fired on 2 of 278 findings
# while "smaller than most" was a strong predictor of failure (D-032).
SMALL_OBJECT_AREA_FRACTION: Final[float] = float(os.getenv("MD_SMALL_AREA", "0.0012"))

# Percentile of the run's own object areas below which an object counts as
# small. Chosen from measurement, not convention: lift rises monotonically as
# the threshold tightens — 1.3x at the 75th percentile, 1.7x at the 50th,
# 2.2-2.7x at the 25th — and the 25th keeps enough findings to be worth
# reporting. Replicated across two splits (D-032).
SMALL_OBJECT_PERCENTILE: Final[float] = float(os.getenv("MD_SMALL_PCT", "0.25"))

# Percentile of the run's own aspect ratios above which an object counts as a
# thin structure. Thin objects are what this project's model handles worst —
# `door_frame` mask mAP50-95 is 0.246 against `door` at 0.599 — and nothing in
# the factor set named it. Lift is 1.7-2.2x at the 75th percentile and vanishes
# below the median, so the effect lives at the extreme (D-032).
THIN_STRUCTURE_PERCENTILE: Final[float] = float(os.getenv("MD_THIN_PCT", "0.75"))

# Fallback aspect ratio above which an object counts as thin, used when the
# run's own distribution is unavailable.
THIN_STRUCTURE_RATIO: Final[float] = float(os.getenv("MD_THIN_RATIO", "4.0"))

# Distance from the frame, as a fraction of image size, within which a box is
# treated as truncated by the edge.
EDGE_TRUNCATION_MARGIN: Final[float] = float(os.getenv("MD_EDGE_MARGIN", "0.01"))

# IoU with a neighbouring annotation above which a region is considered
# crowded. Used as an occlusion proxy: true occlusion is not observable from
# annotations alone, but heavy overlap is the condition under which it happens.
CROWDING_IOU_THRESHOLD: Final[float] = float(os.getenv("MD_CROWD_IOU", "0.25"))

# A class holding less than this share of instances is flagged as
# under-represented relative to an even split across classes.
CLASS_IMBALANCE_RATIO: Final[float] = float(os.getenv("MD_IMBALANCE", "0.5"))

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
DEVICE_OVERRIDE: Final[str] = os.getenv("MD_DEVICE", "").strip()


def resolve_device() -> str:
    """Pick the best available compute device for this machine.

    Preference order is ``cuda`` (NVIDIA GPU) → ``mps`` (Apple Silicon GPU) →
    ``cpu``. Set the ``MD_DEVICE`` environment variable to force a specific
    device, which is useful when debugging a suspected backend bug.

    This function imports :mod:`torch` lazily and catches every exception. That
    is intentional: ``config`` must stay importable in an environment where the
    ML stack is not installed yet, so tooling and tests can still read paths.

    Returns:
        A device string suitable for handing to Ultralytics, e.g. ``"mps"``.
    """
    if DEVICE_OVERRIDE:
        return DEVICE_OVERRIDE

    try:
        import torch
    except ImportError:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
        # `mps.is_built()` guards against a PyTorch wheel compiled without MPS.
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:  # pragma: no cover - defensive, backend probes can throw
        return "cpu"

    return "cpu"


DEVICE: Final[str] = resolve_device()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: Final[str] = os.getenv("MD_LOG_LEVEL", "INFO").upper()
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%H:%M:%S"


def ensure_output_dirs() -> None:
    """Create the directories Model Doctor writes into.

    Only *output* directories are created. ``models/`` and ``datasets/`` are
    deliberately excluded: those hold resources the user supplies, and silently
    conjuring an empty ``models/`` directory would turn a clear "you have not
    added a model yet" error into a confusing "directory is empty" one.
    """
    for directory in (RESULTS_DIR, PREDICTIONS_DIR, VALIDATION_DIR):
        directory.mkdir(parents=True, exist_ok=True)
