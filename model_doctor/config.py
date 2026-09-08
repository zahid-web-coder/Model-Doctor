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
from typing import TYPE_CHECKING, Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is derived from this file's location rather than from the
# current working directory, so the project resolves to the same root whether
# it is run by pytest, by `python -m`, or by Streamlit.
#
# It is the directory *containing* the package, not the package itself, so the
# defaults below keep naming the same `db/`, `results/`, `models/` and
# `datasets/` directories they named before this code moved under
# `model_doctor/`. An installed copy puts those defaults inside site-packages,
# which is no place to write: a deployment sets the `MD_*` variables instead,
# and every path below is overridable for exactly that reason.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


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
# Detector selection
# ---------------------------------------------------------------------------
# Which detector family loads the weights. Two are supported, and the default
# is deliberate rather than arbitrary.
#
# **YOLO stays the default because it is the only one that runs anywhere.**
# Measured on this project's columns test set, CPU, one process each:
#
#     YOLO26 @672    47.8 ms   20.9 FPS    390 MB RSS
#     RF-DETR @480  579.4 ms    1.7 FPS   2833 MB RSS
#
# RF-DETR is more accurate on that set — better recall, better masks, and a
# low-light advantage that survived a resolution control — but it is twelve
# times slower and holds seven times the memory on CPU, and its memory floor
# does not fall when resolution does. It is an accuracy option for GPU and
# batch work, not a drop-in replacement.
DETECTOR_FAMILY: Final[str] = os.getenv("MD_DETECTOR", "yolo").strip().lower()

# Input resolution for RF-DETR, which accepts any multiple of 12 and
# interpolates its positional encodings to match.
#
# 480 is the efficiency knee measured on the columns set: it captures 68% of
# the total accuracy available across the 312-768 sweep for 18% of the extra
# latency. 672 is the accuracy and F1 optimum if throughput does not matter;
# past that, accuracy creeps up while precision falls.
#
# Separate from IMAGE_SIZE because the two models want different values and
# sharing one would silently mis-size whichever was not being tuned.
RFDETR_IMAGE_SIZE: Final[int] = int(os.getenv("MD_RFDETR_IMGSZ", "480"))

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

# Factors permitted to form a *discriminating* failure group. Everything else
# still gets attributed and stored; it simply does not partition failures.
#
# This is a decision recorded from evidence, not a value recomputed per run —
# recomputing it is what makes groups stop being comparable between runs
# (D-033). Membership rule: a factor joins only if it shows lift > 1 and
# p < 0.05 on **at least two splits**.
#
# As measured on the 136-image test and 258-image validation splits:
#
#   small_object     2.77x p<0.001  |  3.45x p<0.001   -> qualifies
#   thin_structure   2.14x p<0.001  |  2.32x p<0.001   -> qualifies
#   crowding         1.28x p=0.041  |  1.12x p=0.310   -> failed to replicate
#   blur             0.81x p=0.252  |  1.17x p=0.305   -> no signal
#   edge_truncation  0.93x p=0.340  |  1.04x p=0.542   -> no signal, pooled 1.00x
#   low_light        1.43x p=0.504  |  1.29x p=0.462   -> no signal
#
# Re-measure before changing this. Adding classes or a new dataset changes the
# failure population, so today's members are not permanently qualified.
DISCRIMINATING_FACTORS: Final[frozenset[str]] = frozenset(
    factor.strip()
    for factor in os.getenv(
        "MD_DISCRIMINATING_FACTORS", "small_object,thin_structure"
    ).split(",")
    if factor.strip()
)

# Smallest failure group a recommendation may be based on. Below this, an
# outcome split of three against two is not a pattern — the group's dominant
# outcome would change with a single reclassified finding. Groups under the
# threshold are still reported, with status `insufficient_evidence`, so they
# stay visible rather than silently vanishing (D-035).
MIN_RECOMMENDATION_GROUP_SIZE: Final[int] = int(os.getenv("MD_MIN_REC_GROUP", "10"))

# Share of a group's failures that must carry one outcome before a
# recall- or precision-shaped action is proposed. At 0.60 a group has to lean
# clearly one way; a near-even split describes two problems, not one.
RECOMMENDATION_OUTCOME_SHARE: Final[float] = float(os.getenv("MD_REC_SHARE", "0.60"))

# ---------------------------------------------------------------------------
# Image-level diagnosis
# ---------------------------------------------------------------------------
# Share of a ground-truth object's mask a prediction must cover before it
# counts as having *found* that object. Used to detect one prediction spanning
# several objects (merged) and several predictions on one object (split).
#
# **Chosen from the data, and deliberately conservative.** Across 430 objects
# in two runs the distribution of best-coverage is strongly bimodal: 72% above
# 0.90, 15% below 0.05, and only 2% between 0.05 and 0.50. 0.50 sits at the top
# of that valley, so a merge is reported only when a prediction really does
# cover most of a second object.
#
# Merge and split counts *are* sensitive to this value — on one run they move
# from 8 to 2 as it goes from 0.30 to 0.90 — so it is stored with every
# diagnosis and the raw pairwise coverages are kept, letting a consumer
# re-threshold without re-running anything.
IMAGE_COVER_HIT: Final[float] = float(os.getenv("MD_IMAGE_COVER_HIT", "0.50"))

# Share below which a prediction is treated as not touching an object at all.
#
# Well supported: sweeping 0.05 to 0.30 changes the count of untouched objects
# by zero on one run and by two on another, because almost nothing lives in
# that band. Any value in it gives the same answer.
IMAGE_COVER_MISS: Final[float] = float(os.getenv("MD_IMAGE_COVER_MISS", "0.25"))

# Pairs weaker than this are not persisted. Storing every (object, prediction)
# combination would be quadratic in a crowded image for rows that say only
# "these do not overlap", which the absence of a row already says.
IMAGE_COVERAGE_FLOOR: Final[float] = float(os.getenv("MD_IMAGE_COVER_FLOOR", "0.01"))

# ---------------------------------------------------------------------------
# API layer
# ---------------------------------------------------------------------------
# Most runs one MCP `get_analysis` call may compare. A comparison is only
# readable at a handful of runs, and the response grows linearly with each;
# a caller wanting more is better served by two calls than by one that cannot
# be read. Not a security limit — the server is read-only — but a size one.
MCP_MAX_RUNS: Final[int] = int(os.getenv("MD_MCP_MAX_RUNS", "10"))

# Browser origins permitted to call the read-only API. A Next.js development
# server runs on port 3000 while the API runs on 8000, so without this every
# request fails CORS. Never widened to "*": the API serves local file contents,
# and any page in any tab could then read them.
CORS_ORIGINS: Final[tuple[str, ...]] = tuple(
    origin.strip()
    for origin in os.getenv(
        "MD_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
)

# ---------------------------------------------------------------------------
# Self-service workspace
# ---------------------------------------------------------------------------
# Where files uploaded through the web UI are stored.
#
# **Deliberately outside the source tree.** An upload is untrusted input, and
# the default must not put it anywhere a mistake could reach `app/`, `models/`,
# `datasets/` or the database. Under the user's home by default; redirect it at
# a mounted volume in a container.

#: Where uploads land when nothing overrides it.
#:
#: Named separately from :data:`WORKSPACE_DIR` because both have to be servable.
#: A database accumulates runs across sessions, and a session that set
#: ``MD_WORKSPACE_DIR`` writes its images somewhere a later session started
#: without it cannot read. See :data:`API_FILE_ROOTS`.
DEFAULT_WORKSPACE_DIR: Final[Path] = Path.home() / ".model-doctor" / "workspace"

WORKSPACE_DIR: Final[Path] = _path_from_env(
    "MD_WORKSPACE_DIR", DEFAULT_WORKSPACE_DIR
)

# Upload ceilings. Generous enough for a real detection dataset, bounded
# enough that one request cannot fill the disk.
MAX_MODEL_BYTES: Final[int] = int(os.getenv("MD_MAX_MODEL_MB", "2048")) * 1_000_000
MAX_DATASET_BYTES: Final[int] = int(os.getenv("MD_MAX_DATASET_MB", "8192")) * 1_000_000

# How many upload workspaces to keep. Each holds a full copy of the dataset it
# was run with, so without a bound the disk fills at one dataset per analysis.
# The most recent are kept so a just-finished job's inputs are still there to
# look at; everything older is removed once a new analysis is queued.
WORKSPACE_KEEP: Final[int] = int(os.getenv("MD_WORKSPACE_KEEP", "5"))

# How much of a failed stage's output to keep for the operator. The tail, not
# the head: a traceback ends with the reason.
LOG_TAIL_CHARS: Final[int] = int(os.getenv("MD_LOG_TAIL", "4000"))

# Seconds a single pipeline stage may run before the worker abandons it. A
# stuck stage must not hold the queue forever.
STAGE_TIMEOUT_S: Final[int] = int(os.getenv("MD_STAGE_TIMEOUT", "10800"))

# Directories the API may read image and heatmap files from. A stored path is
# resolved and checked against these before it is opened, so a database
# containing a path to somewhere else cannot make the API serve it (D-037).
#
# The API must be started with the same `MD_DATASETS_DIR` the runs were
# diagnosed with, or their images will resolve outside every allowed root and
# be refused. Extra roots can be added here when images live in several places.
# `dict.fromkeys` deduplicates while keeping order: with `MD_WORKSPACE_DIR`
# unset the two workspace entries coincide, and a list that repeats itself is
# misleading to read in a 403's error message.
API_FILE_ROOTS: Final[tuple[Path, ...]] = tuple(
    dict.fromkeys(
        Path(root).expanduser().resolve()
        for root in (
            *(
                part.strip()
                for part in os.getenv("MD_API_FILE_ROOTS", "").split(",")
                if part.strip()
            ),
            str(DATASETS_DIR),
            str(RESULTS_DIR),
            # Runs started from the browser read their images out of the
            # upload workspace, not out of `datasets/`. Without this the API
            # refuses every image in every self-service run with a 403, and the
            # dashboard renders a complete diagnosis in which no picture loads.
            str(WORKSPACE_DIR),
            # **And the default workspace, even when overridden.** One database
            # accumulates runs from many sessions, and `MD_WORKSPACE_DIR` is set
            # per session. A run analysed under the default is unreadable the
            # moment a later session sets an override, and vice versa — every
            # image in that run returns 403 while the diagnosis around it renders
            # perfectly, reading as a broken dashboard not a path mismatch.
            #
            # Both are this installation's own upload directory, so allowing
            # both widens nothing meaningful. Anywhere else still requires
            # `MD_API_FILE_ROOTS` to say so explicitly.
            str(DEFAULT_WORKSPACE_DIR),
        )
    )
)


def _parse_path_remap(raw: str) -> tuple[tuple[str, str], ...]:
    """Read ``old=new`` rewrite rules from a semicolon-separated string.

    Malformed entries are skipped rather than raising. This is start-up
    configuration for a read-only service, and refusing to boot because one
    rule of several has a typo would deny access to the rules that are fine.
    The skip is logged by the caller's ``describe`` output instead.
    """
    rules: list[tuple[str, str]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        old, _, new = part.partition("=")
        old, new = old.strip(), new.strip()
        if old and new:
            rules.append((old.rstrip("/"), new.rstrip("/")))
    return tuple(rules)


# Path prefixes rewritten before a stored file path is opened.
#
# **Why this exists.** `images.path` and `heatmaps.path` are absolute and were
# written on the machine that ran the diagnosis. Copy a database to a second
# machine — a colleague's laptop, a CI runner, this machine after the source
# directory moved — and every path in it points somewhere that does not exist,
# so the tables render and every image 404s. Nothing in the schema is wrong;
# the paths are simply someone else's.
#
# Rewriting at serve time rather than migrating the database keeps the record
# of where a run was actually diagnosed intact. The database stays a true
# statement about the machine that produced it, and relocation is the reader's
# configuration rather than an edit to history.
#
# Format: ``MD_PATH_REMAP="/old/prefix=/new/prefix;/another=/somewhere"``.
# The rewritten path is still checked against `API_FILE_ROOTS`, so this widens
# nothing — a remap pointing outside the permitted roots is refused exactly as
# an unremapped one would be.
PATH_REMAP: Final[tuple[tuple[str, str], ...]] = _parse_path_remap(
    os.getenv("MD_PATH_REMAP", "")
)


def remap_path(stored: str) -> str:
    """Apply the first matching remap rule to a stored path.

    First match wins, so a more specific prefix listed before a general one
    takes precedence. Returns the input unchanged when no rule matches, which
    is the common case and the one that must stay cheap.

    Both the stored path and the rule prefixes are normalised to forward
    slashes before comparison so that databases created on one OS can be
    remapped on another (e.g. Windows backslashes vs POSIX forward slashes).
    """
    stored_posix = stored.replace("\\", "/")
    for old, new in PATH_REMAP:
        old_posix = old.replace("\\", "/")
        new_posix = new.replace("\\", "/")
        if stored_posix == old_posix:
            return new_posix
        if stored_posix.startswith(old_posix + "/"):
            return new_posix + stored_posix[len(old_posix) :]
    return stored

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


# Resolved on first access rather than at import. `resolve_device()` imports
# torch, and every module in this project imports `config` — so an eager call
# here pulled ~500 MB of ML stack into any process that merely wanted a path.
# The read-only API is exactly that process: it never touches a model, but
# `import model_doctor.app.api` loaded torch before serving a request, taking a
# deployable
# container from roughly 100 MB to 2.5 GB.
#
# `config.DEVICE` still reads identically for every caller. Only the moment of
# computation moved, which is what the docstring above already intended when it
# said config must stay importable without the ML stack (D-038).
_DEVICE: str | None = None

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    DEVICE: str


def __getattr__(name: str) -> str:
    """Compute module attributes that are deliberately deferred to first use.

    ``DEVICE`` is the only one. It cannot be annotated ``Final`` because it is
    not a module-level assignment; the value is nevertheless resolved once and
    cached, so it is constant for the life of the process exactly as before.

    Args:
        name: Attribute being accessed.

    Returns:
        The resolved device string.

    Raises:
        AttributeError: For any other name, matching normal module behaviour.
    """
    if name == "DEVICE":
        global _DEVICE
        if _DEVICE is None:
            _DEVICE = resolve_device()
        return _DEVICE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
