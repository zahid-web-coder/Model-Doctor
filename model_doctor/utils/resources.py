"""Resource discovery and verification.

Model Doctor depends on two things it does not own: a trained model file and a
dataset. Either may be absent while the project is under development. This
module is the single place that answers "is it here?", so that:

* no other module has to guess or re-implement the check;
* a missing resource produces an actionable message, never a stack trace from
  deep inside a third-party library;
* the code path for "resource present" and "resource absent" is decided once,
  explicitly, at a known boundary.

Nothing in here fabricates a model or a dataset. If a resource is missing, the
functions say so and the caller decides what to do.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from model_doctor import config
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResourceStatus:
    """The outcome of checking one resource.

    Attributes:
        name: Human-readable label, e.g. ``"Trained model"``.
        path: Where the resource was expected. ``None`` for non-path checks.
        available: Whether the resource is usable.
        detail: One line explaining the result, including what to do next when
            the resource is missing.
        required: If ``True``, inference cannot run without it.
    """

    name: str
    path: Path | None
    available: bool
    detail: str
    required: bool = True

    @property
    def symbol(self) -> str:
        """Return a compact status glyph for console output."""
        if self.available:
            return "OK"
        return "MISSING" if self.required else "OPTIONAL"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_model(explicit_path: Path | None = None) -> Path | None:
    """Locate a YOLO weights file without hardcoding its name.

    Resolution order:

    1. ``explicit_path`` if the caller supplied one.
    2. :data:`config.MODEL_PATH` (by default ``models/best.pt``).
    3. Any ``*.pt`` file inside :data:`config.MODELS_DIR`, newest first.

    Step 3 is what makes the project tolerant of real-world naming. A training
    run may emit ``best.pt``, ``last.pt``, or ``yolov8s_run12.pt``; the user
    should not have to rename files to satisfy us.

    Args:
        explicit_path: An override supplied by the caller, e.g. from a CLI flag.

    Returns:
        Path to a weights file, or ``None`` if nothing suitable was found.
    """
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser()
        return candidate if candidate.is_file() else None

    if config.MODEL_PATH.is_file():
        return config.MODEL_PATH

    if not config.MODELS_DIR.is_dir():
        return None

    # Sort by modification time so the most recently trained model wins.
    candidates = sorted(
        (p for p in config.MODELS_DIR.glob("*.pt") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    chosen = candidates[0]
    if chosen != config.MODEL_PATH:
        # DEBUG, not INFO: discovery runs on every resource check, and the
        # chosen weights file is already reported once in the status report.
        logger.debug(
            "Using discovered model %s (%s was not present)",
            chosen.name,
            config.MODEL_PATH.name,
        )
    return chosen


def find_images(directory: Path, recursive: bool = True) -> list[Path]:
    """Collect image files under ``directory``.

    Extensions come from :data:`config.IMAGE_EXTENSIONS` rather than a literal
    list, and matching is case-insensitive so ``.JPG`` is not silently skipped.

    Args:
        directory: Root to search. A missing directory yields ``[]`` rather
            than raising — callers report the absence via the status report.
        recursive: Whether to descend into subdirectories.

    Returns:
        Image paths sorted by name, for deterministic, reproducible ordering.
    """
    if not directory.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    ]
    return sorted(images)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def check_model(explicit_path: Path | None = None) -> ResourceStatus:
    """Report whether a usable model file is present.

    Only existence and non-emptiness are checked here; actually loading the
    weights is expensive and is deferred to the detector. A zero-byte file is
    treated as missing because it is the classic signature of an interrupted
    download or a Git LFS pointer that was never fetched.
    """
    found = discover_model(explicit_path)

    if found is None:
        return ResourceStatus(
            name="Trained model",
            path=config.MODEL_PATH,
            available=False,
            detail=(
                f"No .pt weights found. Place a trained YOLO model at "
                f"{config.MODEL_PATH}, or any *.pt file inside "
                f"{config.MODELS_DIR}/"
            ),
        )

    size_bytes = found.stat().st_size
    if size_bytes == 0:
        return ResourceStatus(
            name="Trained model",
            path=found,
            available=False,
            detail=f"{found.name} is 0 bytes — the file is truncated or a stub.",
        )

    return ResourceStatus(
        name="Trained model",
        path=found,
        available=True,
        detail=f"{found.name} ({size_bytes / 1_048_576:.1f} MB)",
    )


def check_data_yaml() -> ResourceStatus:
    """Report whether the dataset descriptor is present and parseable."""
    path = config.DATA_YAML_PATH

    if not path.is_file():
        return ResourceStatus(
            name="Dataset config (data.yaml)",
            path=path,
            available=False,
            detail=f"Not found. Expected the YOLO dataset descriptor at {path}",
        )

    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except Exception as exc:
        return ResourceStatus(
            name="Dataset config (data.yaml)",
            path=path,
            available=False,
            detail=f"Present but could not be parsed: {exc}",
        )

    if not isinstance(parsed, dict) or "names" not in parsed:
        return ResourceStatus(
            name="Dataset config (data.yaml)",
            path=path,
            available=False,
            detail="Present but has no 'names:' key, so class names are unknown.",
        )

    class_count = len(parsed["names"])
    return ResourceStatus(
        name="Dataset config (data.yaml)",
        path=path,
        available=True,
        detail=f"{class_count} classes declared",
    )


def check_test_images() -> ResourceStatus:
    """Report whether any images are available to run inference on.

    The dataset directory is searched as a whole rather than assuming a
    ``test/images`` layout, because splits differ between exports (Roboflow,
    CVAT, and hand-built datasets all name things differently).
    """
    images = find_images(config.DATASETS_DIR)

    if not images:
        return ResourceStatus(
            name="Dataset images",
            path=config.DATASETS_DIR,
            available=False,
            detail=f"No images found under {config.DATASETS_DIR}/",
        )

    return ResourceStatus(
        name="Dataset images",
        path=config.DATASETS_DIR,
        available=True,
        detail=f"{len(images)} images found",
    )


def check_dependencies() -> ResourceStatus:
    """Report whether the ML stack can be imported.

    Import failure here is a genuinely different problem from a missing model —
    it means the environment is not set up — so it gets its own status line.
    """
    missing: list[str] = []
    for module_name in ("torch", "ultralytics", "cv2", "yaml"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        return ResourceStatus(
            name="Python dependencies",
            path=None,
            available=False,
            detail=(
                f"Not importable: {', '.join(missing)}. "
                f"Run: pip install -r requirements.txt"
            ),
        )

    import torch

    return ResourceStatus(
        name="Python dependencies",
        path=None,
        available=True,
        detail=f"torch {torch.__version__}, device '{config.DEVICE}'",
    )


def verify_all(explicit_model: Path | None = None) -> list[ResourceStatus]:
    """Run every resource check and return the results in report order.

    Never raises. The caller inspects the returned statuses and decides whether
    to proceed, which keeps "what is broken" separate from "what to do about
    it".
    """
    return [
        check_dependencies(),
        check_model(explicit_model),
        check_data_yaml(),
        check_test_images(),
    ]


def format_report(statuses: Sequence[ResourceStatus]) -> str:
    """Render statuses as an aligned, human-readable block of text.

    Args:
        statuses: Results from :func:`verify_all`.

    Returns:
        A multi-line string ready to hand to ``print`` or a logger.
    """
    width = max((len(s.name) for s in statuses), default=0)
    lines = ["", "Model Doctor — resource check", "-" * 64]
    for status in statuses:
        lines.append(f"  [{status.symbol:^7}] {status.name:<{width}}  {status.detail}")
    lines.append("-" * 64)

    blocking = [s for s in statuses if not s.available and s.required]
    if blocking:
        lines.append(
            f"  {len(blocking)} required resource(s) missing — "
            "inference cannot run until they are provided."
        )
    else:
        lines.append("  All required resources present.")
    lines.append("")
    return "\n".join(lines)
