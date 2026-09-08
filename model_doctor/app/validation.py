"""Checks an upload must pass before anything is run on it.

The point of this module is to **refuse clearly rather than fail obscurely**.
A dataset with no labels, a checkpoint from a detector family this project does
not implement, or a split that exists in the descriptor but not on disk are all
ordinary mistakes, and each should produce a sentence naming the problem — not
a traceback three stages later from inside a detector.

**On inspecting checkpoints.** A PyTorch checkpoint is a pickle, and
unpickling one executes code in it. Every inspection here uses
``weights_only=True``, which restricts loading to tensors and plain data, so
identifying a file cannot execute it.

That protects *validation*. It does not make running an arbitrary model safe:
inference has to construct the model, and for Ultralytics that means a full
unpickle. So this workflow assumes an operator who trusts the weights they
uploaded — the same assumption as running the CLI. It is stated here rather
than implied, because the difference matters if this is ever exposed beyond a
trusted user, and no amount of validation closes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_doctor.app.detectors import SUPPORTED_FAMILIES
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Substrings that identify a family from a checkpoint's parameter names. Drawn
# from the two adapters this project implements; a family with no entry here is
# simply not auto-detected, which is honest rather than a guess.
_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "rfdetr": ("transformer.decoder", "class_embed", "bbox_embed"),
    "yolo": ("model.model.", "model.24.", "model.22."),
}

# Byte sequences that identify a family from the *unopened* pickle stream.
# A checkpoint that serialises a live model object — which is what Ultralytics
# writes — names the classes it will reconstruct, and those names appear
# literally in the pickle. Searching for them reads the file as data and never
# runs it, which is the only way to identify such a checkpoint safely.
_PICKLE_MARKERS: tuple[tuple[str, bytes], ...] = (
    ("rfdetr", b"rfdetr"),
    ("rfdetr", b"LWDETR"),
    ("yolo", b"ultralytics"),
)

# Directories a YOLO split is expected to contain.
_IMAGES_DIR = "images"
_LABELS_DIR = "labels"


@dataclass
class Check:
    """One validated property, and what to tell the operator about it."""

    name: str
    ok: bool
    detail: str


@dataclass
class ValidationResult:
    """The outcome of validating one upload."""

    ok: bool
    checks: list[Check] = field(default_factory=list)
    #: Facts worth showing the operator, e.g. detected family, class names.
    facts: dict[str, Any] = field(default_factory=dict)

    def failures(self) -> list[Check]:
        """Only the checks that did not pass."""
        return [c for c in self.checks if not c.ok]

    def add(self, name: str, ok: bool, detail: str) -> None:
        """Record one check and update the overall verdict."""
        self.checks.append(Check(name=name, ok=ok, detail=detail))
        if not ok:
            self.ok = False


def detect_model_family(weights: Path) -> tuple[str | None, str]:
    """Identify which detector family a checkpoint belongs to.

    Two passes, both of which read the file as data rather than running it.

    First the pickle stream is **scanned as bytes** for the class names a
    checkpoint would reconstruct. This is what identifies an Ultralytics
    checkpoint, which cannot be read any other way without executing it:
    ``weights_only=True`` refuses it precisely because it serialises a live
    model object.

    Failing that, the file is loaded with ``weights_only=True`` and its
    parameter names inspected. That covers a plain state dict, which is what
    RF-DETR writes.

    Returns:
        ``(family, explanation)``. The family is ``None`` when the checkpoint
        matches no family this project implements — including when it is a
        perfectly good checkpoint for a detector that has no adapter here. That
        distinction is the operator's to act on and is not guessed at.
    """
    import zipfile

    # Pass one: read the pickle without unpickling it.
    if zipfile.is_zipfile(weights):
        try:
            with zipfile.ZipFile(weights) as bundle:
                for entry in bundle.namelist():
                    if not entry.endswith("data.pkl"):
                        continue
                    blob = bundle.read(entry)
                    for family, marker in _PICKLE_MARKERS:
                        if marker in blob:
                            return family, (
                                f"The checkpoint names {marker.decode()!r} "
                                f"classes, which is the {family} layout."
                            )
                    break
        except Exception as error:  # noqa: BLE001 - fall through to pass two
            logger.debug("Could not scan %s as an archive: %s", weights, error)

    # Pass two: a plain state dict can be read safely and inspected by key.
    try:
        import torch

        blob = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - any failure means "cannot tell"
        return None, (
            "The checkpoint could not be identified without executing it, so "
            "it is not one of the supported families. Supported: "
            f"{', '.join(SUPPORTED_FAMILIES)}. ({type(error).__name__})"
        )

    state = blob
    if isinstance(blob, dict):
        for key in ("model", "state_dict", "ema"):
            inner = blob.get(key)
            if isinstance(inner, dict):
                state = inner
                break

    if not isinstance(state, dict):
        return None, "The checkpoint holds no recognisable parameter dictionary."

    keys = " ".join(str(k) for k in state)
    for family, markers in _FAMILY_MARKERS.items():
        if any(marker in keys for marker in markers):
            return family, f"Parameter names match the {family} layout."

    return None, (
        "The parameter names match neither supported family. Supported: "
        f"{', '.join(SUPPORTED_FAMILIES)}."
    )


def validate_model(
    weights: Path, declared_family: str | None = None
) -> ValidationResult:
    """Check that a checkpoint is one this project can actually run.

    Args:
        weights: The uploaded checkpoint.
        declared_family: What the operator said it is. When given, a mismatch
            with what the file looks like is reported rather than silently
            preferring one — the operator may know something the heuristic
            does not, but they should be told they disagree.
    """
    result = ValidationResult(ok=True)

    result.add(
        "File is present",
        weights.is_file(),
        str(weights.name) if weights.is_file() else "The checkpoint is missing.",
    )
    if not weights.is_file():
        return result

    size_mb = weights.stat().st_size / 1e6
    result.add("File is non-empty", size_mb > 0, f"{size_mb:.1f} MB")

    detected, explanation = detect_model_family(weights)
    result.facts["detected_family"] = detected

    if declared_family:
        family = declared_family.strip().lower()
        result.add(
            "Family is supported",
            family in SUPPORTED_FAMILIES,
            f"{family!r}" if family in SUPPORTED_FAMILIES else (
                f"{family!r} has no adapter in this project. Supported: "
                f"{', '.join(SUPPORTED_FAMILIES)}."
            ),
        )
        if detected is None:
            # An unidentifiable file is not a match — it is an unknown, and
            # passing it here told the operator a garbage checkpoint had been
            # validated. The run would then be started and fail during
            # inference, having wasted their time to reach the same verdict
            # this check could have given immediately.
            result.add(
                "Declared family matches the file",
                False,
                explanation,
            )
        elif detected != family:
            result.add(
                "Declared family matches the file",
                False,
                f"You selected {family!r} but the checkpoint looks like "
                f"{detected!r}. Running it as {family!r} would either fail to "
                "load or produce meaningless predictions.",
            )
        else:
            result.add("Declared family matches the file", True, explanation)
        result.facts["family"] = family
    else:
        result.add(
            "Family could be identified",
            detected is not None,
            explanation,
        )
        result.facts["family"] = detected

    return result


def validate_dataset(
    root: Path, *, require_split: str | None = None
) -> ValidationResult:
    """Check that an unpacked dataset is one the pipeline can read.

    This validates the **YOLO layout** — a ``data.yaml`` naming classes and
    splits, with each split holding ``images/`` and ``labels/``. That is the
    only layout the diagnosis pipeline reads, so anything else is refused with
    a message saying so rather than half-processed.

    Args:
        root: Directory the upload was extracted into.
        require_split: When given, that split must exist and be non-empty.
    """
    result = ValidationResult(ok=True)

    descriptor = _find_descriptor(root)
    result.add(
        "data.yaml found",
        descriptor is not None,
        str(descriptor.relative_to(root)) if descriptor else (
            "No data.yaml anywhere in the upload. This project reads the YOLO "
            "dataset layout: a data.yaml naming the classes, beside split "
            "directories each containing images/ and labels/."
        ),
    )
    if descriptor is None:
        return result
    result.facts["data_yaml"] = str(descriptor)

    try:
        from model_doctor.utils.dataset import load_dataset_config

        dataset = load_dataset_config(descriptor)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim to the operator
        result.add("data.yaml is readable", False, str(error))
        return result

    result.add(
        "data.yaml is readable",
        True,
        f"{len(dataset.class_names)} class(es): "
        f"{', '.join(list(dataset.class_names.values())[:6])}",
    )
    result.facts["class_names"] = dataset.class_names
    result.facts["splits"] = sorted(dataset.splits)

    result.add(
        "At least one split exists on disk",
        bool(dataset.splits),
        ", ".join(sorted(dataset.splits)) if dataset.splits else (
            "The descriptor declares splits, but none of their directories "
            "exist in the upload."
        ),
    )
    if not dataset.splits:
        return result

    target = require_split or next(iter(sorted(dataset.splits)))
    images_dir = dataset.splits.get(target)
    result.add(
        f"Split {target!r} is present",
        images_dir is not None,
        str(images_dir) if images_dir else (
            f"{target!r} is not among the available splits: "
            f"{', '.join(sorted(dataset.splits))}."
        ),
    )
    if images_dir is None:
        return result

    images = _count_images(images_dir)
    result.add(
        "Split contains images",
        images > 0,
        f"{images} image(s)" if images else f"{images_dir} holds no readable images.",
    )
    result.facts["image_count"] = images

    labels_dir = images_dir.parent / _LABELS_DIR
    result.add(
        "Labels directory exists",
        labels_dir.is_dir(),
        str(labels_dir.name) if labels_dir.is_dir() else (
            f"No {_LABELS_DIR}/ beside {images_dir.name}/. Ground truth is "
            "required: without it there is nothing to diagnose failures "
            "against."
        ),
    )
    if not labels_dir.is_dir():
        return result

    parsed, bad = _inspect_labels(labels_dir, max(dataset.class_names, default=0))
    result.add(
        "Label files parse as YOLO annotations",
        not bad,
        f"{parsed} label file(s) read" if not bad else "; ".join(bad[:3]),
    )
    result.facts["label_files"] = parsed
    return result


def _find_descriptor(root: Path) -> Path | None:
    """Locate a data.yaml at, or just below, the upload root."""
    direct = root / "data.yaml"
    if direct.is_file():
        return direct
    # One level down covers the common "dataset zipped inside its own folder"
    # case without walking an arbitrarily deep tree from an untrusted archive.
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "data.yaml").is_file():
            return child / "data.yaml"
    return None


def _count_images(directory: Path) -> int:
    """Count files with an extension the pipeline treats as an image."""
    from model_doctor import config

    return sum(
        1
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def _inspect_labels(directory: Path, max_class_id: int) -> tuple[int, list[str]]:
    """Read a sample of label files, returning the count and any complaints.

    A sample rather than the whole split: this runs inside a request, and
    reading fifty files is enough to catch a wrong format, which is what this
    is for. A per-object error would be found by the diagnosis pass anyway.
    """
    problems: list[str] = []
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".txt")
    for path in files[:50]:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                problems.append(
                    f"{path.name}:{number} has {len(parts)} field(s); a YOLO "
                    "annotation needs at least 5 (class and a box)."
                )
                break
            try:
                class_id = int(parts[0])
                values = [float(v) for v in parts[1:]]
            except ValueError:
                problems.append(f"{path.name}:{number} is not numeric.")
                break
            if class_id < 0 or class_id > max_class_id:
                problems.append(
                    f"{path.name}:{number} uses class id {class_id}, which is "
                    f"outside the {max_class_id + 1} class(es) in data.yaml."
                )
                break
            if any(v < -0.01 or v > 1.01 for v in values):
                problems.append(
                    f"{path.name}:{number} has coordinates outside 0-1. YOLO "
                    "annotations are normalised to the image size."
                )
                break
    return len(files), problems
