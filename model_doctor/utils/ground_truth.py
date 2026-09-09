"""Ground truth in whichever format the dataset is actually in.

Model Doctor's whole job is comparing what a model predicted against what was
annotated, so *reading annotations* is the one place a dataset's format can
matter. Everything downstream — matching, the outcome taxonomy, storage, root
causes, clustering, recommendations — consumes
:class:`~model_doctor.utils.annotations.ObjectAnnotation` and has never known
where those came from. This module is what keeps that true for a second format.

Two layouts are read, and which one a split is in is decided by looking at the
split, never by guessing:

``yolo``
    A ``labels/`` directory beside ``images/``, one ``.txt`` per image. This is
    the original path and its reader is unchanged — :func:`.dataset.load_ground_truth`
    is called as it always was.

``coco``
    Images sitting directly in the split directory beside a single
    ``_annotations.coco.json``. This is what Roboflow exports and what RF-DETR
    trains on, so it is what an RF-DETR model's own dataset looks like.

**On class ids.** COCO category ids are arbitrary integers; a model's class ids
are contiguous indices. The mapping between them is *sorted position*, which is
what ``rfdetr`` itself applies when it builds a dataset with
``remap_category_ids=True``. Agreeing with the training-time mapping is the
whole point: a different rule here would silently relabel every annotation and
still report success.

**On segmentation.** A COCO annotation may carry several polygons, or a
run-length encoded mask. ``ObjectAnnotation`` holds one outline, so the largest
piece is kept for diagnosis and display — the same reduction the prediction side
already makes, so both are lossy in the same direction. The *whole* segmentation
is preserved separately for the evaluator, which can score every component.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from model_doctor.utils.annotations import ObjectAnnotation
from model_doctor.utils.exceptions import DatasetConfigError
from model_doctor.utils.logging_utils import get_logger

logger = get_logger(__name__)

#: The annotation file Roboflow writes and RF-DETR reads. Listed rather than
#: matched by glob so an unrelated JSON in a split is never mistaken for one.
COCO_ANNOTATION_NAMES: tuple[str, ...] = (
    "_annotations.coco.json",
    "annotations.coco.json",
    "instances.json",
)

FORMAT_YOLO = "yolo"
FORMAT_COCO = "coco"

#: A polygon needs three vertices to enclose area. Fewer is a line, which would
#: make every overlap against it zero — worse than admitting there is no outline.
MIN_POLYGON_POINTS = 3


@runtime_checkable
class GroundTruthSource(Protocol):
    """One split's annotations, however they happen to be stored."""

    #: ``"yolo"`` or ``"coco"``. Recorded so a report can say what it read.
    format: str
    #: ``{label index: name}``, in the model's own index space.
    class_names: dict[int, str]

    def annotations_for(
        self, image_path: Path, image_width: int, image_height: int
    ) -> list[ObjectAnnotation]:
        """Return one image's ground truth, in absolute pixels."""

    def coco_ground_truth(self, images_dir: Path) -> dict[str, Any]:
        """Return the split as a COCO dictionary, for the shared evaluator."""


# ---------------------------------------------------------------------------
# YOLO — the original path, unchanged
# ---------------------------------------------------------------------------
@dataclass
class YoloGroundTruth:
    """Reads ``.txt`` label files beside the images.

    A thin shell around the functions that have always done this, so the format
    that works today keeps running exactly the code that made it work.
    """

    split_dir: Path
    class_names: dict[int, str]
    format: str = FORMAT_YOLO

    def annotations_for(
        self, image_path: Path, image_width: int, image_height: int
    ) -> list[ObjectAnnotation]:
        """Delegate to the existing reader, verbatim."""
        from model_doctor.utils.dataset import load_ground_truth

        return load_ground_truth(
            image_path, image_width, image_height, self.class_names
        )

    def coco_ground_truth(self, images_dir: Path) -> dict[str, Any]:
        """Delegate to the existing converter, verbatim."""
        from model_doctor.app.evaluation import build_ground_truth

        return build_ground_truth(images_dir, self.class_names)


# ---------------------------------------------------------------------------
# COCO — read natively, never converted on disk
# ---------------------------------------------------------------------------
@dataclass
class CocoGroundTruth:
    """Reads a split's ``_annotations.coco.json`` directly.

    The file is parsed once and indexed by image file name, because that is the
    only key both halves of the comparison share: a prediction knows the file it
    came from, not the id the annotation file gave it.
    """

    split_dir: Path
    annotation_path: Path
    class_names: dict[int, str]
    #: Original COCO ``category_id`` to contiguous label index.
    category_to_label: dict[int, int]
    #: File name to that image's record and annotations.
    _by_name: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    format: str = FORMAT_COCO

    def annotations_for(
        self,
        image_path: Path,
        image_width: int = 0,  # noqa: ARG002 - signature parity with the YOLO reader
        image_height: int = 0,  # noqa: ARG002 - COCO geometry is already absolute
    ) -> list[ObjectAnnotation]:
        """Return one image's annotations, converted to absolute pixels.

        COCO geometry is already absolute, so nothing is denormalised. The
        image's own dimensions are accepted for signature parity with the YOLO
        reader, whose labels are normalised and cannot be read without them.
        """
        found = self._by_name.get(image_path.name)
        if found is None:
            # Not an error: an image the annotation file does not mention is a
            # negative sample, exactly as a missing .txt is on the YOLO side.
            return []

        _record, raw = found
        annotations: list[ObjectAnnotation] = []
        for entry in raw:
            try:
                annotations.append(self._to_annotation(entry))
            except (ValueError, KeyError, TypeError) as error:
                # One malformed annotation must not discard the rest of the
                # file, the same rule the YOLO reader follows.
                logger.warning(
                    "%s: annotation %s unusable (%s) — skipping.",
                    self.annotation_path.name,
                    entry.get("id"),
                    error,
                )
        return annotations

    def _to_annotation(self, entry: dict[str, Any]) -> ObjectAnnotation:
        """Convert one COCO annotation into the shared representation."""
        category_id = int(entry["category_id"])
        if category_id not in self.category_to_label:
            raise ValueError(
                f"category_id {category_id} is not declared in "
                f"{self.annotation_path.name}"
            )
        label = self.category_to_label[category_id]
        name = self.class_names.get(label, f"id:{label}")

        polygon = _outline_of(entry)
        if polygon is not None:
            # from_polygon derives the box, so box-level analysis works on a
            # segmentation dataset with no special-casing downstream.
            return ObjectAnnotation.from_polygon(
                class_id=label, class_name=name, polygon=polygon
            )

        x, y, width, height = (float(v) for v in entry["bbox"])
        return ObjectAnnotation.from_box(
            class_id=label,
            class_name=name,
            x1=x,
            y1=y,
            x2=x + width,
            y2=y + height,
        )

    def coco_ground_truth(self, images_dir: Path) -> dict[str, Any]:
        """Return the annotation file's own content, in the model's id space.

        Nothing is recomputed from the geometry: boxes, areas and segmentation
        are carried across exactly as annotated, which is the point of reading
        the format natively. Only two things change, and both are mappings
        rather than measurements — category ids become the model's label
        indices, so a prediction's class id means the same thing as an
        annotation's, and images are restricted to those actually on disk.
        """
        from model_doctor.app.evaluation import EvaluationError, list_images

        present = {path.name for path in list_images(images_dir)}
        if not present:
            raise EvaluationError(f"No images found in {images_dir}")

        images: list[dict[str, Any]] = []
        kept_ids: set[int] = set()
        for name in sorted(present):
            found = self._by_name.get(name)
            if found is None:
                continue
            record, _ = found
            images.append({
                "id": int(record["id"]),
                "file_name": name,
                "width": int(record["width"]),
                "height": int(record["height"]),
            })
            kept_ids.add(int(record["id"]))

        if not images:
            raise EvaluationError(
                f"{self.annotation_path} describes none of the images in "
                f"{images_dir}. Refusing to evaluate against empty ground truth."
            )

        annotations: list[dict[str, Any]] = []
        for _name, (record, raw) in self._by_name.items():
            if int(record["id"]) not in kept_ids:
                continue
            for entry in raw:
                category_id = int(entry["category_id"])
                label = self.category_to_label.get(category_id)
                if label is None:
                    continue
                annotations.append({
                    "id": int(entry["id"]),
                    "image_id": int(entry["image_id"]),
                    "category_id": label,
                    "bbox": [float(v) for v in entry["bbox"]],
                    "area": float(
                        entry.get("area")
                        or (float(entry["bbox"][2]) * float(entry["bbox"][3]))
                    ),
                    "iscrowd": int(entry.get("iscrowd", 0)),
                    # Carried through untouched — polygons stay polygons and an
                    # RLE stays an RLE, so the evaluator scores every component
                    # rather than the one this module keeps for display.
                    "segmentation": entry.get("segmentation", []),
                })

        return {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": label, "name": name, "supercategory": "none"}
                for label, name in sorted(self.class_names.items())
            ],
        }


# ---------------------------------------------------------------------------
# Segmentation: several polygons, or a run-length encoded mask
# ---------------------------------------------------------------------------
def _outline_of(entry: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    """Return one outline for an annotation, or ``None`` when it has none.

    ``segmentation`` is either a list of flat ``[x, y, x, y, ...]`` polygons or
    an RLE mask. Both are handled; the largest piece wins, because the shared
    annotation model holds one outline per object and the prediction side
    already reduces the same way.
    """
    segmentation = entry.get("segmentation")
    if not segmentation:
        return None

    if isinstance(segmentation, dict):
        return _outline_from_rle(segmentation)

    rings: list[tuple[tuple[float, float], ...]] = []
    for piece in segmentation:
        if not isinstance(piece, (list, tuple)) or len(piece) < MIN_POLYGON_POINTS * 2:
            continue
        values = [float(v) for v in piece]
        ring = tuple(
            (values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)
        )
        if len(ring) >= MIN_POLYGON_POINTS:
            rings.append(ring)

    if not rings:
        return None
    return max(rings, key=_ring_area)


def _ring_area(ring: tuple[tuple[float, float], ...]) -> float:
    """Return a ring's absolute area by the shoelace formula."""
    total = 0.0
    for index in range(len(ring)):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _outline_from_rle(rle: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    """Trace an RLE mask back to its largest outline.

    Decoded with pycocotools, which is what wrote it, and traced with OpenCV,
    which is what the prediction side already uses for the same reduction.
    Returns ``None`` rather than raising when either is unavailable: an outline
    that cannot be recovered is a missing measurement, not a broken dataset, and
    the annotation still yields a box.
    """
    try:
        import cv2
        import numpy as np
        from pycocotools import mask as coco_mask
    except ImportError:  # pragma: no cover - both are declared dependencies
        logger.warning("pycocotools or opencv is unavailable; RLE outline skipped.")
        return None

    try:
        counts = rle.get("counts")
        encoded = dict(rle)
        if isinstance(counts, str):
            encoded["counts"] = counts.encode("utf-8")
        decoded = coco_mask.decode(encoded)
    except Exception as error:  # noqa: BLE001 - a bad mask is one lost outline
        logger.warning("Could not decode an RLE mask (%s); outline skipped.", error)
        return None

    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    contours, _ = cv2.findContours(
        np.ascontiguousarray(decoded.astype(np.uint8)),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(points) < MIN_POLYGON_POINTS:
        return None
    return tuple((float(x), float(y)) for x, y in points)


# ---------------------------------------------------------------------------
# Which format is this split in?
# ---------------------------------------------------------------------------
def annotation_file(split_dir: Path) -> Path | None:
    """Return the split's COCO annotation file, or ``None``.

    Looked for beside the images, and in the parent only when the descriptor
    pointed at an ``images/`` subdirectory — that is the one layout where the
    file legitimately sits a level up. Searching the parent unconditionally
    would let a stray file at the dataset root bind to every split at once.
    """
    directories = [split_dir]
    if split_dir.name == "images":
        directories.append(split_dir.parent)
    for directory in directories:
        for name in COCO_ANNOTATION_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def labels_dir(split_dir: Path) -> Path | None:
    """Return the split's YOLO ``labels/`` directory, or ``None``."""
    for candidate in (split_dir / "labels", split_dir.parent / "labels"):
        if candidate.is_dir():
            return candidate
    return None


def detect_format(split_dir: Path) -> str | None:
    """Say which layout a split is in, by what is actually on disk.

    YOLO is checked first and wins outright when both are present: a dataset
    carrying label files has been prepared for that reader, and preferring it
    keeps every dataset that works today reading exactly as it did.

    Returns ``None`` when neither is found, which callers report rather than
    treat as an empty dataset — a split whose annotations cannot be located
    must never be read as one that legitimately has none.
    """
    if labels_dir(split_dir) is not None:
        return FORMAT_YOLO
    if annotation_file(split_dir) is not None:
        return FORMAT_COCO
    return None


def read_coco_categories(
    annotation_path: Path,
) -> tuple[dict[int, str], dict[int, int]]:
    """Return ``({label: name}, {category_id: label})`` for one annotation file.

    Ids are mapped to contiguous labels by **sorted position**, which is the
    mapping ``rfdetr`` applies when training on the same file. Category ids are
    not assumed contiguous, zero-based, or ordered in the file.
    """
    try:
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetConfigError(
            f"{annotation_path} is not readable as COCO JSON: {error}"
        ) from error

    categories = raw.get("categories") or []
    if not categories:
        raise DatasetConfigError(
            f"{annotation_path} declares no categories, so class names are unknown."
        )

    ordered = sorted(categories, key=lambda c: int(c["id"]))
    category_to_label = {int(c["id"]): index for index, c in enumerate(ordered)}
    class_names = {index: str(c["name"]) for index, c in enumerate(ordered)}
    return class_names, category_to_label


def open_ground_truth(
    split_dir: Path, class_names: dict[int, str] | None = None
) -> GroundTruthSource:
    """Return a reader for whichever format this split is in.

    Args:
        split_dir: The directory holding the split's images.
        class_names: Names from the dataset descriptor. Used for a YOLO split,
            where the descriptor is the only source. Ignored for a COCO split,
            whose annotation file carries its own categories and is the
            authority on them.

    Raises:
        DatasetConfigError: The split is in neither layout, or its annotation
            file cannot be read. Refusing here is the point: reading an
            unrecognised split as one with no annotations would report every
            prediction as spurious and every image as empty, and call it a
            successful analysis.
    """
    split_dir = Path(split_dir)
    layout = detect_format(split_dir)

    if layout == FORMAT_YOLO:
        return YoloGroundTruth(split_dir=split_dir, class_names=dict(class_names or {}))

    if layout == FORMAT_COCO:
        annotation_path = annotation_file(split_dir)
        assert annotation_path is not None  # detect_format found it
        names, category_to_label = read_coco_categories(annotation_path)
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))

        by_id: dict[int, dict[str, Any]] = {}
        for record in raw.get("images") or []:
            by_id[int(record["id"])] = record

        grouped: dict[int, list[dict[str, Any]]] = {}
        for entry in raw.get("annotations") or []:
            grouped.setdefault(int(entry["image_id"]), []).append(entry)

        by_name: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for image_id, record in by_id.items():
            # The file name may carry a directory prefix from the exporter; the
            # basename is what a prediction will know itself by.
            name = Path(str(record["file_name"])).name
            by_name[name] = (record, grouped.get(image_id, []))

        logger.info(
            "Reading COCO ground truth from %s: %d image(s), %d annotation(s), "
            "%d class(es)",
            annotation_path.name,
            len(by_name),
            sum(len(v) for _, v in by_name.values()),
            len(names),
        )
        return CocoGroundTruth(
            split_dir=split_dir,
            annotation_path=annotation_path,
            class_names=names,
            category_to_label=category_to_label,
            _by_name=by_name,
        )

    raise DatasetConfigError(
        f"{split_dir} is in neither layout Model Doctor reads. Expected either "
        f"a labels/ directory beside the images, or one of "
        f"{', '.join(COCO_ANNOTATION_NAMES)} in the split."
    )


def verify_class_names(
    source: GroundTruthSource, model_names: dict[int, str]
) -> None:
    """Refuse a dataset whose classes are not the model's classes.

    A mismatch means every finding would be attributed to the wrong class while
    the run still reported success, which is the most expensive kind of quiet
    wrongness this project can produce. Compared by name at each index, because
    that is what a reader of the results will see.

    Raises:
        DatasetConfigError: The two disagree.
    """
    if not source.class_names or not model_names:
        return

    dataset = {int(k): str(v) for k, v in source.class_names.items()}
    model = {int(k): str(v) for k, v in model_names.items()}
    shared = sorted(set(dataset) & set(model))
    disagreements = [
        f"{index}: dataset says {dataset[index]!r}, model says {model[index]!r}"
        for index in shared
        if dataset[index] != model[index]
    ]
    if disagreements:
        raise DatasetConfigError(
            "The dataset's classes are not this model's classes, so every "
            "finding would be attributed to the wrong one:\n  "
            + "\n  ".join(disagreements)
        )

    if len(dataset) != len(model):
        logger.warning(
            "The dataset declares %d class(es) and the model %d. The names they "
            "share agree, so this is a subset rather than a mismatch.",
            len(dataset),
            len(model),
        )


__all__ = [
    "COCO_ANNOTATION_NAMES",
    "FORMAT_COCO",
    "FORMAT_YOLO",
    "CocoGroundTruth",
    "GroundTruthSource",
    "YoloGroundTruth",
    "annotation_file",
    "detect_format",
    "labels_dir",
    "open_ground_truth",
    "read_coco_categories",
    "verify_class_names",
]
