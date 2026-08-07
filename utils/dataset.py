"""Dataset descriptor parsing and YOLO label handling.

This module owns everything about *ground truth*: where the splits live, what
the classes are called, and how to read a ``.txt`` label file. Keeping it
separate from inference matters because Model Doctor's core job — comparing
predictions against ground truth — needs both halves to be independently
correct and independently testable.

Class names are always read from ``data.yaml``. They are never written down in
source code, so the same codebase serves a 3-class dataset and a 80-class one
without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config
from utils.exceptions import DatasetConfigError
from utils.geometry import BoxGeometryMixin
from utils.logging_utils import get_logger
from utils.resources import find_images

logger = get_logger(__name__)

# Standard YOLO split names, in the order a report should present them.
SPLIT_KEYS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class GroundTruthBox(BoxGeometryMixin):
    """One line of a YOLO ``.txt`` label file, in absolute pixel coordinates.

    YOLO stores boxes *normalised* (every value in ``[0, 1]``, expressed as a
    fraction of image width/height) and *centre-based*. That format is
    resolution-independent, which is what lets the same label file survive an
    image being resized. It is, however, the wrong format for drawing or for
    computing IoU, so we convert once on read and work in pixels thereafter.

    Attributes:
        class_id: Zero-based index into the dataset's class list.
        x1, y1: Top-left corner in pixels.
        x2, y2: Bottom-right corner in pixels.
    """

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    # Geometry (xyxy, width, height, area, center) comes from BoxGeometryMixin
    # so that predictions and ground truth cannot drift apart.


@dataclass(frozen=True)
class DatasetConfig:
    """Parsed contents of a YOLO ``data.yaml``.

    Attributes:
        root: Directory used to resolve relative split paths.
        class_names: Mapping of class id to class name, e.g. ``{0: "crack"}``.
        splits: Mapping of split name to its resolved image directory. Only
            splits that are declared *and* exist on disk appear here.
        source_path: The ``data.yaml`` this was read from.
    """

    root: Path
    class_names: dict[int, str]
    splits: dict[str, Path]
    source_path: Path

    @property
    def num_classes(self) -> int:
        """Return the number of classes, derived — never hardcoded."""
        return len(self.class_names)

    def name_for(self, class_id: int) -> str:
        """Return the display name for ``class_id``.

        Falls back to ``"id:<n>"`` for an unknown id rather than raising, so a
        model/dataset class-count mismatch shows up visibly in output instead
        of crashing a long batch run.
        """
        return self.class_names.get(class_id, f"id:{class_id}")


def _normalise_names(raw_names: object) -> dict[int, str]:
    """Convert the ``names:`` field into a ``{id: name}`` mapping.

    ``data.yaml`` files in the wild use two shapes::

        names: ['crack', 'spall']        # list — index is the class id
        names: {0: 'crack', 1: 'spall'}  # dict — explicit ids

    Both are accepted so the user never has to hand-edit an exported dataset.

    Raises:
        DatasetConfigError: If the field is neither a list nor a dict.
    """
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise DatasetConfigError(
        f"'names' must be a list or dict, got {type(raw_names).__name__}"
    )


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    """Read and validate ``data.yaml``.

    Split paths inside a YOLO descriptor are relative to the descriptor's own
    directory (or to its ``path:`` key when present). We resolve them here so
    that no caller ever has to reason about relative paths again.

    Args:
        path: Descriptor location. Defaults to :data:`config.DATA_YAML_PATH`.

    Returns:
        A fully resolved :class:`DatasetConfig`.

    Raises:
        DatasetConfigError: If the file is missing, unparseable, or has no
            ``names`` key.
    """
    import yaml

    yaml_path = Path(path) if path is not None else config.DATA_YAML_PATH

    if not yaml_path.is_file():
        raise DatasetConfigError(
            f"data.yaml not found at {yaml_path}. Provide the YOLO dataset "
            f"descriptor before running dataset-level operations."
        )

    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise DatasetConfigError(f"{yaml_path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise DatasetConfigError(f"{yaml_path} did not parse to a mapping.")
    if "names" not in raw:
        raise DatasetConfigError(f"{yaml_path} has no 'names:' key.")

    class_names = _normalise_names(raw["names"])

    # `path:` is an optional root that split entries are relative to. When it
    # is absent, splits are relative to the descriptor's own directory.
    declared_root = raw.get("path")
    root = (
        (yaml_path.parent / str(declared_root)).resolve()
        if declared_root
        else yaml_path.parent.resolve()
    )

    splits: dict[str, Path] = {}
    for key in SPLIT_KEYS:
        entry = raw.get(key)
        if not entry:
            continue
        # A split may be given as a list of directories; take the first.
        entry_str = str(entry[0] if isinstance(entry, (list, tuple)) else entry)
        candidate = Path(entry_str)
        resolved = candidate if candidate.is_absolute() else (root / candidate)
        resolved = resolved.resolve()
        if resolved.is_dir():
            splits[key] = resolved
        else:
            logger.warning(
                "Split '%s' points to %s, which does not exist", key, resolved
            )

    # Cross-check the declared class count against the actual names list. A
    # mismatch is a real dataset bug that otherwise surfaces much later as
    # baffling label errors, so we surface it now.
    declared_nc = raw.get("nc")
    if declared_nc is not None and int(declared_nc) != len(class_names):
        logger.warning(
            "data.yaml declares nc=%s but lists %d names — trusting the names list.",
            declared_nc,
            len(class_names),
        )

    logger.info(
        "Loaded dataset config: %d classes, splits present: %s",
        len(class_names),
        ", ".join(splits) or "none",
    )
    return DatasetConfig(
        root=root,
        class_names=class_names,
        splits=splits,
        source_path=yaml_path,
    )


def label_path_for_image(image_path: Path) -> Path:
    """Return the ``.txt`` label file corresponding to ``image_path``.

    YOLO's convention is a parallel directory tree where ``images/`` is
    replaced by ``labels/`` and the extension becomes ``.txt``::

        dataset/test/images/frame_001.jpg
        dataset/test/labels/frame_001.txt

    The replacement targets the *last* ``images`` path segment, so a dataset
    that happens to live under a folder called ``images/`` higher up the tree
    is not corrupted.
    """
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def load_ground_truth(
    image_path: Path, image_width: int, image_height: int
) -> list[GroundTruthBox]:
    """Read the YOLO label file for an image and convert it to pixel boxes.

    Each line of a YOLO label file is::

        <class_id> <x_center> <y_center> <width> <height>

    with the four geometry values normalised to ``[0, 1]``. Conversion to
    corner form is::

        x1 = (x_center - width / 2)  * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2)  * image_width
        y2 = (y_center + height / 2) * image_height

    Args:
        image_path: The image whose labels should be read.
        image_width: Image width in pixels, used to denormalise.
        image_height: Image height in pixels, used to denormalise.

    Returns:
        Ground-truth boxes. An image with no label file is a legitimate
        *negative* sample, so ``[]`` is returned rather than raising.
    """
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        return []

    boxes: list[GroundTruthBox] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue

        fields = line.split()
        # Segmentation datasets store polygons (many coordinate pairs) on the
        # same kind of line. We only handle detection boxes here, and warn
        # rather than silently misreading a polygon as a box.
        if len(fields) != 5:
            logger.warning(
                "%s line %d: expected 5 fields, got %d — skipping.",
                label_path.name,
                line_number,
                len(fields),
            )
            continue

        try:
            class_id = int(float(fields[0]))
            x_c, y_c, width, height = (float(value) for value in fields[1:])
        except ValueError:
            logger.warning(
                "%s line %d: non-numeric value — skipping.",
                label_path.name,
                line_number,
            )
            continue

        boxes.append(
            GroundTruthBox(
                class_id=class_id,
                x1=(x_c - width / 2) * image_width,
                y1=(y_c - height / 2) * image_height,
                x2=(x_c + width / 2) * image_width,
                y2=(y_c + height / 2) * image_height,
            )
        )

    return boxes


def describe(dataset: DatasetConfig) -> str:
    """Render a dataset summary for console output."""
    lines = [
        "",
        f"Dataset: {dataset.source_path}",
        "-" * 64,
        f"  Root        : {dataset.root}",
        f"  Classes     : {dataset.num_classes}",
    ]
    for class_id in sorted(dataset.class_names):
        lines.append(f"      {class_id:>3}  {dataset.class_names[class_id]}")

    lines.append("  Splits      :")
    for key in SPLIT_KEYS:
        if key in dataset.splits:
            count = len(find_images(dataset.splits[key]))
            lines.append(f"      {key:<6} {count:>6} images   {dataset.splits[key]}")
        else:
            lines.append(f"      {key:<6} {'—':>6}          not declared")
    lines.append("-" * 64)
    lines.append("")
    return "\n".join(lines)
