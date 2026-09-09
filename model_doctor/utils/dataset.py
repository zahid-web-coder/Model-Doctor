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

from model_doctor import config
from model_doctor.utils.annotations import ObjectAnnotation
from model_doctor.utils.exceptions import DatasetConfigError
from model_doctor.utils.logging_utils import get_logger
from model_doctor.utils.resources import find_images

logger = get_logger(__name__)

# Standard YOLO split names, in the order a report should present them.
SPLIT_KEYS: tuple[str, ...] = ("train", "val", "test")

# A YOLO label line is either a detection box or a segmentation outline:
#
#   detection     <class_id> <x_c> <y_c> <w> <h>              -> 5 fields
#   segmentation  <class_id> <x1> <y1> <x2> <y2> ...          -> 1 + 2n fields
#
# Both are normalised to [0, 1]. The two are told apart by field count, which
# is how the wider YOLO ecosystem does it too. A polygon needs at least three
# vertices to enclose any area, giving a minimum of 7 fields.
DETECTION_FIELD_COUNT: int = 5
MIN_POLYGON_FIELDS: int = 7


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

    class_names = _class_names_for(class_names, splits, yaml_path)

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


def _class_names_for(
    declared: dict[int, str], splits: dict[str, Path], yaml_path: Path
) -> dict[int, str]:
    """Return the names to use, preferring a COCO split's own categories.

    A YOLO dataset's ``data.yaml`` is written by whoever made the dataset and is
    the only record of its classes, so it is trusted unchanged — nothing about
    the existing path moves.

    A COCO dataset is different. Ramanujan generates a descriptor for one so
    that the rest of the system has a file to point at, and that generated
    ``names`` is a single placeholder rather than the real classes. Believing it
    would label every finding ``object``. The annotation file beside the images
    is the authority for such a split, and it is read here.

    Args:
        declared: ``names`` as the descriptor gave them.
        splits: Resolved split directories.
        yaml_path: The descriptor, named in the log so a surprising answer can
            be traced back to the file it came from.
    """
    from model_doctor.utils import ground_truth

    for _key, split_dir in sorted(splits.items()):
        if ground_truth.detect_format(split_dir) != ground_truth.FORMAT_COCO:
            continue
        annotation = ground_truth.annotation_file(split_dir)
        if annotation is None:  # pragma: no cover - detect_format found it
            continue
        try:
            names, _ = ground_truth.read_coco_categories(annotation)
        except DatasetConfigError as error:
            logger.warning(
                "%s is a COCO split but its categories are unreadable (%s); "
                "keeping the descriptor's names.",
                split_dir,
                error,
            )
            return declared
        if names != declared:
            logger.info(
                "%s is a COCO split; taking class names from %s rather than "
                "%s. Descriptor said %s; the annotations say %s.",
                split_dir,
                annotation.name,
                yaml_path.name,
                list(declared.values())[:6],
                list(names.values())[:6],
            )
        return names
    return declared


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


def _parse_label_line(
    fields: list[str],
    image_width: int,
    image_height: int,
    class_names: dict[int, str] | None,
) -> ObjectAnnotation:
    """Convert one label line into an annotation in absolute pixels.

    Handles both YOLO label forms, chosen by field count. The two are
    distinguished per *line*, not per file, so a dataset that mixes boxes and
    outlines parses correctly without configuration.

    Args:
        fields: Whitespace-split tokens of the line. Never empty.
        image_width: Image width in pixels, used to denormalise.
        image_height: Image height in pixels, used to denormalise.
        class_names: Class id to name mapping, or ``None`` when unavailable.

    Returns:
        A populated :class:`~utils.annotations.ObjectAnnotation`.

    Raises:
        ValueError: If the field count matches neither form, a value is not
            numeric, or a polygon has too few vertices. The caller turns this
            into a warning and skips the line.
    """
    class_id = int(float(fields[0]))
    name = (class_names or {}).get(class_id, f"id:{class_id}")
    count = len(fields)

    if count == DETECTION_FIELD_COUNT:
        # Detection: centre-based and normalised. Converting to absolute corner
        # form is the same arithmetic a polygon's extent already produces, so
        # both paths end in one coordinate convention (D-006).
        x_c, y_c, width, height = (float(value) for value in fields[1:])
        return ObjectAnnotation.from_box(
            class_id=class_id,
            class_name=name,
            x1=(x_c - width / 2) * image_width,
            y1=(y_c - height / 2) * image_height,
            x2=(x_c + width / 2) * image_width,
            y2=(y_c + height / 2) * image_height,
        )

    # Segmentation: an odd field count of at least 7, being one class id plus
    # x/y pairs. An even count means a coordinate is missing — a truncated
    # line, which must not be silently reinterpreted.
    if count >= MIN_POLYGON_FIELDS and count % 2 == 1:
        values = [float(value) for value in fields[1:]]
        polygon = tuple(
            (values[i] * image_width, values[i + 1] * image_height)
            for i in range(0, len(values), 2)
        )
        # from_polygon derives the bounding box, so box-based analysis works on
        # segmentation data with no special-casing anywhere downstream.
        return ObjectAnnotation.from_polygon(
            class_id=class_id, class_name=name, polygon=polygon
        )

    raise ValueError(
        f"expected {DETECTION_FIELD_COUNT} fields (box) or an odd count of at "
        f"least {MIN_POLYGON_FIELDS} (polygon), got {count}"
    )


def load_ground_truth(
    image_path: Path,
    image_width: int,
    image_height: int,
    class_names: dict[int, str] | None = None,
) -> list[ObjectAnnotation]:
    """Read an image's YOLO label file as annotations in absolute pixels.

    A YOLO label file holds one object per line, in one of two forms::

        <class_id> <x_center> <y_center> <width> <height>      detection
        <class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...           segmentation

    Both store geometry normalised to ``[0, 1]``, which is what lets a label
    file survive its image being resized. Normalised values are the wrong
    format for drawing or for computing overlap, so conversion to absolute
    pixels happens once, here, and everything downstream works in pixels.

    A segmentation line also yields a bounding box, derived from the outline's
    extent. Box-based analysis therefore works unchanged on a segmentation
    dataset.

    Args:
        image_path: The image whose labels should be read.
        image_width: Image width in pixels, used to denormalise.
        image_height: Image height in pixels, used to denormalise.
        class_names: Class id to name mapping, typically
            :attr:`DatasetConfig.class_names`. When omitted, names fall back to
            ``"id:<n>"`` so parsing still succeeds without a descriptor.

    Returns:
        One annotation per valid line. An image with no label file is a
        legitimate *negative* sample, so ``[]`` is returned rather than raising.
    """
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        return []

    annotations: list[ObjectAnnotation] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        try:
            annotations.append(
                _parse_label_line(
                    line.split(), image_width, image_height, class_names
                )
            )
        except ValueError as exc:
            # One malformed line must not discard the rest of the file. A
            # dataset of 1,000 images should not be unusable because of a
            # single truncated row.
            logger.warning(
                "%s line %d: %s — skipping.", label_path.name, line_number, exc
            )

    return annotations


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
