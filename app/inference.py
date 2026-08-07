"""YOLO inference wrapper — the foundation every later module builds on.

This module turns a trained detector into *structured, inspectable data*. That
framing matters: Model Doctor does not exist to draw boxes, it exists to reason
about why boxes are wrong. Every design choice here serves that end.

* Detections are returned as :class:`Detection` objects, not raw tensors, so
  the error-analysis module can consume them without knowing that Ultralytics
  exists. Swapping YOLO for another detector later means reimplementing this
  file and nothing else.
* Class names come from the model's own ``names`` mapping. They are never
  hardcoded, and the model is the authority for what its own class ids mean.
* A missing model produces a clear, actionable message — never a traceback.

Run it directly to check the project's health and, when resources allow, to
produce annotated predictions::

    python -m app.inference --check
    python -m app.inference --image path/to/image.jpg
    python -m app.inference --split test
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow `python app/inference.py` in addition to `python -m app.inference` by
# putting the project root on sys.path before the first local import.
if __package__ in (None, ""):  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils.dataset import DatasetConfig, describe, load_dataset_config
from utils.exceptions import (
    DatasetConfigError,
    ModelLoadError,
    ResourceNotFoundError,
)
from utils.geometry import BoxGeometryMixin
from utils.logging_utils import get_logger
from utils.resources import check_model, find_images, format_report, verify_all

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection(BoxGeometryMixin):
    """A single predicted bounding box.

    Coordinates are absolute pixels in ``xyxy`` (corner) form, matching the
    space :class:`~utils.dataset.GroundTruthBox` uses. Holding predictions and
    ground truth in one coordinate system is what makes IoU computation in the
    future error-analysis module a two-line function instead of a source of
    subtle bugs.

    Attributes:
        class_id: Model's class index for this detection.
        class_name: Human-readable name resolved from the model's own mapping.
        confidence: Model's certainty in ``[0, 1]``.
        x1, y1, x2, y2: Box corners in pixels.
    """

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    # Geometry (xyxy, width, height, area, center) comes from BoxGeometryMixin
    # so predictions and ground truth share one implementation.


@dataclass
class ImagePrediction:
    """Everything produced for one image.

    Attributes:
        image_path: Source image.
        detections: Boxes that survived confidence and NMS filtering.
        inference_ms: Wall-clock time for the forward pass, in milliseconds.
        image_width: Source width in pixels.
        image_height: Source height in pixels.
        annotated_path: Where the rendered image was written, if it was.
        error: Populated when this image failed, leaving the batch intact.
    """

    image_path: Path
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    annotated_path: Path | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether this image was processed without error."""
        return self.error is None


def _safe_float(source: Any, attribute: str) -> float | None:
    """Read ``attribute`` from ``source`` and coerce it to ``float``.

    Returns ``None`` if the attribute is absent or not numeric. Metric objects
    come from a third-party library whose attribute names can change between
    versions; a missing metric should degrade to "not reported", never crash a
    validation run that has already done the expensive work.
    """
    value = getattr(source, attribute, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ValidationMetrics:
    """Evaluation results, normalised away from any library's object model.

    This exists for the same reason :class:`Detection` does (DECISIONS D-005):
    without it, the third-party metrics object leaks outward and every consumer
    becomes coupled to that library's attribute names. Here, exactly one
    function knows them.

    Every field is optional because a metric a future library version stops
    reporting should surface as "unavailable", not as an exception.

    Attributes:
        precision: Mean precision across classes.
        recall: Mean recall across classes.
        map50: Mean average precision at IoU 0.50.
        map50_95: Mean average precision averaged over IoU 0.50-0.95.
        per_class: Class name to its individual metrics.
        artefacts_dir: Where plots and the confusion matrix were written.
    """

    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    artefacts_dir: Path | None = None

    @classmethod
    def from_raw(
        cls,
        raw: Any,
        class_names: dict[int, str] | None = None,
        artefacts_dir: Path | None = None,
    ) -> ValidationMetrics:
        """Extract metrics from a library evaluation result.

        This is the single point of coupling to the evaluation library's shape.
        It is deliberately total: any missing or renamed attribute yields
        ``None`` for that field rather than raising, so a partial result stays
        usable.

        Args:
            raw: The object returned by the library's evaluation call.
            class_names: Class id to name mapping, used to label per-class rows.
            artefacts_dir: Directory the library wrote plots into.

        Returns:
            A populated :class:`ValidationMetrics`.
        """
        box = getattr(raw, "box", None)
        if box is None:
            return cls(artefacts_dir=artefacts_dir)

        per_class: dict[str, dict[str, float]] = {}
        names = class_names or {}
        # `ap_class_index` maps each per-class row back to its class id. The
        # arrays are positional, so they must be read together or not at all —
        # the same alignment hazard as the detection tensors (D-011).
        indices = getattr(box, "ap_class_index", None)
        if indices is not None:
            for row, class_id in enumerate(indices):
                label = names.get(int(class_id), f"id:{int(class_id)}")
                row_metrics: dict[str, float] = {}
                for key, attribute in (
                    ("precision", "p"),
                    ("recall", "r"),
                    ("map50", "ap50"),
                    ("map50_95", "ap"),
                ):
                    series = getattr(box, attribute, None)
                    if series is None or row >= len(series):
                        continue
                    try:
                        row_metrics[key] = float(series[row])
                    except (TypeError, ValueError):
                        continue
                if row_metrics:
                    per_class[label] = row_metrics

        return cls(
            precision=_safe_float(box, "mp"),
            recall=_safe_float(box, "mr"),
            map50=_safe_float(box, "map50"),
            map50_95=_safe_float(box, "map"),
            per_class=per_class,
            artefacts_dir=artefacts_dir,
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class Detector:
    """Loads a YOLO model once and runs inference with it.

    The model is loaded lazily on first use rather than in ``__init__``. This
    keeps construction cheap and side-effect free, so the object can be created
    in a CLI or a Streamlit script before we know whether the user actually
    intends to run inference.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        device: str | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        image_size: int | None = None,
    ) -> None:
        """Configure a detector without loading anything yet.

        Args:
            model_path: Explicit weights path. When ``None``, the model is
                discovered via :func:`~utils.resources.discover_model`.
            device: Compute device. Defaults to :data:`config.DEVICE`.
            confidence: Minimum confidence to keep a detection.
            iou: IoU threshold for Non-Maximum Suppression.
            image_size: Longest-side input resolution.
        """
        self._explicit_model_path = Path(model_path) if model_path else None
        self.device = device or config.DEVICE
        self.confidence = (
            confidence if confidence is not None else config.CONFIDENCE_THRESHOLD
        )
        self.iou = iou if iou is not None else config.NMS_IOU_THRESHOLD
        self.image_size = image_size or config.IMAGE_SIZE

        self._model: Any | None = None
        self._model_path: Path | None = None

    # -- loading ------------------------------------------------------------
    @property
    def model_path(self) -> Path | None:
        """Return the weights path in use, or ``None`` before loading."""
        return self._model_path

    def load(self) -> None:
        """Load the YOLO weights into memory.

        Idempotent: calling it twice is a no-op, so callers may invoke it
        defensively without paying the cost twice.

        Raises:
            ResourceNotFoundError: No usable ``.pt`` file was found.
            ModelLoadError: The file exists but Ultralytics could not load it,
                which usually means it is corrupt or not a YOLO checkpoint.
        """
        if self._model is not None:
            return

        # check_model() already performed discovery; reuse its result rather
        # than scanning the models directory a second time.
        status = check_model(self._explicit_model_path)
        if not status.available or status.path is None:
            raise ResourceNotFoundError(status.detail)
        weights = status.path

        # Imported here rather than at module scope so that this file remains
        # importable (for --check, tests, and tooling) in an environment where
        # the heavy ML stack is not installed.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading model %s on device '%s'", weights.name, self.device)
        started = time.perf_counter()
        try:
            self._model = YOLO(str(weights))
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load {weights} as a YOLO model: {exc}"
            ) from exc

        self._model_path = weights
        logger.info(
            "Model ready in %.2fs — %d classes",
            time.perf_counter() - started,
            len(self.class_names),
        )

    @property
    def class_names(self) -> dict[int, str]:
        """Return the model's own ``{class_id: name}`` mapping.

        This is the authoritative source for what the model's outputs *mean*.
        A checkpoint carries the names it was trained with, so reading them
        here keeps the code correct for any dataset without configuration.

        Raises:
            ModelLoadError: If called before the model is loaded.
        """
        if self._model is None:
            raise ModelLoadError("Model is not loaded — call load() first.")
        return {int(k): str(v) for k, v in self._model.names.items()}

    # -- inference ----------------------------------------------------------
    def predict_image(
        self, image_path: Path, save_annotated: bool = True
    ) -> ImagePrediction:
        """Run the detector on a single image.

        Never raises for per-image problems (unreadable file, corrupt JPEG).
        Those are captured in :attr:`ImagePrediction.error` so that a batch of
        5,000 images is not aborted by one bad file.

        Args:
            image_path: Image to process.
            save_annotated: Whether to write a rendered copy into
                :data:`config.PREDICTIONS_DIR`.

        Returns:
            An :class:`ImagePrediction`, successful or carrying an error.
        """
        image_path = Path(image_path)
        prediction = ImagePrediction(image_path=image_path)

        if not image_path.is_file():
            prediction.error = f"File not found: {image_path}"
            return prediction

        try:
            self.load()
        except (ResourceNotFoundError, ModelLoadError) as exc:
            prediction.error = str(exc)
            return prediction

        try:
            started = time.perf_counter()
            # verbose=False suppresses Ultralytics' per-image console banner;
            # our own logging reports what we actually care about.
            results = self._model.predict(
                source=str(image_path),
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                max_det=config.MAX_DETECTIONS,
                device=self.device,
                verbose=False,
            )
            prediction.inference_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            prediction.error = f"Inference failed: {exc}"
            logger.error("Inference failed on %s: %s", image_path.name, exc)
            return prediction

        if not results:
            prediction.error = "Model returned no result object."
            return prediction

        result = results[0]
        height, width = result.orig_shape  # Ultralytics reports (h, w)
        prediction.image_width = int(width)
        prediction.image_height = int(height)
        prediction.detections = self._extract_detections(result)

        if save_annotated:
            prediction.annotated_path = self._save_annotated(result, image_path)

        return prediction

    def _extract_detections(self, result: Any) -> list[Detection]:
        """Convert one Ultralytics ``Results`` object into :class:`Detection`s.

        The three parallel tensors on ``result.boxes`` are read together:
        ``xyxy`` (geometry), ``conf`` (certainty), and ``cls`` (class index).
        Row *i* of each describes the same detection. We move them to CPU and
        to plain Python floats immediately so nothing downstream has to hold a
        GPU tensor or import torch.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = self.class_names
        coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # strict=True enforces the parallel-tensor invariant rather than
        # trusting it. Without it, three tensors of differing length would zip
        # to the shortest and silently discard detections — a data-corruption
        # bug with no error message. If Ultralytics ever violates the
        # alignment, we want a loud ValueError, not quiet wrong output.
        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, class_id in zip(
            coords, confidences, class_ids, strict=True
        ):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=names.get(int(class_id), f"id:{int(class_id)}"),
                    confidence=float(conf),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

        # Highest confidence first: the most consequential detections are the
        # ones a human reviewing the output should see at the top.
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _save_annotated(self, result: Any, image_path: Path) -> Path | None:
        """Write the rendered prediction image and return its path.

        We call ``result.plot()`` and write the array ourselves rather than
        using Ultralytics' ``save=True`` so that output lands in our configured
        ``results/predictions/`` tree instead of a library-chosen ``runs/``
        directory. Owning the output location keeps the project self-contained.

        Returns ``None`` on failure — a rendering problem should never destroy
        the numeric results we already computed.
        """
        try:
            import cv2

            config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            # `.plot()` returns a BGR array, which is exactly what cv2 expects.
            annotated = result.plot()
            output_path = config.PREDICTIONS_DIR / f"{image_path.stem}_pred.jpg"
            if not cv2.imwrite(str(output_path), annotated):
                logger.warning("cv2 could not write %s", output_path)
                return None
            return output_path
        except Exception as exc:
            logger.warning(
                "Could not save annotated image for %s: %s", image_path.name, exc
            )
            return None

    def predict_many(
        self, image_paths: Sequence[Path], save_annotated: bool = True
    ) -> Iterator[ImagePrediction]:
        """Run inference over many images, yielding results as they complete.

        A generator rather than a list so a caller can stream progress and so
        memory stays flat across a large dataset.

        Args:
            image_paths: Images to process.
            save_annotated: Whether to render each prediction to disk.

        Yields:
            One :class:`ImagePrediction` per input, in order.
        """
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            prediction = self.predict_image(image_path, save_annotated=save_annotated)
            if prediction.succeeded:
                logger.info(
                    "[%d/%d] %s — %d detection(s), %.0f ms",
                    index,
                    total,
                    image_path.name,
                    len(prediction.detections),
                    prediction.inference_ms,
                )
            else:
                logger.error(
                    "[%d/%d] %s — %s",
                    index,
                    total,
                    image_path.name,
                    prediction.error,
                )
            yield prediction

    # -- validation ---------------------------------------------------------
    def validate(
        self, data_yaml: Path | None = None, split: str = "val"
    ) -> ValidationMetrics:
        """Run the detector's built-in evaluation against labelled data.

        This is the *metrics* view (precision, recall, mAP) that Model Doctor
        is ultimately built to go beyond: it tells you how much the model is
        failing, never which images or why.

        Args:
            data_yaml: Dataset descriptor. Defaults to :data:`config.DATA_YAML_PATH`.
            split: Which split to evaluate.

        Returns:
            Normalised :class:`ValidationMetrics`. The library's own result
            object is deliberately not returned, so no caller becomes coupled
            to its shape.

        Raises:
            ResourceNotFoundError: If the descriptor is absent.
            ModelLoadError: If the model cannot be loaded.
        """
        self.load()

        yaml_path = Path(data_yaml) if data_yaml else config.DATA_YAML_PATH
        if not yaml_path.is_file():
            raise ResourceNotFoundError(
                f"Cannot validate: data.yaml not found at {yaml_path}"
            )

        config.ensure_output_dirs()
        logger.info("Validating on split '%s' using %s", split, yaml_path)
        raw = self._model.val(
            data=str(yaml_path),
            split=split,
            imgsz=self.image_size,
            device=self.device,
            project=str(config.VALIDATION_DIR),
            name=split,
            exist_ok=True,
            verbose=False,
        )
        return ValidationMetrics.from_raw(
            raw,
            class_names=self.class_names,
            artefacts_dir=config.VALIDATION_DIR / split,
        )


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------
def format_metrics(metrics: ValidationMetrics) -> str:
    """Render validation metrics as a readable report."""
    lines = ["", "Validation metrics", "-" * 64]

    overall = (
        ("Precision", metrics.precision),
        ("Recall", metrics.recall),
        ("mAP@50", metrics.map50),
        ("mAP@50-95", metrics.map50_95),
    )
    for label, value in overall:
        rendered = f"{value:.4f}" if value is not None else "not reported"
        lines.append(f"  {label:<12} {rendered}")

    if metrics.per_class:
        lines.append("")
        lines.append(
            f"  {'CLASS':<24} {'PREC':>7} {'RECALL':>7} {'mAP50':>7} {'mAP50-95':>9}"
        )
        for name, values in sorted(metrics.per_class.items()):
            lines.append(
                f"  {name:<24} "
                f"{values.get('precision', float('nan')):>7.3f} "
                f"{values.get('recall', float('nan')):>7.3f} "
                f"{values.get('map50', float('nan')):>7.3f} "
                f"{values.get('map50_95', float('nan')):>9.3f}"
            )

    if metrics.artefacts_dir is not None:
        lines.append("")
        lines.append(f"  Artefacts: {metrics.artefacts_dir}")
    lines.append("-" * 64)
    lines.append("")
    return "\n".join(lines)



def format_detections(prediction: ImagePrediction) -> str:
    """Render one image's detections as an aligned table."""
    header = f"\n{prediction.image_path.name}  "
    if not prediction.succeeded:
        return f"{header}\n  ERROR: {prediction.error}\n"

    meta = (
        f"({prediction.image_width}x{prediction.image_height}, "
        f"{prediction.inference_ms:.0f} ms)"
    )
    lines = [header + meta, "-" * 78]

    if not prediction.detections:
        lines.append("  No detections above the confidence threshold.")
        lines.append("-" * 78)
        return "\n".join(lines) + "\n"

    lines.append(
        f"  {'ID':>3}  {'CLASS':<20} {'CONF':>6}   "
        f"{'X1':>7} {'Y1':>7} {'X2':>7} {'Y2':>7}"
    )
    for det in prediction.detections:
        lines.append(
            f"  {det.class_id:>3}  {det.class_name:<20} {det.confidence:>6.3f}   "
            f"{det.x1:>7.1f} {det.y1:>7.1f} {det.x2:>7.1f} {det.y2:>7.1f}"
        )
    lines.append("-" * 78)
    if prediction.annotated_path:
        lines.append(f"  Saved: {prediction.annotated_path}")
    return "\n".join(lines) + "\n"


def summarise(predictions: Sequence[ImagePrediction]) -> str:
    """Render aggregate statistics for a batch run."""
    successful = [p for p in predictions if p.succeeded]
    failed = [p for p in predictions if not p.succeeded]
    total_detections = sum(len(p.detections) for p in successful)

    lines = ["", "Batch summary", "-" * 64, f"  Images processed : {len(successful)}"]
    lines.append(f"  Images failed    : {len(failed)}")
    lines.append(f"  Total detections : {total_detections}")

    if successful:
        mean_ms = sum(p.inference_ms for p in successful) / len(successful)
        lines.append(f"  Mean latency     : {mean_ms:.1f} ms/image")
        empty = sum(1 for p in successful if not p.detections)
        lines.append(f"  Images with none : {empty}")

        # A per-class tally is the first genuinely diagnostic signal available:
        # a class that never fires is already a lead worth chasing.
        tally: dict[str, int] = {}
        for prediction in successful:
            for det in prediction.detections:
                tally[det.class_name] = tally.get(det.class_name, 0) + 1
        if tally:
            lines.append("  Detections by class:")
            for name, count in sorted(tally.items(), key=lambda kv: -kv[1]):
                lines.append(f"      {name:<24} {count:>6}")

    for prediction in failed[:10]:
        lines.append(f"  ! {prediction.image_path.name}: {prediction.error}")
    if len(failed) > 10:
        lines.append(f"  ... and {len(failed) - 10} more failures")

    lines.append("-" * 64)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    """Turn CLI arguments into a concrete list of images to process."""
    if args.image:
        return [Path(args.image)]

    if args.source:
        source = Path(args.source)
        return find_images(source) if source.is_dir() else [source]

    # No explicit source: use the requested split from data.yaml.
    try:
        dataset: DatasetConfig = load_dataset_config()
    except DatasetConfigError as exc:
        logger.error("%s", exc)
        return []

    split_dir = dataset.splits.get(args.split)
    if split_dir is None:
        logger.error(
            "Split '%s' is not available. Declared splits: %s",
            args.split,
            ", ".join(dataset.splits) or "none",
        )
        return []
    return find_images(split_dir)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="model-doctor-inference",
        description="Run YOLO inference and report structured detections.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify resources and exit."
    )
    parser.add_argument("--image", type=str, help="Run on a single image.")
    parser.add_argument("--source", type=str, help="Run on a file or directory.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to run on (default: test).",
    )
    parser.add_argument("--model", type=str, help="Path to specific .pt weights.")
    parser.add_argument("--conf", type=float, help="Confidence threshold override.")
    parser.add_argument(
        "--device", type=str, help="Device override, e.g. cpu, mps, cuda."
    )
    parser.add_argument("--limit", type=int, help="Process at most N images.")
    parser.add_argument(
        "--no-save", action="store_true", help="Skip writing annotated images."
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run model.val() and exit."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns a process exit code rather than calling ``sys.exit`` directly, so
    the function stays callable from tests.

    Returns:
        ``0`` on success, ``1`` when required resources are missing, ``2`` when
        an operation failed.
    """
    args = build_parser().parse_args(argv)
    config.ensure_output_dirs()

    model_override = Path(args.model) if args.model else None
    statuses = verify_all(model_override)
    print(format_report(statuses))

    if args.check:
        # When a dataset descriptor is present, show what was actually parsed.
        # Seeing the discovered class list and per-split image counts is how a
        # user confirms the dataset is wired up as they intended.
        # Absence is already reported in the status table above, so there is
        # nothing to add here when the descriptor is missing.
        with contextlib.suppress(DatasetConfigError):
            print(describe(load_dataset_config()))
        return 0 if all(s.available for s in statuses if s.required) else 1

    # Everything past this point needs a model. Fail fast and clearly.
    model_status = next(s for s in statuses if s.name == "Trained model")
    if not model_status.available:
        logger.error("Cannot run inference: %s", model_status.detail)
        return 1

    detector = Detector(
        model_path=model_override, device=args.device, confidence=args.conf
    )

    if args.validate:
        try:
            metrics = detector.validate()
        except (ResourceNotFoundError, ModelLoadError) as exc:
            logger.error("%s", exc)
            return 1
        print(format_metrics(metrics))
        return 0

    images = _resolve_inputs(args)
    if not images:
        logger.error("No images to process.")
        return 1
    if args.limit:
        images = images[: args.limit]

    logger.info("Running inference on %d image(s)", len(images))
    predictions = list(detector.predict_many(images, save_annotated=not args.no_save))

    # Print full detail for a single image; a table would be unreadable for
    # hundreds, so batches get the aggregate view instead.
    if len(predictions) == 1:
        print(format_detections(predictions[0]))
    print(summarise(predictions))

    return 0 if any(p.succeeded for p in predictions) else 2


if __name__ == "__main__":
    raise SystemExit(main())
