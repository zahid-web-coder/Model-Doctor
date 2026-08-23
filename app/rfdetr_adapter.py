"""RF-DETR segmentation behind Model Doctor's :class:`~app.inference.Detector`.

The point of this module is that **nothing downstream changes**. Matching,
the outcome taxonomy, storage, root-cause analysis and the schema all consume
:class:`~app.inference.Detection` objects and have no idea which library
produced them — which is exactly what ``app/inference.py``'s docstring promised
when it said swapping detectors "means reimplementing this file and nothing
else". This is that reimplementation, as a subclass rather than an edit, so the
YOLO path is not touched at all.

Three things genuinely differ from Ultralytics and are handled here:

* **Loading.** RF-DETR ships a Lightning checkpoint and its own
  ``from_checkpoint`` constructor rather than a callable model class.
* **Result shape.** It returns a ``supervision.Detections`` with boolean raster
  masks, where Ultralytics returns polygons. Model Doctor's annotation model is
  polygon-based, so the raster is traced back to a contour here — see
  :func:`_mask_to_polygon` for why that is lossy in one specific way.
* **Evaluation.** RF-DETR has no ``.val()``. :meth:`validate` therefore refuses
  rather than inventing something, because a metric computed by a different
  evaluator than YOLO's is not comparable to YOLO's, and a silently
  incomparable number is worse than no number.

Class ids are read from the checkpoint's own ``class_names``, never hardcoded,
exactly as the YOLO path reads ``model.names``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import config
from app.inference import Detection, Detector, ImagePrediction, ValidationMetrics
from utils.exceptions import ModelLoadError, ResourceNotFoundError
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# A contour needs three points to bound an area. Below that the "polygon" is a
# point or a line segment, which has no interior and would make every mask IoU
# against it zero — worse than admitting there is no usable outline.
MIN_POLYGON_POINTS = 3

# RF-DETR requires a square resolution divisible by `patch_size * num_windows`.
# For the Nano segmentation variant that product is 12.
RESOLUTION_MULTIPLE = 12


def _mask_to_polygon(mask: np.ndarray) -> list[list[float]] | None:
    """Trace a boolean raster mask back to a single polygon outline.

    Model Doctor's annotation model carries one polygon per object, so a mask
    that resolves into several disconnected blobs — a column split by an
    occluding beam — cannot be represented faithfully. **The largest component
    is kept and the rest are dropped**, which is the same simplification the
    dataset's own YOLO export made, so predictions and ground truth are at
    least lossy in the same direction.

    Returns ``None`` rather than an empty list when nothing traceable is found,
    preserving the distinction the codebase already draws between "this model
    does not segment" and "it segmented nothing here".
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv is a hard dependency
        raise ModelLoadError("opencv-python is required to trace masks") from exc

    binary = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2)
    if len(points) < MIN_POLYGON_POINTS:
        return None
    return [[float(x), float(y)] for x, y in points]


class RFDetrDetector(Detector):
    """Loads an RF-DETR segmentation checkpoint and runs inference with it.

    Substitutable for :class:`~app.inference.Detector` wherever a detector is
    expected. Callers see the same ``load`` / ``class_names`` / ``predict_image``
    / ``predict_many`` surface and the same :class:`Detection` objects.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._class_names: dict[int, str] = {}

    # -- loading ------------------------------------------------------------
    def load(self) -> None:
        """Load the RF-DETR weights into memory.

        Idempotent, like the base implementation.

        Raises:
            ResourceNotFoundError: The checkpoint path does not exist.
            ModelLoadError: ``rfdetr`` is missing, or the file is not an
                RF-DETR checkpoint.
        """
        if self._model is not None:
            return

        weights = Path(self._explicit_model_path) if self._explicit_model_path else None
        if weights is None or not weights.is_file():
            raise ResourceNotFoundError(
                f"RF-DETR checkpoint not found: {weights}. Pass an explicit path."
            )

        # Imported lazily for the same reason the YOLO path does it: this
        # module must stay importable where the heavy stack is absent.
        try:
            from rfdetr import RFDETRSegNano
        except ImportError as exc:
            raise ModelLoadError(
                "rfdetr is not installed. Run: pip install rfdetr==1.8.3"
            ) from exc

        # Resolution is settable, and the checkpoint's training value is not
        # binding at inference — RF-DETR interpolates its positional encodings.
        # It must be a multiple of `patch_size * num_windows`, which is 12 for
        # this variant; an illegal value fails inside the backbone with a shape
        # error that says nothing about resolution, so it is checked here.
        kwargs: dict[str, Any] = {}
        if self.image_size:
            if self.image_size % RESOLUTION_MULTIPLE:
                raise ModelLoadError(
                    f"RF-DETR resolution must be a multiple of "
                    f"{RESOLUTION_MULTIPLE}; got {self.image_size}."
                )
            kwargs["resolution"] = int(self.image_size)

        logger.info(
            "Loading RF-DETR checkpoint %s%s",
            weights.name,
            f" at resolution {kwargs['resolution']}" if kwargs else " at its native resolution",
        )
        started = time.perf_counter()
        try:
            self._model = RFDETRSegNano.from_checkpoint(str(weights), **kwargs)
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load {weights} as an RF-DETR checkpoint: {exc}"
            ) from exc

        # The checkpoint is the authority on what its own class ids mean, the
        # same rule the YOLO path follows. RF-DETR exposes an ordered list
        # where Ultralytics exposes a mapping; index is the class id.
        names = list(getattr(self._model, "class_names", []) or [])
        self._class_names = {index: str(name) for index, name in enumerate(names)}

        self._model_path = weights
        logger.info(
            "RF-DETR ready in %.2fs — %d classes",
            time.perf_counter() - started,
            len(self._class_names),
        )

    @property
    def class_names(self) -> dict[int, str]:
        """Return the checkpoint's own ``{class_id: name}`` mapping."""
        if self._model is None:
            raise ModelLoadError("Model is not loaded — call load() first.")
        return dict(self._class_names)

    # -- inference ----------------------------------------------------------
    def predict_image(
        self, image_path: Path, save_annotated: bool = False
    ) -> ImagePrediction:
        """Run RF-DETR on a single image.

        Mirrors the base contract exactly: per-image failures are captured on
        the returned :class:`ImagePrediction` rather than raised, so one bad
        file cannot abort a batch.

        ``save_annotated`` defaults to ``False`` here because rendering is
        Ultralytics' ``result.plot()`` in the base class and RF-DETR has no
        equivalent; passing ``True`` is accepted and ignored rather than
        pretending to write a file that will not exist.
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
            from PIL import Image

            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
                prediction.image_width, prediction.image_height = image.size

                started = time.perf_counter()
                detections = self._model.predict(image, threshold=self.confidence)
                prediction.inference_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            prediction.error = f"Inference failed: {exc}"
            logger.error("Inference failed on %s: %s", image_path.name, exc)
            return prediction

        prediction.detections = self._extract_detections(detections)
        return prediction

    def _extract_detections(self, result: Any) -> list[Detection]:
        """Convert one ``supervision.Detections`` into :class:`Detection`s.

        The parallel arrays ``xyxy`` / ``confidence`` / ``class_id`` describe
        the same detection at row *i*, and ``mask[i]`` is that detection's
        boolean raster at full image resolution. Boxes are already in original
        image pixels, the same space as ground truth, so nothing is rescaled.
        """
        boxes = getattr(result, "xyxy", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = self.class_names
        confidences = getattr(result, "confidence", None)
        class_ids = getattr(result, "class_id", None)
        masks = getattr(result, "mask", None)

        if confidences is None or class_ids is None:
            logger.warning("RF-DETR returned boxes without confidence or class; skipped")
            return []

        detections: list[Detection] = []
        for index, (box, conf, class_id) in enumerate(
            zip(boxes, confidences, class_ids, strict=True)
        ):
            outline = None
            if masks is not None and index < len(masks):
                outline = _mask_to_polygon(masks[index])

            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=names.get(int(class_id), f"id:{int(class_id)}"),
                    confidence=float(conf),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    polygon=outline,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # -- evaluation ---------------------------------------------------------
    def validate(
        self, data_yaml: Path | None = None, split: str = "val"
    ) -> ValidationMetrics:
        """Refuse to evaluate, and say why.

        The base implementation delegates to Ultralytics' ``model.val()``.
        RF-DETR has no equivalent, and wiring in a *different* evaluator here
        would produce numbers that look like YOLO's and are not comparable to
        them — different matching rules, different NMS, different area
        thresholds. That is precisely the mistake this comparison exists to
        avoid, so it fails loudly instead.

        Both models are scored together, by one evaluator, in
        ``scripts/compare_detectors.py``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "RF-DETR has no built-in validator. Evaluate both models with the "
            "shared evaluator in scripts/compare_detectors.py so the numbers "
            "are comparable."
        )


__all__ = ["RFDetrDetector"]
