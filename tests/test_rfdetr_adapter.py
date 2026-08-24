"""Tests for the RF-DETR adapter and detector selection.

The adapter's job is to make a second library produce the same
:class:`~app.inference.Detection` objects as the first, so the parts worth
testing are the conversions and the boundaries — not the network.

Following the pattern in ``test_metrics.py``: the result shape is read off a
plain object, so a stub stands in for ``supervision.Detections`` and no model,
dataset, GPU or ``rfdetr`` install is required. The one test that genuinely
needs opencv skips itself when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from app.detectors import (
    SUPPORTED_FAMILIES,
    build_detector,
    default_image_size,
    detect_family,
)
from app.inference import Detector
from app.rfdetr_adapter import RESOLUTION_MULTIPLE, RFDetrDetector, _mask_to_polygon
from utils.exceptions import ModelLoadError


class _Detections:
    """Minimal stand-in for ``supervision.Detections``."""

    def __init__(self, xyxy, confidence, class_id, mask=None) -> None:
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.mask = mask


def _loaded(names: dict[int, str] | None = None) -> RFDetrDetector:
    """An adapter with its class map populated, without loading weights."""
    detector = RFDetrDetector(model_path="unused.pt")
    detector._model = object()  # marks it loaded for class_names' guard
    detector._class_names = names if names is not None else {0: "column"}
    return detector


# ---------------------------------------------------------------------------
# Detection extraction
# ---------------------------------------------------------------------------
def test_extracts_boxes_confidence_and_class() -> None:
    """Parallel arrays at row i describe the same detection."""
    detector = _loaded()
    result = _Detections(
        xyxy=[[10.0, 20.0, 30.0, 40.0]], confidence=[0.9], class_id=[0]
    )

    detections = detector._extract_detections(result)

    assert len(detections) == 1
    only = detections[0]
    assert (only.x1, only.y1, only.x2, only.y2) == (10.0, 20.0, 30.0, 40.0)
    assert only.confidence == pytest.approx(0.9)
    assert only.class_id == 0
    assert only.class_name == "column"


def test_no_boxes_yields_no_detections() -> None:
    """An image with nothing found returns an empty list, not an error."""
    detector = _loaded()
    assert detector._extract_detections(_Detections([], [], [])) == []


def test_detections_are_sorted_by_confidence() -> None:
    """Most consequential detections come first, as the YOLO path does."""
    detector = _loaded()
    result = _Detections(
        xyxy=[[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]],
        confidence=[0.2, 0.9, 0.5],
        class_id=[0, 0, 0],
    )

    confidences = [d.confidence for d in detector._extract_detections(result)]

    assert confidences == sorted(confidences, reverse=True)


def test_unknown_class_id_does_not_crash() -> None:
    """An id outside the checkpoint's map degrades to a readable label."""
    detector = _loaded({0: "column"})
    result = _Detections(xyxy=[[0, 0, 1, 1]], confidence=[0.5], class_id=[7])

    assert detector._extract_detections(result)[0].class_name == "id:7"


def test_missing_confidence_is_refused_not_guessed() -> None:
    """Boxes without confidence are dropped rather than given a made-up one.

    ``Detection`` requires a confidence by construction, so inventing one to
    satisfy it would put a fabricated number into the analysis.
    """
    detector = _loaded()
    result = _Detections(xyxy=[[0, 0, 1, 1]], confidence=None, class_id=[0])

    assert detector._extract_detections(result) == []


def test_mismatched_array_lengths_raise() -> None:
    """Misaligned parallel arrays fail loudly rather than zipping short.

    Silently truncating would discard detections with no error anywhere, which
    is a data-corruption bug rather than a visible one.
    """
    detector = _loaded()
    result = _Detections(
        xyxy=[[0, 0, 1, 1], [0, 0, 2, 2]], confidence=[0.5], class_id=[0]
    )

    with pytest.raises(ValueError):
        detector._extract_detections(result)


# ---------------------------------------------------------------------------
# Mask tracing
# ---------------------------------------------------------------------------
def test_mask_traces_to_a_polygon() -> None:
    """A solid rectangle in a raster mask becomes a usable outline."""
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    mask = numpy.zeros((40, 40), dtype=bool)
    mask[10:30, 5:25] = True

    polygon = _mask_to_polygon(mask)

    assert polygon is not None
    assert len(polygon) >= 3
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    assert min(xs) == pytest.approx(5, abs=1)
    assert max(xs) == pytest.approx(24, abs=1)
    assert min(ys) == pytest.approx(10, abs=1)
    assert max(ys) == pytest.approx(29, abs=1)


def test_empty_mask_gives_no_polygon() -> None:
    """Nothing traceable returns None — absent, not an empty outline.

    The codebase distinguishes "does not segment" from "segmented nothing";
    an empty list would collapse the two.
    """
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    assert _mask_to_polygon(numpy.zeros((20, 20), dtype=bool)) is None


def test_largest_component_wins_for_split_masks() -> None:
    """A mask in two pieces keeps the larger, matching the dataset's own export.

    Model Doctor's annotation model carries one polygon per object, so a
    multi-part mask cannot be represented faithfully either way.
    """
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    mask = numpy.zeros((60, 60), dtype=bool)
    mask[5:10, 5:10] = True      # small blob
    mask[20:50, 20:50] = True    # large blob

    polygon = _mask_to_polygon(mask)

    assert polygon is not None
    xs = [point[0] for point in polygon]
    assert min(xs) >= 19  # the small blob at x=5 was not the one kept


# ---------------------------------------------------------------------------
# Resolution guard
# ---------------------------------------------------------------------------
def test_illegal_resolution_is_rejected_with_a_useful_message() -> None:
    """A resolution off the multiple fails here, not inside the backbone.

    Left to the backbone it surfaces as a tensor shape mismatch that says
    nothing about resolution. This guard caught a real bug on its first run:
    the comparison harness was not passing a resolution at all.
    """
    detector = RFDetrDetector(model_path="unused.pt", image_size=500)

    with pytest.raises(ModelLoadError, match=f"multiple of {RESOLUTION_MULTIPLE}"):
        detector.load()


def test_missing_checkpoint_reports_the_path() -> None:
    """A missing file is a clear message, never a traceback."""
    from utils.exceptions import ResourceNotFoundError

    detector = RFDetrDetector(model_path="does/not/exist.pt", image_size=480)

    with pytest.raises(ResourceNotFoundError, match="does/not/exist.pt"):
        detector.load()


# ---------------------------------------------------------------------------
# Evaluation boundary
# ---------------------------------------------------------------------------
def test_validate_refuses_rather_than_inventing_a_number() -> None:
    """RF-DETR has no built-in validator, and a different one is not comparable.

    Returning a metric produced by another evaluator would look like YOLO's and
    silently not be comparable to it, which is worse than no metric.
    """
    detector = RFDetrDetector(model_path="unused.pt", image_size=480)

    with pytest.raises(NotImplementedError, match="comparable"):
        detector.validate()


# ---------------------------------------------------------------------------
# Detector selection
# ---------------------------------------------------------------------------
def test_default_family_is_yolo() -> None:
    """YOLO stays the default: it is the only CPU-viable option of the two."""
    assert config.DETECTOR_FAMILY == "yolo"
    assert isinstance(build_detector(), Detector)
    assert not isinstance(build_detector(), RFDetrDetector)


def test_rfdetr_can_be_selected_explicitly() -> None:
    """Asking for RF-DETR gets RF-DETR, without changing any default."""
    detector = build_detector("rfdetr", model_path="unused.pt")
    assert isinstance(detector, RFDetrDetector)


def test_family_selection_is_case_insensitive() -> None:
    """Configuration read from the environment should not be case-sensitive."""
    assert isinstance(build_detector("RFDETR", model_path="x.pt"), RFDetrDetector)


def test_unknown_family_is_refused() -> None:
    """An unsupported name fails immediately, listing what is supported."""
    with pytest.raises(ModelLoadError, match="Unknown detector family"):
        build_detector("detectron2")


def test_each_family_gets_its_own_resolution() -> None:
    """The two models want different input sizes and must not share one.

    Sharing would silently mis-size whichever model was not being tuned, and
    mis-sizing degrades accuracy without raising anything.
    """
    assert default_image_size("rfdetr") == config.RFDETR_IMAGE_SIZE
    assert default_image_size("yolo") == config.IMAGE_SIZE
    rf = build_detector("rfdetr", model_path="x.pt")
    assert rf.image_size == config.RFDETR_IMAGE_SIZE
    assert build_detector("yolo").image_size == config.IMAGE_SIZE


def test_explicit_image_size_overrides_the_family_default() -> None:
    """A caller sweeping resolutions must be able to say what it wants."""
    assert build_detector("rfdetr", model_path="x.pt", image_size=672).image_size == 672


def test_rfdetr_default_resolution_is_the_measured_knee() -> None:
    """480 is the efficiency knee measured on the columns sweep, not a guess."""
    assert config.RFDETR_IMAGE_SIZE == 480
    assert config.RFDETR_IMAGE_SIZE % RESOLUTION_MULTIPLE == 0


def test_supported_families_are_the_two_documented_ones() -> None:
    """Adding a family should be a deliberate change, visible in the diff."""
    assert set(SUPPORTED_FAMILIES) == {"yolo", "rfdetr"}


# ---------------------------------------------------------------------------
# Family detection — why no schema column was needed
# ---------------------------------------------------------------------------
def test_family_is_read_from_the_checkpoint(tmp_path) -> None:
    """An RF-DETR checkpoint identifies itself; a YOLO one does not.

    This is what let the run pipeline stay on the existing schema: a run
    already records its weights, and the weights already say what they are, so
    a `detector_family` column would have duplicated a stored fact — and two
    copies of a fact can disagree.
    """
    torch = pytest.importorskip("torch")

    rf = tmp_path / "rf.pt"
    torch.save({"rfdetr_version": "1.8.3", "state_dict": {}}, rf)
    assert detect_family(rf) == "rfdetr"

    yolo = tmp_path / "y.pt"
    torch.save({"epoch": 1, "state_dict": {}}, yolo)
    assert detect_family(yolo) == "yolo"


def test_family_of_a_missing_file_falls_back_to_the_default() -> None:
    """A path that is not there defers to the loader's own error, not ours."""
    assert detect_family("no/such/file.pt") == "yolo"


def test_family_of_an_unreadable_file_does_not_raise(tmp_path) -> None:
    """A corrupt checkpoint must not crash family detection.

    It should fail later, in the loader, with a message about the checkpoint —
    not here, with one about family detection.
    """
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    assert detect_family(junk) == "yolo"


# ---------------------------------------------------------------------------
# The mask pass must reproduce the run it is re-examining
# ---------------------------------------------------------------------------
def test_mask_pass_defaults_to_the_run_s_own_checkpoint(monkeypatch, tmp_path) -> None:
    """With no detector injected, the run's own weights and family are used.

    This pass re-runs inference and matches outlines back to findings already
    stored, so it must reproduce that run's predictions. The previous default
    built a YOLO detector by discovery — scanning the models directory — which
    for a run made with any other checkpoint would measure outlines belonging
    to predictions the run never made, and report them as its own.
    """
    torch = pytest.importorskip("torch")
    from app import mask_diagnosis

    weights = tmp_path / "rf.pt"
    torch.save({"rfdetr_version": "1.8.3", "state_dict": {}}, weights)

    class _Run:
        model_path = str(weights)
        image_size = 480

    class _StopError(Exception):
        """Halt the pass once the detector has been chosen.

        Nothing beyond that choice is under test here, and stopping avoids
        loading real weights.
        """

    built: dict[str, object] = {}

    def _spy(family=None, *, model_path=None, image_size=None, **_kwargs):
        built.update(family=family, model_path=model_path, image_size=image_size)
        raise _StopError

    monkeypatch.setattr(mask_diagnosis.storage, "load_run", lambda _c, _r: _Run())
    monkeypatch.setattr(
        mask_diagnosis.storage,
        "load_findings_for_embedding",
        lambda _c, _r, **_kwargs: [{"finding_id": 1}],
    )
    monkeypatch.setattr(mask_diagnosis, "build_detector", _spy)

    with pytest.raises(_StopError):
        mask_diagnosis.diagnose_masks(object(), 1)

    assert built["family"] == "rfdetr"
    assert built["model_path"] == str(weights)
    assert built["image_size"] == 480


# ---------------------------------------------------------------------------
# Heatmaps must refuse for RF-DETR, however they are invoked
# ---------------------------------------------------------------------------
def test_heatmaps_refuse_for_an_rfdetr_run_without_a_model_flag(
    monkeypatch, tmp_path
) -> None:
    """The refusal reads the run's checkpoint, not the ``--model`` flag.

    Checking the flag alone left a hole found in review: asking for heatmaps on
    an RF-DETR run without ``--model`` fell through to weight discovery, found
    the YOLO checkpoint, and attached YOLO Grad-CAM maps to RF-DETR findings.
    Grad-CAM against the wrong architecture does not error — it produces a map
    that means nothing, which is worse than none because it gets believed.
    """
    torch = pytest.importorskip("torch")
    from app import explainability

    weights = tmp_path / "rf.pt"
    torch.save({"rfdetr_version": "1.8.3", "state_dict": {}}, weights)

    class _Run:
        id = 1
        model_path = str(weights)

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(explainability.storage, "connect", lambda _p: _Conn())
    monkeypatch.setattr(explainability.storage, "load_run", lambda _c, _r: _Run())

    # No --model: the family must still be resolved from the run itself.
    exit_code = explainability.main(["--run", "1"])

    assert exit_code == 1, "expected a refusal, not a generated heatmap"
