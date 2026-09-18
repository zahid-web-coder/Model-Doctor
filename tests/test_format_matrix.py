"""The four combinations Ramanujan can produce, each read on its own terms.

A model is analysed with its own family's inference and its dataset in its own
native format. Nothing is converted on disk, and no combination borrows another
one's reader:

    YOLO detection      + YOLO dataset  -> YOLO inference,   YOLO labels
    YOLO segmentation   + YOLO dataset  -> YOLO inference,   YOLO labels
    RF-DETR detection   + COCO dataset  -> RF-DETR inference, COCO annotations
    RF-DETR segmentation+ COCO dataset  -> RF-DETR inference, COCO annotations

The detector is a stub in every case. What is under test is the seam between a
family's predictions and a format's ground truth — not the networks, which are
the libraries' business and need a GPU to say anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor.app.diagnosis import Outcome, diagnose_split
from model_doctor.app.inference import Detection, ImagePrediction
from model_doctor.utils.dataset import load_dataset_config
from model_doctor.utils.exceptions import DatasetConfigError

SIZE = (100, 100)
#: A box and an outline covering the same square, so a paired prediction and
#: annotation agree whichever geometry the pair happens to carry.
SQUARE = (20.0, 20.0, 60.0, 60.0)
RING = [20, 20, 60, 20, 60, 60, 20, 60]


class _StubDetector:
    """A detector of a stated family that returns whatever it is given."""

    def __init__(self, detections, *, segments: bool, names):
        self._detections = detections
        self.segments = segments
        self.class_names = names
        self.image_size = 448
        self.loaded = False

    def load(self):
        self.loaded = True

    def predict_image(self, image_path, save_annotated=False):  # noqa: ARG002
        prediction = ImagePrediction(image_path=Path(image_path))
        prediction.image_width, prediction.image_height = SIZE
        prediction.detections = list(self._detections)
        return prediction


def _detection(class_id, name, *, polygon=None):
    x1, y1, x2, y2 = SQUARE
    return Detection(
        class_id=class_id, class_name=name, confidence=0.9,
        x1=x1, y1=y1, x2=x2, y2=y2,
        polygon=[[p, q] for p, q in zip(polygon[::2], polygon[1::2])]
        if polygon else None,
    )


def _yolo_dataset(root: Path, *, segmentation: bool) -> Path:
    """A YOLO dataset whose single label is a box or an outline."""
    images = root / "test" / "images"
    labels = root / "test" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", SIZE).save(images / "a.jpg")
    if segmentation:
        ring = " ".join(f"{v / 100:.3f}" for v in RING)
        (labels / "a.txt").write_text(f"0 {ring}\n")
    else:
        (labels / "a.txt").write_text("0 0.4 0.4 0.4 0.4\n")
    (root / "data.yaml").write_text(
        "names:\n  0: column\n  1: beam\ntest: test/images\n"
    )
    return root / "data.yaml"


def _coco_dataset(root: Path, *, segmentation: bool) -> Path:
    """A Roboflow-shaped COCO dataset, with the placeholder descriptor
    Ramanujan generates for one."""
    split = root / "test"
    split.mkdir(parents=True)
    Image.new("RGB", SIZE).save(split / "a.jpg")
    annotation = {
        "id": 1, "image_id": 7, "category_id": 4,
        "bbox": [20, 20, 40, 40], "area": 1600.0, "iscrowd": 0,
    }
    if segmentation:
        annotation["segmentation"] = [RING]
    (split / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 7, "file_name": "a.jpg",
                    "width": SIZE[0], "height": SIZE[1]}],
        # Ids that are neither contiguous nor zero-based, as a real export's are.
        "categories": [
            {"id": 4, "name": "column", "supercategory": "none"},
            {"id": 11, "name": "beam", "supercategory": "none"},
        ],
        "annotations": [annotation],
    }))
    (root / "data.yaml").write_text("names:\n  0: object\ntest: test\n")
    return root / "data.yaml"


NAMES = {0: "column", 1: "beam"}


class TestTheFourCombinations:
    """Each one produces a correct finding, from its own family and format."""

    def _run(self, yaml_path, detector):
        dataset = load_dataset_config(yaml_path)
        return dataset, diagnose_split(detector, dataset, "test")

    def test_yolo_detection_on_a_yolo_dataset(self, tmp_path):
        yaml_path = _yolo_dataset(tmp_path / "ds", segmentation=False)
        detector = _StubDetector(
            [_detection(0, "column")], segments=False, names=NAMES
        )
        dataset, summary = self._run(yaml_path, detector)

        assert dataset.class_names == NAMES
        findings = summary.succeeded[0].findings
        assert [f.outcome for f in findings] == [Outcome.CORRECT]
        assert findings[0].truth.polygon is None, "a box dataset yields boxes"

    def test_yolo_segmentation_on_a_yolo_dataset(self, tmp_path):
        yaml_path = _yolo_dataset(tmp_path / "ds", segmentation=True)
        detector = _StubDetector(
            [_detection(0, "column", polygon=RING)], segments=True, names=NAMES
        )
        _dataset, summary = self._run(yaml_path, detector)

        findings = summary.succeeded[0].findings
        assert [f.outcome for f in findings] == [Outcome.CORRECT]
        assert findings[0].truth.polygon is not None, "outlines survive"
        assert findings[0].prediction.polygon is not None

    def test_rfdetr_detection_on_a_coco_dataset(self, tmp_path):
        yaml_path = _coco_dataset(tmp_path / "ds", segmentation=False)
        detector = _StubDetector(
            [_detection(0, "column")], segments=False, names=NAMES
        )
        dataset, summary = self._run(yaml_path, detector)

        assert dataset.class_names == NAMES, "read from the annotations, not the yaml"
        findings = summary.succeeded[0].findings
        assert [f.outcome for f in findings] == [Outcome.CORRECT]
        assert findings[0].truth.class_name == "column"
        assert findings[0].truth.polygon is None

    def test_rfdetr_segmentation_on_a_coco_dataset(self, tmp_path):
        yaml_path = _coco_dataset(tmp_path / "ds", segmentation=True)
        detector = _StubDetector(
            [_detection(0, "column", polygon=RING)], segments=True, names=NAMES
        )
        _dataset, summary = self._run(yaml_path, detector)

        findings = summary.succeeded[0].findings
        assert [f.outcome for f in findings] == [Outcome.CORRECT]
        assert findings[0].truth.polygon is not None, (
            "COCO segmentation reaches diagnosis as an outline, not a box"
        )
        assert len(findings[0].truth.polygon) == 4


class TestTheSeamsThatCouldGoWrongQuietly:
    def _run(self, yaml_path, detector):
        dataset = load_dataset_config(yaml_path)
        return dataset, diagnose_split(detector, dataset, "test")

    def test_a_coco_dataset_is_never_read_as_an_empty_yolo_one(self, tmp_path):
        """The failure this whole layer exists to prevent."""
        yaml_path = _coco_dataset(tmp_path / "ds", segmentation=True)
        detector = _StubDetector(
            [_detection(0, "column")], segments=False, names=NAMES
        )
        _dataset, summary = self._run(yaml_path, detector)
        outcomes = [f.outcome for f in summary.succeeded[0].findings]
        assert Outcome.FALSE_POSITIVE not in outcomes, (
            "ground truth was found, so the prediction is not spurious"
        )

    def test_a_split_with_no_annotations_at_all_refuses(self, tmp_path):
        root = tmp_path / "ds"
        split = root / "test"
        split.mkdir(parents=True)
        Image.new("RGB", SIZE).save(split / "a.jpg")
        (root / "data.yaml").write_text("names:\n  0: column\ntest: test\n")
        dataset = load_dataset_config(root / "data.yaml")
        detector = _StubDetector([], segments=False, names=NAMES)

        with pytest.raises(DatasetConfigError, match="neither layout"):
            diagnose_split(detector, dataset, "test")

    def test_a_class_name_mismatch_stops_the_run(self, tmp_path):
        """Every finding would carry the wrong class, and the run would succeed."""
        yaml_path = _coco_dataset(tmp_path / "ds", segmentation=False)
        detector = _StubDetector(
            [_detection(0, "slab")], segments=False, names={0: "slab", 1: "beam"}
        )
        dataset = load_dataset_config(yaml_path)
        with pytest.raises(DatasetConfigError, match="not this model's classes"):
            diagnose_split(detector, dataset, "test")

    def test_the_reader_used_is_reported(self, tmp_path):
        """A run should be able to say which format it read."""
        from model_doctor.utils.ground_truth import open_ground_truth

        coco = load_dataset_config(_coco_dataset(tmp_path / "c", segmentation=False))
        yolo = load_dataset_config(_yolo_dataset(tmp_path / "y", segmentation=False))
        assert open_ground_truth(coco.splits["test"]).format == "coco"
        assert open_ground_truth(yolo.splits["test"], yolo.class_names).format == "yolo"


class TestTheGuardIsReachable:
    """The class-name check must run against a real detector, not only a stub.

    A detector does not know its class names until it is loaded, and every
    caller in the pipeline hands `diagnose_split` an unloaded one. Reading the
    names before loading raised, the check was skipped, and a mismatch went
    through unnoticed — while a stub that exposes `class_names` as a plain
    attribute made the test suite say otherwise.
    """

    class _RealisticDetector(_StubDetector):
        """Refuses its class names until loaded, exactly as the real ones do."""

        def __init__(self, detections, *, names):
            super().__init__(detections, segments=False, names=names)
            self._names = names
            self.loaded = False

        def load(self):
            self.loaded = True

        @property
        def class_names(self):
            if not self.loaded:
                from model_doctor.utils.exceptions import ModelLoadError

                raise ModelLoadError("Model is not loaded — call load() first.")
            return self._names

        @class_names.setter
        def class_names(self, value):
            self._names = value

    def test_a_mismatch_is_caught_on_an_unloaded_detector(self, tmp_path):
        yaml_path = _coco_dataset(tmp_path / "ds", segmentation=False)
        dataset = load_dataset_config(yaml_path)
        detector = self._RealisticDetector(
            [_detection(0, "slab")], names={0: "slab", 1: "beam"}
        )
        assert not detector.loaded, "the pipeline hands over an unloaded detector"

        with pytest.raises(DatasetConfigError, match="not this model's classes"):
            diagnose_split(detector, dataset, "test")

    def test_matching_names_still_run(self, tmp_path):
        yaml_path = _coco_dataset(tmp_path / "ds", segmentation=False)
        dataset = load_dataset_config(yaml_path)
        detector = self._RealisticDetector([_detection(0, "column")], names=NAMES)

        summary = diagnose_split(detector, dataset, "test")
        assert detector.loaded, "the guard loads the model rather than skipping"
        assert len(summary.succeeded) == 1
