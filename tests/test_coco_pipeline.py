"""A COCO dataset through the applicable pipeline, end to end.

This is the acceptance shape for an RF-DETR segmentation model on the dataset
it was trained against: diagnose, persist, evaluate, and measure outlines —
reading the annotation file directly at every step.

The network is a stub, because what is under test is the seam between a
family's predictions and a format's ground truth. Everything else is real: the
same ``diagnose_split``, the same SQLite schema, the same mask pass, the same
evaluator.

The property this file exists to hold is the negative one. **No YOLO label file
is ever written.** A conversion would make the analysis partly a description of
the conversion, and it would leave files in a directory the user did not ask
anyone to write to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor.app import storage
from model_doctor.app.diagnosis import diagnose_split
from model_doctor.app.inference import Detection, ImagePrediction
from model_doctor.utils.dataset import load_dataset_config
from model_doctor.utils.ground_truth import open_ground_truth

SIZE = (200, 160)
CLASSES = {0: "column", 1: "beam"}
#: Two objects on each image, so a pairing that silently collapsed would show.
OBJECTS = [
    # (category_id, x, y, w, h)
    (4, 20.0, 20.0, 60.0, 60.0),
    (11, 110.0, 40.0, 50.0, 70.0),
]


def _ring(x, y, w, h):
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _dataset(root: Path, images=("a.jpg", "b.jpg")) -> Path:
    """A Roboflow-shaped COCO dataset, with Ramanujan's generated descriptor."""
    split = root / "test"
    split.mkdir(parents=True)

    records, annotations = [], []
    annotation_id = 1
    for index, name in enumerate(images, start=1):
        Image.new("RGB", SIZE, color=(90, 90, 90)).save(split / name)
        image_id = index * 17  # neither contiguous nor zero-based
        records.append({
            "id": image_id, "file_name": name,
            "width": SIZE[0], "height": SIZE[1],
        })
        for category_id, x, y, w, h in OBJECTS:
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [_ring(x, y, w, h)],
            })
            annotation_id += 1

    (split / "_annotations.coco.json").write_text(json.dumps({
        "images": records,
        "categories": [
            {"id": 4, "name": "column", "supercategory": "none"},
            {"id": 11, "name": "beam", "supercategory": "none"},
        ],
        "annotations": annotations,
    }))
    # What Ramanujan writes for a COCO upload: a placeholder that must not win.
    (root / "data.yaml").write_text("names:\n  0: object\ntest: test\n")
    return root / "data.yaml"


class _SegDetector:
    """Stands in for a loaded RF-DETR segmentation model.

    Predicts both objects on every image, one of them displaced, so the run has
    a correct finding and a failing one rather than a uniform answer that would
    hide a pairing mistake.
    """

    image_size = 456  # the snapped Seg Medium resolution
    class_names = CLASSES

    def load(self):
        return None

    def predict_image(self, image_path, save_annotated=False):  # noqa: ARG002
        prediction = ImagePrediction(image_path=Path(image_path))
        prediction.image_width, prediction.image_height = SIZE
        detections = []
        for label, (_cid, x, y, w, h) in enumerate(OBJECTS):
            shift = 0.0 if label == 0 else 18.0
            ring = _ring(x + shift, y, w, h)
            detections.append(Detection(
                class_id=label,
                class_name=CLASSES[label],
                confidence=0.88,
                x1=x + shift, y1=y, x2=x + shift + w, y2=y + h,
                polygon=[[p, q] for p, q in zip(ring[::2], ring[1::2])],
            ))
        prediction.detections = detections
        return prediction


@pytest.fixture
def analysed(tmp_path):
    """One saved run over a COCO split, ready for the later stages."""
    yaml_path = _dataset(tmp_path / "ds")
    dataset = load_dataset_config(yaml_path)
    detector = _SegDetector()
    summary = diagnose_split(detector, dataset, "test")

    database = tmp_path / "md.db"
    (tmp_path / "best.pt").write_bytes(b"stub")  # hashed by build_run_context
    with storage.connect(database) as connection:
        context = storage.build_run_context(
            model_path=tmp_path / "best.pt",
            dataset_yaml=yaml_path,
            split="test",
            confidence=0.25,
            match_iou=0.5,
            localization_floor=0.1,
            image_size=detector.image_size,
        )
        run_id = storage.save_dataset_diagnosis(connection, context, summary)
    return tmp_path, yaml_path, dataset, summary, database, run_id


class TestTheCocoPipeline:
    def test_classes_come_from_the_annotations(self, analysed):
        _tmp, _yaml, dataset, _summary, _db, _run = analysed
        assert dataset.class_names == CLASSES, (
            "the generated placeholder must not survive contact with real "
            "categories"
        )

    def test_every_image_is_diagnosed_against_real_ground_truth(self, analysed):
        _tmp, _yaml, _dataset, summary, _db, _run = analysed
        assert len(summary.succeeded) == 2
        for diagnosis in summary.succeeded:
            assert len(diagnosis.findings) == 2, (
                "two annotations paired with two predictions, not four "
                "unmatched ones"
            )

    def test_the_run_records_both_a_success_and_a_failure(self, analysed):
        """A uniform verdict would hide a pairing that matched everything."""
        _tmp, _yaml, _dataset, summary, _db, _run = analysed
        outcomes = {
            f.outcome for d in summary.succeeded for f in d.findings
        }
        assert len(outcomes) > 1, outcomes

    def test_segmentation_reaches_the_findings_as_outlines(self, analysed):
        _tmp, _yaml, _dataset, _summary, database, run_id = analysed
        with storage.connect(database) as connection:
            rows = storage.load_findings_for_embedding(
                connection, run_id, failures_only=False
            )
        assert rows, "the run persisted findings"
        assert any(row["truth_polygon"] for row in rows), (
            "COCO polygons survived diagnosis and persistence"
        )

    def test_class_names_are_stored_as_the_dataset_named_them(self, analysed):
        _tmp, _yaml, _dataset, _summary, database, run_id = analysed
        with storage.connect(database) as connection:
            rows = storage.load_findings_for_embedding(
                connection, run_id, failures_only=False
            )
        assert {row["class_name"] for row in rows} <= set(CLASSES.values())

    def test_the_evaluator_scores_against_the_annotation_file(self, analysed):
        _tmp, _yaml, dataset, _summary, _db, _run = analysed
        source = open_ground_truth(dataset.splits["test"])
        ground_truth = source.coco_ground_truth(dataset.splits["test"])

        assert source.format == "coco"
        assert len(ground_truth["images"]) == 2
        assert len(ground_truth["annotations"]) == 4
        assert [c["id"] for c in ground_truth["categories"]] == [0, 1]
        assert all(a["segmentation"] for a in ground_truth["annotations"]), (
            "segmentation carried into scoring, not dropped to boxes"
        )

    def test_the_mask_pass_measures_outlines(self, analysed):
        """A segmentation model must still do real mask work, not skip."""
        from model_doctor.app import mask_diagnosis

        _tmp, _yaml, _dataset, _summary, database, run_id = analysed
        with storage.connect(database) as connection:
            report = mask_diagnosis.diagnose_masks(
                connection, run_id, detector=_SegDetector()
            )
        assert report.applicable is True
        assert report.measured > 0
        assert report.mean_mask_iou is not None

    def test_no_yolo_labels_are_ever_written(self, analysed):
        """The requirement, stated as a test.

        Not one ``.txt`` and not one ``labels/`` directory anywhere under the
        dataset — before, during or after the whole pipeline.
        """
        tmp, _yaml, _dataset, _summary, _db, _run = analysed
        root = tmp / "ds"
        assert list(root.rglob("*.txt")) == []
        assert [p for p in root.rglob("labels") if p.is_dir()] == []

    def test_the_annotation_file_is_left_byte_for_byte_alone(self, tmp_path):
        """Reading a dataset must never rewrite it."""
        yaml_path = _dataset(tmp_path / "ds")
        annotation = tmp_path / "ds" / "test" / "_annotations.coco.json"
        before = annotation.read_bytes()

        dataset = load_dataset_config(yaml_path)
        diagnose_split(_SegDetector(), dataset, "test")
        source = open_ground_truth(dataset.splits["test"])
        source.coco_ground_truth(dataset.splits["test"])

        assert annotation.read_bytes() == before
