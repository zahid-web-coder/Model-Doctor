"""Tests for the shared COCO evaluator and its persistence.

The point of this module is that *both* detector families are scored by one
evaluator against one ground truth. So the tests concentrate on the properties
that make a comparison valid — same encoder, same settings, all mask components
preserved — and on the refusals that stop an invalid one being recorded.

Following the pattern in ``test_rfdetr_adapter.py``: everything here runs on
synthetic arrays and temporary databases, so no model, dataset or GPU is
needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import evaluation, storage

pytest.importorskip("pycocotools", reason="pycocotools is a core requirement")


# ---------------------------------------------------------------------------
# Mask encoding — the decision that segmentation scoring rests on
# ---------------------------------------------------------------------------
def _decoded(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_utils

    payload = dict(rle)
    payload["counts"] = payload["counts"].encode("ascii")
    return mask_utils.decode(payload).astype(bool)


def test_encode_mask_round_trips_exactly():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 6:12] = True
    assert np.array_equal(_decoded(evaluation.encode_mask(mask)), mask)


def test_encode_mask_preserves_every_component():
    """The whole reason evaluation does not use the stored polygon.

    Model Doctor keeps one polygon per object, so both families reduce a mask
    to its largest blob on the way in. Scoring against that would charge a
    model for a representation choice, so the raster must survive intact.
    """
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:10, 2:10] = True  # large component
    mask[30:34, 30:33] = True  # smaller, separate component

    restored = _decoded(evaluation.encode_mask(mask))

    assert np.array_equal(restored, mask)
    assert restored[30:34, 30:33].all(), "the smaller component was dropped"
    assert restored.sum() == mask.sum()


def test_encode_mask_returns_none_when_empty():
    """An empty mask is not a prediction and must not become one."""
    assert evaluation.encode_mask(np.zeros((8, 8), dtype=bool)) is None


def test_encode_mask_rejects_wrong_rank():
    with pytest.raises(evaluation.EvaluationError):
        evaluation.encode_mask(np.zeros((2, 8, 8), dtype=bool))


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def _write_split(root: Path) -> Path:
    from PIL import Image

    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(images / "a.jpg")
    # A triangle, normalised, class 0.
    (labels / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.3 0.6\n")
    return images


def test_ground_truth_categories_come_from_the_dataset(tmp_path):
    """Never a hardcoded class list: a two-class dataset evaluates as two."""
    images = _write_split(tmp_path)
    gt = evaluation.build_ground_truth(images, {0: "column", 1: "beam"})

    assert [c["name"] for c in gt["categories"]] == ["column", "beam"]
    assert len(gt["images"]) == 1
    assert len(gt["annotations"]) == 1
    assert gt["annotations"][0]["category_id"] == 0


def test_ground_truth_denormalises_against_the_real_image_size(tmp_path):
    images = _write_split(tmp_path)
    gt = evaluation.build_ground_truth(images, {0: "column"})

    x, y, width, height = gt["annotations"][0]["bbox"]
    assert x == pytest.approx(10.0)  # 0.1 * 100
    assert y == pytest.approx(5.0)  # 0.1 * 50
    assert width == pytest.approx(40.0)  # (0.5 - 0.1) * 100
    assert height == pytest.approx(25.0)  # (0.6 - 0.1) * 50


def test_ground_truth_refuses_an_empty_directory(tmp_path):
    empty = tmp_path / "images"
    empty.mkdir()
    with pytest.raises(evaluation.EvaluationError):
        evaluation.build_ground_truth(empty, {0: "column"})


# ---------------------------------------------------------------------------
# Predictions -> COCO
# ---------------------------------------------------------------------------
def _record(mask: np.ndarray | None = None) -> evaluation.PredictionRecord:
    record = evaluation.PredictionRecord(file_name="a.jpg", width=100, height=50)
    record.boxes.append((10.0, 5.0, 50.0, 30.0))
    record.scores.append(0.9)
    record.class_ids.append(0)
    record.mask_rles.append(None if mask is None else evaluation.encode_mask(mask))
    return record


def test_boxes_convert_to_coco_xywh(tmp_path):
    images = _write_split(tmp_path)
    gt = evaluation.build_ground_truth(images, {0: "column"})

    entries = evaluation.to_coco_detections([_record()], gt, evaluation.TASK_BBOX)

    assert len(entries) == 1
    assert entries[0]["bbox"] == [10.0, 5.0, 40.0, 25.0]
    assert entries[0]["score"] == pytest.approx(0.9)


def test_instances_without_a_mask_are_skipped_not_invented(tmp_path):
    """A missing mask means no segmentation prediction, not an empty one."""
    images = _write_split(tmp_path)
    gt = evaluation.build_ground_truth(images, {0: "column"})

    assert evaluation.to_coco_detections([_record()], gt, evaluation.TASK_SEGM) == []


def test_unknown_image_is_skipped_rather_than_guessed(tmp_path):
    images = _write_split(tmp_path)
    gt = evaluation.build_ground_truth(images, {0: "column"})

    stray = _record()
    stray.file_name = "not-in-the-split.jpg"

    assert evaluation.to_coco_detections([stray], gt, evaluation.TASK_BBOX) == []


def test_unknown_task_is_rejected(tmp_path):
    images = _write_split(tmp_path)
    gt = evaluation.build_ground_truth(images, {0: "column"})
    with pytest.raises(evaluation.EvaluationError):
        evaluation.to_coco_detections([_record()], gt, "keypoints")


def test_a_perfect_prediction_scores_one(tmp_path):
    """End to end through the real evaluator, box and mask together."""
    images = _write_split(tmp_path)
    gt = evaluation.build_ground_truth(images, {0: "column"})

    truth = gt["annotations"][0]
    x, y, width, height = truth["bbox"]
    mask = np.zeros((50, 100), dtype=bool)
    mask[int(y) : int(y + height), int(x) : int(x + width)] = True

    record = evaluation.PredictionRecord(file_name="a.jpg", width=100, height=50)
    record.boxes.append((x, y, x + width, y + height))
    record.scores.append(0.99)
    record.class_ids.append(0)
    record.mask_rles.append(evaluation.encode_mask(mask))

    # Ground truth is a triangle, so only the box can score a perfect 1.0; the
    # mask is scored against the polygon and is deliberately not asserted to.
    boxes = evaluation.coco_evaluate(
        gt,
        evaluation.to_coco_detections([record], gt, evaluation.TASK_BBOX),
        evaluation.TASK_BBOX,
    )
    assert boxes["map50"] == pytest.approx(1.0)


def test_no_predictions_yields_no_metrics(tmp_path):
    images = _write_split(tmp_path)
    gt = evaluation.build_ground_truth(images, {0: "column"})
    assert evaluation.coco_evaluate(gt, [], evaluation.TASK_BBOX) == {}


def test_settings_are_stated_not_implied():
    """These constants are persisted with every result, so they are contract."""
    assert evaluation.SWEEP_CONFIDENCE == 0.01
    assert evaluation.MAX_DETECTIONS == 100
    assert evaluation.IOU_THRESHOLDS == "0.50:0.05:0.95"
    assert evaluation.TASKS == ("bbox", "segm")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _run(connection) -> int:
    return storage.save_run(
        connection,
        storage.RunContext(
            model_path="m.pt",
            model_sha256="abc",
            dataset_yaml="d.yaml",
            split="test",
            confidence_threshold=0.25,
            match_iou_threshold=0.5,
            localization_iou_floor=0.1,
            image_size=672,
        ),
    )


def _save_evaluation(connection, run_id: int, task: str, **overrides):
    payload = {
        "evaluator": "pycocotools COCOeval",
        "evaluator_version": "2.0.11",
        "sweep_confidence": 0.01,
        "iou_thresholds": "0.50:0.05:0.95",
        "max_detections": 100,
        "ground_truth": "YOLO labels, split 'test'",
        "gt_images": 151,
        "gt_annotations": 228,
        "prediction_count": 581,
        "metrics": {"map50": 0.836, "map50_95": 0.709},
    }
    payload.update(overrides)
    return storage.save_evaluation(connection, run_id=run_id, task=task, **payload)


def test_schema_version_is_recorded(tmp_path):
    with storage.connect(tmp_path / "t.db") as connection:
        version = connection.execute("SELECT version FROM schema_info").fetchone()
        assert version["version"] == storage.SCHEMA_VERSION


def test_evaluation_round_trips(tmp_path):
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        _save_evaluation(connection, run_id, "bbox")

        rows = storage.load_evaluations(connection, run_id)

        assert len(rows) == 1
        assert rows[0].task == "bbox"
        assert rows[0].metrics["map50"] == pytest.approx(0.836)
        assert rows[0].sweep_confidence == 0.01
        assert rows[0].max_detections == 100
        assert rows[0].iou_thresholds == "0.50:0.05:0.95"


def test_re_evaluating_supersedes_rather_than_accumulates(tmp_path):
    """Two contradictory scores for one run and task would be unreadable."""
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        _save_evaluation(connection, run_id, "bbox")
        _save_evaluation(connection, run_id, "bbox", metrics={"map50": 0.9})

        rows = storage.load_evaluations(connection, run_id)
        assert len(rows) == 1
        assert rows[0].metrics["map50"] == pytest.approx(0.9)


def test_tasks_are_stored_separately(tmp_path):
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        _save_evaluation(connection, run_id, "bbox")
        _save_evaluation(connection, run_id, "segm", metrics={"map50": 0.83})

        assert [r.task for r in storage.load_evaluations(connection, run_id)] == [
            "bbox",
            "segm",
        ]


def test_missing_metrics_are_null_never_zero(tmp_path):
    """A statistic COCOeval did not report is absent, not a score of nothing."""
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        _save_evaluation(connection, run_id, "bbox", metrics={"map50": 0.5})

        rows = storage.load_evaluations(connection, run_id)
        assert rows[0].metrics["map50"] == pytest.approx(0.5)
        assert rows[0].metrics["ar_small"] is None


def test_evaluations_are_empty_for_an_unevaluated_run(tmp_path):
    with storage.connect(tmp_path / "t.db") as connection:
        assert storage.load_evaluations(connection, _run(connection)) == []


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def test_benchmark_requires_a_device(tmp_path):
    """The refusal that stops a latency being compared with another machine."""
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        for bad in ("", "   "):
            with pytest.raises(ValueError, match="device"):
                storage.save_benchmark(connection, run_id, bad, 672, 3, 30, {})


def test_benchmarks_are_kept_per_device(tmp_path):
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        storage.save_benchmark(connection, run_id, "cpu", 672, 3, 30, {"fps": 22.2})
        storage.save_benchmark(connection, run_id, "mps", 672, 3, 30, {"fps": 35.2})

        rows = storage.load_benchmarks(connection, run_id)
        assert [(r.device, r.measurements["fps"]) for r in rows] == [
            ("cpu", 22.2),
            ("mps", 35.2),
        ]


def test_re_benchmarking_one_device_supersedes(tmp_path):
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        storage.save_benchmark(connection, run_id, "cpu", 672, 3, 30, {"fps": 10.0})
        storage.save_benchmark(connection, run_id, "cpu", 672, 3, 30, {"fps": 22.2})

        rows = storage.load_benchmarks(connection, run_id)
        assert len(rows) == 1
        assert rows[0].measurements["fps"] == pytest.approx(22.2)


def test_unmeasured_gpu_memory_is_null_not_zero(tmp_path):
    """A CPU run has no GPU counter; that is absence, not a measurement."""
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        storage.save_benchmark(connection, run_id, "cpu", 672, 3, 30, {"fps": 22.2})

        rows = storage.load_benchmarks(connection, run_id)
        assert rows[0].measurements["gpu_driver_mb"] is None
        assert rows[0].measurements["gpu_allocated_mb"] is None


def test_device_is_stripped_but_preserved(tmp_path):
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        storage.save_benchmark(connection, run_id, " cpu ", 672, 3, 30, {"fps": 1.0})
        assert storage.load_benchmarks(connection, run_id)[0].device == "cpu"


def test_benchmark_rejects_impossible_image_counts(tmp_path):
    """A latency over zero images is not a measurement."""
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        with pytest.raises(ValueError, match="measured image"):
            storage.save_benchmark(connection, run_id, "cpu", 672, 3, 0, {})
        with pytest.raises(ValueError, match="measured image"):
            storage.save_benchmark(connection, run_id, "cpu", 672, -1, 30, {})


def test_benchmark_counts_are_read_back(tmp_path):
    """The sample size travels with the mean, not beside it."""
    with storage.connect(tmp_path / "t.db") as connection:
        run_id = _run(connection)
        storage.save_benchmark(connection, run_id, "cpu", 672, 3, 30, {"fps": 22.2})

        row = storage.load_benchmarks(connection, run_id)[0]
        assert row.warmup_images == 3
        assert row.measured_images == 30
