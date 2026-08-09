"""Tests for the diagnosis engine.

Each of the five outcomes is pinned down by a case constructed to produce only
that outcome, plus the accounting invariant that every prediction and every
ground truth is classified exactly once.

The poor-localisation cases matter most: they encode the decision that
separates this project from a metrics calculation (D-017).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.diagnosis import (
    DatasetDiagnosis,
    Outcome,
    diagnose_image,
    format_dataset_diagnosis,
    format_image_diagnosis,
)
from utils.annotations import ObjectAnnotation

IMAGE = Path("frame.jpg")


def _truth(
    x1: float, y1: float, x2: float, y2: float, cls: int = 0
) -> ObjectAnnotation:
    """Build a ground-truth annotation."""
    names = {0: "door", 1: "door_frame"}
    return ObjectAnnotation.from_box(cls, names[cls], x1, y1, x2, y2)


def _pred(
    x1: float, y1: float, x2: float, y2: float, cls: int = 0, conf: float = 0.9
) -> ObjectAnnotation:
    """Build a prediction."""
    names = {0: "door", 1: "door_frame"}
    return ObjectAnnotation.from_box(cls, names[cls], x1, y1, x2, y2, confidence=conf)


# ---------------------------------------------------------------------------
# The five outcomes
# ---------------------------------------------------------------------------
def test_correct_detection() -> None:
    """A well-placed, correctly-named prediction is correct."""
    d = diagnose_image(IMAGE, [_pred(0, 0, 10, 10)], [_truth(0, 0, 10, 10)])
    assert d.counts[Outcome.CORRECT] == 1
    assert d.failures == []
    assert d.is_clean


def test_false_positive() -> None:
    """A prediction where nothing exists is a false positive."""
    d = diagnose_image(IMAGE, [_pred(0, 0, 10, 10)], [])
    assert d.counts[Outcome.FALSE_POSITIVE] == 1
    assert d.findings[0].truth is None


def test_false_negative() -> None:
    """An object with no prediction is a false negative."""
    d = diagnose_image(IMAGE, [], [_truth(0, 0, 10, 10)])
    assert d.counts[Outcome.FALSE_NEGATIVE] == 1
    assert d.findings[0].prediction is None


def test_wrong_class() -> None:
    """A well-placed prediction with the wrong label is a wrong class."""
    d = diagnose_image(
        IMAGE, [_pred(0, 0, 10, 10, cls=1)], [_truth(0, 0, 10, 10, cls=0)]
    )
    assert d.counts[Outcome.WRONG_CLASS] == 1
    finding = d.findings[0]
    assert finding.prediction.class_name == "door_frame"
    assert finding.truth.class_name == "door"


def test_poor_localization() -> None:
    """Right class, real object, overlap below the match threshold.

    This is the decision that defines the engine: one explained failure rather
    than a false positive plus a false negative.
    """
    # 10x10 boxes offset by 6: intersection 40, union 160, IoU 0.25.
    d = diagnose_image(IMAGE, [_pred(6, 0, 16, 10)], [_truth(0, 0, 10, 10)])

    assert d.counts[Outcome.POOR_LOCALIZATION] == 1
    assert d.counts[Outcome.FALSE_POSITIVE] == 0
    assert d.counts[Outcome.FALSE_NEGATIVE] == 0
    assert d.findings[0].iou is not None and 0.2 < d.findings[0].iou < 0.3


def test_poor_localization_not_used_below_the_floor() -> None:
    """A barely-touching prediction is not excused as a near miss.

    Below the localisation floor the pairing is not credible, so it becomes a
    genuine false positive and a genuine false negative.
    """
    # 10x10 boxes offset by 9.5: IoU well under the 0.10 floor.
    d = diagnose_image(IMAGE, [_pred(9.5, 0, 19.5, 10)], [_truth(0, 0, 10, 10)])
    assert d.counts[Outcome.POOR_LOCALIZATION] == 0
    assert d.counts[Outcome.FALSE_POSITIVE] == 1
    assert d.counts[Outcome.FALSE_NEGATIVE] == 1


def test_weak_overlap_with_wrong_class_reports_wrong_class() -> None:
    """Naming the object wrongly outranks outlining it badly.

    Reporting this as poor localisation would hide the more consequential
    error.
    """
    d = diagnose_image(
        IMAGE, [_pred(6, 0, 16, 10, cls=1)], [_truth(0, 0, 10, 10, cls=0)]
    )
    assert d.counts[Outcome.WRONG_CLASS] == 1
    assert d.counts[Outcome.POOR_LOCALIZATION] == 0


# ---------------------------------------------------------------------------
# Accounting invariants
# ---------------------------------------------------------------------------
def test_every_annotation_is_classified_exactly_once() -> None:
    """Nothing is dropped and nothing is double-counted.

    Each finding consumes one prediction, one ground truth, or one of each. The
    totals must reconcile, or the report would silently under-report failures.
    """
    predictions = [
        _pred(0, 0, 10, 10),  # correct
        _pred(200, 200, 210, 210),  # false positive
        _pred(106, 100, 116, 110, cls=1),  # wrong class, weak overlap
    ]
    truths = [
        _truth(0, 0, 10, 10),
        _truth(100, 100, 110, 110, cls=0),
        _truth(500, 500, 510, 510),  # false negative
    ]
    d = diagnose_image(IMAGE, predictions, truths)

    used_predictions = sum(1 for f in d.findings if f.prediction is not None)
    used_truths = sum(1 for f in d.findings if f.truth is not None)
    assert used_predictions == len(predictions)
    assert used_truths == len(truths)


def test_duplicate_prediction_becomes_a_false_positive() -> None:
    """Two predictions on one object: one correct, one spurious."""
    d = diagnose_image(
        IMAGE, [_pred(0, 0, 10, 10), _pred(0, 0, 10, 10)], [_truth(0, 0, 10, 10)]
    )
    assert d.counts[Outcome.CORRECT] == 1
    assert d.counts[Outcome.FALSE_POSITIVE] == 1


def test_empty_image_produces_no_findings() -> None:
    """No predictions and no labels is clean, not an error."""
    d = diagnose_image(IMAGE, [], [])
    assert d.findings == []
    assert d.is_clean
    assert d.mean_iou is None


def test_thresholds_are_configurable() -> None:
    """Raising the match threshold turns a hit into poor localisation."""
    prediction, truth = [_pred(1, 1, 11, 11)], [_truth(0, 0, 10, 10)]
    lenient = diagnose_image(IMAGE, prediction, truth, match_threshold=0.5)
    strict = diagnose_image(IMAGE, prediction, truth, match_threshold=0.95)

    assert lenient.counts[Outcome.CORRECT] == 1
    assert strict.counts[Outcome.POOR_LOCALIZATION] == 1


def test_custom_similarity_is_honoured() -> None:
    """The similarity function reaches the matcher — the mask seam."""
    d = diagnose_image(
        IMAGE,
        [_pred(0, 0, 1, 1)],
        [_truth(500, 500, 501, 501)],
        similarity=lambda _a, _b: 1.0,
    )
    assert d.counts[Outcome.CORRECT] == 1


def test_mean_iou_covers_only_paired_findings() -> None:
    """Unpaired findings have no overlap and must not drag the mean to zero."""
    d = diagnose_image(
        IMAGE, [_pred(0, 0, 10, 10), _pred(900, 900, 910, 910)], [_truth(0, 0, 10, 10)]
    )
    assert d.mean_iou == 1.0


# ---------------------------------------------------------------------------
# Dataset aggregation
# ---------------------------------------------------------------------------
def _sample_dataset() -> DatasetDiagnosis:
    """Build a small dataset diagnosis with a known outcome mix."""
    return DatasetDiagnosis(
        diagnoses=[
            diagnose_image(
                Path("a.jpg"), [_pred(0, 0, 10, 10)], [_truth(0, 0, 10, 10)]
            ),
            diagnose_image(
                Path("b.jpg"),
                [_pred(6, 0, 16, 10, cls=1)],
                [_truth(0, 0, 10, 10, cls=1)],
            ),
            diagnose_image(Path("c.jpg"), [], [_truth(0, 0, 10, 10, cls=1)]),
            diagnose_image(Path("d.jpg"), [_pred(0, 0, 10, 10, cls=1)], []),
        ]
    )


def test_dataset_totals_sum_the_images() -> None:
    """Aggregate counts equal the sum of their parts."""
    summary = _sample_dataset()
    assert summary.counts[Outcome.CORRECT] == 1
    assert summary.counts[Outcome.POOR_LOCALIZATION] == 1
    assert summary.counts[Outcome.FALSE_NEGATIVE] == 1
    assert summary.counts[Outcome.FALSE_POSITIVE] == 1
    assert summary.total_predictions == 3
    assert summary.total_truths == 3
    assert summary.clean_images == 1


def test_per_class_statistics_split_by_class() -> None:
    """Failures are attributed to the class that should have been found."""
    stats = _sample_dataset().per_class
    assert set(stats) == {"door", "door_frame"}
    assert stats["door"].counts.get(Outcome.CORRECT) == 1
    assert stats["door_frame"].counts.get(Outcome.POOR_LOCALIZATION) == 1
    assert stats["door_frame"].counts.get(Outcome.FALSE_NEGATIVE) == 1


def test_worst_images_are_ranked_and_exclude_clean_ones() -> None:
    """The report names specific images to inspect, worst first."""
    worst = _sample_dataset().worst_images(limit=10)
    assert all(d.failures for d in worst)
    counts = [len(d.failures) for d in worst]
    assert counts == sorted(counts, reverse=True)


def test_reports_render_without_error() -> None:
    """Both report formats render for a populated dataset."""
    summary = _sample_dataset()
    text = format_dataset_diagnosis(summary)
    assert "Model Doctor" in text and "PER CLASS" in text
    assert "door_frame" in text

    per_image = format_image_diagnosis(summary.diagnoses[1])
    assert "Poor localisation" in per_image


def test_empty_dataset_renders_safely() -> None:
    """A dataset with nothing in it still produces a valid report."""
    text = format_dataset_diagnosis(DatasetDiagnosis())
    assert "Images diagnosed   : 0" in text
