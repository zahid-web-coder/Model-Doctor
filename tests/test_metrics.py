"""Tests for validation-metric normalisation.

`ValidationMetrics.from_raw` is the project's single point of coupling to the
evaluation library's object shape. Because it takes a plain object and reads
attributes off it, it can be tested with lightweight stubs — no model, no
dataset, no network.

That is the practical payoff of the wrapper: the code that would otherwise be
untestable until a dataset exists is now covered today.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor.app.inference import ValidationMetrics, format_metrics


class _Box:
    """Minimal stand-in for the library's per-metric container."""

    def __init__(self, **attributes: object) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


class _Raw:
    """Minimal stand-in for the library's evaluation result."""

    def __init__(self, box: object | None) -> None:
        if box is not None:
            self.box = box


def test_extracts_overall_metrics() -> None:
    """The four headline metrics are read from their source attributes."""
    raw = _Raw(_Box(mp=0.91, mr=0.77, map50=0.85, map=0.62))
    metrics = ValidationMetrics.from_raw(raw)

    assert metrics.precision == 0.91
    assert metrics.recall == 0.77
    assert metrics.map50 == 0.85
    assert metrics.map50_95 == 0.62


def test_missing_metrics_object_does_not_raise() -> None:
    """A result with no metrics container yields an empty, valid object.

    A library version that renames or drops the container must not crash a
    validation run that already did the expensive work.
    """
    metrics = ValidationMetrics.from_raw(_Raw(None), artefacts_dir=Path("/tmp/x"))

    assert metrics.precision is None
    assert metrics.per_class == {}
    assert metrics.artefacts_dir == Path("/tmp/x")


def test_renamed_attributes_degrade_to_none() -> None:
    """An unrecognised attribute name is reported as unavailable, not fatal."""
    metrics = ValidationMetrics.from_raw(_Raw(_Box(mean_precision=0.9)))
    assert metrics.precision is None
    assert metrics.map50 is None


def test_non_numeric_metric_is_ignored() -> None:
    """A non-numeric value yields None rather than propagating a bad type."""
    metrics = ValidationMetrics.from_raw(_Raw(_Box(mp="n/a", mr=0.5)))
    assert metrics.precision is None
    assert metrics.recall == 0.5


def test_per_class_rows_are_labelled_by_class_name() -> None:
    """Per-class arrays are joined to names via the class-index mapping."""
    box = _Box(
        mp=0.8,
        mr=0.7,
        map50=0.75,
        map=0.5,
        ap_class_index=[0, 2],
        p=[0.9, 0.6],
        r=[0.8, 0.5],
        ap50=[0.85, 0.55],
        ap=[0.6, 0.4],
    )
    metrics = ValidationMetrics.from_raw(
        _Raw(box), class_names={0: "first", 1: "second", 2: "third"}
    )

    assert set(metrics.per_class) == {"first", "third"}
    assert metrics.per_class["first"]["precision"] == 0.9
    assert metrics.per_class["third"]["map50_95"] == 0.4


def test_per_class_unknown_id_falls_back_to_label() -> None:
    """A class id with no name degrades visibly instead of crashing."""
    box = _Box(ap_class_index=[7], p=[0.5])
    metrics = ValidationMetrics.from_raw(_Raw(box), class_names={0: "only"})
    assert "id:7" in metrics.per_class


def test_short_per_class_array_does_not_index_out_of_range() -> None:
    """Arrays of differing length are read defensively, never out of bounds.

    Same alignment hazard as the detection tensors: positional arrays that
    disagree in length must not raise or silently mis-pair.
    """
    box = _Box(ap_class_index=[0, 1], p=[0.9])  # p is one short
    metrics = ValidationMetrics.from_raw(_Raw(box), class_names={0: "a", 1: "b"})

    assert metrics.per_class["a"]["precision"] == 0.9
    assert "b" not in metrics.per_class


def test_format_metrics_renders_unavailable_values() -> None:
    """Rendering an empty result reports 'not reported', not a crash."""
    rendered = format_metrics(ValidationMetrics())
    assert "not reported" in rendered
    assert "Validation metrics" in rendered


def test_format_metrics_includes_per_class_table() -> None:
    """Per-class rows appear when present."""
    metrics = ValidationMetrics(
        precision=0.9,
        recall=0.8,
        map50=0.85,
        map50_95=0.6,
        per_class={"alpha": {"precision": 0.9, "recall": 0.8}},
    )
    rendered = format_metrics(metrics)
    assert "alpha" in rendered
    assert "0.9000" in rendered
