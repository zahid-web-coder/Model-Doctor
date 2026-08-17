"""Tests for the group-versus-control statistics.

These functions decide whether Model Doctor claims a condition is associated
with failure. Getting them wrong would not raise — it would produce confident,
specific, unfounded advice — so they are pinned against known values rather
than against their own output.
"""

from __future__ import annotations

import pytest

from utils.statistics import fisher_exact_two_sided, lift


# ---------------------------------------------------------------------------
# lift
# ---------------------------------------------------------------------------
def test_lift_of_one_means_the_condition_explains_nothing() -> None:
    """Equal rates in both groups is the case that must read as "no signal"."""
    assert lift(50, 100, 50, 100) == pytest.approx(1.0)


def test_lift_above_one_means_more_common_among_failures() -> None:
    """Twice as common in the group as in the control."""
    assert lift(60, 100, 30, 100) == pytest.approx(2.0)


def test_lift_below_one_means_more_common_among_successes() -> None:
    """The case that exposed edge_truncation: 71% of failures, 76% of correct."""
    assert lift(90, 127, 115, 151) == pytest.approx(0.9305, abs=1e-4)


def test_lift_is_undefined_rather_than_infinite_when_no_control_has_it() -> None:
    """``inf`` would be rendered as a number and read as a measurement."""
    assert lift(2, 127, 0, 151) is None


def test_lift_is_undefined_for_an_empty_group() -> None:
    """A rate over zero observations is not a small rate, it is no rate."""
    assert lift(0, 0, 5, 10) is None
    assert lift(5, 10, 0, 0) is None


# ---------------------------------------------------------------------------
# Fisher's exact test
# ---------------------------------------------------------------------------
def test_identical_rates_give_a_p_value_of_one() -> None:
    """No difference at all cannot be evidence of a difference."""
    assert fisher_exact_two_sided(50, 100, 50, 100) == pytest.approx(1.0)


def test_a_complete_separation_is_significant() -> None:
    """Every failure has it and no success does — as strong as this test gets."""
    assert fisher_exact_two_sided(20, 20, 0, 20) < 0.0001


def test_reproduces_the_reference_run_values() -> None:
    """Pinned against SciPy's fisher_exact, checked to 1e-15 during development.

    These four are the measurements that drove D-031, so a change in them is a
    change in the project's conclusions and should fail loudly.
    """
    assert fisher_exact_two_sided(73, 127, 68, 151) == pytest.approx(0.041, abs=5e-4)
    assert fisher_exact_two_sided(90, 127, 115, 151) == pytest.approx(0.340, abs=5e-4)
    assert fisher_exact_two_sided(38, 127, 56, 151) == pytest.approx(0.252, abs=5e-4)
    assert fisher_exact_two_sided(12, 127, 10, 151) == pytest.approx(0.504, abs=5e-4)


def test_small_counts_do_not_reach_significance() -> None:
    """2 of 127 versus 0 of 151 looks like infinite lift and proves nothing."""
    assert fisher_exact_two_sided(2, 127, 0, 151) > 0.05


def test_the_test_is_symmetric_in_its_groups() -> None:
    """Swapping group and control tests the same difference."""
    assert fisher_exact_two_sided(30, 100, 10, 100) == pytest.approx(
        fisher_exact_two_sided(10, 100, 30, 100)
    )


def test_an_empty_group_gives_no_evidence_rather_than_certainty() -> None:
    """Returning a small p from no data would be the worst possible answer."""
    assert fisher_exact_two_sided(0, 0, 5, 10) == 1.0


def test_a_count_exceeding_its_total_is_caller_error() -> None:
    """This means mismatched inputs, not a surprising observation."""
    with pytest.raises(ValueError, match="cannot exceed its total"):
        fisher_exact_two_sided(11, 10, 5, 10)


def test_negative_counts_are_rejected() -> None:
    """Guarding the arithmetic rather than returning a meaningless number."""
    with pytest.raises(ValueError, match="must not be negative"):
        fisher_exact_two_sided(-1, 10, 5, 10)


def test_p_value_never_exceeds_one() -> None:
    """Floating-point summation must not produce an impossible probability."""
    for group_count in range(0, 21):
        assert fisher_exact_two_sided(group_count, 20, 10, 20) <= 1.0
