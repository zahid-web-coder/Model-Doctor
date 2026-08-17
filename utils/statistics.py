"""Small statistical helpers for comparing a group against a control group.

A count on its own cannot support a claim about cause. "90 of 127 failures were
edge-truncated" sounds damning until you learn that 115 of 151 *correct*
detections were too. The functions here supply the missing half of that
comparison: how much more common a condition is among failures than among
successes, and whether the difference is distinguishable from chance.

Fisher's exact test is computed directly rather than taken from SciPy. SciPy is
not a declared dependency — it arrives transitively through scikit-learn, and
relying on a transitive dependency is how a working install breaks later. The
2x2 case is a short exact calculation with no approximation, so there is
nothing to gain from the import.

Nothing here knows about detection, findings, or factors. It takes four counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

# Guards against a table one floating-point ulp above the observed probability
# being excluded from the two-sided sum, which would understate the p-value.
_TOLERANCE: float = 1e-9


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return the value below which ``fraction`` of the sample falls.

    Used to derive a threshold from the data being measured rather than from a
    constant. A fixed cut-off does not transfer between datasets: this
    project's ``small_object`` threshold of 0.12% of image area was calibrated
    for benchmarks where objects are small, and fired on 2 of 278 findings for
    a dataset photographed close up — while "smaller than a quarter of the
    others" turned out to be one of its strongest predictors of failure
    (D-032).

    Linear interpolation between the two neighbouring order statistics, which
    is the common definition and behaves sensibly on small samples.

    Args:
        values: The sample. Order does not matter; it is sorted internally.
        fraction: Position in ``[0, 1]``. ``0.25`` returns the lower quartile.

    Returns:
        The interpolated value, or ``None`` for an empty sample — a percentile
        of nothing is undefined, not zero.

    Raises:
        ValueError: If ``fraction`` lies outside ``[0, 1]``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be within [0, 1], got {fraction}.")
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])

    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def lift(
    group_count: int, group_total: int, control_count: int, control_total: int
) -> float | None:
    """Return how many times more common a condition is in a group than control.

    Args:
        group_count: Members of the group with the condition.
        group_total: Size of the group.
        control_count: Members of the control with the condition.
        control_total: Size of the control.

    Returns:
        The ratio of the two rates. ``1.0`` means the condition is equally
        common in both and therefore explains nothing about group membership.
        ``None`` when it cannot be computed — either group is empty, or the
        control rate is zero, which would make the ratio infinite rather than
        large. ``None`` is returned instead of ``inf`` so a caller must decide
        how to present "undefined" rather than rendering a number that will be
        read as a measurement.
    """
    if group_total <= 0 or control_total <= 0:
        return None
    control_rate = control_count / control_total
    if control_rate == 0.0:
        return None
    return (group_count / group_total) / control_rate


def fisher_exact_two_sided(
    group_count: int, group_total: int, control_count: int, control_total: int
) -> float:
    """Return the two-sided p-value for a 2x2 contingency table.

    Tests whether a condition appears at different rates in two groups. The
    table is::

                       has condition    lacks condition
        group          group_count      group_total - group_count
        control        control_count    control_total - control_count

    Exact rather than approximate, so it stays valid at the small counts this
    project works with — a chi-squared test is unreliable when an expected
    cell count falls below about five, which happens here routinely.

    Args:
        group_count: Members of the group with the condition.
        group_total: Size of the group.
        control_count: Members of the control with the condition.
        control_total: Size of the control.

    Returns:
        Probability of observing a table at least as extreme as this one if
        the condition were equally common in both groups. ``1.0`` when either
        group is empty, which is the honest answer: no evidence either way.

    Raises:
        ValueError: If a count exceeds its total, which means the caller has
            mismatched its inputs rather than observed something surprising.
    """
    if group_count > group_total or control_count > control_total:
        raise ValueError(
            "A count cannot exceed its total: "
            f"{group_count}/{group_total}, {control_count}/{control_total}."
        )
    if min(group_count, group_total, control_count, control_total) < 0:
        raise ValueError("Counts and totals must not be negative.")
    if group_total == 0 or control_total == 0:
        return 1.0

    with_condition = group_count + control_count
    total = group_total + control_total

    def probability(count_in_group: int) -> float:
        """Hypergeometric probability of one particular table."""
        return (
            comb(group_total, count_in_group)
            * comb(control_total, with_condition - count_in_group)
            / comb(total, with_condition)
        )

    observed = probability(group_count)
    lowest = max(0, with_condition - control_total)
    highest = min(with_condition, group_total)

    threshold = observed * (1 + _TOLERANCE)
    tail = sum(
        candidate
        for candidate in (probability(x) for x in range(lowest, highest + 1))
        if candidate <= threshold
    )
    return min(1.0, tail)
