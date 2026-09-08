"""Generic one-to-one matching between two sets of annotations.

Given predictions and ground truth for one image, decide which prediction
corresponds to which label. Everything downstream — every failure category,
every statistic — rests on that pairing being right.

**Why this is its own module.** Matching is a general assignment problem, not a
Model Doctor concept. It knows nothing about YOLO, about failure taxonomies, or
about what an unmatched prediction *means*. Keeping it separate means the
diagnosis layer can be rewritten, or a second one added, without touching the
algorithm — and the algorithm can be tested on its own terms.

**Why it takes a similarity function.** The comparison is a parameter, not a
hardcoded call. Box IoU is the default; mask IoU becomes a one-line change at
the call site rather than a rewrite here. That is what "architecture-ready for
masks" means in practice: the seam already exists and is exercised by tests.

The engine is deliberately *unopinionated*. It reports what paired with what
and how strongly. It never says "false positive" — that is an interpretation,
and interpretation belongs to the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from model_doctor.utils.annotations import ObjectAnnotation
from model_doctor.utils.geometry import box_iou

# A comparison between two annotations, returning overlap in [0, 1]. Any
# function of this shape can be substituted — box IoU today, mask IoU later.
SimilarityFn = Callable[[ObjectAnnotation, ObjectAnnotation], float]


@dataclass(frozen=True)
class Match:
    """One prediction paired with one ground-truth annotation.

    Attributes:
        prediction_index: Position in the predictions sequence.
        truth_index: Position in the ground-truth sequence.
        prediction: The paired prediction.
        truth: The paired ground-truth annotation.
        similarity: Overlap score that produced the pairing.
    """

    prediction_index: int
    truth_index: int
    prediction: ObjectAnnotation
    truth: ObjectAnnotation
    similarity: float

    @property
    def same_class(self) -> bool:
        """Return whether the pair agrees on class.

        Reported rather than judged: a pairing with disagreeing classes is
        still a pairing. What it *means* is the caller's decision.
        """
        return self.prediction.class_id == self.truth.class_id


@dataclass(frozen=True)
class MatchResult:
    """The outcome of matching one image's predictions against its labels.

    Attributes:
        matches: Accepted pairings, strongest first.
        unmatched_prediction_indices: Predictions that paired with nothing.
        unmatched_truth_indices: Ground truths that paired with nothing.
    """

    matches: list[Match] = field(default_factory=list)
    unmatched_prediction_indices: list[int] = field(default_factory=list)
    unmatched_truth_indices: list[int] = field(default_factory=list)

    @property
    def mean_similarity(self) -> float | None:
        """Return the average overlap across matches, or ``None`` if none."""
        if not self.matches:
            return None
        return sum(m.similarity for m in self.matches) / len(self.matches)


def match_annotations(
    predictions: Sequence[ObjectAnnotation],
    truths: Sequence[ObjectAnnotation],
    threshold: float,
    similarity: SimilarityFn = box_iou,
    ignore_class: bool = True,
) -> MatchResult:
    """Pair predictions with ground truth, one to one.

    Every prediction/truth combination is scored, then pairs are accepted
    strongest-first. Once an annotation is used it cannot pair again, so each
    prediction matches at most one label and each label at most one prediction.

    **Why strongest-first rather than confidence-first.** Confidence ordering is
    what evaluation metrics use, because they simulate a detector's own ranking.
    Model Doctor is not scoring the model — it is explaining a *specific*
    image — so the question is which prediction genuinely describes which
    object. The best geometric correspondence answers that; a confident
    prediction that overlaps poorly does not become the right pairing by being
    confident.

    Greedy rather than globally optimal (Hungarian) assignment: greedy is
    deterministic, obvious to read, and differs only in crowded scenes where
    several objects overlap heavily. Should that become a real limitation, this
    function is the only place that changes.

    **Ties.** Equal similarity is broken by higher prediction confidence, then
    by earlier index. Without this a run could produce different pairings on
    identical input, which would make diagnoses irreproducible.

    Args:
        predictions: Predicted annotations for one image.
        truths: Ground-truth annotations for the same image.
        threshold: Minimum similarity for a pair to be accepted.
        similarity: Comparison function. Defaults to box IoU; pass a mask-based
            function to match on outlines instead.
        ignore_class: When ``True``, pairs may form across differing classes,
            which is what lets a caller detect a wrong-class prediction rather
            than losing it as an unrelated false positive. When ``False``, only
            same-class pairs are considered.

    Returns:
        A :class:`MatchResult` describing what paired with what.
    """
    candidates: list[tuple[float, float, int, int]] = []
    for p_index, prediction in enumerate(predictions):
        for t_index, truth in enumerate(truths):
            if not ignore_class and prediction.class_id != truth.class_id:
                continue
            score = similarity(prediction, truth)
            if score < threshold:
                continue
            # Confidence participates only as a tie-break. Ground truth has
            # none, so fall back to 0.0 rather than special-casing.
            confidence = prediction.confidence or 0.0
            candidates.append((score, confidence, p_index, t_index))

    # Sort by similarity, then confidence, both descending; index ascending is
    # the final tie-break, achieved by negating the first two keys so a plain
    # ascending sort orders everything correctly and deterministically.
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))

    matches: list[Match] = []
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    for score, _confidence, p_index, t_index in candidates:
        if p_index in used_predictions or t_index in used_truths:
            continue
        used_predictions.add(p_index)
        used_truths.add(t_index)
        matches.append(
            Match(
                prediction_index=p_index,
                truth_index=t_index,
                prediction=predictions[p_index],
                truth=truths[t_index],
                similarity=score,
            )
        )

    return MatchResult(
        matches=matches,
        unmatched_prediction_indices=[
            i for i in range(len(predictions)) if i not in used_predictions
        ],
        unmatched_truth_indices=[
            i for i in range(len(truths)) if i not in used_truths
        ],
    )
