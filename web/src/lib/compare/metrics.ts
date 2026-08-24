import type { Finding, MaskFinding, Outcomes } from "@/lib/api/rows";
import { REASONS, derived, stored, unavailable, type Value } from "./provenance";

/**
 * The comparison's metric set, computed only from what Model Doctor stores.
 *
 * **Every counting rule is written down here rather than assumed**, because
 * Model Doctor's taxonomy is deliberately finer than the two-bucket one that
 * precision and recall come from, and collapsing it involves a choice. Stating
 * the choice next to the number is the difference between a derived metric and
 * an unexplained one — the formulas below are surfaced in the UI, not hidden
 * in this file.
 */

/**
 * Fill in outcomes the API omitted.
 *
 * `/runs/{id}/outcomes` groups by outcome, so an outcome that never occurred is
 * **absent from the payload entirely** rather than present as zero — despite
 * the `Outcomes` type declaring all five keys as required. On a single-class
 * dataset there are no `wrong_class` findings, so that key simply does not
 * arrive, and arithmetic over it yields `NaN` rather than a wrong number.
 *
 * A missing outcome genuinely means none occurred, so zero is the correct
 * reading here — unlike an absent *metric*, which means not measured and must
 * never be shown as zero.
 */
export function normaliseOutcomes(outcomes: Partial<Outcomes>): Outcomes {
  return {
    correct: outcomes.correct ?? 0,
    false_negative: outcomes.false_negative ?? 0,
    false_positive: outcomes.false_positive ?? 0,
    poor_localization: outcomes.poor_localization ?? 0,
    wrong_class: outcomes.wrong_class ?? 0,
  };
}

/**
 * How the five outcomes map onto true/false positives and negatives.
 *
 * `poor_localization` and `wrong_class` each count as **both** a false
 * positive and a false negative, and that is the substantive decision here.
 * The model produced a detection that should not stand (a false positive) and
 * left a ground-truth object unaccounted for (a false negative). Counting
 * either one alone would flatter the model on one axis: score them only as
 * false positives and recall silently improves, only as false negatives and
 * precision does.
 *
 * This is why these figures do not equal the ones mAP implies, which is a
 * property of the project as a whole — see `app/diagnosis.py`.
 */
export function confusion(raw: Partial<Outcomes>) {
  const outcomes = normaliseOutcomes(raw);
  const truePositives = outcomes.correct;
  const falsePositives =
    outcomes.false_positive + outcomes.poor_localization + outcomes.wrong_class;
  const falseNegatives =
    outcomes.false_negative + outcomes.poor_localization + outcomes.wrong_class;
  return { truePositives, falsePositives, falseNegatives };
}

const ratio = (numerator: number, denominator: number): number | null =>
  denominator > 0 ? numerator / denominator : null;

export interface RunMetrics {
  precision: Value<number | null>;
  recall: Value<number | null>;
  f1: Value<number | null>;
  meanIou: Value<number | null>;
  meanMaskIou: Value<number | null>;
  correct: Value<number>;
  falseNegative: Value<number>;
  falsePositive: Value<number>;
  poorLocalization: Value<number>;
  wrongClass: Value<number>;
  mapBox50: Value<number>;
  mapBox5095: Value<number>;
  mapMask50: Value<number>;
  mapMask5095: Value<number>;
}

const CONFUSION_NOTE =
  "poor localisation and wrong class each count as both a false positive and a false negative";

export function computeMetrics(
  raw: Partial<Outcomes>,
  findings: Finding[],
  maskFindings: MaskFinding[],
): RunMetrics {
  const outcomes = normaliseOutcomes(raw);
  const { truePositives: tp, falsePositives: fp, falseNegatives: fn } =
    confusion(outcomes);

  const precision = ratio(tp, tp + fp);
  const recall = ratio(tp, tp + fn);
  const f1 =
    precision !== null && recall !== null && precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null;

  const mean = (values: number[]): number | null =>
    values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

  // Only findings that actually carry a measurement contribute. A null IoU is
  // an absent measurement, not a zero — averaging it in would drag the mean
  // down in proportion to how many objects the model never found.
  const ious = findings
    .map((f) => f.iou)
    .filter((v): v is number => typeof v === "number");
  const maskIous = maskFindings
    .map((m) => m.mask_iou)
    .filter((v): v is number => typeof v === "number");

  return {
    precision: derived(precision, `TP / (TP + FP), where ${CONFUSION_NOTE}`),
    recall: derived(recall, `TP / (TP + FN), where ${CONFUSION_NOTE}`),
    f1: derived(f1, "harmonic mean of precision and recall"),
    meanIou: derived(mean(ious), `mean of ${ious.length} stored box IoU values`),
    meanMaskIou: derived(
      mean(maskIous),
      `mean of ${maskIous.length} stored mask IoU values`,
    ),

    correct: stored(outcomes.correct),
    falseNegative: stored(outcomes.false_negative),
    falsePositive: stored(outcomes.false_positive),
    poorLocalization: stored(outcomes.poor_localization),
    wrongClass: stored(outcomes.wrong_class),

    // Not shown as zero, not borrowed from the offline evaluator's artefacts.
    mapBox50: unavailable(REASONS.MAP),
    mapBox5095: unavailable(REASONS.MAP),
    mapMask50: unavailable(REASONS.MAP),
    mapMask5095: unavailable(REASONS.MAP),
  };
}

/** Which side wins a metric, or `null` when it is a tie or not comparable. */
export function winner(
  a: Value<number | null>,
  b: Value<number | null>,
  higherIsBetter: boolean,
): "a" | "b" | null {
  if (a.provenance === "unavailable" || b.provenance === "unavailable") return null;
  const left = a.value;
  const right = b.value;
  if (typeof left !== "number" || typeof right !== "number") return null;
  if (left === right) return null;
  const aWins = higherIsBetter ? left > right : left < right;
  return aWins ? "a" : "b";
}
