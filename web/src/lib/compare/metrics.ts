import type { Evaluation, Finding, MaskFinding, Outcomes } from "@/lib/api/rows";
import { normaliseOutcomes } from "@/lib/api/rows";
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

export { normaliseOutcomes };

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
  // Nullable like the derived ratios: a stored evaluation may omit a statistic
  // (COCOeval reports -1 for an area band with no ground truth), and that is
  // absence, not zero.
  mapBox50: Value<number | null>;
  mapBox5095: Value<number | null>;
  mapMask50: Value<number | null>;
  mapMask5095: Value<number | null>;
}

const CONFUSION_NOTE =
  "poor localisation and wrong class each count as both a false positive and a false negative";

/**
 * The settings an evaluation must share before two of them can be compared.
 *
 * mAP is only meaningful against a stated protocol. Two runs scored at
 * different confidence sweeps, IoU ranges or detection caps produce numbers
 * that look alike and measure different things, so a mismatch is reported as
 * such rather than quietly rendered side by side.
 */
export function comparableEvaluations(a: Evaluation, b: Evaluation): boolean {
  return (
    a.sweep_confidence === b.sweep_confidence &&
    a.iou_thresholds === b.iou_thresholds &&
    a.max_detections === b.max_detections &&
    a.evaluator === b.evaluator
  );
}

/** One run's evaluations, indexed by task. */
export type EvaluationsByTask = Map<string, Evaluation>;

export const indexEvaluations = (rows: Evaluation[]): EvaluationsByTask =>
  new Map(rows.map((row) => [row.task, row]));

/**
 * Read one stored metric, or say why it is absent.
 *
 * `stored`, never `derived`: this figure was produced by COCOeval and written
 * to `run_evaluations` verbatim. Nothing here recomputes it, and nothing here
 * falls back to an approximation when it is missing.
 */
function evaluationMetric(
  evaluations: EvaluationsByTask,
  task: string,
  key: string,
): Value<number | null> {
  const row = evaluations.get(task);
  if (!row) return unavailable(REASONS.NOT_EVALUATED);
  const value = row.metrics?.[key];
  return typeof value === "number" ? stored(value) : unavailable(REASONS.NOT_EVALUATED);
}

export function computeMetrics(
  raw: Partial<Outcomes>,
  findings: Finding[],
  maskFindings: MaskFinding[],
  evaluations: Evaluation[] = [],
): RunMetrics {
  const outcomes = normaliseOutcomes(raw);
  const byTask = indexEvaluations(evaluations);
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

    // From `run_evaluations`, written by the shared COCO evaluator — or
    // absent, saying so. Never zero, and never borrowed from a script artefact
    // that cannot be attributed to this run.
    mapBox50: evaluationMetric(byTask, "bbox", "map50"),
    mapBox5095: evaluationMetric(byTask, "bbox", "map50_95"),
    mapMask50: evaluationMetric(byTask, "segm", "map50"),
    mapMask5095: evaluationMetric(byTask, "segm", "map50_95"),
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
