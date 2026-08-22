import type { FactorRate, Run } from "./api/rows";

/**
 * Values the UI needs that the API does not return.
 *
 * The screens previously showed a "reading", an "impact" band and a coverage
 * figure. None of those are columns in the API — they were invented in the
 * mock. Rather than drop them from the design or fabricate them, they are
 * derived here from figures the API *does* return, in one place, so the rule
 * behind each is visible and testable.
 */

/** Rate of a factor among failures, from the counts the API gives. */
export const failureRate = (f: FactorRate) =>
  f.failure_total === 0 ? null : f.failure_count / f.failure_total;

/** Rate of the same factor among correct detections. */
export const correctRate = (f: FactorRate) =>
  f.correct_total === 0 ? null : f.correct_count / f.correct_total;

const SIGNIFICANT = 0.05;

/**
 * Plain-English reading of a factor.
 *
 * Significance gates everything. A factor with no lift has too little data; a
 * factor above the threshold is not distinguishable from chance however large
 * its raw count — which is exactly the edge_truncation case, present on most
 * failures and on most correct detections alike.
 */
export function reading(f: FactorRate): string {
  if (f.lift === null || f.p_value === null) return "insufficient data to calculate lift";
  if (f.p_value > SIGNIFICANT) return "no significant difference";
  if (f.lift >= 2.5) return "highly over-represented in failures";
  if (f.lift >= 1.5) return "over-represented in failures";
  if (f.lift > 1) return "slightly over-represented";
  return "not over-represented";
}

/** Impact band, or null when the factor is not significant. */
export function impact(f: FactorRate): "High" | "Medium" | "Low" | null {
  if (f.lift === null || f.p_value === null || f.p_value > SIGNIFICANT) return null;
  if (f.lift >= 2.0) return "High";
  if (f.lift >= 1.4) return "Medium";
  return "Low";
}

/**
 * A run's model, as the checkpoint file name.
 *
 * This is `best.pt` for every run, because that is Ultralytics' default output
 * name and it is genuinely what `runs.model_path` holds. The *architecture* —
 * `yolo26-seg` — is recorded only in `heatmaps.target_layers`, and no endpoint
 * exposes that column, so the frontend cannot show it without inventing a
 * field the API does not return. Exposing it is a backend change and is
 * flagged rather than worked around.
 */
export const modelName = (run: Run | undefined) =>
  run ? (run.model_path.split("/").pop() ?? run.model_path) : null;

/** Short checkpoint fingerprint, so two runs on different weights are
 *  distinguishable when the file name is always the same. */
export const modelFingerprint = (run: Run | undefined) =>
  run ? run.model_sha256.slice(0, 7) : null;

/** Likewise the dataset, taken from its data.yaml path. */
export const datasetName = (run: Run | undefined) => {
  if (!run) return null;
  const parts = run.dataset_yaml.split("/");
  return parts[parts.length - 2] ?? run.dataset_yaml;
};
