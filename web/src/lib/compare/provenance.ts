/**
 * Where a comparison number came from.
 *
 * The comparison page shows figures of three quite different kinds, and the
 * difference matters more here than anywhere else in the product: a stored
 * count and an unavailable metric look identical once they are both rendered
 * as text in a table. Tagging every value at the point it is produced makes
 * "we measured this", "we calculated this from what we measured", and "we do
 * not know this" impossible to confuse — structurally, rather than by a
 * convention someone has to remember.
 *
 * Nothing on this page is ever read from `results/comparison/*.json`. Those
 * artefacts are keyed by script configuration name, not by run id, so they
 * cannot honestly be attributed to a run even when the numbers are real.
 */

export type Provenance =
  /** Read directly from a Model Doctor table. */
  | "stored"
  /** Computed from stored values by a formula stated alongside it. */
  | "derived"
  /** Not obtainable for this run. Carries the reason. */
  | "unavailable";

export interface Measured<T> {
  provenance: "stored" | "derived";
  value: T;
  /** For derived values: the formula, shown to the reader. */
  formula?: string;
}

export interface Unavailable {
  provenance: "unavailable";
  /** Why, in one short clause. Rendered next to "Not available". */
  reason: string;
}

export type Value<T> = Measured<T> | Unavailable;

export const stored = <T,>(value: T): Measured<T> => ({
  provenance: "stored",
  value,
});

export const derived = <T,>(value: T, formula: string): Measured<T> => ({
  provenance: "derived",
  value,
  formula,
});

export const unavailable = (reason: string): Unavailable => ({
  provenance: "unavailable",
  reason,
});

export function isAvailable<T>(value: Value<T>): value is Measured<T> {
  return value.provenance !== "unavailable";
}

/**
 * The reasons used in v1, written once so the same metric never gets two
 * different explanations in two different places.
 */
export const REASONS = {
  /**
   * The decisive fact about mAP, and the reason no amount of backend work on
   * the existing tables would produce it: `findings` holds detections that
   * have already been thresholded at the run's confidence and already matched
   * to ground truth. mAP needs the whole precision-recall curve over
   * unthresholded, unmatched detections. It is not stored, and it is not
   * recoverable from what is.
   */
  MAP: "needs unthresholded detections; findings are stored post-threshold and post-matching",
  /**
   * Latency is measured during a run — `ImagePrediction.inference_ms` — but
   * `ImageDiagnosis` has no field for it, so it is dropped before storage sees
   * it. Persisting it is a small change, deliberately not made in v1.
   */
  TIMING: "measured during inference but not persisted by the current schema",
  MEMORY: "never measured per run",
  CHECKPOINT: "checkpoint not readable from this host",
} as const;
