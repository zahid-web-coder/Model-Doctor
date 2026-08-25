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
   * mAP is never reconstructed from `findings`, which hold detections already
   * thresholded at the run's confidence and already matched to ground truth —
   * on this project's own data, 252 of 5,454 predictions for one model and 224
   * of 581 for the other. mAP integrates over the whole precision/recall
   * curve, so scoring the survivors would measure the operating point rather
   * than the model. It comes from `run_evaluations`, written by a separate
   * pass that re-runs inference unthresholded, or it does not come at all.
   */
  NOT_EVALUATED:
    "this run has not been evaluated — run scripts/evaluate_run.py --run-id <id>",
  /**
   * Two evaluations are comparable only if they agree on the confidence sweep,
   * the IoU range, the detection cap and the ground truth. When they do not,
   * showing them side by side would invite a comparison the numbers cannot
   * support.
   */
  EVAL_MISMATCH:
    "the two runs were evaluated under different settings, so their scores are not comparable",
  /**
   * Latency and memory are properties of a model *on a device*. A run records
   * no device — it is resolved at run time — so the benchmark carries its own,
   * and figures from different devices are never compared.
   */
  NOT_BENCHMARKED:
    "this run has not been benchmarked — run scripts/benchmark_run.py --run-id <id> --device <device>",
  DEVICE_MISMATCH:
    "benchmarked on different devices, which measure different machines rather than different models",
  /**
   * Recorded during benchmarking, because that is the pass that provably has
   * the checkpoint open.
   */
  CHECKPOINT: "checkpoint size is recorded when a run is benchmarked",
  /** Present only where the device exposes a counter for it. */
  NO_GPU_COUNTER: "no GPU memory counter on the device this was measured on",
} as const;
