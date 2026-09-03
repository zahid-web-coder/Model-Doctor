import type { paths } from "@/types/api";

/**
 * Row shapes for the read-only API.
 *
 * **Why these are hand-written when `src/types/api.d.ts` is generated.** The
 * generated file is the source of truth for paths, path parameters and query
 * parameters — all of those are used below. It cannot type response *bodies*,
 * because the FastAPI handlers declare no `response_model`, so the OpenAPI
 * document describes every 200 as a bare object and openapi-typescript
 * faithfully emits `{ [key: string]: unknown }`.
 *
 * The fix belongs in the backend — a `response_model` per endpoint — and that
 * is a backend change, so it is flagged rather than made here. Until then
 * these interfaces are transcribed from live responses and every field has
 * been observed. Nothing is invented: if the API does not return it, it is
 * not here.
 */

/** Compile-time proof that the endpoints below still exist in the schema. */
export type ApiPaths = keyof paths;
type Assert<T extends true> = T;
export type _RoutesExist = Assert<
  "/runs" extends ApiPaths
    ? "/runs/{run_id}/findings" extends ApiPaths
      ? "/runs/{run_id}/factor-rates" extends ApiPaths
        ? "/groups/{cluster_id}/members" extends ApiPaths
          ? "/images/{image_id}" extends ApiPaths
            ? "/findings/{finding_id}/heatmap" extends ApiPaths
              ? true
              : false
            : false
          : false
        : false
      : false
    : false
>;

export type FindingsQuery =
  paths["/runs/{run_id}/findings"]["get"] extends { parameters: { query?: infer Q } } ? Q : never;

export interface Run {
  id: number;
  created_at: string;
  model_path: string;
  model_sha256: string;
  dataset_yaml: string;
  split: string;
  confidence_threshold: number;
  match_iou_threshold: number;
  localization_iou_floor: number;
  image_size: number;
}

/** The five outcome buckets the analysis pass records. */
export interface Outcomes {
  correct: number;
  false_negative: number;
  false_positive: number;
  poor_localization: number;
  wrong_class: number;
}

export type Outcome = keyof Outcomes;

/** Every outcome the analysis pass can record, in a fixed order. */
export const ALL_OUTCOMES: Outcome[] = [
  "correct",
  "false_negative",
  "false_positive",
  "poor_localization",
  "wrong_class",
];

/**
 * Fill in outcomes the API omitted.
 *
 * `/runs/{id}/outcomes` groups by outcome, so an outcome that never occurred is
 * **absent from the payload entirely** rather than present as zero — despite
 * :type:`Outcomes` declaring all five keys as required. On a single-class
 * dataset there are no `wrong_class` findings, so that key simply does not
 * arrive, and arithmetic over it yields `NaN` rather than a wrong number.
 *
 * A missing outcome genuinely means none occurred, so zero is the correct
 * reading here — unlike an absent *metric*, which means not measured and must
 * never be shown as zero. That distinction is why this repair belongs to the
 * outcome type and nothing else: it is safe precisely because the API's
 * omission carries a known meaning.
 *
 * Applied once at the client boundary, so every consumer receives the shape
 * `Outcomes` already promises and no screen has to remember to repair it.
 * Idempotent, so calling it again costs nothing but a copy.
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

export interface Finding {
  id: number;
  run_id: number;
  image_id: number;
  outcome: Outcome;
  class_id: number | null;
  class_name: string | null;
  confidence: number | null;
  iou: number | null;
  pred_box: number[] | null;
  truth_box: number[] | null;
  truth_polygon: number[][] | null;
}

/** `/runs/{id}/findings` pages server-side and returns the total with it. */
export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface FactorRate {
  run_id: number;
  factor: string;
  failure_count: number;
  failure_total: number;
  correct_count: number;
  correct_total: number;
  lift: number | null;
  p_value: number | null;
}

export interface Group {
  id: number;
  run_id: number;
  method: string;
  label: string;
  size: number;
}

/** Members carry the image path and image_id, so a member thumbnail can
 *  be requested from `/images/{id}`. */
export interface GroupMember {
  finding_id: number;
  image_id: number;
  outcome: Outcome;
  class_name: string | null;
  confidence: number | null;
  iou: number | null;
  filename: string;
  path: string;
}

export interface MaskFinding {
  finding_id: number;
  run_id: number;
  outcome: Outcome;
  class_name: string | null;
  box_iou: number | null;
  mask_iou: number | null;
  mask_outcome: string | null;
  pred_polygon: number[][] | null;
}

/**
 * One task's evaluation of one run, from the shared COCO evaluator.
 *
 * The settings travel with the numbers because two mAPs are comparable only if
 * they agree on all of them. A caller that shows `map50` without checking
 * `sweep_confidence`, `iou_thresholds` and `max_detections` match is comparing
 * two different measurements.
 */
export interface Evaluation {
  id: number;
  run_id: number;
  /** `"bbox"` or `"segm"`. */
  task: string;
  evaluator: string;
  evaluator_version: string | null;
  sweep_confidence: number;
  iou_thresholds: string;
  max_detections: number;
  ground_truth: string;
  gt_images: number;
  gt_annotations: number;
  prediction_count: number;
  /** What segmentation was scored from; null for box tasks. */
  mask_source: string | null;
  metrics: Record<string, number | null>;
  created_at: string;
}

/**
 * One controlled compute measurement of a run, on one device.
 *
 * `device` is never absent. Latency and memory are properties of a model *on a
 * device*, so a figure from one is not evidence about another — comparing
 * across devices is the mistake this field exists to prevent.
 */
export interface Benchmark {
  id: number;
  run_id: number;
  device: string;
  image_size: number;
  created_at: string;
  measurements: Record<string, number | string | null>;
}

/** One stored explanation, joined to the finding it explains. */
export interface HeatmapRow {
  finding_id: number;
  run_id: number;
  outcome: Outcome;
  class_name: string | null;
  path: string;
  method: string;
  target_layers: string;
}

/**
 * One suggested investigation, derived from a failure group.
 *
 * `actionable` is the field that matters: a recommendation the evidence does
 * not yet support is still shown, marked, rather than hidden — a reader who
 * only sees the confident ones cannot tell whether the quiet groups were
 * examined and dismissed or never examined at all.
 */
export interface Recommendation {
  id: number;
  run_id: number;
  cluster_id: number | null;
  cluster_label: string | null;
  /** Which rule produced this, e.g. `recall_on_factor`. */
  rule: string;
  action: string;
  rationale: string;
  /** e.g. `replicated` when another run agreed. */
  status: string | null;
  actionable: boolean;
  /** How many failures the group holds. */
  affected: number;
  priority: number;
}

/**
 * One factor attributed to one finding, with the measurement behind it.
 *
 * Distinct from `FactorRate`, which aggregates across a run. This is the
 * per-object attribution: which conditions *this* failure carried, and what
 * was measured to decide that.
 */
export interface RootCause {
  finding_id: number;
  run_id: number;
  outcome: Outcome;
  class_name: string | null;
  factor: string;
  /**
   * Attribution strength in [0, 1] — the detectors clamp to that range. It is
   * not a probability, and the panel presents it as a score rather than a
   * likelihood.
   */
  score: number;
  /** The measurement in words, e.g. "3.7:1 wide, above 1.8:1". */
  evidence: string | null;
}

/** One object/prediction overlap, measured on masks. */
export interface ImageCoverage {
  truth_finding_id: number;
  pred_finding_id: number;
  /** Share of the ground-truth mask the prediction covers, 0-1. */
  coverage: number;
}

/**
 * What shape the model's mistake took on one image.
 *
 * A second lens over the same findings, never a replacement: `outcomes` are
 * the finding-level counts, unchanged. `verdict` says whether the image was
 * handled cleanly, or whether one prediction was stretched across several
 * objects (`merged`) or several predictions landed on one (`split`).
 */
export interface ImageDiagnosis {
  id: number;
  run_id: number;
  image_id: number;
  gt_count: number;
  pred_count: number;
  outcomes: Record<string, number>;
  verdict: string;
  merged: boolean;
  split: boolean;
  objects_untouched: number;
  predictions_on_nothing: number;
  /** Stored with the row: merge and split counts are sensitive to it. */
  cover_hit: number;
  cover_miss: number;
  method: string;
  created_at: string;
  coverage: ImageCoverage[];
}

export interface ImageRow {
  id: number;
  run_id: number;
  path: string;
  filename: string;
  width: number | null;
  height: number | null;
  prediction_count: number;
  truth_count: number;
  error: string | null;
}

export const OUTCOME_LABEL: Record<Outcome, string> = {
  correct: "Correct",
  false_negative: "False Negative",
  false_positive: "False Positive",
  poor_localization: "Poor Localization",
  wrong_class: "Wrong Class",
};

/** Everything except `correct`. */
export const FAILURE_OUTCOMES: Outcome[] = [
  "false_negative",
  "wrong_class",
  "poor_localization",
  "false_positive",
];
