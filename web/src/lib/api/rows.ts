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

/** Members carry the image path but no image_id, so a member thumbnail cannot
 *  be requested from `/images/{id}`. See the note in the clusters screen. */
export interface GroupMember {
  finding_id: number;
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
