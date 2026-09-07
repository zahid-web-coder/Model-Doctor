/**
 * The write half of the backend: uploads, validation, and starting analyses.
 *
 * **A separate base URL from the read API on purpose.** `app.api` is read-only
 * in a structural sense — it opens SQLite through `mode=ro` and declares no
 * non-GET route — and `app.control` is where writes live. Keeping the two
 * apart here mirrors that split rather than blurring it: nothing in this file
 * can be reached from a screen that only imports the reader.
 *
 * Everything here runs in the browser. The read API is consumed by server
 * components; uploads are a file picker and a progress poll, which are not.
 */

export const CONTROL_BASE =
  process.env.NEXT_PUBLIC_CONTROL_BASE?.replace(/\/$/, "") ?? "http://127.0.0.1:8001";

/** One thing that was checked about an upload, and what to say about it. */
export interface ValidationCheck {
  name: string;
  ok: boolean;
  detail: string;
}

/** The verdict on one upload, with the facts worth showing beside it. */
export interface ValidationReport {
  ok: boolean;
  checks: ValidationCheck[];
  facts: Record<string, unknown>;
}

export interface ModelUpload extends ValidationReport {
  path: string;
  name: string;
}

export interface DatasetUpload extends ValidationReport {
  root: string;
  name: string;
}

/** What this build can actually accept — read rather than assumed. */
export interface Capabilities {
  detectors: {
    family: string;
    explainability: boolean;
    explainability_note: string;
  }[];
  dataset_format: { name: string; detail: string; archives: string[] };
  limits: { model_mb: number; dataset_mb: number };
  queue_depth: number;
}

/** What a run owns, so a confirmation can quote quantities rather than ask blind. */
export interface Footprint {
  run_id: number;
  /** Row counts per table, zero-count tables omitted by the server. */
  rows: Record<string, number>;
  total_rows: number;
  heatmap_files: number;
}

/** What a delete actually removed. */
export interface Deletion extends Footprint {
  files_deleted: number;
  /** Files left alone for sitting outside the server's results directory. */
  files_refused: number;
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface Job {
  token: string;
  status: JobStatus;
  stage: string | null;
  /** The stages this run will actually perform, in order. */
  stages: string[];
  stage_index: number | null;
  run_id: number | null;
  detector: string;
  split: string;
  model_name: string;
  dataset_name: string;
  image_size: number;
  confidence: number;
  error: string | null;
  log_tail: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  explainability_supported: boolean;
}

/**
 * A failure carrying whatever the server said, rather than a status code.
 *
 * The status is assigned in the body rather than declared as a constructor
 * parameter property: this file is imported by a test, and Node's strip-only
 * TypeScript mode cannot compile parameter properties. Writing it out keeps
 * the module runnable by the test runner with no build step.
 */
export class ControlError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ControlError";
    this.status = status;
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  let detail: unknown = response.statusText;
  try {
    detail = (await response.json())?.detail ?? detail;
  } catch {
    // A non-JSON error body is not worth failing over.
  }
  // `detail` is an object when validation failed with a per-input breakdown.
  const message =
    typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
  throw new ControlError(response.status, message);
}

export const control = {
  capabilities: async (): Promise<Capabilities> =>
    unwrap(await fetch(`${CONTROL_BASE}/capabilities`, { cache: "no-store" })),

  /** Open a workspace and return its token. Every later call needs it. */
  session: async (): Promise<{ token: string }> =>
    unwrap(await fetch(`${CONTROL_BASE}/uploads`, { method: "POST" })),

  uploadModel: async (
    token: string,
    file: File,
    family?: string
  ): Promise<ModelUpload> => {
    const body = new FormData();
    body.append("file", file);
    if (family) body.append("family", family);
    return unwrap(
      await fetch(`${CONTROL_BASE}/uploads/${token}/model`, { method: "POST", body })
    );
  },

  uploadDataset: async (
    token: string,
    file: File,
    split: string
  ): Promise<DatasetUpload> => {
    const body = new FormData();
    body.append("file", file);
    body.append("split", split);
    return unwrap(
      await fetch(`${CONTROL_BASE}/uploads/${token}/dataset`, { method: "POST", body })
    );
  },

  start: async (payload: {
    token: string;
    detector: string;
    model_path: string;
    data_yaml: string;
    split: string;
    image_size?: number;
    confidence?: number;
  }): Promise<Job> =>
    unwrap(
      await fetch(`${CONTROL_BASE}/analyses`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    ),

  status: async (token: string): Promise<Job> =>
    unwrap(await fetch(`${CONTROL_BASE}/analyses/${token}`, { cache: "no-store" })),

  recent: async (): Promise<Job[]> =>
    unwrap(await fetch(`${CONTROL_BASE}/analyses`, { cache: "no-store" })),

  /** What deleting this run would destroy, asked before anything is destroyed. */
  footprint: async (runId: number): Promise<Footprint> =>
    unwrap(await fetch(`${CONTROL_BASE}/runs/${runId}/footprint`, { cache: "no-store" })),

  /**
   * Delete a run and everything it owns. Irreversible.
   *
   * On the control API rather than the reader, which opens SQLite read-only
   * and declares no non-GET route.
   */
  deleteRun: async (runId: number): Promise<Deletion> =>
    unwrap(await fetch(`${CONTROL_BASE}/runs/${runId}`, { method: "DELETE" })),

  /**
   * Set or clear a run's name.
   *
   * Blank clears it. This is the one writable field on a run: the rest records
   * what an analysis pass did, and it goes to the control API because the
   * reader opens SQLite read-only and declares no non-GET route.
   */
  renameRun: async (
    runId: number,
    name: string | null
  ): Promise<{ run_id: number; name: string | null }> =>
    unwrap(
      await fetch(`${CONTROL_BASE}/runs/${runId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      })
    ),
};

/** Human labels for the pipeline stages the backend names. */
export const STAGE_LABEL: Record<string, string> = {
  "inference+diagnosis": "Inference and diagnosis",
  evaluation: "Evaluation",
  "mask-diagnosis": "Mask diagnosis",
  "root-cause": "Root causes",
  clustering: "Clustering",
  recommendations: "Recommendations",
  "image-diagnosis": "Whole-image verdicts",
  relations: "Finding relationships",
  explainability: "Attention heatmaps",
};

/**
 * Where a job has got to, as a fraction.
 *
 * A finished job is complete whatever its last stage was; a running one has
 * finished the stages before the one it is on. Deliberately not derived from
 * elapsed time — the stages take wildly different lengths, and a bar that
 * moves smoothly and lies is worse than one that jumps and is accurate.
 */
export function progressOf(job: Job): number {
  if (job.status === "succeeded") return 1;
  if (job.stage_index === null) return job.status === "running" ? 0.02 : 0;
  return Math.min(0.98, job.stage_index / Math.max(1, job.stages.length));
}
