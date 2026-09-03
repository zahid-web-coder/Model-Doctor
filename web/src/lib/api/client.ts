import type {
  Run, Outcomes, Finding, Page, FactorRate, Group, GroupMember,
  MaskFinding, ImageRow, Evaluation, Benchmark, HeatmapRow,
  Recommendation, RootCause, ImageDiagnosis,
} from "./rows";
import { normaliseOutcomes } from "./rows";

/**
 * The one place the frontend talks to the backend.
 *
 * Server components fetch on the server, so the base URL has to be absolute —
 * a relative path has no origin there. It comes from the environment so a
 * deployment can point at a different host without a rebuild.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(readonly status: number, readonly detail: string, readonly path: string) {
    super(`${status} ${detail}`);
    this.name = "ApiError";
  }
}

async function get<T>(
  path: string,
  init?: RequestInit & { fresh?: boolean },
): Promise<T> {
  const { fresh, ...rest } = init ?? {};
  const response = await fetch(`${API_BASE}${path}`, {
    // Most reads tolerate a short revalidate: the expensive analysis passes
    // write the database, and re-rendering a chart against data fifteen
    // seconds old costs nothing.
    //
    // **`fresh` exists because that is no longer true everywhere.** Runs are
    // now created and deleted through the control API while this dashboard is
    // open, so a page that both shows a list and mutates it cannot serve a
    // cached copy — a deleted row lingering for fifteen seconds reads as a
    // failed delete, and invites a second attempt that 404s. Cache and
    // revalidate are mutually exclusive in Next, so exactly one is set.
    ...(fresh ? { cache: "no-store" as const } : { next: { revalidate: 15 } }),
    ...rest,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // A non-JSON error body is not worth failing over.
    }
    throw new ApiError(response.status, detail, path);
  }
  return (await response.json()) as T;
}

/**
 * Call an endpoint that is allowed to be absent.
 *
 * Optional tables — heatmaps, clusters, mask findings — only exist in a
 * database saved at or above the schema version that introduced them, and the
 * API answers 404 for those. That is an ordinary state, not a fault, so it
 * returns the fallback instead of throwing (SCHEMA.md §6).
 */
async function getOptional<T>(path: string, fallback: T): Promise<T> {
  try {
    return await get<T>(path);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return fallback;
    throw error;
  }
}

export const api = {
  health: () => get<{ status: string; schema_version: number }>("/health"),

  /**
   * Every run, newest first.
   *
   * `fresh` for any view that can delete or create one — see `get`.
   */
  runs: (fresh = false) => get<Run[]>("/runs", { fresh }),
  run: (id: string | number) => get<Run>(`/runs/${id}`),
  /**
   * Outcome counts, with the buckets the API omitted filled in as zero.
   *
   * The endpoint groups by outcome, so an outcome that never occurred does not
   * appear at all. Repairing it here rather than in each screen is what makes
   * the `Outcomes` type honest: every caller downstream can rely on all five
   * keys existing, which is what the type says. A rejected request still
   * rejects — this fills gaps in a payload, it never invents one.
   */
  outcomes: async (id: string | number): Promise<Outcomes> =>
    normaliseOutcomes(await get<Partial<Outcomes>>(`/runs/${id}/outcomes`)),

  findings: (id: string | number, limit = 50, offset = 0) =>
    get<Page<Finding>>(`/runs/${id}/findings?limit=${limit}&offset=${offset}`),

  images: (id: string | number, limit = 200, offset = 0) =>
    get<ImageRow[]>(`/runs/${id}/images?limit=${limit}&offset=${offset}`),

  factorRates: (id: string | number) =>
    getOptional<FactorRate[]>(`/runs/${id}/factor-rates`, []),

  groups: (id: string | number) => getOptional<Group[]>(`/runs/${id}/groups`, []),
  groupMembers: (groupId: string | number) =>
    getOptional<GroupMember[]>(`/groups/${groupId}/members`, []),

  maskFindings: (id: string | number) =>
    getOptional<MaskFinding[]>(`/runs/${id}/mask-findings`, []),

  /**
   * Measured performance. Both default to empty rather than failing, because
   * empty is the correct answer for a run that has not been evaluated or
   * benchmarked — and for a database written before either table existed.
   */
  evaluation: (id: string | number) =>
    getOptional<Evaluation[]>(`/runs/${id}/evaluation`, []),
  benchmarks: (id: string | number) =>
    getOptional<Benchmark[]>(`/runs/${id}/benchmarks`, []),

  /**
   * Suggested investigations for a run, and the per-object attribution behind
   * them. Both default to empty: a run analysed before those passes existed
   * legitimately has neither.
   */
  recommendations: (id: string | number) =>
    getOptional<Recommendation[]>(`/runs/${id}/recommendations`, []),
  rootCauses: (id: string | number) =>
    getOptional<RootCause[]>(`/runs/${id}/root-causes`, []),

  /** Which findings in a run have an explanation. Empty is a real answer. */
  /**
   * Every finding on one image — all annotated objects and all predictions,
   * including those the matcher paired with a different finding.
   *
   * That last part is the point: a merged detection is only visible when the
   * prediction assigned to a neighbouring finding is drawn beside the object
   * it also covers.
   */
  imageFindings: (runId: string | number, imageId: number) =>
    get<Page<Finding>>(
      `/runs/${runId}/findings?image_id=${imageId}&limit=200`,
    ).then((page) => page.items),

  /**
   * Image-level verdicts. Empty for a run analysed before the pass existed,
   * which the Images screen reports as "not measured" rather than as clean.
   */
  imageDiagnoses: (id: string | number, coverage = false) =>
    getOptional<ImageDiagnosis[]>(
      `/runs/${id}/image-diagnoses?coverage=${coverage}`,
      [],
    ),

  heatmaps: (id: string | number) =>
    getOptional<HeatmapRow[]>(`/runs/${id}/heatmaps`, []),

  /** Image and heatmap bytes are served by id; these are `src` values. */
  imageUrl: (imageId: number) => `${API_BASE}/images/${imageId}`,
  /**
   * `preview` asks for the downscaled companion, which is what a grid of
   * tiles should request: the full-resolution overlay is around twenty times
   * larger and a tile cannot show the difference. Any close view omits it.
   */
  heatmapUrl: (findingId: number, preview = false) =>
    `${API_BASE}/findings/${findingId}/heatmap${preview ? "?preview=true" : ""}`,
};
