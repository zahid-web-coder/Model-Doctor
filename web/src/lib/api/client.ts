import type {
  Run, Outcomes, Finding, Page, FactorRate, Group, GroupMember,
  MaskFinding, ImageRow, Evaluation, Benchmark,
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

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    // The database is written by an offline analysis pass, not by requests, so
    // a short revalidate is enough and avoids hammering it on every render.
    next: { revalidate: 15 },
    ...init,
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

  runs: () => get<Run[]>("/runs"),
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

  /** Image and heatmap bytes are served by id; these are `src` values. */
  imageUrl: (imageId: number) => `${API_BASE}/images/${imageId}`,
  heatmapUrl: (findingId: number) => `${API_BASE}/findings/${findingId}/heatmap`,
};
