import { redirect } from "next/navigation";
import { api } from "@/lib/api/client";
import { FailuresTable } from "@/components/dashboard/FailuresTable";
import { resolvePageSize } from "@/lib/paging";
import { indexByFinding } from "@/lib/rootCauses";

/**
 * Findings from `/runs/{id}/findings`, which pages server-side and returns the
 * total with each page. Outcome counts come from `/runs/{id}/outcomes`.
 *
 * Page *and* page size live in the URL. Both are part of what is being viewed,
 * so a link reproduces the view and a reload does not silently reset it. The
 * size is validated against the offered set rather than trusted: it becomes a
 * `limit` on the API, and an arbitrary number from a query string is not
 * something to forward.
 */
export default async function FailuresPage({
  params, searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ page?: string; size?: string }>;
}) {
  const { runId } = await params;
  const { page, size } = await searchParams;
  const current = Math.max(1, Number(page ?? 1) || 1);
  const pageSize = resolvePageSize(size);

  const [findings, outcomes, rootCauses, masks, images] = await Promise.all([
    api.findings(runId, pageSize, (current - 1) * pageSize),
    api.outcomes(runId).catch(() => null),
    // Root causes come back for the whole run — `/runs/{id}/root-causes` filters
    // by factor, not by finding, so there is no per-page request to make. The
    // set is one row per attributed factor per failure, so it scales with
    // failures rather than with images, and indexing it here keeps the client
    // from issuing a request per selection. Absent for a run analysed before
    // the root-cause pass existed, which `api.rootCauses` returns as empty.
    api.rootCauses(runId),
    // The predicted outline lives in `mask_findings`, not on the finding —
    // separate tables, because the mask pass covers a run independently of
    // diagnosis. Fetched for the run rather than per selection, the same way
    // the root causes above are: the endpoint has no per-finding filter, and a
    // request per click would be worse than one indexed here.
    api.maskFindings(runId),
    // Dimensions, so the overlay can put the photograph and the geometry in
    // one coordinate space. Stored on the image row rather than inferred from
    // the bitmap, which is what lets a stored box be interpreted without
    // re-opening the file.
    api.images(runId).catch(() => []),
  ]);

  // A page past the end returns nothing, and the footer then reports a range
  // that runs backwards ("showing 326-285 of 265"). That is reachable by
  // shrinking the page size on a bookmarked deep link, so land the reader on
  // the last real page instead. Only the out-of-range case pays for the second
  // request, and only once — the redirect target is in range by construction.
  const pageCount = Math.max(1, Math.ceil(findings.total / pageSize));
  if (findings.total > 0 && current > pageCount) {
    redirect(`/runs/${runId}/failures?page=${pageCount}&size=${pageSize}`);
  }

  return (
    <FailuresTable
      runId={runId}
      page={current}
      pageSize={pageSize}
      findings={findings}
      outcomes={outcomes}
      rootCauses={indexByFinding(rootCauses)}
      polygons={Object.fromEntries(
        masks
          .filter((m) => m.pred_polygon !== null)
          .map((m) => [m.finding_id, m.pred_polygon as number[][]])
      )}
      dimensions={Object.fromEntries(
        images
          .filter((i) => typeof i.width === "number" && typeof i.height === "number")
          .map((i) => [i.id, [i.width as number, i.height as number] as [number, number]])
      )}
    />
  );
}
