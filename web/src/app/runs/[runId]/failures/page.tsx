import { redirect } from "next/navigation";
import { api } from "@/lib/api/client";
import { FailuresTable } from "@/components/dashboard/FailuresTable";
import { resolvePageSize } from "@/lib/paging";

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

  const [findings, outcomes] = await Promise.all([
    api.findings(runId, pageSize, (current - 1) * pageSize),
    api.outcomes(runId).catch(() => null),
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
    />
  );
}
