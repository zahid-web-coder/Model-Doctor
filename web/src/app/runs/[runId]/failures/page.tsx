import { api } from "@/lib/api/client";
import { FailuresTable } from "@/components/dashboard/FailuresTable";

/**
 * Findings from `/runs/{id}/findings`, which pages server-side and returns the
 * total with each page. Outcome counts come from `/runs/{id}/outcomes`.
 */
export default async function FailuresPage({
  params, searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ page?: string }>;
}) {
  const { runId } = await params;
  const { page } = await searchParams;
  const current = Math.max(1, Number(page ?? 1) || 1);
  const pageSize = 10;

  const [findings, outcomes] = await Promise.all([
    api.findings(runId, pageSize, (current - 1) * pageSize),
    api.outcomes(runId).catch(() => null),
  ]);

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
