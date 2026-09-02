import { redirect } from "next/navigation";
import { api } from "@/lib/api/client";
import { HeatmapGrid } from "@/components/dashboard/HeatmapGrid";
import { resolvePageSize } from "@/lib/paging";

/**
 * Grad-CAM attention, keyed on findings rather than on mask findings.
 *
 * The two optional tables do not cover the same runs — in the reference
 * database, heatmaps exist only for run 1 and mask findings only for runs 2
 * and 3. Driving this grid off mask findings alone therefore guaranteed that
 * no heatmap could ever appear. Findings always exist, so the grid is built
 * from those and the mask row is merged in by finding id where present.
 *
 * **Which findings carry an explanation is passed down, not just how many.**
 * Grad-CAM runs on failures, so a run is expected to have far fewer heatmaps
 * than findings — run 5 has 73 against 231. Sending only the count left every
 * tile to discover its own absence by requesting a heatmap and taking the 404,
 * which meant dozens of pointless round trips per page and a tile that
 * announced itself as a Grad-CAM until the failure came back. The set costs one
 * request and settles it before anything renders.
 *
 * Paged, because the grid used to show the first sixty findings with no way to
 * reach the rest — on this run that hid three quarters of them. Page and size
 * live in the URL for the same reason they do on the failures table: they are
 * part of what is being viewed, so a link reproduces it.
 */
export default async function HeatmapsPage({
  params, searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ page?: string; size?: string }>;
}) {
  const { runId } = await params;
  const { page, size } = await searchParams;
  const current = Math.max(1, Number(page ?? 1) || 1);
  const pageSize = resolvePageSize(size);

  const [findings, masks, heatmaps] = await Promise.all([
    api.findings(runId, pageSize, (current - 1) * pageSize),
    api.maskFindings(runId),
    // Asked for explicitly rather than inferred from a tile failing to load:
    // "this model has no explanation" and "this image is missing" look the
    // same to an <img>, and only one of them is worth telling the reader.
    api.heatmaps(runId),
  ]);

  // Same guard as the failures table: shrinking the page size on a bookmarked
  // deep link can land past the end, where the footer reports a range running
  // backwards. Send the reader to the last real page instead.
  const pageCount = Math.max(1, Math.ceil(findings.total / pageSize));
  if (findings.total > 0 && current > pageCount) {
    redirect(`/runs/${runId}/heatmaps?page=${pageCount}&size=${pageSize}`);
  }

  const byFinding = new Map(masks.map((m) => [m.finding_id, m]));
  const explainedIds = heatmaps.map((h) => h.finding_id);
  const explained = new Set(explainedIds);

  const tiles = findings.items.map((finding) => ({
    finding,
    mask: byFinding.get(finding.id) ?? null,
    explained: explained.has(finding.id),
  }));

  return (
    <HeatmapGrid
      runId={runId}
      page={current}
      pageSize={pageSize}
      tiles={tiles}
      total={findings.total}
      maskCount={masks.length}
      explained={explainedIds.length}
    />
  );
}
