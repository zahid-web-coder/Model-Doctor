import { api } from "@/lib/api/client";
import { HeatmapGrid } from "@/components/dashboard/HeatmapGrid";

/**
 * Grad-CAM attention, keyed on findings rather than on mask findings.
 *
 * The two optional tables do not cover the same runs — in the reference
 * database, heatmaps exist only for run 1 and mask findings only for runs 2
 * and 3. Driving this grid off mask findings alone therefore guaranteed that
 * no heatmap could ever appear. Findings always exist, so the grid is built
 * from those and the mask row is merged in by finding id where present.
 */
export default async function HeatmapsPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const [page, masks] = await Promise.all([
    api.findings(runId, 60, 0),
    api.maskFindings(runId),
  ]);

  const byFinding = new Map(masks.map((m) => [m.finding_id, m]));
  const tiles = page.items.map((finding) => ({
    finding,
    mask: byFinding.get(finding.id) ?? null,
  }));

  return <HeatmapGrid tiles={tiles} total={page.total} maskCount={masks.length} />;
}
