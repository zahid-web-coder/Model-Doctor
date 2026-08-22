"use client";

import { useMemo, useState } from "react";
import { Download } from "lucide-react";
import { FilterSelect } from "@/components/shared/FilterSelect";
import { api } from "@/lib/api/client";
import { OUTCOME_LABEL, type Finding, type MaskFinding } from "@/lib/api/rows";
import { NOT_MEASURED } from "@/lib/format";

export interface Tile {
  finding: Finding;
  mask: MaskFinding | null;
}

/**
 * One tile: the Grad-CAM if the explainability pass covered this finding, the
 * source image otherwise.
 *
 * Falling back to the source image rather than an empty panel is deliberate —
 * the heatmap tables are optional and cover a subset, so most tiles legitimately
 * have no attention map, and a grid of empty boxes would read as broken rather
 * than as partial coverage.
 */
function HeatTile({ tile }: { tile: Tile }) {
  const [heatmapFailed, setHeatmapFailed] = useState(false);
  const { finding, mask } = tile;
  const src = heatmapFailed ? api.imageUrl(finding.image_id) : api.heatmapUrl(finding.id);

  return (
    <div className="bg-card border border-border/40 rounded-lg p-2 hover:border-brass/50 transition-colors">
      <div className="w-full aspect-square rounded mb-2 overflow-hidden bg-black/10 relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={heatmapFailed ? `Source image for finding ${finding.id}` : `Grad-CAM for finding ${finding.id}`}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={() => setHeatmapFailed(true)}
        />
        {heatmapFailed && (
          <span className="absolute bottom-1 left-1 text-[9px] px-1.5 py-0.5 rounded bg-black/55 text-white/90">
            source image
          </span>
        )}
      </div>
      <div className="flex items-center justify-between">
        <span className="font-mono text-[11px] text-ink">#{finding.id}</span>
        <span className={`font-mono text-[10px] ${mask?.mask_iou == null ? "text-slate italic" : "text-slate"}`}>
          {mask?.mask_iou == null ? NOT_MEASURED : `mask ${mask.mask_iou.toFixed(2)}`}
        </span>
      </div>
      <p className="font-mono text-[10px] text-slate mt-0.5 truncate">
        {finding.class_name ?? NOT_MEASURED} • {OUTCOME_LABEL[finding.outcome] ?? finding.outcome}
      </p>
    </div>
  );
}

export function HeatmapGrid({
  tiles, total, maskCount,
}: {
  tiles: Tile[];
  total: number;
  maskCount: number;
}) {
  const [outcome, setOutcome] = useState("All Outcomes");
  const [cls, setCls] = useState("All Classes");
  const [agreement, setAgreement] = useState("All");

  const outcomes = useMemo(
    () => ["All Outcomes", ...Array.from(new Set(tiles.map((t) => OUTCOME_LABEL[t.finding.outcome] ?? t.finding.outcome)))],
    [tiles]
  );
  const classes = useMemo(
    () => ["All Classes", ...Array.from(new Set(tiles.map((t) => t.finding.class_name).filter((c): c is string => Boolean(c))))],
    [tiles]
  );

  const visible = tiles.filter(({ finding, mask }) => {
    if (outcome !== "All Outcomes" && (OUTCOME_LABEL[finding.outcome] ?? finding.outcome) !== outcome) return false;
    if (cls !== "All Classes" && finding.class_name !== cls) return false;
    // The disagreement the mask pass exists to surface: the box says correct,
    // the outline does not.
    if (agreement === "Box correct, outline not") {
      if (!mask || mask.box_iou === null || mask.mask_iou === null) return false;
      if (!(mask.box_iou >= 0.5 && mask.mask_iou < 0.5)) return false;
    }
    return true;
  });

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Attention Heatmaps</h3>
          <p className="text-[13px] text-slate">Model focus on findings, with outline agreement where measured.</p>
        </div>
        <div className="flex items-center gap-2">
          <FilterSelect value={outcome} options={outcomes} onChange={setOutcome} width="w-[175px]" />
          <FilterSelect value={cls} options={classes} onChange={setCls} width="w-[140px]" />
          <FilterSelect value={agreement} options={["All", "Box correct, outline not"]} onChange={setAgreement} width="w-[200px]" />
          <button type="button" className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors">
            <Download size={13} className="text-slate" /> Export
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 pb-2">
        <p className="text-[11px] text-slate mb-3">
          Showing {visible.length} of the first {tiles.length} findings ({total} in this run).
          {maskCount === 0
            ? " No outline measurements for this run."
            : ` ${maskCount} carry an outline measurement.`}
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
          {visible.map((t) => <HeatTile key={t.finding.id} tile={t} />)}
        </div>
        {visible.length === 0 && (
          <p className="py-10 text-center text-slate text-[13px]">No findings match these filters.</p>
        )}
      </div>
    </div>
  );
}
