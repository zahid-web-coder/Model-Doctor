"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Pagination } from "@/components/shared/Pagination";
import { FilterSelect } from "@/components/shared/FilterSelect";
import { ExportButton } from "@/components/shared/ExportButton";
import { Lightbox } from "@/components/shared/Lightbox";
import { api } from "@/lib/api/client";
import { OUTCOME_LABEL, type Finding, type MaskFinding } from "@/lib/api/rows";
import { NOT_MEASURED } from "@/lib/format";

export interface Tile {
  finding: Finding;
  mask: MaskFinding | null;
  /**
   * Whether the explainability pass produced a heatmap for this finding.
   *
   * Per-tile, not per-run. Grad-CAM runs on failures, so a run that has
   * heatmaps still has none for most of its findings — asking each tile to
   * discover that by requesting one and taking the 404 cost a round trip per
   * correct detection and mislabelled the tile until it came back.
   */
  explained: boolean;
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
function HeatTile({
  tile, onOpen,
}: {
  tile: Tile;
  onOpen: (tile: Tile, showingHeatmap: boolean) => void;
}) {
  const [heatmapFailed, setHeatmapFailed] = useState(false);
  const { finding, mask, explained } = tile;
  // When the run has no explanations at all, go straight to the source image.
  // Requesting sixty heatmaps that are all known to 404 wastes the round trips
  // and, while they are in flight, labels every tile as a Grad-CAM it will
  // never be.
  const showingHeatmap = explained && !heatmapFailed;
  // The grid asks for the preview: a tile is a couple of hundred pixels wide
  // and the full-resolution overlay is roughly twenty times the bytes. The
  // detail view and the endpoint's default are both unchanged.
  const src = showingHeatmap
    ? api.heatmapUrl(finding.id, true)
    : api.imageUrl(finding.image_id);

  return (
    <div className="bg-card border border-border/40 rounded-lg p-2 hover:border-brass/50 transition-colors">
      <button
        type="button"
        onClick={() => onOpen(tile, showingHeatmap)}
        title="Open full size"
        className="block w-full aspect-square rounded mb-2 overflow-hidden bg-black/10 relative cursor-zoom-in"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={showingHeatmap ? `Grad-CAM for finding ${finding.id}` : `Source image for finding ${finding.id}`}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={() => setHeatmapFailed(true)}
        />
        {!showingHeatmap && (
          <span className="absolute bottom-1 left-1 text-[9px] px-1.5 py-0.5 rounded bg-black/55 text-white/90">
            source image
          </span>
        )}
      </button>
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
  runId, page, pageSize, tiles, total, maskCount, explained,
}: {
  runId: string;
  page: number;
  pageSize: number;
  tiles: Tile[];
  total: number;
  maskCount: number;
  /** How many findings in this run actually have an explanation. */
  explained: number;
}) {
  const [outcome, setOutcome] = useState("All Outcomes");
  const [cls, setCls] = useState("All Classes");
  const router = useRouter();
  const [agreement, setAgreement] = useState("All");
  const [coverage, setCoverage] = useState("All findings");
  // Which tile is open, and whether its tile was showing an attribution map —
  // the modal must not claim a heatmap the grid already fell back from.
  const [open, setOpen] = useState<{ tile: Tile; heatmap: boolean } | null>(null);

  const outcomes = useMemo(
    () => ["All Outcomes", ...Array.from(new Set(tiles.map((t) => OUTCOME_LABEL[t.finding.outcome] ?? t.finding.outcome)))],
    [tiles]
  );
  const classes = useMemo(
    () => ["All Classes", ...Array.from(new Set(tiles.map((t) => t.finding.class_name).filter((c): c is string => Boolean(c))))],
    [tiles]
  );

  const pageCount = Math.max(1, Math.ceil(total / pageSize));

  const visible = tiles.filter(({ finding, mask, explained: hasHeatmap }) => {
    if (coverage === "Explained only" && !hasHeatmap) return false;
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
          <FilterSelect value={coverage} options={["All findings", "Explained only"]} onChange={setCoverage} width="w-[160px]" />
          <ExportButton
            rows={visible.map((t) => ({
              finding_id: t.finding.id,
              image_id: t.finding.image_id,
              outcome: t.finding.outcome,
              class_name: t.finding.class_name,
              confidence: t.finding.confidence,
              box_iou: t.mask?.box_iou ?? null,
              mask_iou: t.mask?.mask_iou ?? null,
            }))}
            columns={["finding_id", "image_id", "outcome", "class_name", "confidence", "box_iou", "mask_iou"]}
            filename="heatmap-findings"
            scope={`the ${visible.length} tiles shown`}
          />
        </div>
      </div>
      {explained === 0 && (
        <div className="mb-4 shrink-0 rounded-md border border-brass/40 bg-brass/5 px-3 py-2.5">
          <p className="text-[12px] text-ink font-medium">
            Attention heatmaps are not available for this model.
          </p>
          <p className="text-[11px] text-slate mt-1 leading-relaxed">
            The images below are the original source images — no attribution
            overlay is being shown. Grad-CAM targets a convolutional detection
            head, and this run was produced by a detector that predicts through
            decoder queries instead, so no explanation is generated rather than
            one that would not correspond to the prediction.
          </p>
        </div>
      )}


      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 pb-2">
        <p className="text-[11px] text-slate mb-3">
          Showing {visible.length} of {tiles.length} on this page — {total} finding
          {total === 1 ? "" : "s"} in this run, {explained} with an attention heatmap.
          {maskCount === 0
            ? " No outline measurements for this run."
            : ` ${maskCount} carry an outline measurement.`}
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
          {visible.map((t) => (
            <HeatTile
              key={t.finding.id}
              tile={t}
              onOpen={(tile, heatmap) => setOpen({ tile, heatmap })}
            />
          ))}
        </div>
        {visible.length === 0 && (
          <p className="py-10 text-center text-slate text-[13px]">No findings match these filters.</p>
        )}
      </div>

      <div className="shrink-0 border-t border-border/40 pt-3 mt-2">
        <Pagination
          page={page} pageCount={pageCount} total={total} pageSize={pageSize}
          onChange={(next) => router.push(`/runs/${runId}/heatmaps?page=${next}&size=${pageSize}`)}
          onPageSizeChange={(size) => router.push(`/runs/${runId}/heatmaps?page=1&size=${size}`)}
        />
      </div>

      <Lightbox
        open={open !== null}
        onClose={() => setOpen(null)}
        title={open ? `Finding #${open.tile.finding.id}` : ""}
        subtitle={
          open
            ? `${open.tile.finding.class_name ?? NOT_MEASURED} • ${
                OUTCOME_LABEL[open.tile.finding.outcome] ?? open.tile.finding.outcome
              } • ${open.heatmap ? "Grad-CAM overlay" : "source image — no attribution overlay"}`
            : undefined
        }
        footer={
          open && (
            <div className="flex items-center gap-6 text-[12px] text-slate">
              <span>
                Mask IoU{" "}
                <span className={open.tile.mask?.mask_iou == null ? "italic" : "text-ink font-mono"}>
                  {open.tile.mask?.mask_iou == null ? NOT_MEASURED : open.tile.mask.mask_iou.toFixed(3)}
                </span>
              </span>
              <span>
                Box IoU{" "}
                <span className={open.tile.mask?.box_iou == null ? "italic" : "text-ink font-mono"}>
                  {open.tile.mask?.box_iou == null ? NOT_MEASURED : open.tile.mask.box_iou.toFixed(3)}
                </span>
              </span>
              <span>
                Confidence{" "}
                <span className={open.tile.finding.confidence == null ? "italic" : "text-ink font-mono"}>
                  {open.tile.finding.confidence == null
                    ? NOT_MEASURED
                    : `${(open.tile.finding.confidence * 100).toFixed(1)}%`}
                </span>
              </span>
            </div>
          )
        }
      >
        {open && (
          /* Full resolution here, not the preview the tile requested — the
             detail view is the one place the extra bytes buy something. */
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={
              open.heatmap
                ? api.heatmapUrl(open.tile.finding.id)
                : api.imageUrl(open.tile.finding.image_id)
            }
            alt={
              open.heatmap
                ? `Grad-CAM for finding ${open.tile.finding.id}`
                : `Source image for finding ${open.tile.finding.id}`
            }
            className="max-w-full max-h-[72vh] object-contain rounded"
          />
        )}
      </Lightbox>
    </div>
  );
}
