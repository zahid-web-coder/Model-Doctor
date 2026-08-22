"use client";

import { use, useState } from "react";
import { Download, LayoutGrid, List } from "lucide-react";
import { mockClusters, CLUSTER_STATS } from "@/lib/mock-data/clusters";
import { StatCard } from "@/components/shared/StatCard";
import { Pagination } from "@/components/shared/Pagination";
import { pct } from "@/lib/format";

const PAGE_SIZE = 6;

/** Failure groups. Matches the design: four summary tiles, a card grid with
 *  impact badges, a grid/list toggle and pagination. */
export default function ClustersPage({ params }: { params: Promise<{ runId: string }> }) {
  use(params);
  const [view, setView] = useState<"grid" | "list">("grid");
  const [page, setPage] = useState(1);

  const pageCount = Math.max(1, Math.ceil(CLUSTER_STATS.totalClusters / PAGE_SIZE));
  const visible = mockClusters.slice(0, PAGE_SIZE);

  const impactTone = (impact: string) =>
    impact === "High" ? "bg-[#B3452F]/10 text-[#B3452F]" : "bg-black/5 text-slate";

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Clusters</h3>
          <p className="text-[13px] text-slate">Group similar failures to identify patterns.</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 text-[12px] text-slate">
            <span className="mr-1">View as</span>
            <button type="button" aria-label="Grid view" onClick={() => setView("grid")}
              className={`w-7 h-7 grid place-items-center rounded border ${view === "grid" ? "border-brass/50 bg-[#EBE6D8]" : "border-border/50"}`}>
              <LayoutGrid size={13} />
            </button>
            <button type="button" aria-label="List view" onClick={() => setView("list")}
              className={`w-7 h-7 grid place-items-center rounded border ${view === "list" ? "border-brass/50 bg-[#EBE6D8]" : "border-border/50"}`}>
              <List size={13} />
            </button>
          </div>
          <button type="button" className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors">
            <Download size={13} className="text-slate" /> Export
          </button>
        </div>
      </div>

      <div className="flex gap-3 mb-5 shrink-0">
        <StatCard label="Total Clusters" value={CLUSTER_STATS.totalClusters.toString()} />
        <StatCard label="High Impact" value={CLUSTER_STATS.highImpact.toString()} tone="brass" />
        <StatCard label="Total Failures" value={CLUSTER_STATS.totalFailures.toLocaleString()} />
        <StatCard label="Coverage" value={pct(CLUSTER_STATS.coveragePct, 0)} />
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-1">
        <div className={view === "grid" ? "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4" : "flex flex-col gap-2"}>
          {visible.map((c, i) => (
            <div key={c.id} className="bg-card border border-border/40 rounded-lg p-3 hover:border-brass/50 transition-colors cursor-pointer">
              {view === "grid" && (
                <div className="w-full aspect-[4/3] rounded mb-3 bg-gradient-to-tr from-slate-700 via-stone-600 to-slate-500" />
              )}
              <div className="flex items-start justify-between gap-2 mb-1">
                <p className="text-[13px] font-medium text-ink">
                  <span className="text-slate mr-1.5">{i + 1}</span>
                  <span className="font-mono text-[12px]">{c.name}</span>
                </p>
                <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${impactTone(c.impact)}`}>
                  {c.impact}
                </span>
              </div>
              <p className="text-[12px] text-slate mb-2">
                {c.memberCount} failures ({pct(c.sharePct)})
              </p>
              <p className="text-[11px] leading-relaxed text-slate">{c.description}</p>
            </div>
          ))}
        </div>
      </div>

      <Pagination page={page} pageCount={pageCount} total={CLUSTER_STATS.totalClusters} pageSize={PAGE_SIZE} onChange={setPage} />
    </div>
  );
}
