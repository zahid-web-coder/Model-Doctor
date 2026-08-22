"use client";

import { use, useMemo, useState } from "react";
import { Download } from "lucide-react";
import { mockHeatmaps } from "@/lib/mock-data/heatmaps";
import { CLASSES } from "@/lib/mock-data/classes";
import { FilterSelect } from "@/components/shared/FilterSelect";
import { pct } from "@/lib/format";

/**
 * Grad-CAM attention, one tile per record.
 *
 * This screen previously rendered blank. The mock built its Tailwind classes
 * by interpolation (`from-slate-${300 + i*100}`), and Tailwind scans source
 * statically — it never saw those strings, so the classes were never generated
 * and every tile had no background. The tones are fixed literals now.
 */
export default function HeatmapsPage({ params }: { params: Promise<{ runId: string }> }) {
  use(params);
  const [typeFilter, setTypeFilter] = useState("All Failure Types");
  const [factorFilter, setFactorFilter] = useState("All Factors");
  const [classFilter, setClassFilter] = useState("All Classes");

  const factors = useMemo(
    () => ["All Factors", ...Array.from(new Set(mockHeatmaps.map((h) => h.factor)))],
    []
  );
  const types = useMemo(
    () => ["All Failure Types", ...Array.from(new Set(mockHeatmaps.map((h) => h.failureType)))],
    []
  );

  const visible = mockHeatmaps.filter((h) => {
    if (typeFilter !== "All Failure Types" && h.failureType !== typeFilter) return false;
    if (factorFilter !== "All Factors" && h.factor !== factorFilter) return false;
    if (classFilter !== "All Classes" && h.className !== classFilter) return false;
    return true;
  });

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Attention Heatmaps</h3>
          <p className="text-[13px] text-slate">Visualising model focus areas on failure instances.</p>
        </div>
        <div className="flex items-center gap-2">
          <FilterSelect value={typeFilter} options={types} onChange={setTypeFilter} width="w-[175px]" />
          <FilterSelect value={factorFilter} options={factors} onChange={setFactorFilter} width="w-[155px]" />
          <FilterSelect value={classFilter} options={["All Classes", ...CLASSES]} onChange={setClassFilter} width="w-[140px]" />
          <button type="button" className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors">
            <Download size={13} className="text-slate" /> Export
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 pb-2">
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
          {visible.map((h) => (
            <div key={h.id} className="bg-card border border-border/40 rounded-lg p-2 hover:border-brass/50 transition-colors cursor-pointer group">
              <div className={`w-full aspect-square rounded mb-2 relative overflow-hidden ${h.tone}`}>
                <div className="absolute inset-0 m-auto w-[60%] h-[60%] bg-gradient-to-tr from-blue-900/50 via-orange-400/50 to-red-600/50 mix-blend-screen rounded-full blur-[8px]" />
              </div>
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] text-ink">{h.id}</span>
                <span className={`font-mono text-[10px] ${h.confidence === null ? "text-slate italic" : "text-slate"}`}>
                  {pct(h.confidence, 0)}
                </span>
              </div>
              <p className="font-mono text-[10px] text-slate mt-0.5 truncate">{h.className} • {h.factor}</p>
            </div>
          ))}
          {visible.length === 0 && (
            <p className="col-span-full py-10 text-center text-slate text-[13px]">No heatmaps match these filters.</p>
          )}
        </div>
      </div>
    </div>
  );
}
