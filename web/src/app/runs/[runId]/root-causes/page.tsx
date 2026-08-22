"use client";

import { use } from "react";
import { Download } from "lucide-react";
import { mockRootCauses } from "@/lib/mock-data/root-causes";
import { pct, lift, pValue, NOT_MEASURED } from "@/lib/format";

/**
 * Factor rates with lift and significance.
 *
 * The design's Impact Overview bar chart is drawn from the failure counts we
 * actually hold. The "Root Cause Evolution" line chart from the design is not
 * here: it plots factors over calendar days, and there is no time series in
 * the schema to draw it from. A chart with invented history is worse than an
 * absent one.
 */
export default function RootCausesPage({ params }: { params: Promise<{ runId: string }> }) {
  use(params);
  const max = Math.max(...mockRootCauses.map((f) => f.failures));

  const impactTone = (impact: string | null) =>
    impact === "High" ? "text-[#B3452F]" : impact === "Medium" ? "text-brass" : "text-slate";

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Root Causes</h3>
          <p className="text-[13px] text-slate">Understand what is causing model failures.</p>
        </div>
        <button type="button" className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors">
          <Download size={13} className="text-slate" /> Export
        </button>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-1 flex flex-col gap-5">
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)] gap-4">
          <section className="bg-card border border-border/40 rounded-lg p-4">
            <h4 className="text-[13px] font-medium text-ink mb-3">Impact Overview</h4>
            <div className="flex flex-col gap-2.5">
              {mockRootCauses.map((f) => (
                <div key={f.factor} className="flex items-center gap-3">
                  <span className="font-mono text-[11px] text-slate w-[110px] shrink-0 truncate">{f.factor}</span>
                  <div className="flex-1 h-3 bg-black/5 rounded-sm overflow-hidden">
                    <div
                      className={`h-full rounded-sm ${f.impact === "High" ? "bg-brass" : "bg-slate/40"}`}
                      style={{ width: `${(f.failures / max) * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[11px] text-ink w-[38px] text-right shrink-0">{f.failures}</span>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-slate mt-3">Number of failures carrying each factor.</p>
          </section>

          <section className="bg-card border border-border/40 rounded-lg overflow-hidden">
            <h4 className="text-[13px] font-medium text-ink px-4 pt-4 pb-2">Root Cause Details</h4>
            <table className="w-full text-[13px]">
              <thead className="border-b border-border/40">
                <tr className="text-left text-slate">
                  <th className="font-medium px-4 py-2">Factor</th>
                  <th className="font-medium px-3 py-2 text-right">Failures</th>
                  <th className="font-medium px-3 py-2 text-right">Lift</th>
                  <th className="font-medium px-3 py-2 text-right">P-value</th>
                  <th className="font-medium px-3 py-2">Impact</th>
                </tr>
              </thead>
              <tbody>
                {mockRootCauses.map((f) => {
                  const inert = f.impact === null;
                  return (
                    <tr key={f.factor} className={`border-b border-border/25 ${inert ? "opacity-55" : ""}`}>
                      <td className="px-4 py-2.5 font-mono text-[12px] text-ink">{f.factor}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-[12px] text-ink">{f.failures}</td>
                      <td className={`px-3 py-2.5 text-right font-mono text-[12px] ${f.lift === null ? "text-slate italic" : "text-ink font-medium"}`}>
                        {lift(f.lift)}
                      </td>
                      <td className={`px-3 py-2.5 text-right font-mono text-[12px] ${f.pValue !== null && f.pValue > 0.05 ? "text-[#B3452F]" : "text-slate"}`}>
                        {pValue(f.pValue)}
                      </td>
                      <td className={`px-3 py-2.5 text-[12px] ${impactTone(f.impact)}`}>
                        {f.impact ?? NOT_MEASURED}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>
        </div>

        <section>
          <h4 className="text-[13px] font-medium text-ink mb-2">Attribution &amp; Evidence</h4>
          <div className="flex flex-col gap-2">
            {mockRootCauses.map((f) => (
              <div key={f.factor} className="bg-card border border-border/40 rounded-lg px-4 py-3">
                <div className="flex items-baseline gap-3 mb-1.5">
                  <span className="font-mono text-[12px] text-ink">{f.factor}</span>
                  <span className="text-[11px] text-slate">
                    {pct(f.rateInFailures)} of failures vs {pct(f.rateInCorrect)} of correct detections — {f.reading}
                  </span>
                </div>
                {f.evidence.map((line) => (
                  <p key={line} className="text-[12px] text-slate leading-relaxed">{line}</p>
                ))}
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
