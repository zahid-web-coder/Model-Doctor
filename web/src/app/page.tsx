import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { PageShell } from "@/components/shared/PageShell";
import { HeroSlot } from "@/components/dashboard/HeroSlot";
import { StatCard } from "@/components/shared/StatCard";
import { mockRuns } from "@/lib/mock-data/runs";
import { mockFailures, FAILURE_LABEL, FailureInstance } from "@/lib/mock-data/failures";
import { mockRootCauses } from "@/lib/mock-data/root-causes";
import { num, pct, lift, pValue, predictedClass } from "@/lib/format";

/**
 * Overview.
 *
 * The panels that were here — Throughput, Live Inference, Model Health,
 * Defect Detected and the trend sparklines — are gone. None of them had a
 * source: nothing streams, there is no health metric in the schema, and there
 * is no time series to draw a trend from. What replaces them is the same
 * layout filled with figures the backend can actually produce.
 */
export default function Home() {
  const run = mockRuns[0];

  const byType = (t: FailureInstance["failureType"]) =>
    mockFailures.filter((f) => f.failureType === t).length;
  const total = mockFailures.length;
  const distribution = [
    { label: FAILURE_LABEL.false_negative, count: byType("false_negative"), tone: "bg-[#B3452F]" },
    { label: FAILURE_LABEL.wrong_class, count: byType("wrong_class"), tone: "bg-brass" },
    { label: FAILURE_LABEL.poor_localization, count: byType("poor_localization"), tone: "bg-slate" },
    { label: FAILURE_LABEL.false_positive, count: byType("false_positive"), tone: "bg-slate/50" },
  ];

  const replicated = mockRootCauses.filter((f) => f.impact === "High");

  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-y-auto custom-scrollbar gap-5 pr-1">
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,2.2fr)_minmax(0,1fr)] gap-5 shrink-0">
          <HeroSlot />

          <div className="flex flex-col gap-4">
            <StatCard label="Findings this run" value={num(run.failureCount)} sub={`of ${num(run.totalImages)} images`} tone="brass" />
            <StatCard label="Model" value={run.model} sub={`${run.dataset} • ${run.split} split`} />
            <div className="bg-card border border-border/40 rounded-lg p-4">
              <p className="text-[11px] font-medium text-slate mb-2">Replicated factors</p>
              {replicated.map((f) => (
                <div key={f.factor} className="flex items-baseline justify-between gap-2 mb-1.5 last:mb-0">
                  <span className="font-mono text-[12px] text-ink truncate">{f.factor}</span>
                  <span className="font-mono text-[12px] text-brass shrink-0">{lift(f.lift)}</span>
                </div>
              ))}
              <p className="text-[10px] text-slate mt-2 leading-relaxed">
                Significant on test and replicated on val. Only these are actionable.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-5 shrink-0">
          <section className="bg-card border border-border/40 rounded-lg p-4">
            <h4 className="text-[13px] font-medium text-ink mb-3">Failure Distribution</h4>
            <div className="flex h-2 rounded-full overflow-hidden mb-4">
              {distribution.map((d) => (
                <div key={d.label} className={d.tone} style={{ width: `${(d.count / total) * 100}%` }} />
              ))}
            </div>
            {distribution.map((d) => (
              <div key={d.label} className="flex items-center gap-2 mb-1.5 text-[12px]">
                <span className={`w-2 h-2 rounded-full ${d.tone}`} />
                <span className="text-slate flex-1">{d.label}</span>
                <span className="font-mono text-ink">{d.count}</span>
                <span className="font-mono text-slate w-[52px] text-right">
                  ({pct(d.count / total)})
                </span>
              </div>
            ))}
          </section>

          <section className="bg-card border border-border/40 rounded-lg p-4 xl:col-span-2">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-[13px] font-medium text-ink">Top Root Cause Factors</h4>
              <Link href={`/runs/${run.id}/root-causes`} className="text-[12px] text-slate hover:text-brass inline-flex items-center gap-1 transition-colors">
                View all <ArrowRight size={11} />
              </Link>
            </div>
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-left text-slate border-b border-border/40">
                  <th className="font-medium py-1.5">Factor</th>
                  <th className="font-medium py-1.5 text-right">Failures</th>
                  <th className="font-medium py-1.5 text-right">Lift</th>
                  <th className="font-medium py-1.5 text-right">P-value</th>
                </tr>
              </thead>
              <tbody>
                {mockRootCauses.slice(0, 6).map((f) => (
                  <tr key={f.factor} className={`border-b border-border/20 ${f.impact === null ? "opacity-55" : ""}`}>
                    <td className="py-1.5 font-mono text-ink">{f.factor}</td>
                    <td className="py-1.5 text-right font-mono text-ink">{f.failures}</td>
                    <td className={`py-1.5 text-right font-mono ${f.lift === null ? "text-slate italic" : "text-ink"}`}>{lift(f.lift)}</td>
                    <td className={`py-1.5 text-right font-mono ${f.pValue !== null && f.pValue > 0.05 ? "text-[#B3452F]" : "text-slate"}`}>{pValue(f.pValue)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>

        <section className="shrink-0">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-[13px] font-medium text-ink">Recent Findings</h4>
            <Link href={`/runs/${run.id}/failures`} className="text-[12px] text-slate hover:text-brass inline-flex items-center gap-1 transition-colors">
              View all <ArrowRight size={11} />
            </Link>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3">
            {mockFailures.slice(0, 8).map((f) => (
              <Link key={f.id} href={`/runs/${run.id}/failures`} className="bg-card border border-border/40 rounded-lg p-2 hover:border-brass/50 transition-colors">
                <div className="w-full aspect-[4/3] rounded mb-2 bg-gradient-to-tr from-slate-700 to-slate-500" />
                <p className="font-mono text-[10px] text-ink truncate">{predictedClass(f.groundTruth)}</p>
                <p className="font-mono text-[10px] text-slate truncate">{f.rootCauseFactor}</p>
              </Link>
            ))}
          </div>
        </section>
      </div>
    </PageShell>
  );
}
