import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { PageShell } from "@/components/shared/PageShell";
import { HeroSlot } from "@/components/dashboard/HeroSlot";
import { StatCard } from "@/components/shared/StatCard";
import { api } from "@/lib/api/client";
import { FAILURE_OUTCOMES, OUTCOME_LABEL } from "@/lib/api/rows";
import { reading, impact, modelName, datasetName } from "@/lib/derive";
import { num, pct, lift as fmtLift, pValue as fmtP } from "@/lib/format";

/**
 * Overview, for the most recent run.
 *
 * Every figure here comes from `/runs/{id}/outcomes` and
 * `/runs/{id}/factor-rates`. The panels that used to sit in this space —
 * Throughput, Live Inference, Model Health, Defect Detected — had no source
 * and are gone.
 */
export default async function Home() {
  const runs = await api.runs().catch(() => []);
  const run = runs[0];
  if (!run) {
    return (
      <PageShell>
        <div className="flex-1 grid place-items-center">
          <p className="text-[13px] text-slate">No runs in this database yet.</p>
        </div>
      </PageShell>
    );
  }

  const [outcomes, factors] = await Promise.all([
    api.outcomes(run.id).catch(() => null),
    api.factorRates(run.id),
  ]);

  // `outcomes` is null only when the request failed. The client fills in
  // buckets the endpoint omitted, so a successful response always carries all
  // five keys and these sums are numbers — a run with nothing to report gives
  // zero, which is a finding, not a gap.
  const totalFailures = outcomes ? FAILURE_OUTCOMES.reduce((s, k) => s + outcomes[k], 0) : null;
  const totalFindings = outcomes ? Object.values(outcomes).reduce((s, n) => s + n, 0) : null;
  const tones: Record<string, string> = {
    false_negative: "bg-[#B3452F]",
    wrong_class: "bg-brass",
    poor_localization: "bg-slate",
    false_positive: "bg-slate/50",
  };
  const replicated = factors.filter((f) => impact(f) === "High");

  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-y-auto custom-scrollbar gap-5 pr-1">
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,2.2fr)_minmax(0,1fr)] gap-5 shrink-0">
          <HeroSlot />
          <div className="flex flex-col gap-4">
            <StatCard label="Failures in latest run" value={num(totalFailures)} sub={`of ${num(totalFindings)} findings`} tone="brass" />
            <StatCard label="Model" value={modelName(run) ?? "n/a"} sub={`${datasetName(run) ?? "n/a"} • ${run.split}`} />
            <div className="bg-card border border-border/40 rounded-lg p-4">
              <p className="text-[11px] font-medium text-slate mb-2">Significant factors</p>
              {replicated.length === 0 && <p className="text-[12px] text-slate">None reached significance.</p>}
              {replicated.map((f) => (
                <div key={f.factor} className="flex items-baseline justify-between gap-2 mb-1.5 last:mb-0">
                  <span className="font-mono text-[12px] text-ink truncate">{f.factor}</span>
                  <span className="font-mono text-[12px] text-brass shrink-0">{fmtLift(f.lift)}</span>
                </div>
              ))}
              <p className="text-[10px] text-slate mt-2 leading-relaxed">
                Lift with p &lt; 0.05. Factors that are common but not significant are not listed.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-5 shrink-0">
          <section className="bg-card border border-border/40 rounded-lg p-4">
            <h4 className="text-[13px] font-medium text-ink mb-3">Failure Distribution</h4>
            {/* Three distinct states, deliberately not collapsed into two. A
                failed request is not the same as a run with no failures, and
                showing "no outcomes recorded" for a clean run would report an
                absence of data where there is an absence of *failures*. The
                old guard tested `totalFailures` for truthiness, which made
                both zero and NaN read as missing data. */}
            {outcomes === null || totalFailures === null ? (
              <p className="text-[12px] text-slate">Outcomes could not be read for this run.</p>
            ) : totalFailures === 0 ? (
              <p className="text-[12px] text-slate">
                No failures in this run — all {num(outcomes.correct)} findings were correct.
              </p>
            ) : (
              <>
                <div className="flex h-2 rounded-full overflow-hidden mb-4">
                  {FAILURE_OUTCOMES.map((k) => (
                    <div key={k} className={tones[k]} style={{ width: `${(outcomes[k] / totalFailures) * 100}%` }} />
                  ))}
                </div>
                {FAILURE_OUTCOMES.map((k) => (
                  <div key={k} className="flex items-center gap-2 mb-1.5 text-[12px]">
                    <span className={`w-2 h-2 rounded-full ${tones[k]}`} />
                    <span className="text-slate flex-1">{OUTCOME_LABEL[k]}</span>
                    <span className="font-mono text-ink">{outcomes[k]}</span>
                    <span className="font-mono text-slate w-[52px] text-right">({pct(outcomes[k] / totalFailures)})</span>
                  </div>
                ))}
                <p className="text-[10px] text-slate mt-3">
                  {num(outcomes.correct)} correct detections are excluded from this split.
                </p>
              </>
            )}
          </section>

          <section className="bg-card border border-border/40 rounded-lg p-4 xl:col-span-2">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-[13px] font-medium text-ink">Top Root Cause Factors</h4>
              <Link href={`/runs/${run.id}/root-causes`} className="text-[12px] text-slate hover:text-brass inline-flex items-center gap-1 transition-colors">
                View all <ArrowRight size={11} />
              </Link>
            </div>
            {factors.length === 0 ? (
              <p className="text-[12px] text-slate">No factor rates for this run.</p>
            ) : (
              <table className="w-full text-[12px]">
                <thead>
                  <tr className="text-left text-slate border-b border-border/40">
                    <th className="font-medium py-1.5">Factor</th>
                    <th className="font-medium py-1.5 text-right">Failures</th>
                    <th className="font-medium py-1.5 text-right">Lift</th>
                    <th className="font-medium py-1.5 text-right">P-value</th>
                    <th className="font-medium py-1.5 pl-3">Reading</th>
                  </tr>
                </thead>
                <tbody>
                  {factors.slice(0, 6).map((f) => (
                    <tr key={f.factor} className={`border-b border-border/20 ${impact(f) === null ? "opacity-55" : ""}`}>
                      <td className="py-1.5 font-mono text-ink">{f.factor}</td>
                      <td className="py-1.5 text-right font-mono text-ink">{f.failure_count}</td>
                      <td className={`py-1.5 text-right font-mono ${f.lift === null ? "text-slate italic" : "text-ink"}`}>{fmtLift(f.lift)}</td>
                      <td className={`py-1.5 text-right font-mono ${f.p_value !== null && f.p_value > 0.05 ? "text-[#B3452F]" : "text-slate"}`}>{fmtP(f.p_value)}</td>
                      <td className="py-1.5 pl-3 text-slate">{reading(f)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </div>
      </div>
    </PageShell>
  );
}
