import { api } from "@/lib/api/client";
import { ExportButton } from "@/components/shared/ExportButton";
import { factorNote } from "@/lib/factors";
import { failureRate, correctRate, reading, impact } from "@/lib/derive";
import { pct, lift as fmtLift, pValue as fmtP, NOT_MEASURED } from "@/lib/format";

/**
 * Factor rates from `/runs/{id}/factor-rates`.
 *
 * The API returns counts, lift and p-value. It does not return a "reading" or
 * an "impact" band — those were invented in the mock, and are now derived in
 * `lib/derive.ts` from the figures the API does give, so the rule behind each
 * is in one place.
 *
 * The design's "Root Cause Evolution" line chart is absent: it plots factors
 * over calendar days, and the schema holds no time series to draw it from.
 */
export default async function RootCausesPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const factors = await api.factorRates(runId);
  const max = Math.max(1, ...factors.map((f) => f.failure_count));

  const tone = (value: string | null) =>
    value === "High" ? "text-[#B3452F]" : value === "Medium" ? "text-brass" : "text-slate";

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Root Causes</h3>
          <p className="text-[13px] text-slate">Factors statistically correlated with model failures.</p>
        </div>
        <ExportButton
          rows={factors.map((f) => ({
            factor: f.factor,
            means: factorNote(f.factor)?.rule ?? "",
            failure_count: f.failure_count,
            failure_total: f.failure_total,
            correct_count: f.correct_count,
            correct_total: f.correct_total,
            lift: f.lift,
            p_value: f.p_value,
            reading: reading(f),
          }))}
          columns={["factor", "means", "failure_count", "failure_total", "correct_count", "correct_total", "lift", "p_value", "reading"]}
          filename={`run-${runId}-root-causes`}
          scope={`all ${factors.length} factors`}
        />
      </div>

      {factors.length === 0 ? (
        <div className="flex-1 grid place-items-center">
          <p className="text-[13px] text-slate max-w-[380px] text-center">
            This run has no factor rates. They are written by the root-cause pass,
            so a database saved before that ran will not have them.
          </p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto custom-scrollbar pr-1 flex flex-col gap-5">
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.7fr)] gap-4">
            <section className="bg-card border border-border/40 rounded-lg p-4">
              <h4 className="text-[13px] font-medium text-ink mb-3">Impact Overview</h4>
              <div className="flex flex-col gap-2.5">
                {factors.map((f) => (
                  <div key={f.factor} className="flex items-center gap-3">
                    <span className="font-mono text-[11px] text-slate w-[118px] shrink-0 truncate">{f.factor}</span>
                    <div className="flex-1 h-3 bg-black/5 rounded-sm overflow-hidden">
                      <div
                        className={`h-full rounded-sm ${impact(f) === "High" ? "bg-brass" : "bg-slate/40"}`}
                        style={{ width: `${(f.failure_count / max) * 100}%` }}
                      />
                    </div>
                    <span className="font-mono text-[11px] text-ink w-[38px] text-right shrink-0">{f.failure_count}</span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-slate mt-3">Findings carrying each factor, of {factors[0]?.failure_total ?? 0} failures.</p>
            </section>

            <section className="bg-card border border-border/40 rounded-lg overflow-hidden">
              <h4 className="text-[13px] font-medium text-ink px-4 pt-4 pb-2">Root Cause Details</h4>
              <table className="w-full text-[13px]">
                <thead className="border-b border-border/40">
                  <tr className="text-left text-slate">
                    <th className="font-medium px-4 py-2">Factor</th>
                    <th className="font-medium px-3 py-2 text-right">In failures</th>
                    <th className="font-medium px-3 py-2 text-right">In correct</th>
                    <th className="font-medium px-3 py-2 text-right">Lift</th>
                    <th className="font-medium px-3 py-2 text-right">P-value</th>
                    <th className="font-medium px-3 py-2">Reading</th>
                  </tr>
                </thead>
                <tbody>
                  {factors.map((f) => {
                    const band = impact(f);
                    return (
                      <tr key={f.factor} className={`border-b border-border/25 ${band === null ? "opacity-55" : ""}`}>
                        <td className="px-4 py-2.5 font-mono text-[12px] text-ink">{f.factor}</td>
                        <td className="px-3 py-2.5 text-right font-mono text-[12px] text-ink">{pct(failureRate(f))}</td>
                        <td className="px-3 py-2.5 text-right font-mono text-[12px] text-slate">{pct(correctRate(f))}</td>
                        <td className={`px-3 py-2.5 text-right font-mono text-[12px] ${f.lift === null ? "text-slate italic" : "text-ink font-medium"}`}>
                          {fmtLift(f.lift)}
                        </td>
                        <td className={`px-3 py-2.5 text-right font-mono text-[12px] ${f.p_value !== null && f.p_value > 0.05 ? "text-[#B3452F]" : "text-slate"}`}>
                          {fmtP(f.p_value)}
                        </td>
                        <td className={`px-3 py-2.5 text-[12px] ${tone(band)}`}>{reading(f)}</td>
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
              {factors.map((f) => (
                <div key={f.factor} className="bg-card border border-border/40 rounded-lg px-4 py-3">
                  <div className="flex items-baseline gap-3 flex-wrap">
                    <span className="font-mono text-[12px] text-ink">{f.factor}</span>
                    {factorNote(f.factor)?.calibrated && (
                      <span className="text-[9px] uppercase tracking-wider text-brass border border-brass/40 rounded px-1 py-0.5">
                        threshold calibrated per dataset
                      </span>
                    )}
                    <span className="text-[11px] text-slate">
                      {f.failure_count} of {f.failure_total} failures ({pct(failureRate(f))}) carry this factor,
                      against {f.correct_count} of {f.correct_total} correct detections ({pct(correctRate(f))}) —{" "}
                      {reading(f)}.
                    </span>
                  </div>
                  {/* What the detector measures, so the statistics above can be
                      read without opening the backend. Unknown factors fall
                      through silently rather than being given an invented rule. */}
                  {factorNote(f.factor) && (
                    <div className="mt-2 pt-2 border-t border-border/30 flex flex-col gap-1">
                      <p className="text-[11px] text-ink leading-relaxed">
                        <span className="text-slate">Fires when: </span>
                        {factorNote(f.factor)!.rule}
                      </p>
                      <p className="text-[11px] text-slate leading-relaxed">
                        <span className="text-slate/70">Why it matters: </span>
                        {factorNote(f.factor)!.why}
                      </p>
                    </div>
                  )}
                  {impact(f) === null && f.lift !== null && (
                    <p className="text-[11px] text-slate mt-1.5 leading-relaxed">
                      Common in failures, but no more common than in correct detections. It describes
                      the dataset rather than the failures, so it is not actionable.
                    </p>
                  )}
                  {f.lift === null && (
                    <p className="text-[11px] text-slate mt-1.5 leading-relaxed">
                      Lift is {NOT_MEASURED} — too few findings carry this factor for a stable estimate.
                    </p>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
