import { api } from "@/lib/api/client";
import { StatCard } from "@/components/shared/StatCard";
import { ExportButton } from "@/components/shared/ExportButton";
import { num } from "@/lib/format";

/**
 * Suggested investigations from `/runs/{id}/recommendations`.
 *
 * **Each one is a direction to investigate, not a fix.** The rationale is
 * shown in full rather than summarised, because the recommendation without its
 * evidence is just an assertion — and the evidence is what an engineer would
 * check before acting.
 *
 * **Non-actionable recommendations are shown, marked.** A group the evidence
 * does not yet support is still a group that was examined; hiding it would
 * leave a reader unable to tell whether the quiet failures were considered and
 * dismissed or never considered at all. Sorted by priority, so the ones with
 * the most behind them are read first.
 */
export default async function RecommendationsPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const recommendations = await api.recommendations(runId);

  const actionable = recommendations.filter((r) => r.actionable);
  const affected = recommendations.reduce((sum, r) => sum + r.affected, 0);
  const replicated = recommendations.filter((r) => r.status === "replicated");

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Recommendations</h3>
          <p className="text-[13px] text-slate">
            What the evidence suggests investigating next.
          </p>
        </div>
        <ExportButton
          rows={recommendations}
          columns={[
            "id", "cluster_label", "rule", "action", "rationale",
            "status", "actionable", "affected", "priority",
          ]}
          filename={`run-${runId}-recommendations`}
          scope={`all ${recommendations.length} recommendations`}
        />
      </div>

      {recommendations.length === 0 ? (
        <div className="flex-1 grid place-items-center">
          <p className="text-[13px] text-slate max-w-[420px] text-center leading-relaxed">
            This run has no recommendations. They are written by the
            recommendations pass, so a database saved before that ran will not
            have them — and a run whose failures form no group the rules
            recognise legitimately produces none.
          </p>
        </div>
      ) : (
        <>
          <div className="flex gap-3 mb-5 shrink-0">
            <StatCard label="Recommendations" value={num(recommendations.length)} tone="brass" />
            <StatCard
              label="Actionable"
              value={num(actionable.length)}
              sub={`${recommendations.length - actionable.length} need more evidence`}
            />
            <StatCard label="Failures covered" value={num(affected)} />
            <StatCard
              label="Replicated"
              value={num(replicated.length)}
              sub="agreed by another run"
            />
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar pr-1 flex flex-col gap-3">
            {recommendations.map((entry) => (
              <article
                key={entry.id}
                className={`bg-card border rounded-lg p-4 ${
                  entry.actionable ? "border-border/40" : "border-border/25 opacity-75"
                }`}
              >
                <div className="flex items-start justify-between gap-4 mb-2">
                  <p
                    className={`text-[14px] leading-snug ${
                      entry.actionable ? "text-ink font-medium" : "text-slate"
                    }`}
                  >
                    {entry.action}
                  </p>
                  <div className="flex items-center gap-2 shrink-0">
                    {entry.status === "replicated" && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#66805A]/10 text-[#66805A] whitespace-nowrap">
                        replicated
                      </span>
                    )}
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap ${
                        entry.actionable
                          ? "bg-brass/10 text-brass"
                          : "bg-black/5 text-slate"
                      }`}
                    >
                      {entry.actionable ? "actionable" : "needs evidence"}
                    </span>
                  </div>
                </div>

                <p className="text-[12px] text-slate leading-relaxed mb-3">
                  {entry.rationale}
                </p>

                <div className="flex items-center gap-4 text-[11px] text-slate border-t border-border/30 pt-2.5">
                  {entry.cluster_label && (
                    <span className="font-mono text-ink">{entry.cluster_label}</span>
                  )}
                  <span>
                    <span className="text-ink font-medium">{num(entry.affected)}</span>{" "}
                    failure{entry.affected === 1 ? "" : "s"}
                  </span>
                  <span className="font-mono text-slate/70">{entry.rule}</span>
                </div>
              </article>
            ))}
          </div>

          <p className="text-[11px] text-slate mt-3 shrink-0 leading-relaxed border-t border-border/40 pt-3">
            Recommendations are directions to investigate, backed by the
            evidence shown with each. They are not guaranteed fixes, and nothing
            acts on them automatically — the engineer decides.
          </p>
        </>
      )}
    </div>
  );
}
