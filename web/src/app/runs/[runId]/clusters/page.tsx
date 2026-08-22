import { Download } from "lucide-react";
import { api } from "@/lib/api/client";
import { StatCard } from "@/components/shared/StatCard";
import { OUTCOME_LABEL } from "@/lib/api/rows";
import { pct, num } from "@/lib/format";

/**
 * Failure groups from `/runs/{id}/groups`, with members from
 * `/groups/{id}/members`.
 *
 * Groups are a deterministic signature over root-cause factors, so the label
 * *is* the definition — "small_object + thin_structure" means exactly the
 * findings carrying both. There is nothing to describe beyond that, which is
 * why the mock's prose descriptions are gone.
 *
 * **No thumbnails.** `/groups/{id}/members` returns `filename` and `path` but
 * no `image_id`, and `/images/{id}` only accepts an id. A member image cannot
 * be requested without one, so the cards show the member list instead of
 * guessing. Adding `image_id` to that endpoint is a backend change and is not
 * made here.
 */
export default async function ClustersPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const [groups, outcomes] = await Promise.all([
    api.groups(runId),
    api.outcomes(runId).catch(() => null),
  ]);

  const members = await Promise.all(groups.map((g) => api.groupMembers(g.id)));
  const totalFailures = outcomes
    ? outcomes.false_negative + outcomes.false_positive + outcomes.poor_localization + outcomes.wrong_class
    : null;
  const grouped = groups.reduce((sum, g) => sum + g.size, 0);
  const named = groups.filter((g) => g.label !== "unexplained").length;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Clusters</h3>
          <p className="text-[13px] text-slate">Failures grouped by their root-cause signature.</p>
        </div>
        <button type="button" className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors">
          <Download size={13} className="text-slate" /> Export
        </button>
      </div>

      <div className="flex gap-3 mb-5 shrink-0">
        <StatCard label="Total Clusters" value={num(groups.length)} />
        <StatCard label="Named signatures" value={num(named)} tone="brass" sub="excluding unexplained" />
        <StatCard label="Total Failures" value={num(totalFailures)} />
        <StatCard
          label="Coverage"
          value={totalFailures ? pct(grouped / totalFailures, 0) : "n/a"}
          sub={`${num(grouped)} grouped`}
        />
      </div>

      {groups.length === 0 ? (
        <div className="flex-1 grid place-items-center">
          <p className="text-[13px] text-slate max-w-[380px] text-center">
            This run has no groups. They are written by the grouping pass, so a database
            saved before that ran will not have them.
          </p>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto custom-scrollbar pr-1">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {groups.map((group, i) => {
              const list = members[i] ?? [];
              const share = totalFailures ? group.size / totalFailures : null;
              const unexplained = group.label === "unexplained";
              return (
                <div key={group.id} className="bg-card border border-border/40 rounded-lg p-4 flex flex-col">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <p className="font-mono text-[12px] text-ink">{group.label}</p>
                    {unexplained ? (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/5 text-slate shrink-0">unexplained</span>
                    ) : (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#B3452F]/10 text-[#B3452F] shrink-0">signature</span>
                    )}
                  </div>
                  <p className="text-[12px] text-slate mb-3">
                    {num(group.size)} findings{share !== null ? ` (${pct(share)})` : ""}
                  </p>

                  <div className="flex flex-col gap-1">
                    {list.slice(0, 4).map((m) => (
                      <div key={m.finding_id} className="flex items-center justify-between gap-2 text-[11px]">
                        <span className="font-mono text-slate truncate">{m.class_name ?? "n/a"}</span>
                        <span className="text-slate shrink-0">{OUTCOME_LABEL[m.outcome] ?? m.outcome}</span>
                      </div>
                    ))}
                    {list.length > 4 && (
                      <p className="text-[11px] text-slate mt-1">+{list.length - 4} more</p>
                    )}
                    {list.length === 0 && <p className="text-[11px] text-slate">No members returned.</p>}
                  </div>

                  {unexplained && (
                    <p className="text-[11px] text-slate mt-3 leading-relaxed border-t border-border/40 pt-2">
                      No discriminating factor accounts for these. Listed rather than hidden.
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
