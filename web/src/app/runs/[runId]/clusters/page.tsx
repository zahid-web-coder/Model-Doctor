import { api } from "@/lib/api/client";
import { StatCard } from "@/components/shared/StatCard";
import { ExportButton } from "@/components/shared/ExportButton";
import { ClusterImages } from "@/components/dashboard/ClusterImages";
import { pct, num } from "@/lib/format";
import { factorNote, factorsInSignature, signatureSummary, UNEXPLAINED } from "@/lib/factors";

/**
 * Failure groups from `/runs/{id}/groups`, with members from
 * `/groups/{id}/members`.
 *
 * Groups are a deterministic signature over root-cause factors, so the label
 * *is* the definition — "small_object + thin_structure" means exactly the
 * findings carrying both. There is nothing to describe beyond that, which is
 * why the mock's prose descriptions are gone.
 *
 * **The cards lead with the photographs.** A signature is a claim about what
 * its failures have in common — "small_object + thin_structure, 20 findings"
 * is a definition, and a list of class names and outcomes repeats the
 * definition rather than evidencing it. Whether those twenty really are small
 * and thin, and whether the `unexplained` ones share something the detectors
 * have no name for, are questions answered by looking. So each card shows its
 * members and opens into the same overlay the Failures and Images screens use.
 *
 * Image dimensions and predicted outlines are fetched once for the run and
 * handed down, rather than per card: the cards share images, and a request per
 * cluster would fetch the same run-wide tables four times over.
 */
export default async function ClustersPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  const [groups, outcomes, images, masks] = await Promise.all([
    api.groups(runId),
    api.outcomes(runId).catch(() => null),
    api.images(runId).catch(() => []),
    api.maskFindings(runId),
  ]);

  const dimensions = Object.fromEntries(
    images
      .filter((i) => typeof i.width === "number" && typeof i.height === "number")
      .map((i) => [i.id, [i.width as number, i.height as number] as [number, number]]),
  );
  const polygons = Object.fromEntries(
    masks
      .filter((m) => m.pred_polygon !== null)
      .map((m) => [m.finding_id, m.pred_polygon as number[][]]),
  );

  const members = await Promise.all(groups.map((g) => api.groupMembers(g.id)));
  const totalFailures = outcomes
    ? outcomes.false_negative + outcomes.false_positive + outcomes.poor_localization + outcomes.wrong_class
    : null;
  const grouped = groups.reduce((sum, g) => sum + g.size, 0);
  const named = groups.filter((g) => g.label !== UNEXPLAINED).length;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-start justify-between mb-4 shrink-0">
        <div>
          <h3 className="text-[17px] font-heading text-ink">Clusters</h3>
          <p className="text-[13px] text-slate">Failures grouped by their root-cause signature.</p>
        </div>
        <ExportButton
          rows={groups.map((g, i) => ({
            cluster_id: g.id,
            label: g.label,
            size: g.size,
            method: g.method,
            means: signatureSummary(g.label),
            members: (members[i] ?? []).map((m) => m.finding_id).join(" "),
          }))}
          columns={["cluster_id", "label", "size", "method", "means", "members"]}
          filename={`run-${runId}-clusters`}
          scope={`all ${groups.length} clusters`}
        />
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
              const unexplained = group.label === UNEXPLAINED;
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
                  <p className="text-[12px] text-slate mb-2">
                    {num(group.size)} findings{share !== null ? ` (${pct(share)})` : ""}
                  </p>

                  {/* What the signature means, assembled from its parts. A
                      combined label is a conjunction, so it is explained as
                      one rather than treated as a category of its own. */}
                  <p className="text-[11px] text-slate leading-relaxed mb-3">
                    {signatureSummary(group.label)}
                  </p>

                  {!unexplained && (
                    <dl className="mb-3 flex flex-col gap-2 border-l-2 border-brass/30 pl-2.5">
                      {factorsInSignature(group.label).map((name) => {
                        const note = factorNote(name);
                        return (
                          <div key={name}>
                            <dt className="text-[11px] text-ink font-medium">
                              {note?.title ?? name}
                              {note?.calibrated && (
                                <span className="ml-1.5 text-[9px] uppercase tracking-wider text-brass">
                                  per-dataset
                                </span>
                              )}
                            </dt>
                            <dd className="text-[10.5px] text-slate leading-relaxed">
                              {note?.rule ?? "This build has no description for this factor."}
                            </dd>
                          </div>
                        );
                      })}
                    </dl>
                  )}

                  <ClusterImages
                    runId={runId}
                    label={group.label}
                    members={list}
                    dimensions={dimensions}
                    polygons={polygons}
                  />

                  {unexplained && (
                    <p className="text-[11px] text-slate mt-3 leading-relaxed border-t border-border/40 pt-2">
                      Worth reading as a gap in the detectors rather than a property of the
                      failures: something made these fail, and none of the measured
                      conditions caught it.
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
