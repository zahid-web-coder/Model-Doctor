import Link from "next/link";
import { ArrowRight, Plus, GitCompare } from "lucide-react";
import { PageShell } from "@/components/shared/PageShell";
import { api } from "@/lib/api/client";
import { modelName, datasetName } from "@/lib/derive";
import { num } from "@/lib/format";
import { DeleteRunButton } from "@/components/runs/DeleteRunButton";
import { RunName } from "@/components/runs/RunName";

/**
 * Run history, from `/runs`.
 *
 * Deleting a run is the one action offered here, and it goes to the control
 * API — the reader this page uses opens SQLite read-only and declares no
 * non-GET route. The page is a server component, so the button is a client
 * island and refreshes this route rather than mutating a local copy.
 *
 * Columns are what the run row actually carries: configuration and identity.
 * There is no accuracy or mAP column — that was excluded, and the row does not
 * hold one. Findings per run come from `/runs/{id}/outcomes`, requested
 * alongside so the table can show a count without inventing one.
 */
export default async function RunsPage() {
  // Uncached: this page deletes runs, and a stale list would show a row
  // that no longer exists.
  const runs = await api.runs(true).catch(() => []);
  const counts = await Promise.all(
    runs.map((run) =>
      api.outcomes(run.id)
        .then((o) => Object.values(o).reduce((sum, n) => sum + n, 0))
        .catch(() => null)
    )
  );

  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-start justify-between mb-5 shrink-0">
          <div>
            <h3 className="text-[17px] font-heading text-ink">Run History</h3>
            <p className="text-[13px] text-slate">Track and compare model runs.</p>
          </div>
          {/* Both of these were `<button>` with no handler: they rendered,
              they highlighted on hover, and they did nothing. The screens they
              imply already exist and are reachable from the rail, so the fix
              is to point at them rather than to build anything — and they are
              links, not buttons, because navigation is what they do. */}
          <div className="flex items-center gap-2">
            <Link
              href="/compare"
              className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors"
            >
              <GitCompare size={13} className="text-slate" /> Compare
            </Link>
            <Link
              href="/analyze"
              className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-brass/90 text-[13px] text-white hover:bg-brass transition-colors"
            >
              <Plus size={13} /> New Run
            </Link>
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden bg-card border border-border/40 rounded-lg">
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <table className="w-full text-[13px]">
              <thead className="sticky top-0 bg-card border-b border-border/40">
                <tr className="text-left text-slate">
                  <th className="font-medium px-4 py-3">Run ID</th>
                  <th className="font-medium px-3 py-3">Name</th>
                  <th className="font-medium px-3 py-3">Split</th>
                  <th className="font-medium px-3 py-3">Model</th>
                  <th className="font-medium px-3 py-3">Dataset</th>
                  <th className="font-medium px-3 py-3">Created</th>
                  <th className="font-medium px-3 py-3 text-right">Image size</th>
                  <th className="font-medium px-3 py-3 text-right">Confidence</th>
                  <th className="font-medium px-3 py-3 text-right">Findings</th>
                  <th className="font-medium px-3 py-3 w-[44px]"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run, i) => (
                  <tr key={run.id} className="border-b border-border/25 hover:bg-black/[0.02] transition-colors">
                    <td className="px-4 py-3">
                      <Link href={`/runs/${run.id}/failures`} className="inline-flex items-center gap-1.5 font-mono text-[12px] text-ink hover:text-brass transition-colors">
                        #{run.id} <ArrowRight size={12} />
                      </Link>
                    </td>
                    <td className="px-3 py-3 max-w-[220px]">
                      <RunName runId={run.id} name={run.name} />
                    </td>
                    <td className="px-3 py-3">
                      <span className="text-[10px] font-semibold uppercase tracking-wide px-2 py-0.5 rounded-full bg-black/5 text-slate">
                        {run.split}
                      </span>
                    </td>
                    <td className="px-3 py-3 font-mono text-[12px] text-ink">{modelName(run)}</td>
                    <td className="px-3 py-3 font-mono text-[12px] text-slate truncate max-w-[220px]">{datasetName(run)}</td>
                    <td className="px-3 py-3 text-slate">{new Date(run.created_at).toLocaleDateString()}</td>
                    <td className="px-3 py-3 text-right font-mono text-[12px] text-slate">{run.image_size}</td>
                    <td className="px-3 py-3 text-right font-mono text-[12px] text-slate">{run.confidence_threshold.toFixed(2)}</td>
                    <td className={`px-3 py-3 text-right font-mono text-[12px] ${counts[i] === null ? "text-slate italic" : "text-ink"}`}>
                      {num(counts[i])}
                    </td>
                    <td className="px-2 py-3 text-right">
                      <DeleteRunButton
                        runId={run.id}
                        label={`#${run.id} · ${run.name ?? modelName(run)}`}
                      />
                    </td>
                  </tr>
                ))}
                {runs.length === 0 && (
                  <tr><td colSpan={9} className="px-4 py-10 text-center text-slate">No runs in this database.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
