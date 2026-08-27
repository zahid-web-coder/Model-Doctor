import Link from "next/link";
import { PageShell } from "@/components/shared/PageShell";
import { api } from "@/lib/api/client";
import { FileText, ArrowRight } from "lucide-react";

/**
 * Reports.
 *
 * **There is no report generation in the backend**, so this does not pretend
 * to list documents. What it does instead is tell a reader where the exports
 * that *do* exist live, because the honest answer to "where are my reports?"
 * is "each screen exports what it holds" rather than an empty table.
 *
 * The per-screen exports are deliberately scoped to what the screen shows —
 * a paged table exports its page. That limit is stated here rather than
 * discovered when a file turns out to hold ten rows.
 */

const EXPORTS = [
  {
    tab: "Failures",
    slug: "failures",
    gives: "Findings with outcome, class, confidence and IoU.",
    scope: "The rows on the current page, after filters.",
  },
  {
    tab: "Root Causes",
    slug: "root-causes",
    gives: "Every factor with its counts, lift, p-value and what it measures.",
    scope: "All factors for the run.",
  },
  {
    tab: "Clusters",
    slug: "clusters",
    gives: "Each signature with its size, meaning and member finding ids.",
    scope: "All clusters for the run.",
  },
  {
    tab: "Heatmaps",
    slug: "heatmaps",
    gives: "The heatmap index — finding, outcome, class and outline agreement.",
    scope: "The tiles currently shown, after filters.",
  },
] as const;

export default async function ReportsPage() {
  // The links need a run to point at. The most recent one is the useful
  // default; if the API is unreachable the page still renders, just without
  // deep links, because explaining where exports live does not depend on it.
  const runs = await api.runs().catch(() => []);
  const latest = runs.at(-1) ?? null;

  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-hidden">
        <h3 className="text-[17px] font-heading text-ink">Reports</h3>
        <p className="text-[13px] text-slate mb-6">Export and share diagnosis results.</p>

        <div className="flex-1 overflow-y-auto custom-scrollbar pr-1">
          <div className="rounded-lg border border-brass/40 bg-brass/5 px-4 py-3 mb-6 max-w-[760px]">
            <p className="text-[13px] text-ink font-medium mb-1">
              Compiled reports are not generated yet
            </p>
            <p className="text-[12px] text-slate leading-relaxed">
              Nothing in the backend writes or stores a report document, so there is no
              list here rather than a list of invented ones. Until there is, every
              analysis screen exports the data it is showing, as CSV you can open
              directly.
            </p>
          </div>

          <h4 className="text-[13px] font-medium text-ink mb-3">Available exports</h4>
          <div className="flex flex-col gap-2 max-w-[760px]">
            {EXPORTS.map((item) => (
              <div
                key={item.slug}
                className="bg-card border border-border/40 rounded-lg px-4 py-3 flex items-start justify-between gap-4"
              >
                <div className="min-w-0">
                  <p className="text-[13px] text-ink font-medium mb-0.5">{item.tab}</p>
                  <p className="text-[12px] text-slate leading-relaxed">{item.gives}</p>
                  <p className="text-[11px] text-slate/80 mt-1">
                    <span className="uppercase tracking-wider text-[9px] mr-1.5">covers</span>
                    {item.scope}
                  </p>
                </div>
                {latest && (
                  <Link
                    href={`/runs/${latest.id}/${item.slug}`}
                    className="shrink-0 flex items-center gap-1.5 text-[12px] text-brass hover:underline"
                  >
                    Open <ArrowRight size={12} />
                  </Link>
                )}
              </div>
            ))}
          </div>

          {!latest && (
            <p className="text-[12px] text-slate mt-4 flex items-center gap-2">
              <FileText size={13} className="text-slate/60" />
              No runs are available, so there is nothing to link to yet.
            </p>
          )}
        </div>
      </div>
    </PageShell>
  );
}
