"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowRight, Plus, GitCompare } from "lucide-react";
import { PageShell } from "@/components/shared/PageShell";
import { Pagination } from "@/components/shared/Pagination";
import { mockRuns } from "@/lib/mock-data/runs";
import { num } from "@/lib/format";

const PAGE_SIZE = 10;

/**
 * Run history.
 *
 * No accuracy or mAP column: that was excluded, and the run row in the schema
 * carries configuration and counts, not a headline score. Every run id routes
 * to that run, and the rest of the app follows the selection.
 */
export default function RunsPage() {
  const [page, setPage] = useState(1);
  const pageCount = Math.max(1, Math.ceil(mockRuns.length / PAGE_SIZE));
  const visible = mockRuns.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-start justify-between mb-5 shrink-0">
          <div>
            <h3 className="text-[17px] font-heading text-ink">Run History</h3>
            <p className="text-[13px] text-slate">Track and compare model runs.</p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors">
              <GitCompare size={13} className="text-slate" /> Compare
            </button>
            <button type="button" className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-brass/90 text-[13px] text-white hover:bg-brass transition-colors">
              <Plus size={13} /> New Run
            </button>
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden bg-card border border-border/40 rounded-lg">
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <table className="w-full text-[13px]">
              <thead className="sticky top-0 bg-card border-b border-border/40">
                <tr className="text-left text-slate">
                  <th className="font-medium px-4 py-3">Run ID</th>
                  <th className="font-medium px-3 py-3">Status</th>
                  <th className="font-medium px-3 py-3">Model</th>
                  <th className="font-medium px-3 py-3">Dataset</th>
                  <th className="font-medium px-3 py-3">Split</th>
                  <th className="font-medium px-3 py-3">Date</th>
                  <th className="font-medium px-3 py-3 text-right">Images</th>
                  <th className="font-medium px-3 py-3 text-right">Findings</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((run) => (
                  <tr key={run.id} className="border-b border-border/25 hover:bg-black/[0.02] transition-colors">
                    <td className="px-4 py-3">
                      <Link href={`/runs/${run.id}/failures`} className="inline-flex items-center gap-1.5 font-mono text-[12px] text-ink hover:text-brass transition-colors">
                        #{run.id} <ArrowRight size={12} />
                      </Link>
                    </td>
                    <td className="px-3 py-3">
                      <span className={`text-[10px] font-semibold uppercase tracking-wide px-2 py-0.5 rounded-full ${
                        run.status === "Failed" ? "bg-[#B3452F]/10 text-[#B3452F]" : "bg-black/5 text-slate"
                      }`}>
                        {run.status}
                      </span>
                    </td>
                    <td className="px-3 py-3 font-mono text-[12px] text-ink">{run.model}</td>
                    <td className="px-3 py-3 font-mono text-[12px] text-slate">{run.dataset}</td>
                    <td className="px-3 py-3 font-mono text-[12px] text-slate">{run.split}</td>
                    <td className="px-3 py-3 text-slate">{run.date}</td>
                    <td className="px-3 py-3 text-right font-mono text-[12px] text-ink">{num(run.totalImages)}</td>
                    <td className="px-3 py-3 text-right font-mono text-[12px] text-ink">{num(run.failureCount)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="px-4 border-t border-border/40">
            <Pagination page={page} pageCount={pageCount} total={mockRuns.length} pageSize={PAGE_SIZE} onChange={setPage} />
          </div>
        </div>
      </div>
    </PageShell>
  );
}
