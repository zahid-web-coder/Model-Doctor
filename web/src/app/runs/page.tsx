import { mockRuns } from "@/lib/mock-data/runs";
import { PageShell } from "@/components/shared/PageShell";
import Link from "next/link";

export default function RunsPage() {
  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-hidden">
        <div className="mb-6">
          <h2 className="text-2xl font-heading text-ink">Run History</h2>
          <p className="text-[13px] text-slate">View and compare past inspection runs.</p>
        </div>

        <div className="flex-1 bg-card border border-border/40 rounded-xl overflow-hidden shadow-sm flex flex-col">
          <div className="overflow-x-auto flex-1 custom-scrollbar">
            <table className="w-full text-[13px] text-left border-collapse whitespace-nowrap">
              <thead className="sticky top-0 bg-muted/90 backdrop-blur-sm z-10 border-b border-border/40">
                <tr>
                  <th className="py-3 px-4 font-medium text-slate">Run ID</th>
                  <th className="py-3 px-4 font-medium text-slate">Status</th>
                  <th className="py-3 px-4 font-medium text-slate">Model</th>
                  <th className="py-3 px-4 font-medium text-slate">Dataset</th>
                  <th className="py-3 px-4 font-medium text-slate">Date</th>
                  <th className="py-3 px-4 font-medium text-slate text-right">Images</th>
                  <th className="py-3 px-4 font-medium text-slate text-right">Failures</th>
                  <th className="py-3 px-4 font-medium text-slate text-right">Accuracy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/20">
                {mockRuns.map((run) => (
                  <tr 
                    key={run.id} 
                    className="hover:bg-muted/30 transition-colors cursor-pointer group"
                  >
                    <td className="py-3 px-4">
                      <Link href={`/runs/${run.id}/root-causes`} className="flex items-center gap-2 group-hover:text-brass transition-colors">
                        <span className="font-mono font-medium text-ink group-hover:text-brass">#{run.id}</span>
                        <span className="text-brass opacity-0 group-hover:opacity-100 transition-opacity">→</span>
                      </Link>
                    </td>
                    <td className="py-3 px-4">
                      {run.status === 'Live' && (
                        <div className="flex items-center gap-1.5 w-max bg-[#4CAF50]/10 px-2 py-0.5 rounded-full">
                          <div className="w-1.5 h-1.5 rounded-full bg-[#4CAF50] animate-pulse" />
                          <span className="text-[#4CAF50] text-[10px] font-bold uppercase">Live</span>
                        </div>
                      )}
                      {run.status === 'Complete' && (
                        <div className="flex items-center gap-1.5 w-max bg-slate/10 px-2 py-0.5 rounded-full">
                          <div className="w-1.5 h-1.5 rounded-full bg-slate" />
                          <span className="text-slate text-[10px] font-bold uppercase">Complete</span>
                        </div>
                      )}
                      {run.status === 'Failed' && (
                        <div className="flex items-center gap-1.5 w-max bg-[#E47260]/10 px-2 py-0.5 rounded-full">
                          <div className="w-1.5 h-1.5 rounded-full bg-[#E47260]" />
                          <span className="text-[#E47260] text-[10px] font-bold uppercase">Failed</span>
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-4 text-ink">{run.model}</td>
                    <td className="py-3 px-4 text-ink">{run.dataset}</td>
                    <td className="py-3 px-4 text-slate">{run.date}</td>
                    <td className="py-3 px-4 text-right font-mono text-ink">{run.totalImages.toLocaleString()}</td>
                    <td className="py-3 px-4 text-right font-mono text-ink">{run.failureCount.toLocaleString()}</td>
                    <td className="py-3 px-4 text-right font-mono font-medium text-ink">{(run.accuracy * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
