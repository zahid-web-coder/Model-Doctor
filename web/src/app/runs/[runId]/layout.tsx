import { ReactNode } from "react";
import { PageShell } from "@/components/shared/PageShell";
import { RunNav } from "@/components/layout/RunNav";
import { api } from "@/lib/api/client";
import { modelName, datasetName } from "@/lib/derive";
import { num } from "@/lib/format";

export default async function RunLayout({
  children, params,
}: {
  children: ReactNode;
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const [run, outcomes] = await Promise.all([
    api.run(runId).catch(() => undefined),
    api.outcomes(runId).catch(() => undefined),
  ]);

  const findings = outcomes
    ? Object.values(outcomes).reduce((sum, n) => sum + n, 0)
    : null;

  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-end justify-between border-b border-border/40 mb-6 shrink-0 pt-2">
          <div className="mb-4">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-heading text-ink">Run #{runId}</h2>
              <span className="text-[10px] font-bold uppercase tracking-wide text-slate px-2.5 py-1 rounded-full bg-black/5">
                {run?.split ?? "n/a"}
              </span>
            </div>
            <p className="text-[13px] text-slate font-mono">
              {modelName(run) ?? "n/a"} • {datasetName(run) ?? "n/a"} • {num(findings)} findings
            </p>
          </div>
          <RunNav runId={runId} />
        </div>
        <div className="flex-1 overflow-hidden">{children}</div>
      </div>
    </PageShell>
  );
}
