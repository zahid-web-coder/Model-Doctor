import { ReactNode } from "react";
import { PageShell } from "@/components/shared/PageShell";
import { RunNav } from "@/components/layout/RunNav";
import { findRun } from "@/lib/mock-data/runs";
import { num } from "@/lib/format";

export default async function RunLayout({
  children,
  params,
}: {
  children: ReactNode;
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const run = findRun(runId);

  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-end justify-between border-b border-border/40 mb-6 shrink-0 pt-2">
          <div className="mb-4">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-heading text-ink">Run #{runId}</h2>
              <span className="text-[10px] font-bold uppercase tracking-wide text-slate px-2.5 py-1 rounded-full bg-black/5">
                {run?.status ?? "unknown"}
              </span>
            </div>
            {/* Real run metadata, from the runs table shape. No LIVE badge:
                nothing streams, and the run is a completed analysis pass. */}
            <p className="text-[13px] text-slate">
              {run?.model ?? "n/a"} • {run?.dataset ?? "n/a"} • {run?.split ?? "n/a"} split •{" "}
              {num(run?.totalImages)} images
            </p>
          </div>
          <RunNav runId={runId} />
        </div>
        <div className="flex-1 overflow-hidden">{children}</div>
      </div>
    </PageShell>
  );
}
