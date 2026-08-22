import { ReactNode } from "react";
import { PageShell } from "@/components/shared/PageShell";
import { RunNav } from "@/components/layout/RunNav";

interface RunLayoutProps {
  children: ReactNode;
  params: Promise<{ runId: string }>;
}

export default async function RunLayout({ children, params }: RunLayoutProps) {
  const resolvedParams = await params;
  
  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-hidden">
        {/* Run Header & Sub Navigation */}
        <div className="flex items-end justify-between border-b border-border/40 pb-0 mb-6 shrink-0 pt-2">
          <div className="mb-4">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-heading text-ink">Run #{resolvedParams.runId}</h2>
              <div className="flex items-center gap-1.5 bg-[#4CAF50]/10 px-2.5 py-1 rounded-full">
                <div className="w-1.5 h-1.5 rounded-full bg-[#4CAF50] animate-pulse" />
                <span className="text-[#4CAF50] text-[10px] font-bold uppercase tracking-wide">Live</span>
              </div>
            </div>
            <p className="text-[13px] text-slate">ResNet50 • Inspection Set • 10k Images</p>
          </div>

          <RunNav runId={resolvedParams.runId} />
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden">
          {children}
        </div>
      </div>
    </PageShell>
  );
}
