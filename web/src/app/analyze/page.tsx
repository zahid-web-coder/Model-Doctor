import { PageShell } from "@/components/shared/PageShell";
import { NewAnalysis } from "@/components/analysis/NewAnalysis";

/**
 * Run a new analysis from the browser.
 *
 * A client component below the shell: this screen uploads files and polls a
 * job, neither of which a server component can do. Every other screen in the
 * dashboard renders on the server from the read-only API, and that is
 * unchanged — this is the one place the browser talks to the write API.
 */
export default function AnalyzePage() {
  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-hidden">
        <h3 className="text-[17px] font-heading text-ink">New Analysis</h3>
        <p className="text-[13px] text-slate mb-6">
          Upload a checkpoint and a labelled dataset, and run the full diagnosis
          pipeline without touching a terminal.
        </p>
        <div className="flex-1 overflow-y-auto custom-scrollbar pr-1 pb-4">
          <NewAnalysis />
        </div>
      </div>
    </PageShell>
  );
}
