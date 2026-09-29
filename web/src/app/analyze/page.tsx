import { PageShell } from "@/components/shared/PageShell";
import { NewAnalysis } from "@/components/analysis/NewAnalysis";
import { AnalysisPreview } from "@/components/analysis/AnalysisPreview";
import { READ_ONLY } from "@/lib/deployment";

/**
 * Run a new analysis from the browser.
 *
 * A client component below the shell: this screen uploads files and polls a
 * job, neither of which a server component can do. Every other screen in the
 * dashboard renders on the server from the read-only API, and that is
 * unchanged — this is the one place the browser talks to the write API.
 *
 * Which is exactly why the public build does not render it. That write API
 * executes the checkpoint it is given, so it is never deployed beside the
 * public read API; `NewAnalysis` is not mounted at all there, so the browser
 * never opens a request to a control service that is not listening. The
 * public build shows `AnalysisPreview` instead: the same set-up, locked, with
 * no network code of its own.
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
          {READ_ONLY ? <AnalysisPreview /> : <NewAnalysis />}
        </div>
      </div>
    </PageShell>
  );
}
