import Link from "next/link";

import { API_BASE } from "@/lib/api/client";
import { READ_ONLY } from "@/lib/deployment";

/**
 * Shown when the backend cannot be reached.
 *
 * Two audiences, two messages. Locally the most likely cause is that the API
 * simply is not running, and the command to start it is worth more than an
 * apology. On the public deployment neither is true: the visitor cannot start
 * anything, the host name and status line mean nothing to them, and a shell
 * command on a product page reads as a site that is broken rather than one
 * that is waiting for its data. See {@link READ_ONLY}.
 */
export function ApiDown({ error }: { error: unknown }) {
  return READ_ONLY ? <Unavailable /> : <Diagnostic error={error} />;
}

/** The public face: what is missing, and what still works. */
function Unavailable() {
  return (
    <main className="min-h-screen bg-background text-foreground grid place-items-center p-6">
      <div className="max-w-[480px] text-center">
        <h1 className="font-heading text-[22px] text-ink mb-3">
          The demo dataset isn&rsquo;t published yet
        </h1>
        <p className="text-[13px] text-slate leading-relaxed mb-6">
          Model Doctor&rsquo;s analysis service is running, but no dataset has been
          published to this deployment yet, so there are no runs to explore.
          This page will fill in once one is.
        </p>
        <Link
          href="/"
          className="inline-block text-[13px] font-medium text-ink border border-border/60 rounded-lg px-4 py-2 hover:bg-black/5 transition-colors"
        >
          Back to the overview
        </Link>
      </div>
    </main>
  );
}

/** The developer face, unchanged: the address tried, the error, the remedy. */
function Diagnostic({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <main className="min-h-screen bg-background text-foreground grid place-items-center p-6">
      <div className="max-w-[520px]">
        <h1 className="font-heading text-[22px] text-ink mb-2">Cannot reach the API</h1>
        <p className="text-[13px] text-slate mb-4">
          Tried <code className="font-mono text-[12px] text-ink">{API_BASE}</code> and got:
          <br />
          <span className="font-mono text-[12px] text-[#B3452F]">{message}</span>
        </p>
        <div className="bg-card border border-border/40 rounded-lg p-4">
          <p className="text-[12px] text-slate mb-2">Start the backend from the repository root:</p>
          <pre className="font-mono text-[11px] text-ink whitespace-pre-wrap leading-relaxed">
{`MD_API_FILE_ROOTS=/path/to/your/dataset \\
  python -m uvicorn model_doctor.app.api:app --port 8000`}
          </pre>
          <p className="text-[11px] text-slate mt-3 leading-relaxed">
            MD_API_FILE_ROOTS must contain the directory the run was diagnosed from,
            or image and heatmap requests return 403.
          </p>
        </div>
      </div>
    </main>
  );
}
