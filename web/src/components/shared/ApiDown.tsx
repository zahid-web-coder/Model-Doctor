import { API_BASE } from "@/lib/api/client";

/**
 * Shown when the backend cannot be reached.
 *
 * Deliberately explicit rather than a blank page or a spinner that never
 * resolves: the most likely cause is that the API simply is not running, and
 * the command to start it is worth more than an apology.
 */
export function ApiDown({ error }: { error: unknown }) {
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
  python -m uvicorn app.api:app --port 8000`}
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
