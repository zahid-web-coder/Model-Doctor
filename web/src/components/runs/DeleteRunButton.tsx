"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Trash2, Loader2 } from "lucide-react";
import { Lightbox } from "@/components/shared/Lightbox";
import { control, ControlError, type Footprint } from "@/lib/api/control";
import { num } from "@/lib/format";

/**
 * Deleting one run, with what that costs stated before it happens.
 *
 * **The confirmation quotes quantities because a bare question cannot be
 * answered honestly.** "Delete this run?" and "delete this run, its 231
 * findings, 155 attributed causes and 73 heatmaps?" are different questions,
 * and only the second lets someone judge. The counts come from the server's
 * own footprint endpoint rather than from anything this component assumes, so
 * what is quoted is what the delete will actually remove.
 *
 * The request goes to the control API. The read API opens SQLite read-only
 * and declares no non-GET route, and delete does not get to be the exception.
 */

/** Table names as a reader would say them. */
const LABEL: Record<string, string> = {
  images: "images",
  findings: "findings",
  mask_findings: "outline measurements",
  root_causes: "attributed causes",
  clusters: "failure groups",
  cluster_members: "group memberships",
  recommendations: "recommendations",
  heatmaps: "heatmaps",
  embeddings: "embeddings",
  factor_rates: "factor rates",
  run_evaluations: "evaluations",
  run_benchmarks: "benchmarks",
};

export function DeleteRunButton({
  runId,
  label,
}: {
  runId: number;
  /** How the run is named in the confirmation, e.g. "#5 · best.pt". */
  label: string;
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [footprint, setFootprint] = useState<Footprint | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function askFirst() {
    setOpen(true);
    setError(null);
    setFootprint(null);
    try {
      setFootprint(await control.footprint(runId));
    } catch (cause) {
      // The dialog still opens. A reader who cannot see the cost should be
      // told that, not shown a confident button over missing information.
      setError(
        cause instanceof ControlError && cause.status === 404
          ? "This run no longer exists."
          : "Could not reach the analysis service, so the cost of this delete is unknown."
      );
    }
  }

  async function confirm() {
    setBusy(true);
    setError(null);
    try {
      await control.deleteRun(runId);
      setOpen(false);
      // The runs page is a server component; refresh re-fetches it rather than
      // mutating a local copy that could disagree with the database.
      router.refresh();
    } catch (cause) {
      setError(
        cause instanceof ControlError
          ? cause.message
          : "The analysis service is not reachable, so nothing was deleted."
      );
    } finally {
      setBusy(false);
    }
  }

  const rows = footprint ? Object.entries(footprint.rows).filter(([, n]) => n > 0) : [];

  return (
    <>
      <button
        type="button"
        onClick={askFirst}
        title={`Delete run #${runId}`}
        aria-label={`Delete run #${runId}`}
        className="p-1.5 rounded text-slate hover:text-[#A65C48] hover:bg-[#A65C48]/10 transition-colors"
      >
        <Trash2 size={14} />
      </button>

      <Lightbox
        open={open}
        onClose={() => !busy && setOpen(false)}
        title={`Delete run ${label}?`}
        subtitle="This cannot be undone."
      >
        <div className="w-[420px] max-w-full flex flex-col gap-4 text-left">
          {error && (
            <p className="text-[12px] leading-relaxed text-[#A65C48] bg-[#A65C48]/8 border border-[#A65C48]/25 rounded px-3 py-2">
              {error}
            </p>
          )}

          {footprint && rows.length > 0 && (
            <div>
              <p className="text-[11px] uppercase tracking-wide text-slate mb-2">
                This removes
              </p>
              <div className="border border-border/40 rounded overflow-hidden">
                {rows.map(([table, n]) => (
                  <div
                    key={table}
                    className="flex items-center justify-between px-3 py-1.5 text-[12px] border-b border-border/25 last:border-b-0"
                  >
                    <span className="text-slate">{LABEL[table] ?? table}</span>
                    <span className="font-mono text-ink tabular-nums">{num(n)}</span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate mt-2 leading-relaxed">
                {num(footprint.total_rows)} rows in total
                {footprint.heatmap_files > 0 &&
                  `, and ${num(footprint.heatmap_files)} heatmap image${footprint.heatmap_files === 1 ? "" : "s"} on disk`}
                . Other runs are unaffected.
              </p>
            </div>
          )}

          {footprint && rows.length === 0 && !error && (
            <p className="text-[12px] text-slate leading-relaxed">
              This run holds no findings — only its own record will be removed.
            </p>
          )}

          {!footprint && !error && (
            <p className="text-[12px] text-slate">Checking what this run holds…</p>
          )}

          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setOpen(false)}
              disabled={busy}
              className="px-3 py-1.5 rounded-md border border-border/60 text-[13px] text-ink hover:border-brass/50 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={confirm}
              // Disabled until the cost is known: confirming a delete whose
              // extent could not be read is consent without information.
              disabled={busy || footprint === null}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[#A65C48] text-[13px] text-white hover:bg-[#94513f] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {busy && <Loader2 size={13} className="animate-spin" />}
              {busy ? "Deleting…" : `Delete run #${runId}`}
            </button>
          </div>
        </div>
      </Lightbox>
    </>
  );
}
