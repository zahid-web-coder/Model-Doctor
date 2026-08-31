"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Check, X, Loader2, Clock, ArrowRight, RotateCw } from "lucide-react";
import { control, STAGE_LABEL, type Job } from "@/lib/api/control";

/**
 * Previous analyses, newest first.
 *
 * **Every analysis is already recorded** — the job row is written before any
 * work starts, precisely so a caller always has something to poll. This screen
 * is what makes that record reachable: without it, closing the tab during a run
 * loses the only handle on it, and a finished analysis is findable only by
 * guessing which run id it produced.
 *
 * A live job is offered for resuming rather than merely listed. Watching
 * progress is the reason someone leaves the tab open, and the work continues on
 * the server whether they do or not, so coming back to it should not require
 * starting again.
 */

const TONE: Record<Job["status"], string> = {
  succeeded: "text-[#66805A]",
  failed: "text-[#A65C48]",
  running: "text-brass",
  queued: "text-slate",
  cancelled: "text-slate",
};

function StatusIcon({ status }: { status: Job["status"] }) {
  if (status === "succeeded") return <Check size={13} className={TONE.succeeded} />;
  if (status === "failed") return <X size={13} className={TONE.failed} />;
  if (status === "running") return <Loader2 size={13} className="animate-spin text-brass" />;
  return <Clock size={13} className="text-slate" />;
}

/** Render a stored UTC timestamp in the reader's own timezone. */
function when(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export function AnalysisHistory({ onResume }: { onResume?: (job: Job) => void }) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  const load = useCallback(() => {
    control
      .recent()
      .then((rows) => {
        setJobs(rows);
        setUnreachable(false);
      })
      .catch(() => setUnreachable(true));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Keep the list fresh only while something in it can still change. Polling a
  // list of finished jobs would be requests spent to learn nothing.
  const live = jobs?.some((j) => j.status === "running" || j.status === "queued");
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(load, 4000);
    return () => clearInterval(timer);
  }, [live, load]);

  if (unreachable || jobs === null) return null;

  if (jobs.length === 0) {
    return (
      <section className="mt-8">
        <div className="flex items-baseline justify-between mb-3">
          <h4 className="text-[13px] font-medium text-ink">Previous analyses</h4>
        </div>
        <p className="text-[12px] text-slate">
          Nothing yet. Analyses you run appear here, including ones still in
          progress.
        </p>
      </section>
    );
  }

  return (
    <section className="mt-8">
      <div className="flex items-baseline justify-between mb-3">
        <h4 className="text-[13px] font-medium text-ink">Previous analyses</h4>
        <button
          type="button"
          onClick={load}
          className="flex items-center gap-1.5 text-[11px] text-slate hover:text-ink transition-colors"
        >
          <RotateCw size={11} /> Refresh
        </button>
      </div>

      <div className="flex flex-col gap-1.5">
        {jobs.map((job) => {
          const inFlight = job.status === "running" || job.status === "queued";
          return (
            <div
              key={job.token}
              className="bg-card border border-border/40 rounded-lg px-4 py-2.5 flex items-center gap-4"
            >
              <StatusIcon status={job.status} />

              <div className="min-w-0 flex-1">
                <p className="text-[12px] font-mono text-ink truncate">
                  {job.model_name} <span className="text-slate">·</span> {job.dataset_name}
                </p>
                <p className="text-[11px] text-slate mt-0.5">
                  {job.detector} · {job.split} · {when(job.created_at)}
                  {inFlight && job.stage && (
                    <span className="text-brass"> · {STAGE_LABEL[job.stage] ?? job.stage}</span>
                  )}
                </p>
              </div>

              <span className={`text-[11px] shrink-0 ${TONE[job.status]}`}>
                {job.status}
              </span>

              {/* A failed job may still have produced a run before it stopped;
                  offering that run is more useful than hiding it, and the
                  status beside it says not to trust it as complete. */}
              {job.run_id !== null && (
                <Link
                  href={`/runs/${job.run_id}/failures`}
                  className="shrink-0 flex items-center gap-1 text-[11px] text-brass hover:underline"
                >
                  Run #{job.run_id} <ArrowRight size={11} />
                </Link>
              )}

              {inFlight && onResume && (
                <button
                  type="button"
                  onClick={() => onResume(job)}
                  className="shrink-0 text-[11px] text-ink border border-border/60 rounded px-2 py-1 hover:border-brass/50 transition-colors"
                >
                  Watch
                </button>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
