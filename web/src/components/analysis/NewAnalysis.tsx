"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Check, X, Loader2, ArrowRight, Upload, AlertTriangle } from "lucide-react";
import {
  control,
  progressOf,
  STAGE_LABEL,
  type Capabilities,
  type DatasetUpload,
  type Job,
  type ModelUpload,
  type ValidationReport,
} from "@/lib/api/control";

/**
 * The self-service workflow: upload, validate, run, watch, open the results.
 *
 * **Structured as an evaluation set-up, not a file manager.** The sections are
 * the questions an engineer answers before a run — which weights, which data,
 * which split, at what confidence — and each one reports what the backend
 * concluded about the file rather than merely that bytes arrived. A green tick
 * here means "this will run", not "this uploaded".
 *
 * Validation is shown per check rather than as a single verdict, because
 * "dataset invalid" is not actionable and "no labels/ beside images/" is.
 */

type Phase = "compose" | "running" | "done";

function CheckList({ report }: { report: ValidationReport }) {
  return (
    <ul className="mt-3 flex flex-col gap-1.5">
      {report.checks.map((check) => (
        <li key={check.name} className="flex items-start gap-2 text-[12px]">
          {check.ok ? (
            <Check size={13} className="text-[#66805A] mt-0.5 shrink-0" />
          ) : (
            <X size={13} className="text-[#A65C48] mt-0.5 shrink-0" />
          )}
          <span className={check.ok ? "text-slate" : "text-ink"}>
            <span className="font-medium">{check.name}</span>
            <span className="text-slate"> — {check.detail}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

function Panel({
  step, title, hint, children,
}: {
  step: number;
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-card border border-border/40 rounded-lg p-5">
      <div className="flex items-baseline gap-3 mb-1">
        <span className="text-[10px] font-mono text-brass">{String(step).padStart(2, "0")}</span>
        <h4 className="text-[14px] font-medium text-ink">{title}</h4>
      </div>
      <p className="text-[12px] text-slate mb-4 leading-relaxed">{hint}</p>
      {children}
    </section>
  );
}

function FilePicker({
  label, accept, busy, chosen, onPick,
}: {
  label: string;
  accept: string;
  busy: boolean;
  chosen: string | null;
  onPick: (file: File) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        disabled={busy}
        onClick={() => input.current?.click()}
        className="flex items-center gap-2 px-3 py-2 rounded-md border border-border/60 bg-canvas text-[13px] text-ink hover:border-brass/60 transition-colors disabled:opacity-50"
      >
        {busy ? <Loader2 size={13} className="animate-spin" /> : <Upload size={13} />}
        {busy ? "Checking…" : label}
      </button>
      <span className="text-[12px] text-slate font-mono truncate">{chosen ?? "none selected"}</span>
      <input
        ref={input}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onPick(file);
          event.target.value = "";
        }}
      />
    </div>
  );
}

export function NewAnalysis() {
  const router = useRouter();
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [unreachable, setUnreachable] = useState<string | null>(null);

  const [token, setToken] = useState<string | null>(null);
  const [detector, setDetector] = useState("yolo");
  const [split, setSplit] = useState("test");
  const [imageSize, setImageSize] = useState(640);
  const [confidence, setConfidence] = useState(0.25);

  const [model, setModel] = useState<ModelUpload | null>(null);
  const [dataset, setDataset] = useState<DatasetUpload | null>(null);
  const [busy, setBusy] = useState<"model" | "dataset" | "start" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [phase, setPhase] = useState<Phase>("compose");
  const [job, setJob] = useState<Job | null>(null);

  useEffect(() => {
    control
      .capabilities()
      .then(setCapabilities)
      .catch((cause) =>
        setUnreachable(
          cause instanceof Error ? cause.message : "The control API is not reachable."
        )
      );
  }, []);

  const session = useCallback(async () => {
    if (token) return token;
    const opened = (await control.session()).token;
    setToken(opened);
    return opened;
  }, [token]);

  const pickModel = async (file: File) => {
    setBusy("model");
    setError(null);
    try {
      const result = await control.uploadModel(await session(), file, detector);
      setModel(result);
      // The file is the authority on what it is. If it disagrees with the
      // selection, follow the file — the operator picked from a menu, the
      // checkpoint was produced by a trainer.
      const detected = result.facts.detected_family;
      if (typeof detected === "string" && detected !== detector) setDetector(detected);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  const pickDataset = async (file: File) => {
    setBusy("dataset");
    setError(null);
    try {
      setDataset(await control.uploadDataset(await session(), file, split));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  const ready = Boolean(token && model?.ok && dataset?.ok);

  const start = async () => {
    if (!ready || !token || !model || !dataset) return;
    setBusy("start");
    setError(null);
    try {
      const started = await control.start({
        token,
        detector,
        model_path: model.path,
        data_yaml: String(dataset.facts.data_yaml ?? ""),
        split,
        image_size: imageSize,
        confidence,
      });
      setJob(started);
      setPhase("running");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  // Poll while a job is live. Stops on a terminal state rather than running
  // forever, and the interval is loose because the stages are minutes long.
  useEffect(() => {
    if (phase !== "running" || !token) return;
    let cancelled = false;
    const timer = setInterval(async () => {
      try {
        const latest = await control.status(token);
        if (cancelled) return;
        setJob(latest);
        if (latest.status === "succeeded" || latest.status === "failed") {
          setPhase("done");
        }
      } catch {
        // A single failed poll is not a failed job; the next tick retries.
      }
    }, 2500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [phase, token]);

  if (unreachable) {
    return (
      <div className="rounded-lg border border-[#A65C48]/40 bg-[#A65C48]/5 p-5 max-w-[620px]">
        <p className="text-[13px] text-ink font-medium mb-1">
          The analysis service is not running
        </p>
        <p className="text-[12px] text-slate leading-relaxed">
          Results already in the database are unaffected — this only stops new
          analyses being started. Launch it with:
        </p>
        <code className="mt-2 block text-[11px] font-mono bg-black/5 rounded px-2 py-1.5 text-ink">
          uvicorn app.control:app --port 8001
        </code>
      </div>
    );
  }

  // -----------------------------------------------------------------------
  if (phase !== "compose" && job) {
    const done = job.status === "succeeded";
    const failed = job.status === "failed";
    return (
      <div className="max-w-[720px] flex flex-col gap-4">
        <section className="bg-card border border-border/40 rounded-lg p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="text-[14px] font-medium text-ink">
                {done ? "Analysis complete" : failed ? "Analysis failed" : "Running analysis"}
              </h4>
              <p className="text-[12px] text-slate font-mono mt-0.5">
                {job.model_name} · {job.dataset_name} · {job.detector} · {job.split}
              </p>
            </div>
            {!done && !failed && <Loader2 size={16} className="animate-spin text-brass" />}
          </div>

          <div className="h-1 rounded bg-black/5 overflow-hidden mb-4">
            <div
              className={`h-full transition-all duration-500 ${failed ? "bg-[#A65C48]" : "bg-brass"}`}
              style={{ width: `${Math.round(progressOf(job) * 100)}%` }}
            />
          </div>

          <ol className="flex flex-col gap-1.5">
            {job.stages.map((stage, index) => {
              const at = job.stage_index;
              const passed = done || (at !== null && index < at);
              const current = !done && at === index;
              return (
                <li key={stage} className="flex items-center gap-2 text-[12px]">
                  {passed ? (
                    <Check size={13} className="text-[#66805A]" />
                  ) : current ? (
                    <Loader2 size={13} className="animate-spin text-brass" />
                  ) : (
                    <span className="w-[13px] h-[13px] rounded-full border border-border/60" />
                  )}
                  <span className={current ? "text-ink font-medium" : "text-slate"}>
                    {STAGE_LABEL[stage] ?? stage}
                  </span>
                </li>
              );
            })}
          </ol>

          {!job.explainability_supported && (
            <p className="mt-4 text-[11px] text-slate leading-relaxed border-t border-border/40 pt-3">
              Attention heatmaps are not generated for {job.detector}. No attribution
              method has been validated for this detector in this project, so none is
              produced rather than one that would not correspond to the prediction.
            </p>
          )}
        </section>

        {failed && job.error && (
          <section className="rounded-lg border border-[#A65C48]/40 bg-[#A65C48]/5 p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle size={13} className="text-[#A65C48]" />
              <span className="text-[12px] font-medium text-ink">{job.error}</span>
            </div>
            {job.log_tail && (
              <pre className="text-[10px] font-mono text-slate whitespace-pre-wrap max-h-[220px] overflow-auto bg-black/5 rounded p-2">
                {job.log_tail}
              </pre>
            )}
          </section>
        )}

        <div className="flex items-center gap-3">
          {job.run_id !== null && (
            <Link
              href={`/runs/${job.run_id}/failures`}
              className="flex items-center gap-2 px-4 py-2 rounded-md bg-ink text-canvas text-[13px] hover:opacity-90 transition-opacity"
            >
              Open run #{job.run_id} <ArrowRight size={13} />
            </Link>
          )}
          {done && job.run_id !== null && (
            <button
              type="button"
              onClick={() => router.push("/compare")}
              className="px-4 py-2 rounded-md border border-border/60 text-[13px] text-ink hover:border-brass/50 transition-colors"
            >
              Compare with another run
            </button>
          )}
        </div>
      </div>
    );
  }

  // -----------------------------------------------------------------------
  const families = capabilities?.detectors ?? [];
  const selected = families.find((d) => d.family === detector);

  return (
    <div className="max-w-[720px] flex flex-col gap-4">
      <Panel
        step={1}
        title="Model checkpoint"
        hint={
          capabilities
            ? `A .pt or .pth checkpoint, up to ${capabilities.limits.model_mb} MB. The family is read from the file itself.`
            : "A trained detector checkpoint."
        }
      >
        <FilePicker
          label="Choose checkpoint"
          accept=".pt,.pth"
          busy={busy === "model"}
          chosen={model?.name ?? null}
          onPick={pickModel}
        />
        {model && <CheckList report={model} />}
      </Panel>

      <Panel
        step={2}
        title="Detector family"
        hint="Read from the checkpoint. Change it only if you know the file better than the heuristic does."
      >
        <div className="flex flex-wrap gap-2">
          {families.map((entry) => (
            <button
              key={entry.family}
              type="button"
              onClick={() => setDetector(entry.family)}
              className={`px-3 py-1.5 rounded-md border text-[13px] font-mono transition-colors ${
                detector === entry.family
                  ? "border-brass bg-brass/10 text-ink"
                  : "border-border/60 text-slate hover:border-brass/40"
              }`}
            >
              {entry.family}
            </button>
          ))}
        </div>
        {selected && (
          <p className="mt-3 text-[11px] text-slate leading-relaxed">
            {selected.explainability_note}
          </p>
        )}
      </Panel>

      <Panel
        step={3}
        title="Dataset"
        hint={
          capabilities
            ? `${capabilities.dataset_format.detail} Upload as ${capabilities.dataset_format.archives.join(", ")}, up to ${capabilities.limits.dataset_mb} MB.`
            : "A labelled dataset to diagnose against."
        }
      >
        <FilePicker
          label="Choose dataset archive"
          accept=".zip,.tar,.tar.gz,.tgz"
          busy={busy === "dataset"}
          chosen={dataset?.name ?? null}
          onPick={pickDataset}
        />
        {dataset && <CheckList report={dataset} />}
      </Panel>

      <Panel step={4} title="Run settings" hint="Recorded with the run, so two runs are comparable only when these agree.">
        <div className="grid grid-cols-3 gap-4">
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] uppercase tracking-wider text-slate">Split</span>
            <input
              value={split}
              onChange={(event) => setSplit(event.target.value)}
              className="h-8 rounded border border-border/60 bg-canvas px-2 text-[13px] font-mono text-ink"
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] uppercase tracking-wider text-slate">Image size</span>
            <input
              type="number" step={32} min={128} max={2048} value={imageSize}
              onChange={(event) => setImageSize(Number(event.target.value))}
              className="h-8 rounded border border-border/60 bg-canvas px-2 text-[13px] font-mono text-ink"
            />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[11px] uppercase tracking-wider text-slate">Confidence</span>
            <input
              type="number" step={0.05} min={0.01} max={0.95} value={confidence}
              onChange={(event) => setConfidence(Number(event.target.value))}
              className="h-8 rounded border border-border/60 bg-canvas px-2 text-[13px] font-mono text-ink"
            />
          </label>
        </div>
      </Panel>

      {error && (
        <div className="rounded-lg border border-[#A65C48]/40 bg-[#A65C48]/5 p-4">
          <pre className="text-[11px] font-mono text-ink whitespace-pre-wrap">{error}</pre>
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          type="button"
          disabled={!ready || busy !== null}
          onClick={start}
          className="flex items-center gap-2 px-4 py-2 rounded-md bg-ink text-canvas text-[13px] hover:opacity-90 transition-opacity disabled:opacity-35"
        >
          {busy === "start" ? <Loader2 size={13} className="animate-spin" /> : null}
          Run analysis
        </button>
        <span className="text-[12px] text-slate">
          {ready
            ? "Inference, evaluation, diagnosis, clustering and recommendations."
            : "Upload a checkpoint and a dataset that both pass validation."}
        </span>
      </div>
    </div>
  );
}
