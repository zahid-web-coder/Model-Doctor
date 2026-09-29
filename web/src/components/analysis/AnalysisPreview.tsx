"use client";

import { useEffect, useRef, useState } from "react";
import { Lock, Upload } from "lucide-react";
import { Panel } from "./NewAnalysis";
import { STAGE_LABEL } from "@/lib/api/control";

/**
 * The New Analysis screen as the public demo shows it: the real set-up,
 * locked.
 *
 * **Shown, not hidden.** A visitor learns more from seeing the workflow —
 * which weights, which data, which split, at what confidence — than from a
 * notice saying it exists. So this renders the same four steps as
 * `NewAnalysis`, with its own `Panel`, and every control in its place.
 *
 * **Locked, and says so where you touch it.** Clicking a button, choosing a
 * file, typing in a field or tabbing into one shows a lock message beside it.
 * Nothing is sent anywhere: this component has no fetch at all, because the
 * control service it would talk to is never deployed with the public site.
 *
 * **The reason given is the true one** (see `ReadOnlyNotice`): there are no
 * accounts or permissions in this project, so the message never says "not
 * authorised". Running an analysis executes the uploaded checkpoint, and that
 * service is kept off the public internet.
 */

const MESSAGE = "Locked in the public demo — run Model Doctor locally to analyse your own models.";

type Toast = { x: number; y: number; id: number };

function LockedButton({ children }: { children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-disabled="true"
      title={MESSAGE}
      className="flex items-center gap-2 px-3 py-2 rounded-md border border-border/60 bg-canvas text-[13px] text-ink/60 cursor-not-allowed"
    >
      {children}
    </button>
  );
}

function LockedField({ label, value }: { label: string; value: string }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[11px] uppercase tracking-wider text-slate">{label}</span>
      <span className="relative">
        <input
          readOnly
          aria-disabled="true"
          title={MESSAGE}
          value={value}
          className="h-8 w-full rounded border border-border/60 bg-canvas pl-2 pr-7 text-[13px] font-mono text-ink/60 cursor-not-allowed"
        />
        <Lock size={11} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate/70" />
      </span>
    </label>
  );
}

export function AnalysisPreview() {
  const [toast, setToast] = useState<Toast | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  const show = (x: number, y: number) => {
    setToast({ x, y, id: Date.now() });
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setToast(null), 2200);
  };

  // Anything interactive inside the form: a press, a keystroke or focus
  // arriving by keyboard all end here instead of doing something.
  const isControl = (target: EventTarget | null) =>
    target instanceof HTMLElement && Boolean(target.closest("button, input, [data-locked]"));

  const onPointerDown = (event: React.PointerEvent) => {
    if (!isControl(event.target)) return;
    event.preventDefault();
    show(event.clientX, event.clientY);
  };
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (!isControl(event.target) || event.key === "Tab") return;
    event.preventDefault();
    const box = (event.target as HTMLElement).getBoundingClientRect();
    show(box.left + box.width / 2, box.top);
  };
  const onFocus = (event: React.FocusEvent) => {
    if (!isControl(event.target)) return;
    const box = (event.target as HTMLElement).getBoundingClientRect();
    show(box.left + box.width / 2, box.top);
  };

  return (
    <div
      className="max-w-[720px] flex flex-col gap-4"
      onPointerDownCapture={onPointerDown}
      onKeyDownCapture={onKeyDown}
      onFocusCapture={onFocus}
    >
      <div className="flex items-start gap-3 rounded-lg border border-brass/40 bg-brass/5 px-4 py-3">
        <Lock size={14} className="text-brass mt-0.5 shrink-0" />
        <div className="text-[12.5px] leading-relaxed">
          <p className="text-ink font-medium">Preview · locked in the public demo</p>
          <p className="text-slate">
            This is the workflow as it runs locally. Running an analysis executes
            the uploaded checkpoint, so that service is intentionally not exposed
            on the public internet. Every result in this dashboard was produced
            by it.
          </p>
        </div>
      </div>

      <Panel
        step={1}
        title="Model checkpoint"
        hint="A .pt or .pth checkpoint. The detector family is read from the file itself."
      >
        <div className="flex items-center gap-3">
          <LockedButton>
            <Upload size={13} /> Choose checkpoint <Lock size={11} className="text-slate/70" />
          </LockedButton>
          <span className="text-[12px] text-slate font-mono">none selected</span>
        </div>
      </Panel>

      <Panel
        step={2}
        title="Detector family"
        hint="Read from the checkpoint. Change it only if you know the file better than the heuristic does."
      >
        <div className="flex flex-wrap gap-2">
          {["yolo", "rfdetr"].map((family, i) => (
            <button
              key={family}
              type="button"
              aria-disabled="true"
              title={MESSAGE}
              className={`px-3 py-1.5 rounded-md border text-[13px] font-mono cursor-not-allowed ${
                i === 0 ? "border-brass/60 bg-brass/10 text-ink/70" : "border-border/60 text-slate/70"
              }`}
            >
              {family}
            </button>
          ))}
        </div>
      </Panel>

      <Panel
        step={3}
        title="Dataset"
        hint="A data.yaml naming the classes, beside split directories each holding images/ and labels/, uploaded as a .zip or .tar archive. Each check is reported before a run can start."
      >
        <div className="flex items-center gap-3">
          <LockedButton>
            <Upload size={13} /> Choose dataset archive <Lock size={11} className="text-slate/70" />
          </LockedButton>
          <span className="text-[12px] text-slate font-mono">none selected</span>
        </div>
      </Panel>

      <Panel step={4} title="Run settings" hint="Recorded with the run, so two runs are comparable only when these agree.">
        <div className="grid grid-cols-3 gap-4">
          <LockedField label="Split" value="test" />
          <LockedField label="Image size" value="640" />
          <LockedField label="Confidence" value="0.25" />
        </div>
      </Panel>

      <section className="bg-card border border-border/40 rounded-lg p-5">
        <h4 className="text-[14px] font-medium text-ink mb-1">What a run does</h4>
        <p className="text-[12px] text-slate mb-3 leading-relaxed">
          One click runs every stage, in order, and the results open in the dashboard.
        </p>
        <ol className="grid grid-cols-2 gap-x-6 gap-y-1.5">
          {Object.values(STAGE_LABEL).map((label, i) => (
            <li key={label} className="flex items-center gap-2 text-[12px] text-slate">
              <span className="text-[10px] font-mono text-brass w-4">{String(i + 1).padStart(2, "0")}</span>
              {label}
            </li>
          ))}
        </ol>
      </section>

      <div className="flex items-center gap-3">
        <button
          type="button"
          aria-disabled="true"
          title={MESSAGE}
          className="flex items-center gap-2 px-4 py-2 rounded-md bg-ink/35 text-canvas text-[13px] cursor-not-allowed"
        >
          <Lock size={12} /> Run analysis
        </button>
        <span className="text-[12px] text-slate">Available when Model Doctor runs locally.</span>
      </div>

      {toast && (
        <div
          key={toast.id}
          role="status"
          aria-live="polite"
          className="pointer-events-none fixed z-50 -translate-x-1/2 -translate-y-full flex items-center gap-2 rounded-md bg-ink text-canvas text-[12px] px-3 py-2 shadow-lg max-w-[320px]"
          style={{ left: toast.x, top: toast.y - 10 }}
        >
          <Lock size={12} className="shrink-0" />
          <span>{MESSAGE}</span>
        </div>
      )}
    </div>
  );
}
