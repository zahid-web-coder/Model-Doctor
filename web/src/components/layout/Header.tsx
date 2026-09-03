"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { ChevronDown, Check } from "lucide-react";
import { useRun } from "@/lib/run-context";
import { modelName, datasetName, modelFingerprint } from "@/lib/derive";

/**
 * The run / model / dataset / split selectors.
 *
 * These read from the selected run rather than filtering it. Model, dataset
 * and split are properties of a completed analysis pass — the backend cannot
 * re-run a diagnosis on a different split from a GET — so choosing a value
 * navigates to the most recent run that used it. A control that looked like a
 * filter but changed nothing would be worse than one that says what it does.
 *
 * **The run selector exists because the other three cannot name a run.** They
 * are the pass's inputs, not its identity: Ultralytics writes every checkpoint
 * to `best.pt`, so several runs share a model name, a dataset and a split, and
 * choosing one of those values lands on whichever matched first. The run id is
 * the thing that distinguishes them.
 */
/**
 * One option. `note` carries what distinguishes it when the label cannot.
 *
 * Runs need this and the other three do not: every run in this project is
 * `best.pt` on the same dataset, so a list of run labels alone would be a
 * column of identical strings.
 */
interface Option {
  key: string;
  label: string;
  note?: string;
}

function Dropdown({
  label, value, options, onSelect, width = "min-w-[220px]",
}: {
  label: string;
  value: string | null;
  options: Option[];
  onSelect: (key: string) => void;
  width?: string;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointer = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="flex flex-col gap-1 relative" ref={root}>
      <label className="text-[10px] font-semibold tracking-wider text-slate uppercase">{label}</label>
      <button
        type="button" aria-haspopup="listbox" aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className={`flex items-center gap-1.5 text-[14px] font-medium text-ink rounded-md px-1.5 py-0.5 -mx-1.5 transition-colors ${
          open ? "bg-black/[0.06]" : "hover:bg-black/[0.04]"
        }`}
      >
        <span className="font-mono text-[13px]">
          {options.find((o) => o.key === value)?.label ?? value ?? "n/a"}
        </span>
        <ChevronDown size={14} className={`text-slate transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <ul role="listbox" className={`absolute top-full left-0 z-50 mt-1 ${width} max-h-[60vh] overflow-y-auto custom-scrollbar rounded-lg border border-border/60 bg-canvas shadow-lg py-1`}>
          {options.map((option) => {
            const selected = option.key === value;
            return (
              <li key={option.key}>
                <button
                  type="button" role="option" aria-selected={selected}
                  onClick={() => { onSelect(option.key); setOpen(false); }}
                  className={`w-full flex items-center justify-between gap-3 px-3 py-2 text-[13px] text-left font-mono transition-colors ${
                    selected ? "text-ink bg-black/[0.04]" : "text-slate hover:bg-black/[0.04]"
                  }`}
                >
                  <span className="min-w-0">
                    <span className="block truncate">{option.label}</span>
                    {option.note && (
                      <span className="block text-[11px] text-slate/80 truncate mt-0.5">
                        {option.note}
                      </span>
                    )}
                  </span>
                  {selected && <Check size={13} className="text-brass shrink-0" />}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export function Header() {
  const router = useRouter();
  const pathname = usePathname();
  const { runs, runId, model, dataset, split, runsMatching } = useRun();

  const distinct = (values: (string | null)[]) =>
    Array.from(new Set(values.filter((v): v is string => Boolean(v))))
      .map((v) => ({ key: v, label: v }));

  /**
   * The section being read, so switching run stays on it.
   *
   * Sending a reader from Clusters to Failures because they changed run
   * discards what they were doing; the whole point of switching is usually to
   * see the same screen for a different run.
   */
  const section = pathname.match(/^\/runs\/[^/]+\/([^/?]+)/)?.[1] ?? "failures";
  const open = (id: string | number) => router.push(`/runs/${id}/${section}`);

  const go = (key: "model" | "dataset" | "split") => (value: string) => {
    const target = runsMatching(key, value);
    if (target) open(target.id);
  };

  return (
    <header className="flex items-center justify-between shrink-0">
      <div className="flex items-center gap-10">
        {/*
          **Run comes first, and it is the only control that identifies one.**
          Model, dataset and split are properties of a pass, and in this project
          they are the same across most runs — every checkpoint is written to
          `best.pt` by Ultralytics, and runs 5, 6, 7 and 10 share a dataset and
          a split. Picking a model therefore lands on whichever run happened to
          match first, with nothing on screen to say which. The run id is what
          distinguishes them, so it is offered directly, with the checkpoint
          fingerprint and date beside it for runs that share a name.
        */}
        <Dropdown
          label="Run"
          value={runId}
          width="min-w-[300px]"
          options={runs.map((r) => ({
            key: String(r.id),
            label: `#${r.id}`,
            note: [
              modelName(r),
              modelFingerprint(r),
              r.split,
              new Date(r.created_at).toLocaleDateString(),
            ]
              .filter(Boolean)
              .join(" · "),
          }))}
          onSelect={open}
        />
        <Dropdown label="Model" value={model} options={distinct(runs.map(modelName))} onSelect={go("model")} />
        <Dropdown label="Dataset" value={dataset} options={distinct(runs.map(datasetName))} onSelect={go("dataset")} />
        <Dropdown label="Split" value={split} options={distinct(runs.map((r) => r.split))} onSelect={go("split")} />
      </div>
    </header>
  );
}
