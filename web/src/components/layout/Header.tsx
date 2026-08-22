"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronDown, Check } from "lucide-react";
import { useRun } from "@/lib/run-context";
import { modelName, datasetName } from "@/lib/derive";

/**
 * The model / dataset / split selectors.
 *
 * These read from the selected run rather than filtering it. Model, dataset
 * and split are properties of a completed analysis pass — the backend cannot
 * re-run a diagnosis on a different split from a GET — so choosing a value
 * navigates to the most recent run that used it. A control that looked like a
 * filter but changed nothing would be worse than one that says what it does.
 */
function Dropdown({
  label, value, options, onSelect,
}: {
  label: string;
  value: string | null;
  options: string[];
  onSelect: (value: string) => void;
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
        <span className="font-mono text-[13px]">{value ?? "n/a"}</span>
        <ChevronDown size={14} className={`text-slate transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <ul role="listbox" className="absolute top-full left-0 z-50 mt-1 min-w-[220px] rounded-lg border border-border/60 bg-canvas shadow-lg py-1">
          {options.map((option) => {
            const selected = option === value;
            return (
              <li key={option}>
                <button
                  type="button" role="option" aria-selected={selected}
                  onClick={() => { onSelect(option); setOpen(false); }}
                  className={`w-full flex items-center justify-between gap-3 px-3 py-2 text-[13px] text-left font-mono transition-colors ${
                    selected ? "text-ink bg-black/[0.04]" : "text-slate hover:bg-black/[0.04]"
                  }`}
                >
                  <span className="truncate">{option}</span>
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
  const { runs, model, dataset, split, runsMatching } = useRun();

  const distinct = (values: (string | null)[]) =>
    Array.from(new Set(values.filter((v): v is string => Boolean(v))));

  const go = (key: "model" | "dataset" | "split") => (value: string) => {
    const target = runsMatching(key, value);
    if (target) router.push(`/runs/${target.id}/failures`);
  };

  return (
    <header className="flex items-center justify-between shrink-0">
      <div className="flex items-center gap-10">
        <Dropdown label="Model" value={model} options={distinct(runs.map(modelName))} onSelect={go("model")} />
        <Dropdown label="Dataset" value={dataset} options={distinct(runs.map(datasetName))} onSelect={go("dataset")} />
        <Dropdown label="Split" value={split} options={distinct(runs.map((r) => r.split))} onSelect={go("split")} />
      </div>
    </header>
  );
}
