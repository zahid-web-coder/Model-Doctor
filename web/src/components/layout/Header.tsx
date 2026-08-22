"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, Check } from "lucide-react";
import { useRun } from "@/lib/run-context";

/**
 * The model / dataset / split selectors.
 *
 * Previously these were `<Select defaultValue="…">` with no `value` and no
 * `onValueChange` — uncontrolled, with nothing listening, so choosing an
 * option changed nothing anywhere in the app. They are real controls now:
 * the whole trigger opens the menu, selection writes to the shared run
 * context, the open menu closes on outside click and on Escape, and the
 * current choice is marked.
 */

interface DropdownProps {
  label: string;
  value: string;
  options: string[];
  onSelect: (value: string) => void;
}

function Dropdown({ label, value, options, onSelect }: DropdownProps) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="flex flex-col gap-1 relative" ref={root}>
      <label className="text-[10px] font-semibold tracking-wider text-slate uppercase">
        {label}
      </label>
      {/* The whole control is the target, not just the chevron. */}
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className={`flex items-center gap-1.5 text-[14px] font-medium text-ink rounded-md px-1.5 py-0.5 -mx-1.5 transition-colors ${
          open ? "bg-black/[0.06]" : "hover:bg-black/[0.04]"
        }`}
      >
        <span>{value}</span>
        <ChevronDown
          size={14}
          className={`text-slate transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <ul
          role="listbox"
          className="absolute top-full left-0 z-50 mt-1 min-w-[180px] rounded-lg border border-border/60 bg-canvas shadow-lg py-1"
        >
          {options.map((option) => {
            const selected = option === value;
            return (
              <li key={option}>
                <button
                  type="button"
                  role="option"
                  aria-selected={selected}
                  onClick={() => {
                    onSelect(option);
                    setOpen(false);
                  }}
                  className={`w-full flex items-center justify-between gap-3 px-3 py-2 text-[13px] text-left transition-colors ${
                    selected ? "text-ink font-medium bg-black/[0.04]" : "text-slate hover:bg-black/[0.04]"
                  }`}
                >
                  <span>{option}</span>
                  {selected && <Check size={13} className="text-brass" />}
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
  const { model, dataset, environment, setModel, setDataset, setEnvironment } = useRun();

  return (
    <header className="flex items-center justify-between shrink-0">
      <div className="flex items-center gap-10">
        <Dropdown
          label="Model"
          value={model}
          options={["yolov8s-seg", "yolov8n-seg", "yolov8m-seg"]}
          onSelect={setModel}
        />
        <Dropdown
          label="Dataset"
          value={dataset}
          options={["door_df_window", "door_df_window_1280"]}
          onSelect={setDataset}
        />
        <Dropdown
          label="Split"
          value={environment}
          options={["test", "val", "train"]}
          onSelect={setEnvironment}
        />
      </div>
    </header>
  );
}
