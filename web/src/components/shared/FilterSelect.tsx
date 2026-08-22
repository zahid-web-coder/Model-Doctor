"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, Check } from "lucide-react";

/** A working filter control: opens on the whole button, closes on outside
 *  click and Escape, and reports the chosen value. */
export function FilterSelect({
  value, options, onChange, width = "w-[150px]",
}: {
  value: string;
  options: string[];
  onChange: (value: string) => void;
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
    <div className={`relative ${width}`} ref={root}>
      <button
        type="button" aria-haspopup="listbox" aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors"
      >
        <span className="truncate">{value}</span>
        <ChevronDown size={13} className={`text-slate shrink-0 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <ul role="listbox" className="absolute top-full left-0 z-50 mt-1 w-full rounded-md border border-border/60 bg-canvas shadow-lg py-1 max-h-64 overflow-y-auto">
          {options.map((option) => (
            <li key={option}>
              <button
                type="button" role="option" aria-selected={option === value}
                onClick={() => { onChange(option); setOpen(false); }}
                className={`w-full flex items-center justify-between gap-2 px-3 py-1.5 text-[13px] text-left transition-colors ${
                  option === value ? "text-ink font-medium bg-black/[0.04]" : "text-slate hover:bg-black/[0.04]"
                }`}
              >
                <span className="truncate">{option}</span>
                {option === value && <Check size={12} className="text-brass shrink-0" />}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
