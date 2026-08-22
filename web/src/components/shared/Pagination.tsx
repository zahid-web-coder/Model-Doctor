"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

/** Page controls, matching the design's "Showing 1-10 of 1,247" footer. */
export function Pagination({
  page, pageCount, total, pageSize, onChange,
}: {
  page: number;
  pageCount: number;
  total: number;
  pageSize: number;
  onChange: (page: number) => void;
}) {
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(page * pageSize, total);

  // First few, then the last, so a long list does not produce a hundred
  // buttons — the design shows 1 2 3 4 5 … 125.
  const pages: (number | "gap")[] = [];
  for (let i = 1; i <= Math.min(5, pageCount); i += 1) pages.push(i);
  if (pageCount > 6) pages.push("gap");
  if (pageCount > 5) pages.push(pageCount);

  const step = (delta: number) =>
    onChange(Math.min(pageCount, Math.max(1, page + delta)));

  return (
    <div className="flex items-center justify-between pt-4 shrink-0">
      <div className="flex items-center gap-1">
        <button
          type="button" aria-label="Previous page" onClick={() => step(-1)} disabled={page === 1}
          className="w-7 h-7 grid place-items-center rounded border border-border/50 text-slate disabled:opacity-40 hover:bg-black/5"
        >
          <ChevronLeft size={14} />
        </button>
        {pages.map((p, i) =>
          p === "gap" ? (
            <span key={`gap-${i}`} className="px-1 text-slate text-[13px]">…</span>
          ) : (
            <button
              key={p} type="button" onClick={() => onChange(p)}
              aria-current={p === page ? "page" : undefined}
              className={`w-7 h-7 grid place-items-center rounded text-[13px] transition-colors ${
                p === page
                  ? "bg-[#EBE6D8] text-ink font-medium border border-brass/40"
                  : "text-slate hover:bg-black/5 border border-transparent"
              }`}
            >
              {p}
            </button>
          )
        )}
        <button
          type="button" aria-label="Next page" onClick={() => step(1)} disabled={page === pageCount}
          className="w-7 h-7 grid place-items-center rounded border border-border/50 text-slate disabled:opacity-40 hover:bg-black/5"
        >
          <ChevronRight size={14} />
        </button>
      </div>
      <p className="text-[12px] text-slate">
        Showing {from}-{to} of {total.toLocaleString()}
      </p>
    </div>
  );
}
