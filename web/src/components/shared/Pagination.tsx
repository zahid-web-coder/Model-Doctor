"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { PAGE_SIZES, pageWindow } from "@/lib/paging";

/** Page controls, with a rows-per-page control when the caller can change it. */
export function Pagination({
  page, pageCount, total, pageSize, onChange, onPageSizeChange,
}: {
  page: number;
  pageCount: number;
  total: number;
  pageSize: number;
  onChange: (page: number) => void;
  /** Omit to hide the rows-per-page control. */
  onPageSizeChange?: (size: number) => void;
}) {
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(page * pageSize, total);
  const pages = pageWindow(page, pageCount);

  const step = (delta: number) =>
    onChange(Math.min(pageCount, Math.max(1, page + delta)));

  return (
    <div className="flex items-center justify-between gap-4 pt-4 shrink-0 flex-wrap">
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
              className={`min-w-7 h-7 px-1.5 grid place-items-center rounded text-[13px] transition-colors ${
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

      <div className="flex items-center gap-4">
        {onPageSizeChange && (
          <label className="flex items-center gap-2 text-[12px] text-slate">
            Rows
            <select
              value={pageSize}
              onChange={(event) => onPageSizeChange(Number(event.target.value))}
              className="h-7 rounded border border-border/50 bg-card px-1.5 text-[12px] text-ink hover:border-brass/50 transition-colors cursor-pointer"
            >
              {PAGE_SIZES.map((size) => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </label>
        )}
        <p className="text-[12px] text-slate">
          Showing {from}-{to} of {total.toLocaleString()}
        </p>
      </div>
    </div>
  );
}
