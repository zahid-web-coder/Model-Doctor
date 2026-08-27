"use client";

import { useState } from "react";
import { Download, Check } from "lucide-react";
import { exportCsv, exportJson } from "@/lib/export";

/**
 * Downloads the rows a screen is showing.
 *
 * A client component so the server-rendered pages can hand it their already
 * fetched rows and stay server components themselves — the data crosses the
 * boundary as props, and no second request is made for something the page
 * already has.
 *
 * `scope` is not decoration. An export of a paged table contains that page,
 * not the run, and a reader who opens the file expecting everything has been
 * misled by silence. Callers pass the truth and it is shown in the title.
 */
export function ExportButton<T extends object>({
  rows, columns, filename, scope, format = "csv",
}: {
  rows: T[];
  /** Column order for CSV. Ignored when `format` is `"json"`. */
  columns?: (keyof T)[];
  /** Without extension — the format's suffix is appended. */
  filename: string;
  /** What the file actually covers, e.g. "this page" or "all 22 clusters". */
  scope: string;
  format?: "csv" | "json";
}) {
  const [done, setDone] = useState(false);
  const empty = rows.length === 0;

  const run = () => {
    if (empty) return;
    if (format === "json" || !columns) {
      exportJson(`${filename}.json`, rows);
    } else {
      exportCsv(`${filename}.csv`, rows, columns);
    }
    setDone(true);
    setTimeout(() => setDone(false), 1600);
  };

  return (
    <button
      type="button"
      onClick={run}
      disabled={empty}
      title={empty ? "Nothing to export" : `Download ${scope} as ${format.toUpperCase()}`}
      className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-border/60 bg-card text-[13px] text-ink hover:border-brass/50 transition-colors disabled:opacity-40 disabled:hover:border-border/60"
    >
      {done ? (
        <Check size={13} className="text-[#66805A]" />
      ) : (
        <Download size={13} className="text-slate" />
      )}
      {done ? "Downloaded" : "Export"}
    </button>
  );
}
