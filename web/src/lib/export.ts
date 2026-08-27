/**
 * Client-side download of data a screen already holds.
 *
 * **No backend involved.** The rows are already in the browser because the
 * screen rendered them, so writing them out is a Blob and an anchor click.
 * Routing this through the API would mean a new endpoint, a second query, and
 * the chance of exporting something other than what the reader is looking at.
 *
 * The trade-off is that an export covers exactly what the page holds, not the
 * whole run — a paged table exports its current page. Callers say so in their
 * filename and their UI rather than letting a reader assume otherwise.
 */

/** Values a cell can hold before it is rendered. */
type Cell = string | number | boolean | null | undefined;

/**
 * Quote one CSV field.
 *
 * Doubling embedded quotes and wrapping anything containing a delimiter,
 * quote or newline is the whole of RFC 4180 that matters here. Class names and
 * file paths routinely contain commas, and an unquoted one silently shifts
 * every later column of that row.
 */
function csvCell(value: Cell): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoking immediately can cancel the download in some browsers; a tick is
  // enough for the navigation to have been started.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Write `rows` as CSV, taking the column order from `columns`.
 *
 * The header is the column keys, so a reader importing this gets the field
 * names the API uses rather than prettified labels that no longer match
 * anything.
 */
export function exportCsv<T extends object>(
  filename: string,
  rows: T[],
  columns: (keyof T)[],
): void {
  const header = columns.map((c) => csvCell(String(c))).join(",");
  const body = rows.map((row) =>
    columns.map((c) => csvCell(row[c] as Cell)).join(",")
  );
  // A trailing newline: POSIX tools treat a file without one as truncated.
  const text = [header, ...body].join("\r\n") + "\r\n";
  download(new Blob([text], { type: "text/csv;charset=utf-8" }), filename);
}

/** Write any serialisable value as indented JSON. */
export function exportJson(filename: string, value: unknown): void {
  const text = JSON.stringify(value, null, 2);
  download(new Blob([text], { type: "application/json" }), filename);
}
