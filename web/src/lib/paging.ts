/**
 * Paging constants shared by server and client.
 *
 * **This lives outside the Pagination component on purpose.** That component
 * is `"use client"`, and a value imported from a client module into a server
 * component arrives as a client reference rather than the value itself — the
 * array is still an array to TypeScript, but calling `.includes` on it at
 * request time throws. Keeping the constant in a neutral module means both
 * sides import the real thing.
 */

/** Row counts offered by the rows-per-page control. First entry is the default. */
export const PAGE_SIZES = [10, 25, 50, 100] as const;

/**
 * Coerce a page size from a query string into one we actually offer.
 *
 * The value becomes a `limit` on an API request, so an arbitrary number out of
 * a URL is not something to forward — a caller asking for a million rows
 * should get the default, not a timeout.
 */
export function resolvePageSize(raw: string | undefined): number {
  const requested = Number(raw);
  return (PAGE_SIZES as readonly number[]).includes(requested)
    ? requested
    : PAGE_SIZES[0];
}

/**
 * Which page buttons to draw, always including the current one.
 *
 * The previous rule was "first five, then the last", which meant that from
 * page six onward the page you were actually on had no button and no
 * highlight — the control claimed you were somewhere in 1–5 while the table
 * showed something else. Windowing around the current page fixes that: first
 * and last are always reachable, the neighbours of the current page are always
 * present, and gaps stand in for the rest.
 */
export function pageWindow(page: number, pageCount: number, radius = 1): (number | "gap")[] {
  if (pageCount <= 1) return pageCount === 1 ? [1] : [];

  const wanted = new Set<number>([1, pageCount]);
  for (let p = page - radius; p <= page + radius; p += 1) {
    if (p >= 1 && p <= pageCount) wanted.add(p);
  }
  // Near the ends there is spare room; spend it on real numbers rather than
  // leaving a gap that hides only one page — "1 … 3" is worse than "1 2 3".
  if (page <= radius + 2) for (let p = 1; p <= Math.min(radius + 3, pageCount); p += 1) wanted.add(p);
  if (page >= pageCount - radius - 1) {
    for (let p = Math.max(1, pageCount - radius - 2); p <= pageCount; p += 1) wanted.add(p);
  }

  const sorted = [...wanted].sort((a, b) => a - b);
  const out: (number | "gap")[] = [];
  let previous = 0;
  for (const p of sorted) {
    // A gap standing in for exactly one page is longer than the page it hides.
    if (previous && p - previous === 2) out.push(previous + 1);
    else if (previous && p - previous > 2) out.push("gap");
    out.push(p);
    previous = p;
  }
  return out;
}
