import type { Outcome, RootCause } from "@/lib/api/rows";

/**
 * Turning a run's root causes into something a finding can be looked up in.
 *
 * `/runs/{id}/root-causes` returns every attributed factor for the run in one
 * list — it filters by factor, not by finding — so the page fetches the set
 * once and indexes it here rather than issuing a request per selection.
 *
 * This lives in a neutral module, not in the table component, because the
 * table is a `"use client"` file and the failures page that does the indexing
 * is a server component. Importing a value across that boundary in that
 * direction yields a client reference that typechecks and then throws at
 * runtime, which is exactly the trap `lib/paging.ts` exists to avoid.
 */

/**
 * Group a run's root causes by the finding they were attributed to.
 *
 * A plain object rather than a Map: the result crosses the server/client
 * boundary as a prop, and a Map does not survive serialisation. Findings with
 * no attributed factor are absent from the result — a real state, and one the
 * caller must render as such rather than as an empty list of causes.
 */
export function indexByFinding(causes: RootCause[]): Record<number, RootCause[]> {
  const index: Record<number, RootCause[]> = {};
  for (const cause of causes) {
    const existing = index[cause.finding_id];
    if (existing) existing.push(cause);
    else index[cause.finding_id] = [cause];
  }
  return index;
}

/**
 * A finding's factors, strongest attribution first.
 *
 * Sorted here rather than trusted from the API. The endpoint orders across the
 * whole run, and slicing one finding out of that ordering does not guarantee
 * the order within the slice is meaningful. Copies before sorting so the
 * caller's array is not mutated.
 */
export function strongestFirst(causes: RootCause[] | undefined): RootCause[] {
  return [...(causes ?? [])].sort((a, b) => b.score - a.score);
}

/**
 * Why a finding shows no attributed factor.
 *
 * **The root-cause pass runs on failures, not on every finding.** A correct
 * detection has no attributed factor because it was never a candidate for
 * attribution — saying "no measured condition applied" there would imply the
 * detectors ran and found nothing, which is a different and stronger claim.
 * Verified against run 5: 155 attributions across 72 findings, every one a
 * false positive, false negative, or poor localisation.
 *
 * For a failure, the absence is real and has two possible causes, and the text
 * names both rather than picking one. The `unexplained_backlog` recommendation
 * exists precisely because failures no detector accounts for are a known,
 * reported state — not an oversight.
 */
export function noCausesReason(outcome: Outcome): string {
  if (outcome === "correct") {
    return (
      "Factors are attributed to failures, so a correct detection has none. " +
      "This finding matched its ground-truth object, and there is nothing to " +
      "explain."
    );
  }
  return (
    "No factor was attributed to this failure. Either none of the measured " +
    "conditions applied — the analysis reports these as unexplained rather " +
    "than guessing — or this run predates the root-cause pass."
  );
}
