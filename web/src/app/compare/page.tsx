import { PageShell } from "@/components/shared/PageShell";
import { ComparisonView } from "@/components/compare/ComparisonView";
import { api } from "@/lib/api/client";
import { alignRuns, checkCompatibility } from "@/lib/compare/align";
import { computeMetrics, normaliseOutcomes } from "@/lib/compare/metrics";
import type { Run } from "@/lib/api/rows";

/**
 * Model comparison.
 *
 * Answers one engineering question — *I trained two models, which should I ship
 * and why?* — from data Model Doctor already stores. Every figure comes from the
 * two selected runs; nothing is imported from the offline evaluator's artefacts,
 * and metrics the schema cannot supply say so rather than appearing as zero.
 *
 * **Fetched on the server, with the selection in the URL**, which is how every
 * other screen in this app works. Fetching from the browser instead would make
 * this the only page that requires the API to be reachable from the client — and
 * therefore the only one needing a CORS origin configured — while a URL-driven
 * selection additionally makes a specific comparison shareable.
 */

const FINDINGS_PAGE = 1000;
const IMAGES_PAGE = 1000;

async function loadRun(run: Run) {
  const [outcomes, findings, images, factors, masks, evaluations, benchmarks] =
    await Promise.all([
      api.outcomes(run.id),
      api.findings(run.id, FINDINGS_PAGE, 0),
      api.images(run.id, IMAGES_PAGE, 0),
      api.factorRates(run.id),
      api.maskFindings(run.id),
      api.evaluation(run.id),
      api.benchmarks(run.id),
    ]);
  return {
    run, outcomes, findings: findings.items, images, factors, masks,
    evaluations, benchmarks,
  };
}

export default async function ComparePage({
  searchParams,
}: {
  searchParams: Promise<{ a?: string; b?: string }>;
}) {
  const runs = await api.runs().catch(() => []);
  const { a: askedA, b: askedB } = await searchParams;

  // Default to the two most recent runs, so the page is useful on arrival.
  const pick = (asked: string | undefined, fallback: Run | undefined) =>
    runs.find((r) => String(r.id) === asked) ?? fallback;
  const runA = pick(askedA, runs[1] ?? runs[0]);
  const runB = pick(askedB, runs[0]);

  const both =
    runA && runB && runA.id !== runB.id
      ? await Promise.all([loadRun(runA), loadRun(runB)]).catch(() => null)
      : null;

  const prepared = both
    ? (() => {
        const [a, b] = both;
        const alignment = alignRuns(a.findings, a.images, b.findings, b.images);
        return {
          alignment,
          compatibility: checkCompatibility(
            a.run,
            b.run,
            alignment,
            a.findings,
            b.findings,
          ),
          metricsA: computeMetrics(a.outcomes, a.findings, a.masks, a.evaluations),
          metricsB: computeMetrics(b.outcomes, b.findings, b.masks, b.evaluations),
          outcomesA: normaliseOutcomes(a.outcomes),
          outcomesB: normaliseOutcomes(b.outcomes),
          factorsA: a.factors,
          factorsB: b.factors,
          masksA: a.masks,
          masksB: b.masks,
          imagesA: a.images,
          imagesB: b.images,
          evaluationsA: a.evaluations,
          evaluationsB: b.evaluations,
          benchmarksA: a.benchmarks,
          benchmarksB: b.benchmarks,
          runA: a.run,
          runB: b.run,
        };
      })()
    : null;

  return (
    <PageShell>
      <div className="flex flex-col h-full overflow-y-auto custom-scrollbar gap-5 pr-1">
        <header className="shrink-0">
          <h2 className="font-heading text-[22px] text-ink">Model comparison</h2>
          <p className="text-[12px] text-slate mt-0.5">
            Two runs over the same data, side by side. Which model wins, where each
            one fails, and what the stored evidence supports.
          </p>
        </header>

        <ComparisonView
          runs={runs}
          idA={runA ? String(runA.id) : ""}
          idB={runB ? String(runB.id) : ""}
          prepared={prepared}
        />
      </div>
    </PageShell>
  );
}
