"use client";

import { useRouter } from "next/navigation";
import type {
  Benchmark, Evaluation, FactorRate, ImageRow, MaskFinding, Outcomes, Run,
} from "@/lib/api/rows";
import type { Alignment, Compatibility } from "@/lib/compare/align";
import type { RunMetrics } from "@/lib/compare/metrics";
import { modelName } from "@/lib/derive";
import { RunSelector } from "./RunSelector";
import { MetricTable } from "./MetricTable";
import { EfficiencyPanel } from "./EfficiencyPanel";
import { FailureDelta } from "./FailureDelta";
import { RootCauseCompare } from "./RootCauseCompare";
import { DiffViewer } from "./DiffViewer";
import { Verdict } from "./Verdict";

/**
 * The comparison's interactive shell.
 *
 * Every figure arrives already computed from the server, which fetched it from
 * endpoints that already existed — five reads per run, no new backend, no
 * schema change. This component owns only what is genuinely interactive:
 * changing the selection, which is a URL change so a comparison can be linked,
 * and stepping through the side-by-side viewer.
 */

export interface Prepared {
  alignment: Alignment;
  compatibility: Compatibility;
  metricsA: RunMetrics;
  metricsB: RunMetrics;
  outcomesA: Outcomes;
  outcomesB: Outcomes;
  factorsA: FactorRate[];
  factorsB: FactorRate[];
  masksA: MaskFinding[];
  masksB: MaskFinding[];
  imagesA: ImageRow[];
  imagesB: ImageRow[];
  evaluationsA: Evaluation[];
  evaluationsB: Evaluation[];
  benchmarksA: Benchmark[];
  benchmarksB: Benchmark[];
  runA: Run;
  runB: Run;
}

export function ComparisonView({
  runs,
  idA,
  idB,
  prepared,
}: {
  runs: Run[];
  idA: string;
  idB: string;
  prepared: Prepared | null;
}) {
  const router = useRouter();

  const select = (which: "a" | "b", id: string) => {
    const next = new URLSearchParams({
      a: which === "a" ? id : idA,
      b: which === "b" ? id : idB,
    });
    router.push(`/compare?${next.toString()}`);
  };

  if (runs.length < 2) {
    return (
      <div className="bg-card border border-border/40 rounded-lg p-6 text-center">
        <p className="text-[13px] text-ink">Comparison needs two runs.</p>
        <p className="text-[12px] text-slate mt-1">
          This database has {runs.length}. Diagnose another model against the same
          split to compare them.
        </p>
      </div>
    );
  }

  const labelFor = (run: Run) => `${modelName(run) ?? "n/a"} · run ${run.id}`;
  const blocked = prepared?.compatibility.level === "blocked";

  return (
    <div className="flex flex-col gap-5">
      <RunSelector
        runs={runs}
        idA={idA}
        idB={idB}
        onChangeA={(id) => select("a", id)}
        onChangeB={(id) => select("b", id)}
        compatibility={
          prepared?.compatibility ??
          (idA === idB
            ? { level: "blocked", headline: "Select two different runs.", details: [] }
            : null)
        }
      />

      {prepared && !blocked && (
        <>
          {/* The answer leads, and the evidence follows it. This page exists to
              settle one question — which model to ship — so burying the verdict
              under four tables would make it read as an analytics dashboard that
              happens to end in a recommendation. Each dimension carries its own
              figures, so the summary is not an assertion ahead of its evidence;
              the sections below are where that evidence is examined. */}
          <Verdict
            labelA={labelFor(prepared.runA)}
            labelB={labelFor(prepared.runB)}
            a={prepared.metricsA}
            b={prepared.metricsB}
            factorsA={prepared.factorsA}
            factorsB={prepared.factorsB}
            benchmarksA={prepared.benchmarksA}
            benchmarksB={prepared.benchmarksB}
          />

          <MetricTable
            labelA={labelFor(prepared.runA)}
            labelB={labelFor(prepared.runB)}
            a={prepared.metricsA}
            b={prepared.metricsB}
          />

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            <FailureDelta
              labelA={labelFor(prepared.runA)}
              labelB={labelFor(prepared.runB)}
              a={prepared.outcomesA}
              b={prepared.outcomesB}
            />
            <EfficiencyPanel
              labelA={labelFor(prepared.runA)}
              labelB={labelFor(prepared.runB)}
              runA={prepared.runA}
              runB={prepared.runB}
              benchmarksA={prepared.benchmarksA}
              benchmarksB={prepared.benchmarksB}
            />
          </div>

          <RootCauseCompare
            labelA={labelFor(prepared.runA)}
            labelB={labelFor(prepared.runB)}
            a={prepared.factorsA}
            b={prepared.factorsB}
          />

          {/* Keyed on the pair, so switching to a different comparison starts at
              its first object. Swapping the same two runs keeps the key and so
              holds position, which is what a reader means by "swap sides". */}
          <DiffViewer
            key={[prepared.runA.id, prepared.runB.id].sort((x, y) => x - y).join("-")}
            labelA={labelFor(prepared.runA)}
            labelB={labelFor(prepared.runB)}
            alignment={prepared.alignment}
            masksA={prepared.masksA}
            masksB={prepared.masksB}
            imagesA={prepared.imagesA}
            imagesB={prepared.imagesB}
          />
        </>
      )}
    </div>
  );
}
