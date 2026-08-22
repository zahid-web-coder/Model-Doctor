"use client";

import { createContext, useContext, useMemo, useState, ReactNode } from "react";
import { usePathname } from "next/navigation";
import type { Run } from "@/lib/api/rows";
import { modelName, datasetName } from "@/lib/derive";

/**
 * The selected run, and the model / dataset / split it was produced with.
 *
 * The URL is the source of truth for which run is selected, because a run is
 * addressable and a reload has to land on the same one. The runs themselves
 * are fetched on the server and handed in — a client context should not own a
 * network call that every page already needs.
 *
 * Model, dataset and split are read from the selected run rather than picked
 * by the user: they are properties of a completed analysis pass, not filters.
 * Choosing a different one selects the run that used it.
 */
interface RunContextValue {
  runId: string;
  runs: Run[];
  run: Run | undefined;
  model: string | null;
  dataset: string | null;
  split: string | null;
  /** Selecting a value routes to the most recent run matching it. */
  runsMatching: (key: "model" | "dataset" | "split", value: string) => Run | undefined;
}

const RunContext = createContext<RunContextValue | null>(null);

export function RunProvider({ runs, children }: { runs: Run[]; children: ReactNode }) {
  const pathname = usePathname();

  const runId = useMemo(() => {
    const match = pathname.match(/^\/runs\/([^/]+)/);
    if (match) return match[1];
    return runs[0] ? String(runs[0].id) : "";
  }, [pathname, runs]);

  const run = useMemo(() => runs.find((r) => String(r.id) === runId), [runs, runId]);

  const value = useMemo<RunContextValue>(() => ({
    runId,
    runs,
    run,
    model: modelName(run),
    dataset: datasetName(run),
    split: run?.split ?? null,
    runsMatching: (key, wanted) =>
      runs.find((candidate) => {
        if (key === "split") return candidate.split === wanted;
        if (key === "model") return modelName(candidate) === wanted;
        return datasetName(candidate) === wanted;
      }),
  }), [runId, runs, run]);

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRun() {
  const ctx = useContext(RunContext);
  if (!ctx) throw new Error("useRun must be used inside RunProvider");
  return ctx;
}
