"use client";

import { createContext, useContext, useMemo, useState, ReactNode } from "react";
import { usePathname } from "next/navigation";
import { mockRuns } from "@/lib/mock-data/runs";

/**
 * The selected run, model, dataset and environment, shared by every screen.
 *
 * Before this the run id was hardcoded as "1287" in six places — the sidebar
 * links, the run layout and the mock data — so nothing could actually be
 * selected. The URL is the source of truth for the run, because the run is
 * addressable and a reload has to land on the same one.
 */

interface RunContextValue {
  runId: string;
  model: string;
  dataset: string;
  environment: string;
  setModel: (v: string) => void;
  setDataset: (v: string) => void;
  setEnvironment: (v: string) => void;
}

const RunContext = createContext<RunContextValue | null>(null);

export function RunProvider({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [model, setModel] = useState("yolov8s-seg");
  const [dataset, setDataset] = useState("door_df_window");
  const [environment, setEnvironment] = useState("test");

  // Read the run out of the path, falling back to the most recent one on
  // pages that are not run-scoped (Overview, Runs, Reports, Settings).
  const runId = useMemo(() => {
    const match = pathname.match(/^\/runs\/([^/]+)/);
    return match ? match[1] : mockRuns[0].id;
  }, [pathname]);

  const value = useMemo(
    () => ({ runId, model, dataset, environment, setModel, setDataset, setEnvironment }),
    [runId, model, dataset, environment]
  );

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRun() {
  const ctx = useContext(RunContext);
  if (!ctx) throw new Error("useRun must be used inside RunProvider");
  return ctx;
}
