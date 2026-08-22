"use client";

import {
  LayoutGrid, Activity, GitMerge, AlertCircle, Clock, Target,
  FileText, Settings,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useRun } from "@/lib/run-context";
import { modelName, datasetName, modelFingerprint } from "@/lib/derive";
import { num } from "@/lib/format";
import { Wordmark } from "@/components/shared/Logo";

/**
 * Primary navigation, plus the current-run card.
 *
 * Two things were broken here. The run-scoped links were hardcoded to run
 * "1287", so the sidebar always pointed at one run whatever you were looking
 * at. And the active state used `pathname.startsWith(href)`, which meant
 * "Runs" (`/runs`) matched every `/runs/1287/...` page — so Runs stayed lit on
 * every screen alongside the real one. Both are fixed below: hrefs are built
 * from the selected run, and matching is exact except where a section
 * genuinely owns a subtree.
 */
export function Sidebar() {
  const pathname = usePathname();
  const { runId, run } = useRun();

  const items = [
    { icon: LayoutGrid,  label: "Overview",    href: "/dashboard" },
    { icon: Activity,    label: "Root Causes", href: `/runs/${runId}/root-causes` },
    { icon: GitMerge,    label: "Clusters",    href: `/runs/${runId}/clusters` },
    { icon: AlertCircle, label: "Failures",    href: `/runs/${runId}/failures` },
    { icon: Clock,       label: "Runs",        href: "/runs" },
    { icon: Target,      label: "Heatmaps",    href: `/runs/${runId}/heatmaps` },
    { icon: FileText,    label: "Reports",     href: "/reports" },
    { icon: Settings,    label: "Settings",    href: "/settings" },
  ];

  // "Runs" owns only the index and a bare /runs/:id, never the tab pages —
  // those belong to their own nav items.
  const isActive = (href: string) => {
    if (href === "/runs") return pathname === "/runs" || /^\/runs\/[^/]+$/.test(pathname);
    return pathname === href;
  };

  return (
    <div className="w-[240px] h-full flex flex-col justify-between shrink-0">
      <div className="flex flex-col gap-8">
        <div className="px-2">
          <Wordmark size={38} />
        </div>

        <nav className="flex flex-col gap-1">
          {items.map(({ icon: Icon, label, href }) => {
            const active = isActive(href);
            return (
              <Link
                href={href}
                key={label}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-3 px-4 py-3 text-[14px] font-medium rounded-lg transition-colors ${
                  active
                    ? "bg-gradient-to-r from-[#EBE6D8] to-transparent text-ink border-l-[3px] border-brass"
                    : "text-slate hover:bg-black/5"
                }`}
              >
                <Icon size={18} className={active ? "text-ink" : "text-slate"} strokeWidth={2} />
                <span>{label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Current run. Reads the selected run rather than a hardcoded one, and
          links back to that run's page. No LIVE badge and no Model Health:
          nothing streams, and there is no such metric in the schema. */}
      <div className="flex flex-col gap-4">
        <Link
          href={`/runs/${runId}/failures`}
          className="block bg-[#EBE6D8]/50 rounded-xl p-4 border border-border/40 hover:border-brass/60 transition-colors"
        >
          <p className="text-slate text-[11px] font-medium mb-1">Current Run</p>
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-ink font-semibold text-[15px]">Run #{runId}</h3>
            <span className="text-[10px] font-semibold uppercase tracking-wide text-slate px-2 py-0.5 rounded-full bg-black/5">
              {run?.split ?? "n/a"}
            </span>
          </div>
          <p className="text-slate text-[12px] mb-1">
            {modelName(run) ?? "n/a"}
            <span className="text-slate/60 font-mono text-[10px] ml-1.5">{modelFingerprint(run)}</span>
          </p>
          <p className="text-slate/70 text-[11px]">{datasetName(run) ?? "n/a"}</p>
        </Link>

        {/* Run configuration, straight off the run row. A findings count
            would need a second request per page for a number the pages
            already show, so the card carries the thresholds the pass ran at
            instead — which nothing else on screen surfaces. */}
        <div className="bg-[#EBE6D8]/50 rounded-xl p-4 border border-border/40">
          <p className="text-slate text-[11px] font-medium mb-2">Run configuration</p>
          <Config label="Confidence" value={run ? run.confidence_threshold.toFixed(2) : "n/a"} />
          <Config label="Match IoU" value={run ? run.match_iou_threshold.toFixed(2) : "n/a"} />
          <Config label="Localisation floor" value={run ? run.localization_iou_floor.toFixed(2) : "n/a"} />
          <Config label="Image size" value={run ? num(run.image_size) : "n/a"} />
        </div>
      </div>
    </div>
  );
}

function Config({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 mb-1 last:mb-0">
      <span className="text-slate text-[11px]">{label}</span>
      <span className="text-ink font-mono text-[11px]">{value}</span>
    </div>
  );
}
