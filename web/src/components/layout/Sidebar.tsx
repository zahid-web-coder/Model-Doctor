"use client";

import {
  LayoutGrid, Activity, GitMerge, AlertCircle, Clock, Target,
  FileText, Settings,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useRun } from "@/lib/run-context";
import { findRun } from "@/lib/mock-data/runs";
import { num } from "@/lib/format";

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
  const { runId } = useRun();
  const run = findRun(runId);

  const items = [
    { icon: LayoutGrid,  label: "Overview",    href: "/" },
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
    if (href === "/") return pathname === "/";
    if (href === "/runs") return pathname === "/runs" || /^\/runs\/[^/]+$/.test(pathname);
    return pathname === href;
  };

  return (
    <div className="w-[240px] h-full flex flex-col justify-between shrink-0">
      <div className="flex flex-col gap-8">
        <div className="flex items-center gap-3 px-2">
          <div className="w-10 h-10 bg-gradient-to-br from-[#D4CFC4] to-[#AFAAA0] rounded-sm flex items-center justify-center shadow-sm">
            <div className="w-5 h-5 border-[2px] border-white/80" />
          </div>
          <div>
            <h2 className="text-[15px] font-bold tracking-wide text-ink/90 leading-tight">
              MODEL<br />DOCTOR
            </h2>
            <p className="text-[9px] font-semibold tracking-wider text-slate uppercase mt-[2px]">
              AI Model Diagnostics
            </p>
          </div>
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
              {run?.status ?? "unknown"}
            </span>
          </div>
          <p className="text-slate text-[12px] mb-1">
            {run?.model ?? "n/a"} • {num(run?.totalImages)} images
          </p>
          <p className="text-slate/70 text-[11px]">
            {run?.dataset ?? "n/a"} • {run?.split ?? "n/a"} split
          </p>
        </Link>

        <div className="bg-[#EBE6D8]/50 rounded-xl p-4 border border-border/40">
          <p className="text-slate text-[11px] font-medium mb-1">Findings</p>
          <h3 className="text-ink font-heading text-4xl mb-1">{num(run?.failureCount)}</h3>
          <p className="text-slate text-[12px]">
            of {num(run?.totalImages)} images in this run
          </p>
        </div>
      </div>
    </div>
  );
}
