"use client";

import { useEffect, useState } from "react";

import {
  LayoutGrid, Activity, GitMerge, AlertCircle, Clock, Target,
  FileText, Settings, PanelLeftClose, PanelLeftOpen, Scale, PlusCircle, Lightbulb, Images,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useRun } from "@/lib/run-context";
import { modelName, datasetName, modelFingerprint } from "@/lib/derive";
import { num } from "@/lib/format";
import { Logo, Wordmark } from "@/components/shared/Logo";

const STORAGE_KEY = "md.sidebar.collapsed";

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
 *
 * **Collapsible to a 68px rail**, remembered across navigations. The mark
 * stays at both widths; the labels and the run cards are what give way.
 */
export function Sidebar() {
  const pathname = usePathname();
  const { runId, run } = useRun();

  // Starts expanded and corrects on mount rather than reading storage during
  // render: the server has no localStorage, so seeding from it directly would
  // render one width on the server and another on the client.
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => {
    setCollapsed(window.localStorage.getItem(STORAGE_KEY) === "1");
  }, []);

  const toggle = () => {
    setCollapsed((was) => {
      const next = !was;
      window.localStorage.setItem(STORAGE_KEY, next ? "1" : "0");
      return next;
    });
  };

  const items = [
    { icon: PlusCircle,  label: "New Analysis", href: "/analyze" },
    { icon: LayoutGrid,  label: "Overview",    href: "/dashboard" },
    { icon: Activity,    label: "Root Causes", href: `/runs/${runId}/root-causes` },
    { icon: GitMerge,    label: "Clusters",    href: `/runs/${runId}/clusters` },
    { icon: AlertCircle, label: "Failures",    href: `/runs/${runId}/failures` },
    { icon: Clock,       label: "Runs",        href: "/runs" },
    { icon: Scale,       label: "Compare",     href: "/compare" },
    { icon: Images,      label: "Images",      href: `/runs/${runId}/images` },
    { icon: Target,      label: "Heatmaps",    href: `/runs/${runId}/heatmaps` },
    { icon: Lightbulb,   label: "Recommendations", href: `/runs/${runId}/recommendations` },
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
    <div
      className={`${collapsed ? "w-[68px]" : "w-[240px]"} h-full flex flex-col justify-between shrink-0 transition-[width] duration-200`}
    >
      <div className="flex flex-col gap-8">
        {/* The mark stays at every width — it is the one thing that should not
            collapse, because a rail with no identity on it reads as chrome
            belonging to the browser rather than to the product. */}
        <div className={`flex items-center ${collapsed ? "flex-col gap-3" : "justify-between"} px-2`}>
          {collapsed ? <Logo size={34} /> : <Wordmark size={38} />}
          <button
            type="button"
            onClick={toggle}
            aria-expanded={!collapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="grid place-items-center w-7 h-7 rounded-md text-slate hover:text-ink hover:bg-black/5 transition-colors shrink-0"
          >
            {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>

        <nav className="flex flex-col gap-1">
          {items.map(({ icon: Icon, label, href }) => {
            const active = isActive(href);
            return (
              <Link
                href={href}
                key={label}
                aria-current={active ? "page" : undefined}
                // The label is the accessible name at full width; collapsed,
                // the icon alone is not, so `title` carries it for pointers and
                // the visually-hidden span carries it for screen readers.
                title={collapsed ? label : undefined}
                className={`flex items-center gap-3 ${collapsed ? "justify-center px-0" : "px-4"} py-3 text-[14px] font-medium rounded-lg transition-colors ${
                  active
                    ? `bg-gradient-to-r from-[#EBE6D8] to-transparent text-ink ${collapsed ? "" : "border-l-[3px] border-brass"}`
                    : "text-slate hover:bg-black/5"
                }`}
              >
                <Icon size={18} className={active ? "text-ink" : "text-slate"} strokeWidth={2} />
                <span className={collapsed ? "sr-only" : undefined}>{label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Current run. Reads the selected run rather than a hardcoded one, and
          links back to that run's page. No LIVE badge and no Model Health:
          nothing streams, and there is no such metric in the schema.

          Collapsed, this becomes a single badge. Squeezing the two cards into
          a 68px rail would leave every value truncated, and a truncated
          fingerprint is worse than no fingerprint — the run number is the part
          that still means something at this width. */}
      {collapsed ? (
        <Link
          href={`/runs/${runId}/failures`}
          title={`Run #${runId} — ${run?.split ?? "n/a"}`}
          className="grid place-items-center mx-auto w-11 h-11 rounded-xl bg-[#EBE6D8]/50 border border-border/40 hover:border-brass/60 transition-colors"
        >
          <span className="text-[9px] font-medium text-slate leading-none">Run</span>
          <span className="text-ink font-semibold text-[13px] leading-none mt-0.5">#{runId}</span>
        </Link>
      ) : (
      <div className="flex flex-col gap-4">
        <Link
          href={`/runs/${runId}/failures`}
          className="block bg-[#EBE6D8]/50 rounded-xl p-4 border border-border/40 hover:border-brass/60 transition-colors"
        >
          <p className="text-slate text-[11px] font-medium mb-1">Current Run</p>
          <div className="flex justify-between items-center gap-2 mb-3">
            {/* The id stays even when the run is named: it is what every URL,
                export and log line refers to, and two runs may share a name. */}
            <h3 className="text-ink font-semibold text-[15px] min-w-0 truncate">
              {run?.name ? (
                <>
                  {run.name}
                  <span className="text-slate/60 font-normal text-[12px] ml-1.5">
                    #{runId}
                  </span>
                </>
              ) : (
                `Run #${runId}`
              )}
            </h3>
            <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wide text-slate px-2 py-0.5 rounded-full bg-black/5">
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
      )}
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
