"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/** The per-run tab strip. Every tab keeps the run you are already looking at. */
export function RunNav({ runId }: { runId: string }) {
  const pathname = usePathname();
  const tabs = [
    { label: "Root Causes", href: `/runs/${runId}/root-causes` },
    { label: "Clusters", href: `/runs/${runId}/clusters` },
    { label: "Failures", href: `/runs/${runId}/failures` },
    { label: "Heatmaps", href: `/runs/${runId}/heatmaps` },
  ];

  return (
    <nav className="flex items-center gap-1">
      {tabs.map(({ label, href }) => {
        const active = pathname === href;
        return (
          <Link
            key={label}
            href={href}
            aria-current={active ? "page" : undefined}
            className={`px-4 py-3 text-[14px] transition-colors border-b-2 -mb-[1px] ${
              active
                ? "text-ink font-medium border-brass"
                : "text-slate border-transparent hover:text-ink/80"
            }`}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
