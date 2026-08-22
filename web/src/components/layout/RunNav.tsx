"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

interface RunNavProps {
  runId: string;
}

export function RunNav({ runId }: RunNavProps) {
  const pathname = usePathname();
  
  const navItems = [
    { name: "Root Causes", path: `/runs/${runId}/root-causes` },
    { name: "Clusters", path: `/runs/${runId}/clusters` },
    { name: "Failures", path: `/runs/${runId}/failures` },
    { name: "Heatmaps", path: `/runs/${runId}/heatmaps` },
  ];

  return (
    <nav className="flex gap-1 relative top-[1px]">
      {navItems.map((item) => {
        const isActive = pathname.startsWith(item.path);
        return (
          <Link
            key={item.name}
            href={item.path}
            className={`px-4 py-2.5 text-[14px] font-medium rounded-t-lg transition-colors border-b-2 ${
              isActive
                ? "text-ink border-brass bg-background/50"
                : "text-slate border-transparent hover:bg-black/5 hover:text-ink"
            }`}
          >
            {item.name}
          </Link>
        );
      })}
    </nav>
  );
}
