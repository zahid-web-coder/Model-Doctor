"use client";

import {
  Home,
  Bell,
  Users,
  GitBranch,
  Map,
  Camera,
  LayoutGrid,
  Activity,
  GitMerge,
  AlertCircle,
  Clock,
  Target,
  FileText,
  Settings,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export function Sidebar() {
  const pathname = usePathname();
  const navItems = [
    { icon: LayoutGrid, label: "Overview", href: "/" },
    { icon: Activity, label: "Root Causes", href: "/runs/1287/root-causes" },
    { icon: GitMerge, label: "Clusters", href: "/runs/1287/clusters" },
    { icon: AlertCircle, label: "Failures", href: "/runs/1287/failures" },
    { icon: Clock, label: "Runs", href: "/runs" },
    { icon: Target, label: "Heatmaps", href: "/runs/1287/heatmaps" },
  ];

  return (
    <div className="w-[240px] h-full flex flex-col justify-between shrink-0">
      <div className="flex flex-col gap-8">
        {/* Logo */}
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

        {/* Navigation */}
        <nav className="flex flex-col gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <Link 
                href={item.href} 
                key={item.label}
                className={`flex items-center gap-3 px-4 py-3 text-[14px] font-medium rounded-lg transition-colors ${
                  isActive
                    ? "bg-gradient-to-r from-[#EBE6D8] to-transparent text-ink border-l-[3px] border-brass"
                    : "text-slate hover:bg-black/5"
                }`}
              >
                <Icon size={18} className={isActive ? "text-ink" : "text-slate"} strokeWidth={2} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Bottom Cards in Sidebar */}
      <div className="flex flex-col gap-4">
        {/* Current Run Card */}
        <div className="bg-[#EBE6D8]/50 rounded-xl p-4 border border-border/40">
          <p className="text-slate text-[11px] font-medium mb-1">Current Run</p>
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-ink font-semibold text-[15px]">Run #1287</h3>
            <div className="flex items-center gap-1.5 bg-[#4CAF50]/10 px-2 py-0.5 rounded-full">
              <div className="w-1.5 h-1.5 rounded-full bg-[#4CAF50] animate-pulse" />
              <span className="text-[#4CAF50] text-[10px] font-semibold uppercase">Live</span>
            </div>
          </div>
          <p className="text-slate text-[12px] mb-1">ResNet50 • 10k Images</p>
          <p className="text-slate/70 text-[11px]">May 21, 2024 • 10:25 AM</p>
        </div>

        {/* Model Health Card */}
        <div className="bg-[#EBE6D8]/50 rounded-xl p-4 border border-border/40 relative overflow-hidden">
          <p className="text-slate text-[11px] font-medium mb-1">Model Health</p>
          <h3 className="text-ink font-heading text-4xl mb-1">92%</h3>
          <p className="text-[#4CAF50] text-[12px] font-medium">↑ 6% vs last run</p>
          
          <div className="absolute -bottom-2 -right-2 w-32 h-16 opacity-40">
            <svg viewBox="0 0 100 30" className="w-full h-full stroke-[#4CAF50] fill-none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="0,25 20,20 40,25 60,10 80,15 100,5" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
