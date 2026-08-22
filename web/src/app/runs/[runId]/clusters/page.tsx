"use client";

import { useState } from "react";
import { mockClusters, Cluster } from "@/lib/mock-data/clusters";

export default function ClustersPage() {
  const [activeCluster, setActiveCluster] = useState<Cluster | null>(mockClusters[0]);

  return (
    <div className="flex gap-6 h-full overflow-hidden">
      {/* Clusters List */}
      <div className="w-[300px] flex flex-col gap-3 h-full overflow-y-auto pr-2 custom-scrollbar shrink-0">
        <h3 className="text-[15px] font-medium text-ink mb-1">Failure Clusters</h3>
        {mockClusters.map((cluster) => (
          <div 
            key={cluster.id}
            onClick={() => setActiveCluster(cluster)}
            className={`p-3 rounded-lg border cursor-pointer transition-all ${
              activeCluster?.id === cluster.id 
                ? 'border-brass bg-background/50 shadow-sm' 
                : 'border-border/40 bg-card hover:bg-muted/30'
            }`}
          >
            <div className="flex gap-3 mb-2">
              <div className={`w-12 h-12 rounded-md bg-gradient-to-tr ${cluster.thumbnail} opacity-80 shrink-0`} />
              <div className="overflow-hidden">
                <h4 className="text-[13px] font-semibold text-ink truncate">{cluster.name}</h4>
                <p className="text-[12px] text-slate font-mono mt-1">{cluster.memberCount} members</p>
              </div>
            </div>
            <div className="flex items-center gap-2 mt-2 pt-2 border-t border-border/20">
              <span className="text-[10px] text-slate uppercase tracking-wider">Dominant Factor:</span>
              <span className="text-[11px] font-mono font-medium text-ink bg-muted/50 px-1.5 py-0.5 rounded">{cluster.dominantFactor}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Cluster Detail View */}
      <div className="flex-1 bg-card rounded-xl border border-border/40 p-6 flex flex-col overflow-hidden">
        {activeCluster ? (
          <>
            <div className="mb-6">
              <h2 className="text-2xl font-heading text-ink mb-2">{activeCluster.name}</h2>
              <p className="text-[14px] text-slate">{activeCluster.description}</p>
            </div>
            
            <h3 className="text-[13px] font-medium text-ink mb-3">Representative Samples</h3>
            <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">
              <div className="grid grid-cols-3 gap-4">
                {Array.from({length: 12}).map((_, i) => (
                  <div key={i} className="aspect-square rounded-lg bg-muted relative overflow-hidden group">
                    <div className={`absolute inset-0 m-auto w-[70%] h-[70%] bg-gradient-to-tr ${activeCluster.thumbnail} rounded-full blur-[8px] opacity-60 mix-blend-multiply`} />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-slate text-[13px]">
            Select a cluster to view details
          </div>
        )}
      </div>
    </div>
  );
}
