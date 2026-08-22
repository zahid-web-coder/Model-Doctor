"use client";

import { mockHeatmaps } from "@/lib/mock-data/heatmaps";

export default function HeatmapsPage() {
  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex justify-between items-center mb-4 shrink-0">
        <div>
          <h3 className="text-[15px] font-medium text-ink mb-1">Attention Heatmaps</h3>
          <p className="text-[13px] text-slate">Visualizing model focus areas on failure instances.</p>
        </div>
        
        <div className="flex gap-2">
          <select className="bg-card border border-border/40 rounded-md text-[13px] text-ink px-3 py-1.5 outline-none">
            <option>All Failure Types</option>
            <option>False Positive</option>
            <option>False Negative</option>
          </select>
          <select className="bg-card border border-border/40 rounded-md text-[13px] text-ink px-3 py-1.5 outline-none">
            <option>All Factors</option>
            <option>blur</option>
            <option>low_light</option>
          </select>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 pb-6">
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
          {mockHeatmaps.map((img) => (
            <div key={img.id} className="bg-card border border-border/40 rounded-lg p-2 flex flex-col hover:border-brass/50 transition-colors cursor-pointer group">
              <div className={`w-full aspect-square rounded mb-2 relative overflow-hidden ${img.thumbnail}`}>
                {/* Heatmap overlay */}
                <div className="absolute inset-0 m-auto w-[60%] h-[60%] bg-gradient-to-tr from-blue-900/40 via-orange-400/40 to-red-600/40 mix-blend-overlay rounded-full blur-[6px] opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
              <div className="flex justify-between items-center mt-auto">
                <span className="font-mono text-[11px] text-ink font-medium">{img.id}</span>
                <span className="font-mono text-[10px] text-slate">{(img.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
