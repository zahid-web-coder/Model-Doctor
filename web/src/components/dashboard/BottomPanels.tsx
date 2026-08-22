import { Card } from "@/components/ui/card";
import { RotateCw, Plus, Minus, Crosshair } from "lucide-react";

export function BottomPanels() {
  const recentFailures = [
    { label: "Crack", conf: "96%" },
    { label: "Missing Part", conf: "94%" },
    { label: "Deformation", conf: "93%" },
    { label: "Rust", conf: "92%" },
    { label: "Leak", conf: "91%" },
    { label: "Break", conf: "90%" },
    { label: "Corrosion", conf: "89%" },
    { label: "Dirt", conf: "88%" },
  ];

  return (
    <div className="flex flex-col gap-4 bg-[#EBE6D8]/50 p-4 rounded-xl border border-border/40 mt-auto shrink-0">
      <div className="flex gap-4">
        {/* Failure Heatmap (3D View) */}
        <div className="flex-1 bg-[#DED9CC]/30 rounded-lg p-4 relative min-h-[220px]">
          <h3 className="text-[12px] font-medium text-ink mb-2">Failure Heatmap (3D View)</h3>
          
          <div className="absolute left-4 top-12 flex flex-col gap-2 z-10">
            <button className="w-6 h-6 flex items-center justify-center text-slate hover:text-ink"><RotateCw size={14} /></button>
            <button className="w-6 h-6 flex items-center justify-center text-slate hover:text-ink"><Plus size={16} /></button>
            <button className="w-6 h-6 flex items-center justify-center text-slate hover:text-ink"><Minus size={16} /></button>
            <button className="w-6 h-6 flex items-center justify-center text-slate hover:text-ink"><Crosshair size={14} /></button>
          </div>
          
          <div className="absolute bottom-4 left-4 bg-white/50 px-2 py-0.5 rounded text-[10px] font-bold text-ink">3D</div>
          
          {/* Mockup for 3D Heatmap - using a pseudo isometric structure */}
          <div className="absolute inset-0 flex items-center justify-center opacity-60 pointer-events-none">
            <div className="w-[180px] h-[120px] bg-gradient-to-tr from-blue-900 via-orange-400 to-red-600 blur-xl opacity-40 rounded-full"></div>
            {/* Voxel grid mockup */}
            <div className="absolute grid grid-cols-5 grid-rows-5 gap-1 transform rotate-x-60 rotate-z-45 scale-150">
              {Array.from({length: 25}).map((_, i) => (
                <div key={i} className="w-4 h-4 bg-[#1E1B16]/5 border border-white/20"></div>
              ))}
            </div>
          </div>
        </div>

        {/* Failure Distribution */}
        <div className="w-[340px] flex flex-col gap-4">
          <div className="flex gap-6">
            <div className="flex-1">
              <h3 className="text-[12px] font-medium text-ink mb-2">Failure Distribution</h3>
              <div className="relative w-20 h-20 mx-auto mt-4">
                {/* SVG Pie Chart Mock */}
                <svg viewBox="0 0 100 100" className="transform -rotate-90 w-full h-full">
                  <circle cx="50" cy="50" r="40" fill="none" stroke="#EAE4D6" strokeWidth="20" />
                  <circle cx="50" cy="50" r="40" fill="none" stroke="#E47260" strokeWidth="20" strokeDasharray="251.2" strokeDashoffset="150" />
                  <circle cx="50" cy="50" r="40" fill="none" stroke="#F5A623" strokeWidth="20" strokeDasharray="251.2" strokeDashoffset="220" />
                  <circle cx="50" cy="50" r="40" fill="none" stroke="#4A90E2" strokeWidth="20" strokeDasharray="251.2" strokeDashoffset="200" />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-white rounded-full m-4">
                  <span className="text-[14px] font-bold text-ink leading-none">1,247</span>
                  <span className="text-[8px] text-slate mt-0.5 leading-none text-center">Total<br/>Failures</span>
                </div>
              </div>
            </div>
            <div className="flex-1 flex flex-col justify-center gap-2 mt-6">
              {[
                { label: "False Negative", count: "504", pct: "40.5%", color: "bg-[#E47260]" },
                { label: "Wrong Class", count: "237", pct: "19.0%", color: "bg-[#F5A623]" },
                { label: "Poor Localization", count: "267", pct: "21.4%", color: "bg-[#4A90E2]" },
                { label: "False Positive", count: "239", pct: "19.1%", color: "bg-[#9B51E0]" },
              ].map(item => (
                <div key={item.label} className="flex items-center text-[10px] whitespace-nowrap">
                  <div className={`w-2 h-2 rounded-sm ${item.color} mr-2`} />
                  <span className="text-slate flex-1 w-24">{item.label}</span>
                  <span className="text-ink font-medium w-6">{item.count}</span>
                  <span className="text-slate ml-1">({item.pct})</span>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="text-[11px] font-medium text-slate mb-2">Top Root Cause Factors</h3>
            <div className="text-[10px]">
              <div className="flex text-slate/70 border-b border-border/30 pb-1 mb-1">
                <span className="flex-[2]">Factor</span>
                <span className="flex-1 text-right">Failures</span>
                <span className="flex-1 text-right">Lift vs. Control</span>
                <span className="flex-1 text-right">P-value</span>
              </div>
              {[
                { factor: "blur", failures: "412", lift: "2.71x", p: "<0.001", sig: true },
                { factor: "low_light", failures: "351", lift: "2.24x", p: "<0.001", sig: true },
                { factor: "occlusion", failures: "232", lift: "1.72x", p: "0.002", sig: true },
                { factor: "small_object", failures: "198", lift: "1.42x", p: "0.006", sig: true },
                { factor: "contrast", failures: "182", lift: "1.27x", p: "0.031", sig: true },
              ].map(row => (
                <div key={row.factor} className="flex py-0.5 items-center">
                  <span className="flex-[2] text-ink font-mono">{row.factor}</span>
                  <span className="flex-1 text-right text-ink font-mono">{row.failures}</span>
                  <span className="flex-1 text-right text-ink font-mono">{row.lift}</span>
                  <span className={`flex-1 text-right font-mono ${row.sig ? 'text-[#A65C48]' : 'text-slate'}`}>{row.p}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Live Inference */}
        <div className="w-[280px]">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-[12px] font-medium text-ink">Live Inference</h3>
            <div className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-[#4CAF50] animate-pulse" />
              <span className="text-[#4CAF50] text-[10px] font-semibold uppercase">Live</span>
            </div>
          </div>
          
          <div className="bg-[#DED9CC]/30 rounded-lg p-1 min-h-[140px] relative overflow-hidden mb-3">
             <div className="absolute inset-0 m-auto w-[80%] h-[60%] bg-gradient-to-tr from-blue-900 via-orange-400 to-red-600 rounded-md opacity-90 blur-[2px]"></div>
          </div>
          
          <div className="flex justify-between items-center text-[11px] mb-1">
            <span className="text-slate">Prediction</span>
            <span className="bg-[#E47260]/20 text-[#E47260] px-1.5 py-0.5 rounded text-[10px] font-bold">NG</span>
          </div>
          <div className="flex justify-between items-center text-[13px] font-medium mb-1">
            <span className="text-[#E47260]">Defect</span>
            <span className="text-ink">92%</span>
          </div>
          <div className="flex justify-between items-center text-[11px]">
            <span className="text-slate">Confidence / Time</span>
            <span className="text-ink font-mono">120 ms</span>
          </div>
        </div>
      </div>

      {/* Recent Worst Failures Strip */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-[11px] font-medium text-slate">Recent Worst Failures</h3>
          <button className="text-[11px] text-ink font-medium hover:underline flex items-center gap-1">
            View all →
          </button>
        </div>
        
        <div className="flex gap-2 justify-between">
          {recentFailures.map((failure, idx) => (
            <div key={idx} className="flex-1 flex flex-col gap-1">
              <div className="bg-[#DED9CC]/50 aspect-video rounded-md relative overflow-hidden">
                <div className="absolute inset-0 m-auto w-[60%] h-[50%] bg-gradient-to-tr from-blue-900 via-yellow-400 to-red-600 rounded opacity-80 blur-[1px]"></div>
              </div>
              <div className="flex justify-between items-center text-[10px]">
                <span className="text-slate font-medium truncate pr-1">{failure.label}</span>
                <span className="text-[#E47260] font-bold">{failure.conf}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
