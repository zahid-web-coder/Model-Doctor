import { Card } from "@/components/ui/card";

export function RightPanels() {
  return (
    <div className="w-[280px] flex flex-col gap-4 shrink-0">
      {/* Defect Detected */}
      <Card className="p-5 bg-[#EBE6D8]/50 border-border/40 shadow-none rounded-xl">
        <h3 className="text-[12px] font-medium text-ink mb-3">Defect Detected</h3>
        <p className="text-[13px] text-slate font-medium mb-1">Crack</p>
        <div className="flex items-baseline gap-2 mb-2">
          <span className="text-4xl font-heading text-ink">92%</span>
          <span className="text-[12px] text-slate">Confidence</span>
        </div>
        {/* Sparkline placeholder */}
        <div className="h-10 mt-4 w-full">
          <svg viewBox="0 0 100 30" className="w-full h-full stroke-[#4CAF50] fill-none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="0,25 10,22 20,26 30,15 40,18 50,20 60,10 70,12 80,8 90,14 100,5" />
          </svg>
        </div>
      </Card>

      {/* Live Heatmap */}
      <Card className="p-5 bg-[#EBE6D8]/50 border-border/40 shadow-none rounded-xl">
        <h3 className="text-[12px] font-medium text-ink mb-3">Live Heatmap</h3>
        <div className="flex gap-2 h-20">
          <div className="flex-1 bg-gradient-to-tr from-blue-900 via-yellow-400 to-red-600 rounded-md opacity-90 border border-black/10"></div>
          <div className="flex-1 bg-gradient-to-tr from-blue-900 via-orange-400 to-red-600 rounded-md opacity-90 border border-black/10 relative">
            <div className="absolute inset-0 m-auto w-3 h-3 bg-white rounded-full shadow-[0_0_10px_white]"></div>
          </div>
          <div className="w-2.5 bg-gradient-to-b from-red-600 via-yellow-400 to-blue-900 rounded-full ml-1"></div>
        </div>
      </Card>

      {/* Throughput */}
      <Card className="p-5 bg-[#EBE6D8]/50 border-border/40 shadow-none rounded-xl">
        <h3 className="text-[12px] font-medium text-ink mb-3">Throughput</h3>
        <div className="flex items-baseline gap-1 mb-1">
          <span className="text-3xl font-heading text-ink">1,256</span>
          <span className="text-[12px] text-slate">img/min</span>
        </div>
        <p className="text-[#4CAF50] text-[12px] font-medium mb-4">↑ 12% vs last run</p>
        {/* Sparkline placeholder */}
        <div className="h-10 w-full">
          <svg viewBox="0 0 100 30" className="w-full h-full stroke-[#4CAF50] fill-none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="0,28 15,25 30,22 45,26 60,15 75,18 90,8 100,10" />
          </svg>
        </div>
      </Card>

      {/* Active Alerts */}
      <Card className="p-5 bg-[#EBE6D8]/50 border-border/40 shadow-none rounded-xl flex-1">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-[12px] font-medium text-ink">Active Alerts</h3>
          <div className="bg-[#E47260] text-white text-[10px] font-bold px-2 py-0.5 rounded-full">
            3
          </div>
        </div>
        <ul className="flex flex-col gap-3">
          {["Anomaly Detected", "Low Confidence", "Data Drift"].map((alert, i) => (
            <li key={i} className="flex items-center gap-2 text-[13px] text-slate">
              <div className="w-1.5 h-1.5 rounded-full bg-[#E47260]" />
              {alert}
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
