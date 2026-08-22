export function TelemetryStrip() {
  const stats = [
    { label: "Conveyor", value: "0.2 m/s", status: "ok" },
    { label: "Scanner", value: "Active", status: "live" },
    { label: "Beam", value: "100%", status: "ok" },
    { label: "Robotic status", value: "Nominal", status: "ok" },
    { label: "Accuracy", value: "99.2%", status: "ok" },
  ];

  return (
    <div className="bg-panel-dark/95 backdrop-blur-md border-t border-white/10 p-3 px-6 flex justify-between items-center text-[11px] font-mono w-full">
      <div className="flex gap-8 w-full justify-between items-center text-slate/80">
        {stats.map((stat, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="uppercase tracking-wider">{stat.label}</span>
            <span className={`font-semibold ${stat.status === 'live' ? 'text-[#4CAF50]' : 'text-[#DED9CC]'}`}>
              {stat.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
