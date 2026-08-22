/** The stat tile used across Failures, Clusters and Root Causes. */
export function StatCard({
  label, value, sub, tone = "ink",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "ink" | "brass" | "slate";
}) {
  const colour =
    tone === "brass" ? "text-brass" : tone === "slate" ? "text-slate" : "text-ink";
  return (
    <div className="bg-card border border-border/40 rounded-lg px-4 py-3 flex-1 min-w-0">
      <p className="text-[11px] font-medium text-slate truncate">{label}</p>
      <p className={`font-heading text-[26px] leading-tight ${colour}`}>{value}</p>
      {sub && <p className="text-[11px] text-slate mt-0.5">{sub}</p>}
    </div>
  );
}
