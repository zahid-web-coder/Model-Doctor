import { mockRootCauses } from "@/lib/mock-data/root-causes";

export default function RootCausesPage() {
  // Sort by lift descending, nulls last
  const sortedFactors = [...mockRootCauses].sort((a, b) => {
    if (a.lift === null) return 1;
    if (b.lift === null) return -1;
    return b.lift - a.lift;
  });

  return (
    <div className="flex flex-col gap-6 h-full overflow-y-auto pr-2 pb-6 custom-scrollbar">
      <div>
        <h3 className="text-[15px] font-medium text-ink mb-1">Root Cause Factors</h3>
        <p className="text-[13px] text-slate mb-6">Factors statistically correlated with model failures.</p>
        
        <div className="bg-card border border-border/40 rounded-xl overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-[13px] text-left border-collapse">
              <thead>
                <tr className="border-b border-border/40 bg-muted/30">
                  <th className="py-3 px-4 font-medium text-slate">Factor</th>
                  <th className="py-3 px-4 font-medium text-slate text-right">Failures Rate</th>
                  <th className="py-3 px-4 font-medium text-slate text-right">Correct Rate</th>
                  <th className="py-3 px-4 font-medium text-slate text-right">Lift</th>
                  <th className="py-3 px-4 font-medium text-slate text-right">P-value</th>
                  <th className="py-3 px-4 font-medium text-slate">Reading</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/20">
                {sortedFactors.map((factor) => {
                  const noEvidence = factor.pValue === null || factor.pValue >= 0.05;
                  
                  return (
                    <tr 
                      key={factor.factor} 
                      className={`hover:bg-muted/20 transition-colors ${noEvidence ? 'text-slate opacity-70' : 'text-ink'}`}
                    >
                      <td className="py-3 px-4 font-mono font-medium">{factor.factor}</td>
                      <td className="py-3 px-4 text-right font-mono">{(factor.rateInFailures * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 text-right font-mono">{(factor.rateInCorrect * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 text-right font-mono font-semibold">
                        {factor.lift !== null ? `${factor.lift.toFixed(2)}x` : 'n/a'}
                      </td>
                      <td className={`py-3 px-4 text-right font-mono ${!noEvidence ? 'text-[#66805A]' : 'text-[#A65C48]'}`}>
                        {factor.pValue !== null ? (factor.pValue < 0.001 ? '<0.001' : factor.pValue.toFixed(3)) : 'n/a'}
                      </td>
                      <td className="py-3 px-4 text-slate">{factor.reading}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-[15px] font-medium text-ink mb-4">Attribution & Evidence</h3>
        <div className="flex flex-col gap-3">
          {sortedFactors.filter(f => f.lift !== null && f.pValue !== null && f.pValue < 0.05).map(factor => (
            <div key={factor.factor} className="bg-muted/30 border border-border/20 rounded-lg p-4">
              <h4 className="font-mono text-[13px] font-semibold text-ink mb-2">{factor.factor}</h4>
              <ul className="list-disc list-inside flex flex-col gap-1 text-[13px] text-slate">
                {factor.evidence.map((ev, i) => (
                  <li key={i}>{ev}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
