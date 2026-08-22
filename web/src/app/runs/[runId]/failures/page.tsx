"use client";

import { useState } from "react";
import { mockFailures, FailureInstance } from "@/lib/mock-data/failures";

export default function FailuresPage() {
  const [selectedFailure, setSelectedFailure] = useState<FailureInstance | null>(mockFailures[0]);

  return (
    <div className="flex gap-6 h-full overflow-hidden">
      {/* Failures Table */}
      <div className="flex-1 bg-card border border-border/40 rounded-xl flex flex-col overflow-hidden shadow-sm">
        <div className="overflow-x-auto flex-1 custom-scrollbar">
          <table className="w-full text-[13px] text-left border-collapse whitespace-nowrap">
            <thead className="sticky top-0 bg-muted/90 backdrop-blur-sm z-10 border-b border-border/40">
              <tr>
                <th className="py-3 px-4 font-medium text-slate w-16">Image</th>
                <th className="py-3 px-4 font-medium text-slate">ID</th>
                <th className="py-3 px-4 font-medium text-slate">Prediction</th>
                <th className="py-3 px-4 font-medium text-slate">Ground Truth</th>
                <th className="py-3 px-4 font-medium text-slate text-right">Confidence</th>
                <th className="py-3 px-4 font-medium text-slate">Failure Type</th>
                <th className="py-3 px-4 font-medium text-slate">Root Cause</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/20">
              {mockFailures.map((failure) => (
                <tr 
                  key={failure.id} 
                  onClick={() => setSelectedFailure(failure)}
                  className={`cursor-pointer transition-colors ${selectedFailure?.id === failure.id ? 'bg-brass/10' : 'hover:bg-muted/30'}`}
                >
                  <td className="py-2 px-4">
                    <div className={`w-10 h-10 rounded ${failure.thumbnail}`} />
                  </td>
                  <td className="py-2 px-4 font-mono font-medium text-ink">{failure.id}</td>
                  <td className="py-2 px-4 text-ink">{failure.prediction}</td>
                  <td className="py-2 px-4 text-ink">{failure.groundTruth}</td>
                  <td className="py-2 px-4 text-right font-mono font-medium text-ink">{(failure.confidence * 100).toFixed(1)}%</td>
                  <td className="py-2 px-4">
                    <span className="bg-muted px-2 py-1 rounded text-slate text-[11px]">{failure.failureType}</span>
                  </td>
                  <td className="py-2 px-4 font-mono text-slate">{failure.rootCauseFactor}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Failure Detail View */}
      {selectedFailure && (
        <div className="w-[320px] bg-card border border-border/40 rounded-xl flex flex-col overflow-hidden shrink-0">
          <div className="p-4 border-b border-border/20">
            <h3 className="font-mono text-[14px] font-semibold text-ink">{selectedFailure.id}</h3>
          </div>
          
          <div className="p-4 flex-1 overflow-y-auto custom-scrollbar">
            <div className={`w-full aspect-square rounded-lg mb-4 relative overflow-hidden ${selectedFailure.thumbnail}`}>
              <div className="absolute inset-0 bg-gradient-to-tr from-blue-900/30 via-orange-400/30 to-red-600/30 mix-blend-overlay" />
            </div>
            
            <div className="space-y-4">
              <div>
                <p className="text-[11px] font-medium text-slate uppercase tracking-wider mb-1">Failure Type</p>
                <p className="text-[13px] text-ink font-medium">{selectedFailure.failureType}</p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[11px] font-medium text-slate uppercase tracking-wider mb-1">Prediction</p>
                  <p className="text-[13px] text-ink font-medium">{selectedFailure.prediction}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium text-slate uppercase tracking-wider mb-1">Ground Truth</p>
                  <p className="text-[13px] text-ink font-medium">{selectedFailure.groundTruth}</p>
                </div>
              </div>
              
              <div>
                <p className="text-[11px] font-medium text-slate uppercase tracking-wider mb-1">Confidence</p>
                <p className="text-[14px] text-ink font-mono font-medium">{(selectedFailure.confidence * 100).toFixed(2)}%</p>
              </div>
              
              <div>
                <p className="text-[11px] font-medium text-slate uppercase tracking-wider mb-1">Correlated Root Cause</p>
                <p className="text-[13px] text-ink font-mono bg-muted/30 px-2 py-1 rounded w-max">{selectedFailure.rootCauseFactor}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
