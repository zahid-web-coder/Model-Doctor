import { PageShell } from "@/components/shared/PageShell";
import { FileText } from "lucide-react";

/**
 * Reports.
 *
 * A real destination so the nav item is not a dead link, and deliberately
 * empty rather than filled with a mock report: there is no report generation
 * in the backend, and a screen showing invented documents is worse than one
 * that says so.
 */
export default function ReportsPage() {
  return (
    <PageShell>
      <div className="flex flex-col h-full">
        <h3 className="text-[17px] font-heading text-ink">Reports</h3>
        <p className="text-[13px] text-slate mb-6">Export and share diagnosis results.</p>
        <div className="flex-1 grid place-items-center">
          <div className="text-center max-w-[380px]">
            <FileText size={28} className="text-slate/50 mx-auto mb-3" />
            <p className="text-[14px] text-ink mb-1.5">Not implemented yet</p>
            <p className="text-[12px] text-slate leading-relaxed">
              The backend has no report generation, so there is nothing to list here.
              Left as a stub rather than filled with placeholder documents.
            </p>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
