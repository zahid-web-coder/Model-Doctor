import { PageShell } from "@/components/shared/PageShell";
import { Settings as SettingsIcon } from "lucide-react";

/**
 * Settings.
 *
 * A real destination for the nav item. The values that would live here —
 * confidence threshold, match IoU, localisation floor, image size — are
 * recorded per run by the analysis pass and are not editable from a read-only
 * API, so this stays a stub until there is something to change.
 */
export default function SettingsPage() {
  return (
    <PageShell>
      <div className="flex flex-col h-full">
        <h3 className="text-[17px] font-heading text-ink">Settings</h3>
        <p className="text-[13px] text-slate mb-6">Analysis configuration.</p>
        <div className="flex-1 grid place-items-center">
          <div className="text-center max-w-[380px]">
            <SettingsIcon size={28} className="text-slate/50 mx-auto mb-3" />
            <p className="text-[14px] text-ink mb-1.5">Nothing editable yet</p>
            <p className="text-[12px] text-slate leading-relaxed">
              Thresholds are recorded per run by the analysis pass. The API is read-only,
              so there is nothing here to change from the browser.
            </p>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
