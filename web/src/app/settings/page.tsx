import { PageShell } from "@/components/shared/PageShell";
import { Settings as SettingsIcon } from "lucide-react";
import { ReadOnlyNotice } from "@/components/shared/ReadOnlyNotice";
import { READ_ONLY } from "@/lib/deployment";

/**
 * Settings.
 *
 * A real destination for the nav item. The values that would live here —
 * confidence threshold, match IoU, localisation floor, image size — are
 * recorded per run by the analysis pass and are not editable from a read-only
 * API, so this stays a stub until there is something to change.
 *
 * On the public deployment it says so in those terms rather than in a
 * developer's, since the reader there cannot run the pass that would change
 * them.
 */
export default function SettingsPage() {
  return (
    <PageShell>
      <div className="flex flex-col h-full">
        <h3 className="text-[17px] font-heading text-ink">Settings</h3>
        <p className="text-[13px] text-slate mb-6">Analysis configuration.</p>
        {READ_ONLY ? (
          <ReadOnlyNotice title="Settings aren&rsquo;t available in the public demo">
            <p>The public deployment is a read-only Model Doctor demo.</p>
          </ReadOnlyNotice>
        ) : (
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
        )}
      </div>
    </PageShell>
  );
}
