import Link from "next/link";
import { Lock } from "lucide-react";

/**
 * What a screen shows on the public deployment when the service behind it is
 * not there.
 *
 * The nav still offers these screens, deliberately — a rail that hides half
 * the product understates it, and the reason they are unavailable is worth
 * reading. So the destination explains itself rather than 404ing.
 *
 * **It claims nothing that is not true.** There is no authentication in this
 * project: no accounts, no sign-in, no permissions. A gate reading "not
 * authorised" would imply something to authorise against, and the first person
 * to look would find it missing. The honest reason — that a service which
 * executes uploaded checkpoints is kept off the public internet — is also the
 * better one.
 */
export function ReadOnlyNotice({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex-1 grid place-items-center">
      <div className="text-center max-w-[440px]">
        <div className="w-10 h-10 rounded-full bg-black/5 grid place-items-center mx-auto mb-4">
          <Lock size={17} className="text-slate" strokeWidth={2} />
        </div>
        <h4 className="text-[15px] font-heading text-ink mb-2.5">{title}</h4>
        <div className="text-[12.5px] text-slate leading-relaxed space-y-2.5">
          {children}
        </div>
        <Link
          href="/dashboard"
          className="inline-block mt-6 text-[13px] font-medium text-ink border border-border/60 rounded-lg px-4 py-2 hover:bg-black/5 transition-colors"
        >
          Back to Dashboard
        </Link>
      </div>
    </div>
  );
}
