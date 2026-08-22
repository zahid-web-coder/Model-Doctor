import Link from "next/link";

/**
 * Placeholder. The landing page (phase 2 onward) replaces this file entirely.
 *
 * It exists so the site root is not a 404 between phases, and it is
 * deliberately plain — no 3D, no scroll story, no layout worth keeping. The
 * dashboard lives at `/dashboard` from this commit onward.
 */
export default function RootPlaceholder() {
  return (
    <main className="min-h-screen grid place-items-center p-8">
      <div className="text-center">
        <h1 className="font-heading text-ink text-[28px] mb-2">Model Doctor</h1>
        <p className="text-[13px] text-slate mb-6">
          The landing page is not built yet.
        </p>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-brass/90 text-white text-[13px] font-medium hover:bg-brass transition-colors"
        >
          Open the dashboard →
        </Link>
      </div>
    </main>
  );
}
