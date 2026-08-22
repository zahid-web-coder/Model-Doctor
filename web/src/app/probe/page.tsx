import { HeroCanvas } from "@/components/landing/HeroCanvas";

/**
 * Phase 0 probe — throwaway.
 *
 * Proves four things the sandbox could not, because the sandbox is Vite and
 * this is Next with SSR:
 *
 *   1. The Draco-compressed GLB decodes with a self-hosted decoder.
 *   2. `ssr: false` keeps three out of the server render.
 *   3. three does not land in a server chunk.
 *   4. The named nodes survive Next's bundler, so the runtime can still drive
 *      them by name.
 *
 * Deleted once the real landing route exists in phase 1.
 */
export default function ProbePage() {
  return (
    <main className="h-screen w-full">
      <HeroCanvas className="h-full w-full" />
    </main>
  );
}
