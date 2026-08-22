"use client";

import dynamic from "next/dynamic";

/**
 * The boundary that keeps three out of the server bundle.
 *
 * `next/dynamic` with `ssr: false` has to be called from a client component in
 * the App Router — doing it from a server component is a build error, and the
 * error does not say so plainly. This file exists for that one reason, so the
 * page underneath can stay a server component.
 */
const HeroModel = dynamic(() => import("./HeroModel"), {
  ssr: false,
  loading: () => <div className="w-full h-full bg-[#d8d2c6]" />,
});

export function HeroCanvas({ className }: { className?: string }) {
  return (
    <div className={className}>
      <HeroModel />
    </div>
  );
}
