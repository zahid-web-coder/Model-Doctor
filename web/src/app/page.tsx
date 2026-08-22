import type { Metadata } from "next";
import { Landing } from "@/components/landing/Landing";

export const metadata: Metadata = {
  title: "Model Doctor — AI that explains vision model failures",
  description:
    "Find why a vision model fails, not just how often. Every claim carries its evidence: lift, significance, and whether it replicates on a second run.",
};

/**
 * The landing page.
 *
 * A server component with one client child, so nothing about the story — three,
 * drei, gsap — reaches the server bundle. The dashboard lives at `/dashboard`.
 */
export default function LandingPage() {
  return <Landing />;
}
