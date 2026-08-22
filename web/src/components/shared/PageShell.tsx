import { ReactNode } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { RunProvider } from "@/lib/run-context";

/**
 * The frame every screen sits in. RunProvider wraps the whole shell so the
 * sidebar, the header selectors and the page body all read the same selected
 * run — previously each of them held its own hardcoded copy.
 */
export function PageShell({ children }: { children: ReactNode }) {
  return (
    <RunProvider>
      <main className="min-h-screen bg-background text-foreground p-6 flex gap-6 overflow-hidden max-h-screen">
        <Sidebar />
        <div className="flex-1 flex flex-col gap-6 overflow-hidden">
          <Header />
          <div className="flex-1 flex flex-col overflow-hidden bg-panel-dark/5 rounded-[12px] border border-border/10 p-6">
            {children}
          </div>
        </div>
      </main>
    </RunProvider>
  );
}
