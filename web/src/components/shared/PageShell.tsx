import { ReactNode } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { RunProvider } from "@/lib/run-context";
import { api } from "@/lib/api/client";
import { ApiDown } from "@/components/shared/ApiDown";

/**
 * The frame every screen sits in.
 *
 * The run list is fetched here, once, on the server, and handed to the
 * provider — so the sidebar, the header selectors and the page body all read
 * the same runs without three separate requests.
 */
export async function PageShell({ children }: { children: ReactNode }) {
  let runs = [];
  try {
    runs = await api.runs();
  } catch (error) {
    return <ApiDown error={error} />;
  }

  return (
    <RunProvider runs={runs}>
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
