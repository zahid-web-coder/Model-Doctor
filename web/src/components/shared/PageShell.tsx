import { ReactNode } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";

interface PageShellProps {
  children: ReactNode;
}

export function PageShell({ children }: PageShellProps) {
  return (
    <main className="min-h-screen bg-background text-foreground p-6 flex gap-6 overflow-hidden max-h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col gap-6 overflow-hidden">
        <Header />
        <div className="flex-1 flex flex-col overflow-hidden bg-panel-dark/5 rounded-[12px] border border-border/10 p-6">
          {children}
        </div>
      </div>
    </main>
  );
}
