import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { HeroVideo } from "@/components/dashboard/HeroVideo";
import { RightPanels } from "@/components/dashboard/RightPanels";
import { BottomPanels } from "@/components/dashboard/BottomPanels";
import { TelemetryStrip } from "@/components/dashboard/TelemetryStrip";

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground p-6 flex gap-6">
      <div className="sticky top-6 h-[calc(100vh-48px)]">
        <Sidebar />
      </div>
      <div className="flex-1 flex flex-col gap-6">
        <Header />
        
        <div className="flex gap-6">
          <div className="flex-1 flex flex-col relative rounded-[12px] overflow-hidden bg-panel-dark/5 border border-border/10 min-h-[650px]">
            {/* Center Area: Hero Text and Video */}
            <div className="absolute inset-0 z-0">
               <HeroVideo />
            </div>
            
            <div className="relative z-10 flex flex-col h-full pointer-events-none p-10">
              <h1 className="text-[52px] font-heading text-ink leading-[1.05] max-w-[400px] tracking-tight">
                Diagnose.<br/>
                Understand.<br/>
                <span className="text-slate/80">Improve.</span>
              </h1>
              <p className="mt-6 text-slate max-w-[280px] text-[15px] leading-relaxed">
                Automatically detect failures, analyze root causes and improve model performance with confidence.
              </p>
              
              <div className="mt-10 flex flex-col gap-3 pointer-events-auto">
                <button className="bg-[#B98952] text-white px-5 py-2.5 rounded-[6px] w-max font-medium text-[13px] hover:brightness-110 transition-all shadow-md shadow-black/10 flex items-center gap-2">
                  Explore Failures <span className="font-sans">→</span>
                </button>
                <button className="bg-[#EAE4D6]/50 backdrop-blur-sm text-ink px-5 py-2.5 rounded-[6px] w-max font-medium text-[13px] flex items-center gap-2 hover:bg-[#EAE4D6]/70 transition-all border border-border/20">
                  View Live Run 
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-slate/70"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                </button>
              </div>
            </div>
          </div>
          
          <RightPanels />
        </div>
        
        <BottomPanels />
      </div>
    </main>
  );
}
