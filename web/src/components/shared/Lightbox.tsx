"use client";

import { useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

/**
 * A modal overlay for viewing one thing at full size.
 *
 * **Rendered through a portal, not in place.** The grids and comparison panes
 * that open this live inside scroll containers with their own stacking
 * contexts; a fixed-position child of one of those is clipped by it. Attaching
 * to `document.body` is what makes "fills the viewport" mean the viewport.
 *
 * Closing is deliberately over-served — Escape, the backdrop, and an explicit
 * button — because a modal that traps a reader who reached it by a stray click
 * is worse than no modal. The backdrop handler checks the event target so a
 * drag that ends outside the panel does not count as a click on it.
 *
 * Body scroll is locked while open: the page behind scrolling under a fixed
 * overlay is the thing that makes a lightbox feel broken.
 */
export function Lightbox({
  open, onClose, title, subtitle, children, footer,
}: {
  open: boolean;
  onClose: () => void;
  /** Shown top-left. Keep it short — this is a label, not a description. */
  title: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [open, onClose]);

  // `document` does not exist while the server renders, and the portal target
  // has to be a real node. Bailing out here rather than guarding at every call
  // site keeps the check in one place.
  if (!open || typeof document === "undefined") return null;

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-8 bg-black/70 backdrop-blur-sm"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
    >
      <div className="bg-card border border-border/60 rounded-xl shadow-2xl max-w-[min(1100px,95vw)] max-h-[92vh] w-full flex flex-col overflow-hidden">
        <div className="flex items-start justify-between gap-4 px-5 py-3.5 border-b border-border/40 shrink-0">
          <div className="min-w-0">
            <div className="text-[14px] text-ink font-medium truncate">{title}</div>
            {subtitle && <div className="text-[12px] text-slate mt-0.5 truncate">{subtitle}</div>}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="shrink-0 w-7 h-7 grid place-items-center rounded border border-border/50 text-slate hover:bg-black/5 hover:text-ink transition-colors"
          >
            <X size={15} />
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-auto custom-scrollbar p-4 grid place-items-center bg-black/[0.03]">
          {children}
        </div>

        {footer && (
          <div className="px-5 py-3 border-t border-border/40 shrink-0">{footer}</div>
        )}
      </div>
    </div>,
    document.body
  );
}
