"use client";

import { useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X, ChevronLeft, ChevronRight } from "lucide-react";

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
 *
 * **Stepping between neighbours lives here, not in each caller.** Every screen
 * that opens this opens it from a list — a page of findings, a filtered table
 * of images, a cluster's members — and closing the panel to pick the next row
 * loses the layer switches and the reader's place. Three copies of the same
 * arrows would also mean three chances for the keys to behave differently.
 * Callers supply `onPrev`/`onNext` and keep ownership of what "next" means;
 * this only draws the controls and binds the keys.
 */
export function Lightbox({
  open, onClose, title, subtitle, children, footer, onPrev, onNext, position,
}: {
  open: boolean;
  onClose: () => void;
  /** Shown top-left. Keep it short — this is a label, not a description. */
  title: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  /**
   * Step to the previous or next item. Omit either to leave that direction
   * unavailable — the control is then disabled rather than absent, so the
   * reader can see they are at the end of the list instead of wondering
   * whether stepping exists at all.
   */
  onPrev?: () => void;
  onNext?: () => void;
  /** Where the reader is, e.g. "3 of 25". Shown between the arrows. */
  position?: string;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      // Arrows belong to whatever the reader is typing in. A filter box or a
      // threshold field inside the panel must still be able to move its own
      // caret, so the keys only step the list when nothing is taking text.
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        target?.isContentEditable
      ) {
        return;
      }
      if (event.key === "ArrowLeft" && onPrev) {
        event.preventDefault();
        onPrev();
      } else if (event.key === "ArrowRight" && onNext) {
        event.preventDefault();
        onNext();
      }
    };
    window.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [open, onClose, onPrev, onNext]);

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
      <div className="relative bg-card border border-border/60 rounded-xl shadow-2xl max-w-[min(1100px,95vw)] max-h-[92vh] w-full flex flex-col overflow-hidden">
        <div className="flex items-start justify-between gap-4 px-5 py-3.5 border-b border-border/40 shrink-0">
          <div className="min-w-0">
            <div className="text-[14px] text-ink font-medium truncate">{title}</div>
            {subtitle && <div className="text-[12px] text-slate mt-0.5 truncate">{subtitle}</div>}
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            {(onPrev || onNext) && (
              <>
                <Step
                  onClick={onPrev}
                  label="Previous"
                  icon={<ChevronLeft size={15} />}
                />
                {position && (
                  <span className="text-[11px] text-slate tabular-nums px-0.5 whitespace-nowrap">
                    {position}
                  </span>
                )}
                <Step
                  onClick={onNext}
                  label="Next"
                  icon={<ChevronRight size={15} />}
                />
                <span className="w-px h-5 bg-border/60 mx-0.5" />
              </>
            )}
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="w-7 h-7 grid place-items-center rounded border border-border/50 text-slate hover:bg-black/5 hover:text-ink transition-colors"
            >
              <X size={15} />
            </button>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-auto custom-scrollbar p-4 sm:px-16 grid place-items-center bg-black/[0.03]">
          {children}
        </div>

        {/* **The arrows sit on the picture, not only in the corner.** A 28px
            control tucked beside the close button is easy to miss and easy to
            mistake for decoration, and a reader who clicks the greyed-out one
            at the start of a list learns nothing about whether stepping works
            at all. These are placed where the eye already is, sized to be
            unmistakable, and the panel gains side padding so they never cover
            the drawing they are next to. */}
        {(onPrev || onNext) && (
          <>
            <Edge side="left" onClick={onPrev} label="Previous">
              <ChevronLeft size={20} />
            </Edge>
            <Edge side="right" onClick={onNext} label="Next">
              <ChevronRight size={20} />
            </Edge>
          </>
        )}

        {footer && (
          <div className="px-5 py-3 border-t border-border/40 shrink-0">{footer}</div>
        )}
      </div>
    </div>,
    document.body
  );
}

/**
 * A large stepping arrow on the edge of the panel.
 *
 * Hidden at the ends rather than merely disabled, unlike its counterpart in the
 * header: a dimmed disc floating over a photograph reads as part of the image,
 * and the header control is already there to say the direction exists and is
 * exhausted.
 */
function Edge({
  side,
  onClick,
  label,
  children,
}: {
  side: "left" | "right";
  onClick?: () => void;
  label: string;
  children: ReactNode;
}) {
  if (!onClick) return null;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={`${label} (arrow key)`}
      className={`absolute top-1/2 -translate-y-1/2 ${
        side === "left" ? "left-2" : "right-2"
      } z-10 w-10 h-10 grid place-items-center rounded-full bg-card/95 border border-border/60 text-ink shadow-lg hover:bg-card hover:border-brass/60 hover:text-brass active:scale-95 transition-all`}
    >
      {children}
    </button>
  );
}

/** One stepping arrow, disabled at the ends of the list rather than removed. */
function Step({
  onClick,
  label,
  icon,
}: {
  onClick?: () => void;
  label: string;
  icon: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={!onClick}
      aria-label={label}
      title={onClick ? `${label} (arrow key)` : `No ${label.toLowerCase()} item`}
      className={`w-7 h-7 grid place-items-center rounded border transition-colors ${
        onClick
          ? "border-border/50 text-slate hover:bg-black/5 hover:text-ink"
          : "border-border/30 text-slate/40 cursor-not-allowed"
      }`}
    >
      {icon}
    </button>
  );
}
