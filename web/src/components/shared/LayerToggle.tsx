"use client";

/**
 * One switch for a drawing layer.
 *
 * Shared by the failures panel and the image viewer rather than written twice:
 * both offer the same four layers over the same overlay, and two copies would
 * drift the first time one gained a state the other lacked.
 *
 * **Disabled rather than hidden when the layer has nothing to show.** A reader
 * who cannot find the "Pred outline" control does not learn that this finding
 * has no predicted outline, and on a false negative that absence is the whole
 * point.
 */
export function LayerToggle({
  label,
  on,
  onClick,
  disabled = false,
  title,
}: {
  label: string;
  on: boolean;
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={on}
      title={title ?? (disabled ? `No ${label.toLowerCase()} stored here` : undefined)}
      className={`text-[10px] px-2 py-1 rounded border transition-colors ${
        disabled
          ? "border-border/30 text-slate/50 cursor-not-allowed"
          : on
            ? "border-brass/50 bg-brass/10 text-brass"
            : "border-border/50 text-slate hover:border-brass/40"
      }`}
    >
      {label}
    </button>
  );
}
