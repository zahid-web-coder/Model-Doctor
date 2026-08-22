/**
 * The Model Doctor mark.
 *
 * Two ideas in one shape. The four corner brackets are a detection box — the
 * thing every vision model draws, and the thing this product interrogates. The
 * trace running through them is a diagnostic pulse: the box says *what was
 * found*, the pulse says *why it failed*, which is the whole difference between
 * this and a metrics dashboard.
 *
 * **Drawn, not an image file.** It has to sit on warm beige in the dashboard
 * and on dark concrete on the landing page, at 20px in a sidebar and at 180px
 * on a title card. An SVG does all of that from one definition, inherits the
 * surrounding colour, and costs no request.
 *
 * The brackets are deliberately open at the corners rather than a closed
 * square: a complete outline reads as a generic app icon, and the gaps are
 * what make it read as a reticle at small sizes.
 */
export function Logo({
  size = 32,
  className,
  accent = "#C9A227",
  title = "Model Doctor",
}: {
  size?: number;
  className?: string;
  /** The brackets. The pulse always takes the surrounding text colour. */
  accent?: string;
  /** Set to null for a purely decorative instance beside a visible wordmark. */
  title?: string | null;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title ?? undefined}
    >
      {title && <title>{title}</title>}

      {/* Detection box, corners only. */}
      <g
        stroke={accent}
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M3.6 10.4V5.6a2 2 0 0 1 2-2h4.8" />
        <path d="M21.6 3.6h4.8a2 2 0 0 1 2 2v4.8" />
        <path d="M28.4 21.6v4.8a2 2 0 0 1-2 2h-4.8" />
        <path d="M10.4 28.4H5.6a2 2 0 0 1-2-2v-4.8" />
      </g>

      {/* Diagnostic trace. Flat, then one asymmetric excursion — a symmetric
          spike reads as a mountain, and it is the offset recovery that reads
          as a reading taken off an instrument. */}
      <path
        d="M7 16h3.4l2.3-5.2L17 21.6l2-5.6h6"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * Mark plus wordmark, as used in the sidebar and the landing header.
 *
 * The two-line setting is deliberate: "MODEL DOCTOR" on one line at sidebar
 * width forces the type down to a size where the mark starts to dominate it.
 */
export function Wordmark({
  size = 40,
  tone = "dark",
}: {
  size?: number;
  /** `dark` for warm light backgrounds, `light` for the concrete scene. */
  tone?: "dark" | "light";
}) {
  const ink = tone === "light" ? "text-white" : "text-ink/90";
  const sub = tone === "light" ? "text-white/50" : "text-slate";
  return (
    <div className="flex items-center gap-3">
      <Logo size={size} className={ink} title={null} />
      <div>
        <h2 className={`text-[15px] font-bold tracking-wide ${ink} leading-tight`}>
          MODEL<br />DOCTOR
        </h2>
        <p className={`text-[9px] font-semibold tracking-wider ${sub} uppercase mt-[2px]`}>
          AI Model Diagnostics
        </p>
      </div>
    </div>
  );
}
