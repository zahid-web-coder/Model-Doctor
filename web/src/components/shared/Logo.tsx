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
 * ## Why the geometry is what it is
 *
 * Everything is on a 24-unit grid at whole-unit coordinates, and every stroke
 * is the same width. The first version mixed 2.4 and 2.2 weights on values
 * like 21.6 and 28.4, which looks fine at 180px and turns to mush at 20px:
 * off-grid coordinates put stroke edges midway across a device pixel, and the
 * renderer resolves that with a grey half-covered row on one side and not the
 * other. The mark reads as slightly out of focus and slightly lopsided without
 * it being obvious why.
 *
 * The width is even (2) and the coordinates are integers on purpose — an even
 * stroke centred on an integer covers whole pixels either side, so the edges
 * land on pixel boundaries at 24px and stay clean at every doubling of it.
 *
 * The brackets are open at the corners rather than a closed square: a complete
 * outline reads as a generic app icon, and the gaps are what make it read as a
 * reticle. Their arms are all 6 units long and their radii all 2, so the four
 * corners are the same shape rotated rather than four similar shapes.
 *
 * The pulse is symmetric about the centre — it spans x 6 to 18, sits on y 12,
 * and its peak and trough are the same 4 units off the line. It reads as an
 * instrument trace rather than as a scribble because it returns to the
 * baseline at both ends.
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
      viewBox="0 0 24 24"
      fill="none"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title ?? undefined}
    >
      {title && <title>{title}</title>}

      {/* Detection box, corners only. Four identical L's: a 4-unit arm, a
          2-unit radius, a 4-unit arm. */}
      <g stroke={accent}>
        <path d="M3 9V5a2 2 0 0 1 2-2h4" />
        <path d="M15 3h4a2 2 0 0 1 2 2v4" />
        <path d="M21 15v4a2 2 0 0 1-2 2h-4" />
        <path d="M9 21H5a2 2 0 0 1-2-2v-4" />
      </g>

      {/* Diagnostic trace. Point-symmetric about the centre: equal 2-unit
          flats at both ends, equal 2×4 rises either side of a 4×8 fall, peak
          and trough the same distance off the baseline and equidistant from
          the middle. */}
      <path d="M6 12h2l2-4 4 8 2-4h2" stroke="currentColor" />
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
