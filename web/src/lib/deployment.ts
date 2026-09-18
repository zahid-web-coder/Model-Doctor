/**
 * Whether this build is the public, read-only deployment.
 *
 * The public site serves the read API only. The control service — which
 * uploads checkpoints and starts training — is never deployed with it, so the
 * screens that drive it have nothing to talk to and must not be offered. The
 * same distinction decides how a failed fetch is explained: a developer wants
 * the status line and the command, a visitor wants neither.
 *
 * Defaulting off `NODE_ENV` rather than a required variable is deliberate.
 * `next dev` keeps the developer affordances with no setup, a deployed build
 * drops them with no setup, and neither depends on someone remembering to set
 * a flag in a dashboard — the failure mode of a required variable is a public
 * site still showing a `uvicorn` command.
 *
 * `NEXT_PUBLIC_READ_ONLY` overrides in both directions for the cases the
 * default gets wrong: `1` to preview the public build locally, `0` to keep the
 * control screens in a production build running against a local control API.
 */
export const READ_ONLY =
  process.env.NEXT_PUBLIC_READ_ONLY === "1" ||
  (process.env.NEXT_PUBLIC_READ_ONLY !== "0" &&
    process.env.NODE_ENV === "production");
