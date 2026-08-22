# Model Doctor — Frontend Build Plan for Jawad (Antigravity Edition)
**Scope:** six screens, then the 3D hero. Built inside Google Antigravity, dispatched as agent tasks, verified against the frozen API contract.

---

## 0 · Guardrails — save these as an Antigravity Rule before you dispatch anything

Antigravity lets you attach persistent **Rules/Memory** to a project, so every agent you spawn — Manager or Editor, Autopilot or Review-driven — inherits them without you repeating yourself per task. Create this rule first, before Phase 1.

> **Rule: Frontend-only scope.**
> Every task in this project is confined to `web/`. Do not modify anything outside the two files each task is explicitly scoped to touch (the component file and its adjacent types/test file, generally).
> The following are permanently off-limits to any agent, on any task, with no exceptions:
> - `app/storage.py`
> - `app/diagnosis.py`
> - `config.py`
> - `docs/SCHEMA.md` — **read this, never edit it.** It is the contract, not a draft.
>
> If a task appears to require a backend change, stop and surface it to Zahid rather than editing backend files. That is the only approved escalation path.

Two more rules worth saving alongside it, both straight from the execution doc:
- **No invented data.** Every number on screen must trace to a documented field in `docs/SCHEMA.md` or the OpenAPI response. Nulls render as `n/a` or "not measured" — never as zero.
- **Types are generated, never hand-written.** Regenerate from `/openapi.json` after any backend-facing change; a diff in generated types means the contract moved and must be justified, not silently absorbed.

---

## 1 · Antigravity project setup

| Step | Detail |
|---|---|
| Install | Antigravity is a standalone IDE (VS Code–based). Point it at the repo root so agents can *read* `docs/SCHEMA.md` and `/openapi.json`, even though they can't touch backend files. |
| Agent mode | **Agent-assisted** (recommended) for Phase 1 (foundation) — the highest-risk phase, where a wrong type-generation setup poisons every later screen. Once Phase 1 is reviewed and merged, switch routine screen work to **Agent-driven/Autopilot** — Phases 3–8 in this plan are repetitive, well-specified CRUD-style screens where autonomy pays off. |
| Terminal policy | `Auto`, with `Agent Decides` for confirmations — lets agents run `npm`, `tsc`, `next lint` without interrupting you for routine commands. |
| Browser-in-the-loop | Antigravity's Chrome extension lets an agent run the dev server, click through the screen it just built, and check it against the acceptance criteria itself before calling the task done. Use this — don't eyeball every screen manually. |
| Artifacts | Every agent task should end with an Artifact: a screenshot of the finished screen, a short walkthrough, and a note on which endpoints it hit. That Artifact is what you hand Zahid at PR review instead of a screen-share. |
| Parallelism | Per the execution plan, Phases 3–6 (Root Causes, Clusters, Failures, Runs) depend **only on Phase 2**, not on each other. Once Phase 2 merges, use the **Agent Manager** to dispatch four agents in parallel — one per screen — each in its own workspace/branch. This is where Antigravity earns its keep over a single-threaded coding session. |

---

## 2 · Design System

Locking this down before Phase 1 means every parallel agent draws from the same tokens instead of improvising.

### 2.1 Color palette

| Token | Hex | Role |
|---|---|---|
| `--canvas` | `#F4EFE3` | Primary background — warm parchment beige, matches the approved direction |
| `--ink` | `#1E1B16` | Primary text, headlines |
| `--panel-dark` | `#15130F` | Telemetry strip, footer, the dark control-panel band under the hero, dark comparison cards |
| `--brass` | `#BB8F51` | Primary accent — CTA buttons, active tab, selected row, lift-bar fill |
| `--slate` | `#8B8272` | Secondary text, hairline borders, disabled states |
| `--evidence-pair` | `#66805A` (moss) / `#A65C48` (rust) | Semantic pair: moss = statistically real (lift ranked, p < 0.05); rust = de-emphasized / no evidence. Reuse everywhere a factor, group, or comparison needs a real-vs-noise read. |

Avoid the generic AI-design terracotta (`#D97757`) — `--brass` is deliberately more muted and gold-leaning so it doesn't read as a template default.

### 2.2 Typography

| Role | Typeface | Notes |
|---|---|---|
| Display | **Fraunces** (variable, optical size low) | The serif in the mockup headline — used sparingly, on hero copy and section titles only |
| Body / UI | **Inter** | Tables, labels, nav, buttons — this is a data tool, body text needs to disappear |
| Utility / data | **IBM Plex Mono** | Lift values, p-values, IoU, confidence, model hashes, telemetry readouts — matches the control-panel aesthetic in the hero and gives statistics a "measured" feel that a proportional font doesn't |

Load all three via `next/font`, self-hosted — no runtime Google Fonts request.

### 2.3 Layout

- Desktop-first, 12-column grid, 8px base spacing unit, usable down to tablet (per the execution doc's responsiveness rule) — no mobile-first stacking.
- Tables scroll horizontally rather than reflow.
- Hairline (1px) borders in `--slate` at 20% opacity, not drop shadows — this is a premium instrument-panel feel, not a card-heavy dashboard.
- Radius: 6px on cards/buttons, 2px on data-table cells. Small radii read as engineering tool, not consumer app.

Rough hero layout, matching the mockup:

```
┌─ nav: logo · Overview · Root Causes · Clusters · Failures · Runs · Heatmaps ─ [View Report] ─┐
│                                                                                                │
│  Find why           ┌──────────────── product render / 3D scanner ────────────────┐          │
│  your model          │                                                              │          │
│  fails.  (serif)      │                                                              │          │
│                                                                                       │
│  body copy, Inter                                                                    │
│  [Explore Insights]                                                                  │
├─ panel-dark telemetry strip: Conveyor · Scanner · Beam · Robotic status · Accuracy ─────────┤
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3 · Build order (per your instruction — screens first, hero last)

This follows the execution doc's phase table exactly, with Antigravity dispatch notes added.

| Phase | Screen | Depends on | Dispatch as |
|---|---|---|---|
| 1 | Foundation — scaffold, Tailwind tokens from §2.1, shadcn, generated types, typed client, routing shell | — | Single Agent-assisted task. Highest scrutiny — everything downstream inherits it. |
| 2 | Overview | Phase 1 | Single task, Agent-assisted still, since this is the template every other screen copies for loading/error/empty states |
| 3 | Root Causes | Phase 2 | Parallel agent #1, Autopilot |
| 4 | Clusters | Phase 2 | Parallel agent #2, Autopilot |
| 5 | Failures | Phase 2 | Parallel agent #3, Autopilot |
| 6 | Runs | Phase 2 | Parallel agent #4, Autopilot |
| 7 | Heatmaps / Outlines | Phase 5 (needs the Failures detail-view pattern for overlays) | Dispatched once Failures merges |
| 8 | **3D Hero** | Phases 1–7 complete | Its own isolated task — see §5 |

Each per-screen task brief should hand the agent: the relevant endpoints from the schema, the "Key UI" and "Data rules" cells from the execution doc, the "Do not misrepresent" cell verbatim, and the "Done when" line as its literal acceptance test — have the agent verify against that line via the browser tool before marking the task complete.

Example task brief (Root Causes, to paste into an Antigravity task):

> Build the Root Causes screen at `web/app/runs/[runId]/root-causes/`. Endpoints: `GET /runs/{id}/factor-rates`, `GET /runs/{id}/root-causes`. Render a factor table (rate in failures, rate in correct, lift, p-value, plain-language reading) ranked by lift, never by count. Render null lift as `n/a`, never as a number. Treat p ≥ 0.05 as no evidence — style that row with `--evidence-pair` rust and visually de-emphasize it. Below the table, a per-factor attribution list showing evidence strings verbatim. Use design tokens from `web/lib/tokens.ts`. Do not touch anything outside `web/`. When done, run the dev server, screenshot the rendered screen against real data from run 1, and confirm: lift-ranked ✓, nulls as n/a ✓, uninformative factors visibly marked ✓, evidence strings verbatim ✓.

---

## 4 · Quality gates (unchanged from the execution doc, still apply per PR)

`pytest` / `ruff` stay green because agents never touch backend files. Every frontend PR still needs: `npm run build`, `tsc --noEmit`, `next lint`, an OpenAPI type diff check, a responsive pass, and the manual "no fake metrics" review at each phase boundary.

---

## 5 · Phase 8 — the 3D Hero

Only starts after Phase 7 merges. If it slips, the six-screen product ships with the static hero image — that path is not a compromise, it's the designed fallback.

### 5.1 Blender — modeling for R3F, not for a render

The whole point of using Blender here instead of AI image-to-3D is independent, correctly-pivoted parts. Model discipline matters more than surface polish:

- **One object per moving part**, named exactly as R3F will reference them:
  `Conveyor`, `Gantry`, `ScannerHead`, `Beam`, `InspectionObject`. Add `HeatmapDecal` for the fade-in glow and `CameraRig` as an empty if you want a dolly path.
- **Pivot placement is the actual deliverable.** `Gantry` origin sits on the rail centerline, at the point it translates from. `ScannerHead` origin sits at its true rotation/tilt axis, not its bounding-box center. `Beam` origin sits at the emitter, scaling downward from there. Get this wrong and every GSAP tween fights the geometry instead of driving it.
- Build each part as an **empty-parented rig**: mesh parented to an empty at the correct pivot, not the mesh's own origin moved by eye. This keeps hierarchy clean for the exporter.
- `Ctrl+A` → apply all transforms on the meshes (not the empties) before final export, so scale/rotation isn't hiding inside the mesh data.
- Materials: matte brushed-aluminum body, `--brass` accent trim on structural highlights — keep it procedural (Principled BSDF, no image textures) wherever it reads fine without one. Every texture you add costs against the 1K/KTX2 budget below.
- **Do not bake any action or NLA animation into the file.** No keyframes on `Conveyor`, `Gantry`, `ScannerHead`, `Beam`, or `InspectionObject`. R3F/GSAP/`useFrame` own all motion in the browser — Blender's job ends at a correctly-pivoted static rig.

### 5.2 Export

- File → Export → glTF 2.0 (`.glb`), binary.
- **Animation: off.** No actions, no NLA tracks selected for export — if Blender's exporter shows an Animation panel, leave it unchecked.
- Apply modifiers on export, +Y up (R3F/three.js convention), export only the current view layer.
- Confirm in the export preview that all six object names survive exactly as authored — no `.001` suffixes, no Blender auto-renaming from duplicated objects.

### 5.3 Optimize the GLB (post-Blender, before it ever touches the repo)

| Step | Command / target |
|---|---|
| Compress | `npx gltf-transform optimize scene.glb scene.optimized.glb --compress meshopt` — meshopt over Draco here, since decode speed in-browser matters more than the last few KB for a hero that has to feel instant on load |
| Textures | If any exist, convert to KTX2/Basis via `gltf-transform`, cap at 1K |
| Triangle budget | 20–50k total, per the execution doc |
| Draw calls | Under 50 |
| File size | Under 200KB |
| **Verify names survived** | `gltf-transform inspect scene.optimized.glb` — confirm `Conveyor`, `Gantry`, `ScannerHead`, `Beam`, `InspectionObject` are still present as named nodes. Optimizers occasionally merge or rename nodes; if any of the five are missing, the R3F side has nothing to grab a ref to and the whole animation layer silently breaks. |

### 5.4 R3F integration

```
web/components/hero/
  ScannerScene.tsx      — canvas, camera, lighting, the loaded GLB
  useSceneAnimation.ts  — GSAP ScrollTrigger timeline + useFrame idle motion
  StaticHeroFallback.tsx — the WebGL/reduced-motion/no-JS fallback image
```

- `useGLTF('/models/scanner.glb')`, then pull `nodes.Conveyor`, `nodes.Gantry`, `nodes.ScannerHead`, `nodes.Beam`, `nodes.InspectionObject` by exact name and attach a `ref` to each.
- `useFrame` drives continuous, non-scroll-linked motion: conveyor belt texture-offset scroll, beam emissive-intensity pulse, idle drift. This runs regardless of scroll position — it's the "machine is alive" layer.
- GSAP `ScrollTrigger` drives the scroll-linked story: `Gantry` travel along the rail, `ScannerHead` tilt as it passes over `InspectionObject`, `InspectionObject` entering/exiting the frame, `HeatmapDecal` opacity fading in as the beam crosses it, `CameraRig` dolly. Build this as one timeline scrubbed by scroll progress, not a pile of independent triggers.
- Framer Motion stays strictly on the 2D UI layer around the canvas (buttons, headline entrance) — never on the 3D objects, per the execution doc's own call.
- Pointer parallax: a small lerp toward pointer position inside `useFrame`, capped in range, skipped entirely under reduced motion.

### 5.5 Fallbacks and performance — non-negotiable, matches the execution doc verbatim

- `next/dynamic(..., { ssr: false })` with a `Suspense` boundary — `StaticHeroFallback` is both the loading state and the permanent no-WebGL fallback, so it's never extra work.
- `IntersectionObserver` pauses `useFrame`/`ScrollTrigger` updates the moment the hero leaves the viewport — this is the most-missed optimization and the one most likely to get flagged in QA.
- `prefers-reduced-motion` freezes to one clean static frame (beam mid-pulse, object centered under the head) rather than disabling the scene.
- `dpr={[1, 2]}`, `frameloop="demand"` when nothing is actively scrolling or animating.

### 5.6 Dispatching the hero task

Isolate it exactly as the execution doc's branch strategy specifies: `web/components/hero/` only, dynamically imported into exactly one route, so it can be developed and merged with almost no conflict surface against the six screens. As an Antigravity task brief:

> Build the scanner hero in `web/components/hero/` only. Load `/models/scanner.glb`, drive `Conveyor`, `Gantry`, `ScannerHead`, `Beam`, `InspectionObject` via `useFrame` (idle motion) and GSAP `ScrollTrigger` (scroll-linked timeline) exactly as described in §5.4. Do not touch any screen file. Build `StaticHeroFallback.tsx` first — it is both the Suspense fallback and the permanent degrade path. Verify offscreen pause with the browser tool by scrolling the hero out of view and confirming the frame loop stops (check DevTools performance, not just visually). Screenshot both the animated state and the reduced-motion frozen frame as your Artifact.

---

## 6 · Definition of done, restated for this plan

Screens: all six built, reading only documented endpoints, loading/error/empty states present, verified against the "Done when" line from the execution doc via Antigravity's browser tool, no backend file touched. Hero: either shipped within the §5.5 performance budget with working fallbacks, or deliberately deferred with `StaticHeroFallback` live — both are acceptable, per the original doc's own asymmetry rule. Either way, `docs/SCHEMA.md` is unchanged, and `git log` shows zero commits touching `app/storage.py`, `app/diagnosis.py`, or `config.py`.
