# Landing Page — Architecture and Plan

**Status: shipped.** All phases are on `main`. The landing page is a
**separate experience** from the dashboard; nothing in it modifies dashboard UI
or API integration.

This note is kept as the record of the plan *and* of where the plan was wrong.
Two sections are marked **superseded** — the second GLB (§4) and the route
groups (§3). Both were planned, both turned out to be unnecessary, and both
are struck through rather than deleted so the reasoning survives. Read those
before treating any other section as instruction: what shipped is the
authority, not what was intended.

Companion to [`HERO_3D_DESIGN.md`](HERO_3D_DESIGN.md), which governs the 3D
asset itself. Where the two disagree, the hero note wins on asset questions and
this note wins on page questions.

---

## 1. What the page is for

The dashboard is where the product is used. The landing page is where it is
*understood*. One job: make a visitor grasp what Model Doctor does before they
click into a screen full of tables.

Feeling: Apple product reveal, industrial design, premium manufacturing.
**Not** dark cyberpunk, not neon, not futuristic-black UI. The palette is the
one the 3D world already uses — warm beige concrete, off-white housing, dark
metal conveyor, one blue inspection beam, restrained heatmap colour.

---

## 2. The story, and the beat that was missing

The original flow was Factory → Inspection → Verification → Dashboard. That is
a *machine* story; any industrial vision vendor could run it. The headline
promises explanation, but detection is what those beats actually show.

**The product's real idea is that the obvious cause is usually wrong.** From
run 3, measured, not illustrative:

| Factor | On failures | On correct | Lift | p |
| --- | --- | --- | --- | --- |
| `edge_truncation` | 148/201 (73.6%) | 209/295 (70.8%) | **1.04** | 0.54 |
| `small_object` | 87/201 (43.3%) | 37/295 (12.5%) | **3.45** | 1.5e-14 |
| `thin_structure` | 76/201 (37.8%) | 48/295 (16.3%) | **2.32** | 9.2e-08 |

The most common factor on failures explains nothing — it is nearly as common on
the plates that passed. The real cause is three times rarer and three times more
predictive, and it **replicates**: run 2 finds `small_object` at lift 2.77,
p=3.4e-06, independently.

So the story gains a beat and loses a lamp:

| Scene | Beat | Why |
| --- | --- | --- |
| 1 | **Hero** — cinematic factory reveal | Establish the world |
| 2 | **Inspection** — plate scans, heatmap resolves | What the model saw |
| 3 | **Diagnosis** — the obvious cause is struck out | **The differentiator** |
| 4 | **Verification** — the claim replicates on a second batch | Why it is trustworthy |
| 5 | **Transition** — camera into the machine display → `/dashboard` | Hand-off |

**Scene 3 in detail.** Two candidate causes surface over the scanned plate.
`edge_truncation` is large and lit — the obvious suspect at 73.6%. Then the
counter-evidence lands: *70.8% of the passing plates have it too*. It dims and
strikes out. `small_object` stays lit. Nothing else on the page has to argue
that Model Doctor is different; this beat does it with arithmetic.

**Why the pass/fail lamp is gone.** A green/red verification light says *we
grade parts*. Model Doctor diagnoses *models*. The honest meaning of
verification here is replication across splits, which is both what the data
supports and a stronger claim. Red/green as a sole status channel is also
inaccessible to colour-blind readers; any status cue pairs colour with shape or
text.

---

## 3. Route architecture

The root layout on `main` carries only fonts and `<body>` — dashboard chrome is
applied per page through `PageShell`. A landing page at `/` therefore inherits
nothing and needs no layout surgery.

```
app/
  page.tsx                  landing
  dashboard/page.tsx        overview, moved from /
  runs/...                  unchanged
  reports/, settings/       unchanged
```

**Superseded: route groups were planned and are not used.** The plan put the
landing page in `(marketing)/` and the dashboard in `(app)/`, justified as a
"bundle boundary". That justification was wrong — route groups in the App
Router are purely organisational, affecting shared layouts and nothing else.
Chunking is per route and derives from imports.

The boundary that actually matters is already there without them: the landing
page never imports dashboard code, the dashboard never imports three, and
`next/dynamic` with `ssr: false` keeps three out of the server bundle. Verified
on the shipped build — three sits in one lazy 1,041 KB chunk that the dashboard
entry does not reference, and no server chunk mentions it at all.

So the flat layout above is what shipped, and adding groups would have been
churn bought for a benefit that does not exist.

**Migration cost is two links.** `Sidebar.tsx` points Overview at `/`, and
`HeroSlot` has a hardcoded `/runs/1287/failures`. Nothing else in the app links
to `/`. The hardcoded run id is a pre-existing bug and is fixed as part of the
move, not left in a new location.

**No auto-redirect for returning visitors.** It breaks the shared-link case,
which is the only reason a landing page exists. `Launch Dashboard` is sticky
from scroll 0 instead.

---

## 4. Asset strategy

The arm rig is validated at 30/30 kinematic assertions, 19,568 triangles,
169.3 KB. **`md_blockout.py`'s arm chain is frozen.** Re-verifying it costs more
than the landing page gains.

The plan was a second GLB:

| File | Contents | Loaded by | Budget |
| --- | --- | --- | --- |
| `hero.glb` | arm, conveyor, rollers — **unchanged** | dashboard + landing | 169 KB, frozen |
| ~~`factory.glb`~~ | ~~verification chamber, hall shell, conveyor extension~~ | ~~landing only~~ | ~~< 250 KB~~ |

**Superseded during implementation: `factory.glb` was never built, and should
not be.** Every object it was going to hold — the verification station, the
infeed and outfeed cabinets, the hall — turned out to be cabinets and frames,
which is boxes and cylinders. R3F builds those for **0 KB of download and no
export step**, against a budgeted 250 KB plus a second asset to keep in step
with the Blender source forever.

The rule that produced the wrong answer here was "new geometry means a new
asset". The better rule is: **model in Blender only what cannot be described
procedurally.** An articulated arm with hand-placed pivots qualifies. A steel
frame does not.

Hall textures follow the same logic — drawn on a canvas at runtime, because a
1K concrete map is larger than the entire compressed machine.

So the shipped asset strategy is one GLB, unchanged, shared by both routes via
drei's loader cache.

**The GLB contains no animation clips.** Plate motion is runtime state
(`objectProgress`, hero note §3). Baking it into Blender would give the same
value two owners, and they would drift.

---

## 5. Ownership — extending the one-owner rule

The hero note's §3 rule holds and extends: **`useFrame` owns what loops, GSAP
owns what responds to scroll, never both.**

| Parameter | Owner |
| --- | --- |
| Roller spin, plate advance, beam flicker | `useFrame` |
| `scrollT` (0..1 across the whole page) | **GSAP ScrollTrigger** |
| Camera position, beam intensity, overlay opacity | derived from `scrollT` |
| Scene 3 candidate opacity | derived from `scrollT` |

GSAP writes exactly one number. Everything cinematic is a pure function of it.
That keeps scrubbing correct in both directions and makes the reduced-motion
path a matter of pinning `scrollT` to fixed values.

---

## 6. Transition into the dashboard

The highest-risk beat. **Do not navigate while the camera is animating.**

1. Camera pushes toward the machine display.
2. Animation completes and **freezes** on a matched frame.
3. `/dashboard` has been prefetched.
4. Route change happens against a static image.

The dashboard's first paint must match the frozen frame's background colour.
Unmounting a canvas mid-animation is what produces the white flash that gives
this effect away.

**What shipped is a cover-fade, not a match cut.** The sequence above holds —
prefetch, cover, then navigate — but step 2 covers the frame with a solid panel
in the dashboard's own background colour rather than freezing on a frame that
visually continues into the dashboard. It is flash-free and honest; it is not
the camera-flies-into-the-screen effect this section implies. Upgrading it is a
separate piece of work and would need the dashboard's first paint designed
against the final frame.

---

## 7. Performance budget

| Check | Target | Shipped |
| --- | --- | --- |
| GLB total, compressed | < 450 KB | **174 KB** — one file, not two |
| three in a dashboard chunk | absent | **absent** — 0 references in the entry chunk |
| three in the server bundle | absent | **absent** — 0 server chunks |
| Scene triangles | < 60k | **not measured** |
| Draw calls | < 80 | **not measured** |
| Frame rate | 60fps sustained | **not measured** |
| `useFrame` paused offscreen | required | implemented, **not verified** |

**Three of these were never measured, and the earlier "measured, not assumed"
claim in this section was wrong.** The review environment reports
`document.visibilityState: "hidden"` and suspends `requestAnimationFrame`: an
instrumented overlay read 0 FPS and 0 draw calls over 2.5 seconds while the
scene was visibly rendering in screenshots. Nothing about the scene — it is the
harness. The bundle figures above are real, because they come from the built
output rather than from the browser.

So the frame-rate and draw-call gates remain **open**, not passed. They need a
local run with a real frame monitor. The two likeliest costs are the
`MeshReflectorMaterial` floor and the 2048 shadow map.

The hero note's warning still applies and is still the one that gets missed:
the offscreen pause is verified by watching CPU, not by reading the code — and
it has not been.

**Mobile does not get the 3D story.** Four scroll-linked scenes on a mid-range
phone is a jank and battery problem. Mobile gets the poster plus the same five
beats as scroll-revealed stills and copy — which is also the `prefers-reduced-motion`
path and the no-WebGL path, so it is built once and required three times.

Reduced motion shows **all** the copy at once. It is not a degraded experience
that hides content.

---

## 8. Phases

| Phase | Deliverable | Gate |
| --- | --- | --- |
| **0** | R3F + Draco GLB renders inside Next, `ssr:false` | three absent from server bundle |
| 1 | Route split, landing shell, poster | dashboard unaffected, links fixed |
| 2 | Scene 1 hero, existing GLB, camera only | 60fps |
| 3 | ~~`factory.glb`~~ → built procedurally in R3F instead | 0 KB, no second asset |
| 4 | Scroll timeline, five beats | scrubs correctly both directions |
| 5 | Scene 3 diagnosis beat, real numbers | figures match the database |
| 6 | Transition, mobile, a11y, perf gates | full budget table passes |

**Phase 0 is not ceremony.** The sandbox runs on Vite; the app runs on Next with
SSR. Draco decoder paths, `ssr:false` and bundle isolation are precisely where
this breaks, and phases 1–6 all assume it works.
