# Landing Page — Architecture and Plan

**Status:** Approved direction, phase 0 in progress. The landing page is a
**separate experience** from the dashboard. Nothing here modifies dashboard UI
or API integration.

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
  (marketing)/
    page.tsx                landing
  (app)/
    dashboard/page.tsx      overview, moved from /
    runs/...                unchanged
    reports/, settings/     unchanged
```

Route groups are for the **bundle boundary**, not the layout: three, drei and
gsap must never enter a dashboard chunk.

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

New geometry lives in a second file:

| File | Contents | Loaded by | Budget |
| --- | --- | --- | --- |
| `hero.glb` | arm, conveyor, rollers — **unchanged** | dashboard + landing | 169 KB, frozen |
| `factory.glb` | verification chamber, hall shell, conveyor extension | landing only | < 250 KB |

"Reuse the asset, do not duplicate" is right, but a shared asset is not the same
as a single file. Merging them would make the dashboard download a verification
chamber it never renders.

**Neither GLB contains animation clips.** Plate motion is runtime state
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

---

## 7. Performance budget

Landing page, measured on the machine that recorded 2.0 ms/frame for the hero:

| Check | Target |
| --- | --- |
| Scene triangles | < 60k |
| Draw calls | < 80 |
| GLB total, compressed | < 450 KB |
| three in a dashboard chunk | **absent** |
| `useFrame` paused offscreen | required |
| Frame rate | 60fps sustained |

Measured, not assumed. The hero note's warning applies unchanged: the offscreen
pause is the one that gets missed, and it is verified by watching CPU, not by
reading the code.

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
| 3 | `factory.glb` — chamber and hall | < 250 KB |
| 4 | Scroll timeline, five beats | scrubs correctly both directions |
| 5 | Scene 3 diagnosis beat, real numbers | figures match the database |
| 6 | Transition, mobile, a11y, perf gates | full budget table passes |

**Phase 0 is not ceremony.** The sandbox runs on Vite; the app runs on Next with
SSR. Draco decoder paths, `ssr:false` and bundle isolation are precisely where
this breaks, and phases 1–6 all assume it works.
