# 3D Hero — Architecture and Preparation

**Status: shipped.** The asset is built and verified (30/30 kinematic
assertions, 19,568 triangles, 174 KB), and the R3F scene is live in both the
dashboard Overview hero and the landing page, on `main`.

The sequencing constraint below — *build the asset alone, integrate only after
the six screens are done* — was honoured and is kept as the record of why. The
six screens shipped first; the hero followed and could not have delayed them.

Sections written in the future tense describe the plan as it stood. Where the
implementation diverged, the companion note
[`LANDING_PAGE_DESIGN.md`](LANDING_PAGE_DESIGN.md) records it — what shipped is
the authority, not what was intended.

---

## 1. What the scene is

A scanner arm on a gantry passes over a conveyor. An object travels along the
belt, the scanner sweeps it, a beam plays across it, and a heatmap resolves onto
its surface as the beam passes.

That last beat is the point: it is the product's actual idea — *the model looked
here* — expressed physically.

---

## 2. Component tree

```
<HeroSection>                    2D layout, headline, CTAs — Framer Motion
  <HeroCanvasLoader>             next/dynamic, ssr:false, IntersectionObserver
    <Suspense fallback={<HeroPoster/>}>
      <Canvas dpr={[1,2]}>
        <Lighting/>              key + fill + rim, soft environment
        <CameraRig/>             parallax on pointer, dolly on scroll
        <Ground/>                shadow catcher only, no visible plane

        <Conveyor>               rails and legs — static
          <Rollers/>             one node each; all spin at conveyorSpeed
          <ConveyorStart/>       landmark: where objectProgress = 0
          <ConveyorEnd/>         landmark: where objectProgress = 1
        </Conveyor>

        <ArmMount>               pedestal, fixed beside the conveyor
          <ArmBase>              yaws about Z — armYaw
            <ArmShoulder>        swings on the shoulder joint
              <ArmForearm>       swings on the elbow joint
                <ScannerHead>    rotates, tilts, on the wrist joint
                  <ScannerLens/>
                  <Beam/>        emissive cone; intensity is a prop
                </ScannerHead>
              </ArmForearm>
            </ArmShoulder>
          </ArmBase>
        </ArmMount>

        <InspectionObject>       travels along the rollers
          <HeatmapOverlay/>      runtime only — see below
        </InspectionObject>
      </Canvas>
    </Suspense>
  </HeroCanvasLoader>
</HeroSection>
```

Every moving part is its own component with its own transform origin. That is
the whole reason for building rather than importing a fused mesh.

**The arm is a kinematic chain, and the nesting is what makes it one.** Yawing
`ArmBase` carries the whole arm; swinging `ArmShoulder` carries everything
below it; tilting `ScannerHead` re-aims the beam. Each parameter is set once,
on one node, and the rest follows — no value is derived twice and nothing has
to be kept in step by hand.

**Rollers are separate nodes, not one mesh.** Each turns about its own axis;
joined into a single mesh they would orbit a shared origin like a carousel.
There is no belt surface — plates ride directly on the rollers, so conveyor
motion reads from the rollers turning and the plates translating, and the asset
carries one fewer texture for it.

**`HeatmapOverlay` exists only at runtime.** Its opacity is derived from the
beam (§3), so it is built in R3F from `InspectionObject`'s own bounds and is
deliberately absent from the exported asset. A placeholder plane would ship
geometry the runtime replaces; a placeholder material would put the appearance
of a measurement into the asset, where nothing can keep it in step with what
the beam is actually doing.

---

## 3. State model — and the rule that keeps it sane

Seven controllable parameters, one store:

```ts
type HeroState = {
  conveyorSpeed:   number   // 0..1, belt and rollers
  armYaw:          number   // -1..1, base rotation sweeping across the work
  headRotation:    number   // radians
  headTilt:        number   // radians
  beamIntensity:   number   // 0..1
  objectProgress:  number   // 0..1 along the belt, wraps
  cameraPreset:    'idle' | 'inspect' | 'wide'
}
```

### Exactly one owner per parameter

This is the rule that matters. When two systems write the same value, the
symptom is jitter that appears only sometimes and is miserable to debug.

| Parameter | Owner | Why |
| --- | --- | --- |
| `conveyorSpeed` | config | Set once; not animated |
| `objectProgress` | `useFrame` | Advances every frame, wraps at 1 |
| Belt offset, roller spin | `useFrame` | Derived from `conveyorSpeed` |
| Beam flicker | `useFrame` | A small multiplier on `beamIntensity` |
| `armYaw` | **GSAP** | Scroll-linked |
| `beamIntensity` (base) | **GSAP** | Scroll-linked |
| `headRotation`, `headTilt` | **GSAP** | Scroll-linked |
| `cameraPreset` | **GSAP** | Scroll-linked |

**`useFrame` owns what loops. GSAP owns what responds to scroll. Never both.**

### Heatmap opacity is derived, never stored

```ts
const heatmapOpacity = clamp01(objectProgress - beamPassPoint) * beamIntensity
```

It cannot desynchronise from the beam because it is computed from the beam. An
eighth stored parameter would be one more thing to keep in step.

---

## 4. Blender or procedural — the decision

**Decided 2026-08-22: Blender.** The comparison below was written while the
question was still open and reads as an argument for procedural. It is kept
because it is still an accurate account of the trade-off — what changed is
which side of it matters most.

| | Procedural R3F | Blender |
| --- | --- | --- |
| Asset pipeline | None | Model → pivots → GLB → Draco → load |
| Download | 0 KB | Target < 200 KB |
| Pivots | Free — each part is a component | Must be set by hand, correctly |
| Iteration | Edit a number, hot reload | Re-export, re-optimise, reload |
| Ceiling on looks | Good, clearly stylised | Higher, with skill |
| Time to something on screen | Hours | Days |

**The reason.** Two things decided it. First, **Jawad is already proficient in
Blender** — the condition this section originally set for choosing it, and the
one that removes its largest risk, since the pipeline stages in the table are
only expensive to someone learning them. Second, **Blender → GLB → R3F delivers
both halves of what the hero needs**: modelled realism a procedural scene
cannot reach, *and* full browser interactivity — the exported asset is not a
video, and every part arrives as a named node the runtime drives through the
parameters in §3.

The approved mockup does not change this and was never a candidate to build
from. It is a photorealistic still from an image model — a **look reference and
nothing else** — whose parts do not connect coherently and whose perspective
does not agree between views (§7). The mockup sets the look, Blender settles the
geometry, R3F controls the motion.

**The cost is carried, not argued away.** Four extra pipeline stages, pivots
placed by hand, and a slower edit loop. Pivots are the specific risk: an origin
at a part's visual centre rather than its joint gives a scanner head that spins
in place instead of swinging on its mount, and that is expensive to correct
once the geometry is finished. It is mitigated by setting and verifying every
origin during blockout, and by exporting a grey-box GLB and confirming names,
hierarchy and pivots survive the round trip **before** any detail work starts.

**Ownership.** The asset is built standalone and does not enter the Next.js
project until the six screens are done, so it cannot block them.

**AI 3D generation (Meshy, Tripo) is out for the mechanical parts.** It produces
a single fused mesh with baked shading and arbitrary origins; splitting and
re-pivoting that costs more than modelling cleanly. Hard surfaces are also its
weakest case.

---

## 5. Performance budget — pass or fail

| Check | Target | |
| --- | --- | --- |
| Scene triangles | 20–50k | ☐ |
| Draw calls | < 50 | ☐ |
| GLB, if any | < 200 KB compressed | ☐ |
| Textures | 1K max, KTX2 — or none | ☐ |
| Canvas DPR | `dpr={[1, 2]}`, never uncapped | ☐ |
| Three.js in server bundle | **absent** — `next/dynamic`, `ssr:false` | ☐ |
| **`useFrame` paused offscreen** | IntersectionObserver | ☐ |
| Idle behaviour | `frameloop="demand"` when static | ☐ |
| Post-processing | none unless measured | ☐ |
| Frame rate, mid-range laptop | 60fps sustained | ☐ |

**The offscreen pause is the one that gets missed.** Without it the render loop
runs for the whole session, draining battery on a page the user scrolled past
minutes ago. Verify it by scrolling away and watching CPU, not by reading the
code.

---

## 6. Fallback strategy

Three paths, all required, all cheap:

1. **No WebGL** → static hero image. Detect on mount, never attempt the canvas.
2. **`prefers-reduced-motion`** → render the scene, freeze the animation. Not a
   blank space; a still frame of a real scene.
3. **Hero not finished** → the same static image, shipped alone.

Path 3 is why the poster image is built **first**, not last. It is required for
paths 1 and 2 regardless, so it is never wasted work — and it means the landing
page is complete before the hero is started.

---

## 7. Look, and the honest target

The approved mockup is a photorealistic still from an image model. **A stylised
version of that scene at 60fps is very achievable; that exact image moving is
not.**

Target: clean premium product render. Soft studio lighting, matte materials, one
restrained emissive beam, slow deliberate motion.

Three things that read as premium in real-time 3D, in order:

1. **Lighting.** A three-point rig with soft shadows does more than any amount
   of geometry.
2. **Restraint.** Fewer objects, more space. The common failure is adding
   geometry to compensate for lighting that is not working yet.
3. **Slow motion.** Fast movement reads as cheap. Real industrial machinery is
   deliberate.

---

## 8. Preparation that can happen now

Without blocking Jawad, and without writing scene code:

- [x] **Blender or procedural — settled.** Blender, for the reasons in §4.
- [ ] Collect 5–10 photographs of real industrial scanners. Reality is a better
      modelling reference than the mockup.
- [ ] Agree the beige palette as tokens, shared with the 2D UI.
- [ ] Decide the scroll narrative: how many beats, and what changes at each.
- [ ] Build the **poster image** — required by two fallback paths anyway.

**Do not build the scene** *(historical — this held until the six screens
shipped, and was then lifted)*. Not because it would break anything — it is an
isolated route — but because a finished hero waiting in a branch creates
pressure to integrate early, and that is exactly how the screens slip.

It worked: the screens went in first, and the scene followed without ever
competing with them for attention.
