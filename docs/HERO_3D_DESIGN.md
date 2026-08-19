# 3D Hero — Architecture and Preparation

**Status:** Design only. **No implementation until all six screens are done.**

The application is complete and useful without this. If the hero slips or
disappoints, the six screens still ship — that constraint governs every decision
below.

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

        <Conveyor>               static frame
          <Rollers/>             rotate with belt speed
          <BeltSurface/>         texture offset, or repeated segments
        </Conveyor>

        <Gantry>                 travels along X
          <Arm/>
          <ScannerHead>          rotates, tilts
            <Lens/>
            <Beam/>              emissive cone; intensity is a prop
          </ScannerHead>
        </Gantry>

        <InspectionObject>       travels along the belt
          <HeatmapOverlay/>      opacity derived, not stored
        </InspectionObject>
      </Canvas>
    </Suspense>
  </HeroCanvasLoader>
</HeroSection>
```

Every moving part is its own component with its own transform origin. That is
the whole reason for building rather than importing a fused mesh.

---

## 3. State model — and the rule that keeps it sane

Seven controllable parameters, one store:

```ts
type HeroState = {
  conveyorSpeed:   number   // 0..1, belt and rollers
  gantryX:         number   // -1..1, normalised travel along the conveyor
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
| `gantryX` | **GSAP** | Scroll-linked |
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

**Open question, and the only blocking one: does Jawad know Blender?**

| | Procedural R3F | Blender |
| --- | --- | --- |
| Asset pipeline | None | Model → pivots → GLB → Draco → load |
| Download | 0 KB | Target < 200 KB |
| Pivots | Free — each part is a component | Must be set by hand, correctly |
| Iteration | Edit a number, hot reload | Re-export, re-optimise, reload |
| Ceiling on looks | Good, clearly stylised | Higher, with skill |
| Time to something on screen | Hours | Days |

**Default: procedural.** A scanner and conveyor are boxes, cylinders and a
gantry. Restraint plus good lighting reads as premium; geometric detail does
not, and detail is what procedural gives up.

**Switch to Blender only if** Jawad already works in it *and* the procedural
version has been tried and judged insufficient. Not before — the four extra
pipeline stages are the most likely source of a slipped week.

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

- [ ] **Ask Jawad about Blender.** The one blocking unknown.
- [ ] Collect 5–10 photographs of real industrial scanners. Reality is a better
      modelling reference than the mockup.
- [ ] Agree the beige palette as tokens, shared with the 2D UI.
- [ ] Decide the scroll narrative: how many beats, and what changes at each.
- [ ] Build the **poster image** — required by two fallback paths anyway.

**Do not build the scene.** Not because it would break anything — it is an
isolated route — but because a finished hero waiting in a branch creates
pressure to integrate early, and that is exactly how the screens slip.
