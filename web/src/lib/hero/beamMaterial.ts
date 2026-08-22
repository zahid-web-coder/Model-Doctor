import * as THREE from "three";

/**
 * The scanning beam, as a volume rather than a translucent cone.
 *
 * A plain transparent mesh reads as a solid object with the lights turned
 * down — you can see its silhouette, which is exactly what a light shaft does
 * not have. Three things fix that, all in the fragment shader:
 *
 * - **Edge glow.** A cone of light is brighter where the eye looks through
 *   more of it, which is at the silhouette. Fresnel-weighting the alpha gives
 *   that for free and is most of what sells the volume.
 * - **Vertical falloff.** Brightest at the aperture, fading toward the work
 *   surface, so the beam has a source rather than just existing.
 * - **Additive blending.** Light adds to what is behind it. Alpha blending
 *   darkens, which is why a transparent cone reads as smoked glass.
 *
 * `depthWrite` stays off so the beam never occludes the plate it lands on.
 */
export function createBeamMaterial() {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    toneMapped: false,
    uniforms: {
      uIntensity: { value: 1.0 },
      uColor: { value: new THREE.Color("#1f7ffb") },
      uCore: { value: new THREE.Color("#7dc0ff") },
      uLength: { value: 0.6 },
      uTime: { value: 0 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vLocal;
      varying vec3 vViewDir;
      varying vec3 vNormalW;
      void main() {
        vLocal = position;
        vec4 world = modelMatrix * vec4(position, 1.0);
        vViewDir = normalize(cameraPosition - world.xyz);
        vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * viewMatrix * world;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform float uIntensity;
      uniform float uLength;
      uniform float uTime;
      uniform vec3 uColor;
      uniform vec3 uCore;
      varying vec3 vLocal;
      varying vec3 vViewDir;
      varying vec3 vNormalW;

      void main() {
        // 0 at the aperture, 1 at the work surface.
        float drop = clamp(-vLocal.z / uLength, 0.0, 1.0);

        // Grazing angles look through more of the cone, so the silhouette is
        // the brightest part. This is the term that reads as volume.
        float facing = abs(dot(normalize(vNormalW), normalize(vViewDir)));
        float rim = pow(1.0 - facing, 2.6);

        // Bright at the source, thinning as it spreads.
        float fall = pow(1.0 - drop, 1.35);

        // A slow travelling band, so the beam is not a static cone.
        float sweep = 0.035 * sin(drop * 26.0 - uTime * 2.2);

        // Dissipate toward the work surface instead of stopping. Light does
        // not have an edge where it ends.
        float tail = 1.0 - smoothstep(0.62, 1.0, drop);

        // Almost nothing in the body, most of it at the grazing silhouette.
        // A flat base fills the cone evenly, which is what makes it read as a
        // translucent slab rather than as air full of light.
        float alpha = (0.012 + rim * 0.58) * (0.18 + fall * 0.62) * tail + sweep * tail;
        alpha = clamp(alpha, 0.0, 1.0) * uIntensity;

        vec3 tint = mix(uColor, uCore, fall * 0.34);
        gl_FragColor = vec4(tint * (0.42 + fall * 0.62) * uIntensity, alpha);
      }
    `,
  });
}
