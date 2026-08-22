/**
 * Where the hero asset and its decoder live.
 *
 * The Draco decoder is served from `/draco`, not from Google's CDN. drei
 * defaults to a gstatic URL, which means the hero silently fails on a blocked
 * network and pins us to a decoder version we do not control. `public/draco`
 * is generated on postinstall from the exact `three` in `package.json` (see
 * `scripts/copy-draco.mjs`), so decoder and loader cannot drift apart.
 *
 * Both constants live here rather than in a component because the loader is
 * keyed on the URL — two spellings of the same path would load and cache the
 * asset twice.
 */
export const MODEL_URL = "/models/hero.glb";
export const DRACO_PATH = "/draco/";
