export interface HeatmapImage {
  id: string;
  thumbnail: string;
  failureType: string;
  confidence: number;
  factor: string;
}

export const mockHeatmaps: HeatmapImage[] = Array.from({ length: 24 }).map((_, i) => {
  const types = ["False Positive", "False Negative", "Wrong Class"];
  const factors = ["blur", "low_light", "occlusion", "small_object"];
  
  return {
    id: `hm-${2000 + i}`,
    thumbnail: `bg-gradient-to-tr from-slate-${300 + (i%5)*100} to-slate-${400 + (i%5)*100}`,
    failureType: types[i % types.length],
    confidence: 0.6 + Math.random() * 0.39,
    factor: factors[i % factors.length]
  };
});
