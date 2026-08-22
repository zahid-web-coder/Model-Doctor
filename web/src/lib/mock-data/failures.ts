export interface FailureInstance {
  id: string;
  prediction: string;
  groundTruth: string;
  confidence: number;
  failureType: "False Positive" | "False Negative" | "Wrong Class" | "Poor Localization";
  rootCauseFactor: string;
  thumbnail: string;
}

export const mockFailures: FailureInstance[] = Array.from({ length: 15 }).map((_, i) => {
  const types = ["False Positive", "False Negative", "Wrong Class", "Poor Localization"] as const;
  const factors = ["blur", "low_light", "occlusion", "small_object", "contrast"];
  const classes = ["Crack", "Deformation", "Rust", "Leak", "Corrosion"];
  
  const type = types[i % types.length];
  const pred = type === "False Negative" ? "OK" : classes[i % classes.length];
  const gt = type === "False Positive" ? "OK" : classes[(i + 1) % classes.length];
  
  return {
    id: `img-${1000 + i}`,
    prediction: pred,
    groundTruth: gt,
    confidence: 0.5 + Math.random() * 0.49,
    failureType: type,
    rootCauseFactor: factors[i % factors.length],
    thumbnail: `bg-gradient-to-tr from-slate-${300 + (i%5)*100} to-slate-${400 + (i%5)*100}`
  };
});
