export interface Run {
  id: string;
  model: string;
  dataset: string;
  environment: string;
  status: "Live" | "Complete" | "Failed";
  date: string;
  totalImages: number;
  accuracy: number;
  failureCount: number;
}

export const mockRuns: Run[] = [
  {
    id: "1287",
    model: "ResNet50",
    dataset: "Inspection Set",
    environment: "Production",
    status: "Live",
    date: "May 21, 2024",
    totalImages: 10452,
    accuracy: 0.92,
    failureCount: 836
  },
  {
    id: "1286",
    model: "ResNet50",
    dataset: "Inspection Set",
    environment: "Production",
    status: "Complete",
    date: "May 20, 2024",
    totalImages: 15000,
    accuracy: 0.86,
    failureCount: 2100
  },
  {
    id: "1285",
    model: "MobileNet V3",
    dataset: "Training Set",
    environment: "Staging",
    status: "Complete",
    date: "May 19, 2024",
    totalImages: 50000,
    accuracy: 0.95,
    failureCount: 2500
  },
  {
    id: "1284",
    model: "ResNet50",
    dataset: "Inspection Set",
    environment: "Production",
    status: "Failed",
    date: "May 18, 2024",
    totalImages: 450,
    accuracy: 0,
    failureCount: 450
  }
];
