export interface Cluster {
  id: string;
  name: string;
  memberCount: number;
  dominantFactor: string;
  thumbnail: string;
  description: string;
}

export const mockClusters: Cluster[] = [
  {
    id: "c-1",
    name: "Edge Deformation",
    memberCount: 142,
    dominantFactor: "blur",
    thumbnail: "from-blue-900 via-orange-400 to-red-600",
    description: "Failures clustered around blurred edges where deformation is present."
  },
  {
    id: "c-2",
    name: "Low Light Corner",
    memberCount: 89,
    dominantFactor: "low_light",
    thumbnail: "from-slate-800 via-slate-600 to-black",
    description: "Dark regions in the bottom-left corner of the inspection area."
  },
  {
    id: "c-3",
    name: "Center Glare",
    memberCount: 76,
    dominantFactor: "contrast",
    thumbnail: "from-yellow-100 via-white to-yellow-200",
    description: "High contrast washout in the center of the part."
  },
  {
    id: "c-4",
    name: "Micro-scratches",
    memberCount: 45,
    dominantFactor: "small_object",
    thumbnail: "from-slate-400 via-slate-300 to-slate-200",
    description: "Very small defects barely visible at current resolution."
  }
];
