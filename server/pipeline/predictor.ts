import type { AppConfig, Marker } from "../../shared/types.js";

export interface ModelOutput {
  antibiotic: string;
  target: string;
  matchedMarkers: string[];
  resistanceProbability: number;
  calibratedConfidence: number;
  failThreshold: number;
  workThreshold: number;
}

function markerProbability(matchCount: number, totalMarkers: number): number {
  if (matchCount <= 0) return 0.18;
  if (matchCount === 1) return 0.82;
  return Math.min(0.94, 0.82 + ((matchCount - 1) * 0.06) / Math.max(totalMarkers, 1));
}

export function runAntibioticModels(markers: Marker[], features: Record<string, number>, config: AppConfig): ModelOutput[] {
  const markerSymbols = new Set(markers.map((marker) => marker.symbol));

  return config.antibiotics.map((antibiotic) => {
    const matchedMarkers = antibiotic.known_markers.filter((marker) => markerSymbols.has(marker));
    let probability = markerProbability(matchedMarkers.length, antibiotic.known_markers.length);

    if ((features["qc::ambiguous_base_fraction"] ?? 0) > 0.05) {
      probability = (probability + 0.5) / 2;
    }

    const confidence = Math.max(probability, 1 - probability);
    return {
      antibiotic: antibiotic.name,
      target: antibiotic.target,
      matchedMarkers,
      resistanceProbability: Number(probability.toFixed(3)),
      calibratedConfidence: Number(confidence.toFixed(3)),
      failThreshold: antibiotic.fail_threshold,
      workThreshold: antibiotic.work_threshold
    };
  });
}
