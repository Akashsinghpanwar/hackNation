import type { AppConfig, GenomeQC, Marker } from "../../shared/types.js";

export function buildFeatureVector(markers: Marker[], qc: GenomeQC, config: AppConfig): Record<string, number> {
  const symbols = new Set(markers.map((marker) => marker.symbol));
  const features: Record<string, number> = {};

  for (const antibiotic of config.antibiotics) {
    for (const marker of antibiotic.known_markers) {
      features[`marker::${marker}`] = symbols.has(marker) ? 1 : 0;
    }
  }

  features["qc::ambiguous_base_fraction"] = qc.ambiguous_base_fraction;
  features["qc::contig_count"] = qc.contig_count;
  features["qc::sequence_length"] = qc.sequence_length;
  features["support::marker_density"] = Math.min(markers.length / 8, 1);
  return features;
}
