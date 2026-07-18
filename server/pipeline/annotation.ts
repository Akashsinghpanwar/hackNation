import type { AppConfig, GenomeRecord, Marker } from "../../shared/types.js";

const markerMetadata: Record<string, { elementType: string; drugClass: string }> = {
  "gyrA_S83L": { elementType: "point_mutation", drugClass: "fluoroquinolone" },
  "parC_S80I": { elementType: "point_mutation", drugClass: "fluoroquinolone" },
  qnrS1: { elementType: "AMR_gene", drugClass: "fluoroquinolone" },
  "blaTEM-1": { elementType: "AMR_gene", drugClass: "beta-lactam" },
  "blaSHV-1": { elementType: "AMR_gene", drugClass: "beta-lactam" },
  tetA: { elementType: "AMR_gene", drugClass: "tetracycline" },
  tetB: { elementType: "AMR_gene", drugClass: "tetracycline" },
  "blaCTX-M-15": { elementType: "AMR_gene", drugClass: "extended-spectrum beta-lactam" },
  "blaCMY-2": { elementType: "AMR_gene", drugClass: "cephalosporin" }
};

export function demoAnnotateMarkers(records: GenomeRecord[], config: AppConfig): Marker[] {
  const haystack = records.map((record) => `${record.header}\n${record.sequence}`).join("\n").toLowerCase();
  const configuredMarkers = new Set(config.antibiotics.flatMap((antibiotic) => antibiotic.known_markers));

  return [...configuredMarkers].sort().flatMap((symbol) => {
    if (!haystack.includes(symbol.toLowerCase())) return [];
    const meta = markerMetadata[symbol] ?? { elementType: "AMR_marker", drugClass: "unknown" };
    return [{
      symbol,
      element_type: meta.elementType,
      drug_class: meta.drugClass,
      evidence_level: "known_marker",
      identity: 99,
      coverage: 100
    }];
  });
}
