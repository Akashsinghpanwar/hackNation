import type { AnalysisResult } from "../../shared/types.js";
import { loadConfig } from "../config.js";
import { demoAnnotateMarkers } from "./annotation.js";
import { buildFeatureVector } from "./features.js";
import { parseFasta, qualityCheck } from "./fasta.js";
import { runAntibioticModels } from "./predictor.js";
import { applySafetyGovernor } from "./safety.js";

export function analyseFasta(content: string, fileName: string, species: string): AnalysisResult {
  const config = loadConfig();
  const warnings: string[] = [];
  const { records, checksum } = parseFasta(content, fileName, config);
  const qc = qualityCheck(records, checksum, config);
  const markers = demoAnnotateMarkers(records, config);
  const features = buildFeatureVector(markers, qc, config);
  const modelOutputs = runAntibioticModels(markers, features, config);
  let predictions = applySafetyGovernor(qc, modelOutputs, config);

  if (species !== config.supported_species) {
    warnings.push(`Unsupported species selected: ${species}. Only ${config.supported_species} is configured.`);
    predictions = predictions.map((prediction) => ({
      ...prediction,
      decision: "no_call",
      evidence_category: "Unsupported species scope",
      reason_codes: [...prediction.reason_codes, "unsupported_species"],
      explanation: `${prediction.antibiotic}: no-call because this prototype is only configured for ${config.supported_species}.`
    }));
  }

  return {
    run_id: qc.genome_id,
    species,
    qc,
    markers,
    predictions,
    warnings: [...warnings, ...qc.warnings],
    disclaimer: config.safety.lab_confirmation_message
  };
}
