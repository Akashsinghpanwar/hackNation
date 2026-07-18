import type { AntibioticPrediction, AppConfig, GenomeQC } from "../../shared/types.js";
import type { ModelOutput } from "./predictor.js";

export function applySafetyGovernor(qc: GenomeQC, modelOutputs: ModelOutput[], config: AppConfig): AntibioticPrediction[] {
  return modelOutputs.map((output) => {
    const reasonCodes: string[] = [];
    const matched = output.matchedMarkers;
    const targetStatus = "present_by_supported_species_scope";
    const oodStatus = matched.length > 0 ? "in_distribution_demo_support" : "low_marker_support";

    let decision: AntibioticPrediction["decision"];
    if (qc.qc_status === "fail") {
      decision = "no_call";
      reasonCodes.push("qc_failed");
    } else if (output.calibratedConfidence < config.safety.low_confidence_cutoff) {
      decision = "no_call";
      reasonCodes.push("low_calibrated_confidence");
    } else if (oodStatus === "low_marker_support" && output.resistanceProbability > output.workThreshold) {
      decision = "no_call";
      reasonCodes.push("weak_or_unsupported_evidence");
    } else if (output.resistanceProbability >= output.failThreshold) {
      decision = "likely_to_fail";
      reasonCodes.push("resistance_marker_detected");
    } else if (output.resistanceProbability <= output.workThreshold) {
      decision = "likely_to_work";
      reasonCodes.push("low_resistance_signal_and_target_present");
    } else {
      decision = "no_call";
      reasonCodes.push("probability_in_no_call_region");
    }

    if (qc.warnings.length > 0) reasonCodes.push("qc_warning_present");

    const evidenceCategory = matched.length > 0
      ? "Known resistance gene or mutation detected"
      : decision === "likely_to_work"
        ? "No known resistance signal found"
        : "Insufficient or out-of-distribution evidence";

    const markerText = matched.length > 0 ? matched.join(", ") : "no configured resistance marker";
    const explanation = `${output.antibiotic}: ${decision.replaceAll("_", " ")} with calibrated confidence ${(output.calibratedConfidence * 100).toFixed(0)}%. Evidence: ${markerText}. This is decision support only and requires standard laboratory confirmation.`;

    return {
      antibiotic: output.antibiotic,
      decision,
      calibrated_confidence: output.calibratedConfidence,
      resistance_probability: output.resistanceProbability,
      evidence_category: evidenceCategory,
      supporting_markers: matched,
      target_status: targetStatus,
      ood_status: oodStatus,
      reason_codes: reasonCodes,
      explanation,
      lab_confirmation_required: true
    };
  });
}
