export type Decision = "likely_to_fail" | "likely_to_work" | "no_call";

export interface AntibioticConfig {
  name: string;
  target: string;
  known_markers: string[];
  fail_threshold: number;
  work_threshold: number;
}

export interface AppConfig {
  product_name: string;
  supported_species: string;
  model_version: string;
  supported_extensions: string[];
  max_file_size_mb: number;
  max_contigs: number;
  quality: {
    min_total_bases: number;
    max_total_bases: number;
    max_ambiguous_fraction: number;
  };
  antibiotics: AntibioticConfig[];
  safety: {
    low_confidence_cutoff: number;
    ood_marker_cutoff: number;
    lab_confirmation_message: string;
  };
}

export interface GenomeRecord {
  header: string;
  sequence: string;
}

export interface GenomeQC {
  genome_id: string;
  qc_status: "pass" | "warn" | "fail";
  sequence_length: number;
  contig_count: number;
  ambiguous_base_fraction: number;
  sha256: string;
  warnings: string[];
}

export interface Marker {
  symbol: string;
  element_type: string;
  drug_class: string;
  evidence_level: string;
  identity: number | null;
  coverage: number | null;
}

export interface AntibioticPrediction {
  antibiotic: string;
  decision: Decision;
  calibrated_confidence: number | null;
  resistance_probability: number | null;
  evidence_category: string;
  supporting_markers: string[];
  target_status: string;
  ood_status: string;
  reason_codes: string[];
  explanation: string;
  lab_confirmation_required: boolean;
}

export interface AnalysisResult {
  run_id: string;
  species: string;
  qc: GenomeQC;
  markers: Marker[];
  predictions: AntibioticPrediction[];
  warnings: string[];
  disclaimer: string;
}
