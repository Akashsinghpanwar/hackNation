import { createHash } from "node:crypto";
import { extname } from "node:path";
import type { AppConfig, GenomeQC, GenomeRecord } from "../../shared/types.js";

const validBases = new Set("ACGTNRYKMSWBDHV".split(""));

export class FastaValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FastaValidationError";
  }
}

export function parseFasta(content: string, fileName: string, config: AppConfig): { records: GenomeRecord[]; checksum: string } {
  const suffix = extname(fileName).toLowerCase();
  if (!config.supported_extensions.includes(suffix)) {
    throw new FastaValidationError(`Unsupported file extension. Use ${config.supported_extensions.join(", ")}.`);
  }

  const maxBytes = config.max_file_size_mb * 1024 * 1024;
  if (Buffer.byteLength(content, "utf8") > maxBytes) {
    throw new FastaValidationError(`File is larger than ${config.max_file_size_mb} MB.`);
  }

  if (!content.trimStart().startsWith(">")) {
    throw new FastaValidationError("FASTA must start with a header line beginning with '>'.");
  }

  const records: GenomeRecord[] = [];
  let header: string | null = null;
  let sequenceParts: string[] = [];

  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith(">")) {
      if (line.length === 1) {
        throw new FastaValidationError("FASTA header cannot be empty.");
      }
      if (header !== null) {
        records.push({ header, sequence: sequenceParts.join("").toUpperCase() });
      }
      header = line.slice(1);
      sequenceParts = [];
      continue;
    }

    const sequence = line.toUpperCase();
    const invalid = [...new Set(sequence.split("").filter((base) => !validBases.has(base)))].sort();
    if (invalid.length > 0) {
      throw new FastaValidationError(`Invalid nucleotide characters found: ${invalid.join(", ")}.`);
    }
    sequenceParts.push(sequence);
  }

  if (header !== null) {
    records.push({ header, sequence: sequenceParts.join("").toUpperCase() });
  }

  if (records.length === 0 || records.some((record) => record.sequence.length === 0)) {
    throw new FastaValidationError("FASTA must contain at least one non-empty sequence.");
  }
  if (records.length > config.max_contigs) {
    throw new FastaValidationError(`Too many contigs. Maximum allowed is ${config.max_contigs}.`);
  }

  const checksum = createHash("sha256").update(content).digest("hex");
  return { records, checksum };
}

export function qualityCheck(records: GenomeRecord[], checksum: string, config: AppConfig): GenomeQC {
  const sequenceLength = records.reduce((total, record) => total + record.sequence.length, 0);
  const ambiguousBases = records.reduce((total, record) => total + (record.sequence.match(/N/g)?.length ?? 0), 0);
  const ambiguousBaseFraction = sequenceLength === 0 ? 1 : ambiguousBases / sequenceLength;
  const headers = records.map((record) => record.header);
  const warnings: string[] = [];

  if (sequenceLength < config.quality.min_total_bases) warnings.push("Genome is shorter than the configured supported-species range.");
  if (sequenceLength > config.quality.max_total_bases) warnings.push("Genome is longer than the configured supported-species range.");
  if (ambiguousBaseFraction > config.quality.max_ambiguous_fraction) warnings.push("Ambiguous base fraction is above the configured threshold.");
  if (new Set(headers).size !== headers.length) warnings.push("Duplicate FASTA headers detected.");

  let qcStatus: GenomeQC["qc_status"] = warnings.length > 0 ? "warn" : "pass";
  if (sequenceLength === 0 || ambiguousBaseFraction > 0.25) qcStatus = "fail";

  return {
    genome_id: `run_${checksum.slice(0, 8)}`,
    qc_status: qcStatus,
    sequence_length: sequenceLength,
    contig_count: records.length,
    ambiguous_base_fraction: Number(ambiguousBaseFraction.toFixed(4)),
    sha256: checksum,
    warnings
  };
}
