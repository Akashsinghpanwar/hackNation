import type { AnalysisResult, AntibioticPrediction, AppConfig, Marker } from "../shared/types.js";

interface StaticAnalysisRequest {
  mode: "fasta" | "tsv" | "bvbrc";
  species: string;
  fileName?: string;
  content?: string;
  genome_id?: string;
}

interface RawMetricBin {
  mean_prob: number;
  frac_positive: number;
  count: number;
}

interface RawDrugMetric {
  auroc: number | null;
  pr_auc: number | null;
  brier_score: number | null;
  no_call_rate: number | null;
  answered_accuracy: number | null;
  balanced_accuracy_called: number | null;
  resistant_recall_called: number | null;
  susceptible_recall_called: number | null;
  reliability?: RawMetricBin[];
  generalization?: {
    n_groups_evaluated: number;
    auroc_mean: number | null;
    auroc_median: number | null;
    auroc_min: number | null;
  } | null;
}

interface KmerIndex {
  k: number;
  families: Record<string, number[][]>;
}

const SUPPORTED_SPECIES = "Escherichia coli";
const DISCLAIMER = "Research prototype only. Confirm every result with standard laboratory susceptibility testing.";
const MIN_KMER_BASES = 5_000;
const MIN_CORE_ECOLI_BASES = 500_000;
const CONTAINMENT_THRESHOLD = 0.5;
const CORE_ECOLI_GENES = new Set(["blaEC", "blaAmpC", "marA", "marR", "marB"]);
const BASE_CODE: Record<string, bigint> = { A: 0n, C: 1n, G: 2n, T: 3n };

const DEFAULT_CONFIG: AppConfig = {
  product_name: "AMRShield Sentinel",
  supported_species: SUPPORTED_SPECIES,
  model_version: "demo-marker-baseline-v0.1",
  supported_extensions: [".fa", ".fasta", ".fna"],
  max_file_size_mb: 20,
  max_contigs: 500,
  quality: {
    min_total_bases: 100,
    max_total_bases: 7_000_000,
    max_ambiguous_fraction: 0.08
  },
  antibiotics: [
    {
      name: "Ciprofloxacin",
      target: "DNA gyrase / topoisomerase IV",
      known_markers: ["gyrA_S83L", "parC_S80I", "qnrS1"],
      fail_threshold: 0.72,
      work_threshold: 0.28
    },
    {
      name: "Ampicillin",
      target: "Penicillin-binding proteins",
      known_markers: ["blaTEM-1", "blaSHV-1"],
      fail_threshold: 0.7,
      work_threshold: 0.3
    },
    {
      name: "Tetracycline",
      target: "30S ribosomal subunit",
      known_markers: ["tetA", "tetB"],
      fail_threshold: 0.7,
      work_threshold: 0.3
    },
    {
      name: "Ceftriaxone",
      target: "Penicillin-binding proteins",
      known_markers: ["blaCTX-M-15", "blaCMY-2"],
      fail_threshold: 0.74,
      work_threshold: 0.26
    }
  ],
  safety: {
    low_confidence_cutoff: 0.6,
    ood_marker_cutoff: 0.2,
    lab_confirmation_message: DISCLAIMER
  }
};

const GENE_PATTERNS: Array<[string, RegExp]> = [
  ["blaCTX-M", /CTX-M/i],
  ["blaTEM", /\bTEM\b|TEM family/i],
  ["blaSHV", /\bSHV\b|SHV family/i],
  ["blaOXA", /OXA family|Class D beta-lactamase/i],
  ["blaCMY", /\bCMY\b/i],
  ["blaEC", /BlaEC family/i],
  ["blaAmpC", /Class C beta-lactamase|AmpC/i],
  ["betaLactamResProtein", /beta-lactam resistance protein/i],
  ["tetA", /Tet\(A\)|TetA\b/i],
  ["tetB", /Tet\(B\)|TetB\b/i],
  ["tetC", /Tet\(C\)|TetC\b/i],
  ["tetM", /Tet\(M\)|TetM\b/i],
  ["tetG", /Tet\(G\)|TetG\b/i],
  ["tetR", /Tetracycline resistance regulatory|TetR\b/i],
  ["qnrA", /qnrA|quinolone resistance.*A/i],
  ["qnrB", /qnrB|quinolone resistance.*B/i],
  ["qnrS", /qnrS|quinolone resistance.*S/i],
  ["qnrD", /qnrD/i],
  ["aac6Ib", /AAC\(6'\).*Ib|aac\(6'\).*Ib/i],
  ["oqxA", /OqxA|oqxA/i],
  ["oqxB", /OqxB|oqxB/i],
  ["aadA", /aadA|ANT\(3''\)/i],
  ["aac3", /AAC\(3\)/i],
  ["ant2", /ANT\(2''\)/i],
  ["aph3", /APH\(3'\)|aminoglycoside 3'-phosphotransferase/i],
  ["sul1", /type-2.*sulfonamide|Sulfonamide resistance|dihydropteroate synthase type-2/i],
  ["dhfr", /dihydrofolate reductase|DHFR/i],
  ["trimethoprimRes", /trimethoprim/i],
  ["marA", /MarA\b/i],
  ["marR", /MarR\b/i],
  ["marB", /MarB\b/i],
  ["acrAB", /AcrA|AcrB|AcrAB/i],
  ["tolC", /TolC\b/i],
  ["mdtEFG", /MdtG|MdtH|MdtE|MdtF/i],
  ["emrAB", /EmrA|EmrB|EmrR/i],
  ["qacE", /QacE/i],
  ["integraseI", /integron integrase|class 1 integron/i],
  ["catA", /chloramphenicol acetyltransferase|cat[ABC]\b/i],
  ["mcr", /\bmcr-/i]
];

const TARGET_PATTERNS: Record<string, RegExp[]> = {
  ampicillin: [/penicillin-binding protein/i, /\bPBP\b/i, /\bftsI\b/i, /\bmrdA\b/i, /\bmrcA\b/i, /\bmrcB\b/i],
  ceftriaxone: [/penicillin-binding protein/i, /\bPBP\b/i, /\bftsI\b/i, /\bmrdA\b/i, /\bmrcA\b/i, /\bmrcB\b/i],
  ciprofloxacin: [/DNA gyrase/i, /\bgyrA\b/i, /\bgyrB\b/i, /topoisomerase IV/i, /\bparC\b/i, /\bparE\b/i],
  tetracycline: [/30S ribosomal/i, /16S ribosomal/i, /ribosomal protein S/i, /\brrs[A-H]?\b/i]
};

const KNOWN_MARKERS: Record<string, Set<string>> = {
  ampicillin: new Set(["blaTEM", "blaCTX-M", "blaSHV", "blaOXA", "blaCMY", "blaAmpC", "betaLactamResProtein"]),
  ciprofloxacin: new Set(["qnrA", "qnrB", "qnrS", "qnrD", "aac6Ib", "oqxA", "oqxB"]),
  ceftriaxone: new Set(["blaCTX-M", "blaCMY", "blaSHV", "blaOXA"]),
  tetracycline: new Set(["tetA", "tetB", "tetC", "tetM", "tetG", "tetR"])
};

const DRUG_CLASS: Record<string, string> = {
  ampicillin: "Beta-lactam",
  ciprofloxacin: "Fluoroquinolone",
  ceftriaxone: "Cephalosporin",
  tetracycline: "Tetracycline"
};

const DRUG_TARGET: Record<string, string> = {
  ampicillin: "Penicillin-binding proteins",
  ciprofloxacin: "DNA gyrase / topoisomerase IV",
  ceftriaxone: "Penicillin-binding proteins",
  tetracycline: "30S ribosomal subunit"
};

const STATIC_BASELINE: Record<string, number> = {
  ampicillin: 0.22,
  ciprofloxacin: 0.18,
  ceftriaxone: 0.12,
  tetracycline: 0.2
};

let kmerIndexPromise: Promise<KmerIndex | null> | undefined;

export async function loadStaticConfig(): Promise<AppConfig> {
  return fetchRelativeJson<AppConfig>("static/app_config.json").catch(() => DEFAULT_CONFIG);
}

export async function loadStaticMetrics(): Promise<unknown> {
  const raw = await fetchRelativeJson<Record<string, RawDrugMetric>>("static/demo_metrics.json");
  const metrics = Object.entries(raw).map(([antibiotic, m]) => ({
    antibiotic: titleCase(antibiotic),
    auroc: m.auroc,
    pr_auc: m.pr_auc,
    balanced_accuracy: m.balanced_accuracy_called,
    resistant_recall: m.resistant_recall_called,
    susceptible_recall: m.susceptible_recall_called,
    brier: m.brier_score,
    no_call_rate: m.no_call_rate,
    answered_accuracy: m.answered_accuracy
  }));

  const reliability: Record<string, Array<{ mean_confidence: number; observed_accuracy: number; count: number }>> = {};
  for (const [antibiotic, m] of Object.entries(raw)) {
    reliability[antibiotic] = (m.reliability ?? []).map((bin) => ({
      mean_confidence: bin.mean_prob,
      observed_accuracy: bin.frac_positive,
      count: bin.count
    }));
  }

  const generalization = Object.entries(raw)
    .filter(([, m]) => Boolean(m.generalization))
    .map(([antibiotic, m]) => ({
      antibiotic: titleCase(antibiotic),
      overall_auroc: m.auroc,
      group_mean_auroc: m.generalization?.auroc_mean ?? null,
      group_min_auroc: m.generalization?.auroc_min ?? null,
      n_groups: m.generalization?.n_groups_evaluated ?? 0
    }));

  const firstKey = Object.keys(reliability)[0] ?? "";
  const reliability_points = reliability.ampicillin ?? reliability[firstKey] ?? [];
  return { metrics, reliability, reliability_points, generalization };
}

export async function loadStaticSample(sample: string): Promise<{ fileName: string; content: string }> {
  const res = await fetch(`demo_samples/${sample}`);
  if (!res.ok) throw new Error("Demo sample is not available in the static build.");
  return { fileName: sample, content: await res.text() };
}

export async function runStaticAnalysis(request: StaticAnalysisRequest, config: AppConfig): Promise<AnalysisResult> {
  const speciesSupported = request.species === config.supported_species;
  const warnings: string[] = ["Static GitHub Pages demo: browser marker rules are being used because the Node/Python backend is unavailable."];
  const genes = new Set<string>();
  const targets = new Set<string>();
  let source = "Static browser analysis";
  let bases = 0;
  let contigs = 0;
  let ambiguousBaseFraction = 0;
  let checksumInput = request.content ?? request.genome_id ?? request.fileName ?? "static-run";
  let runId = request.fileName ?? request.genome_id ?? "static-run";

  if (request.mode === "tsv") {
    const products = parseAmrfinderTsv(request.content ?? "");
    addAll(genes, classifyProducts(products));
    addAll(targets, detectTargets(products));
    source = "AMRFinderPlus TSV (static parser)";
  } else if (request.mode === "bvbrc") {
    runId = request.genome_id ?? "bvbrc";
    checksumInput = runId;
    source = `BV-BRC genome_feature: ${runId}`;
    try {
      const products = await fetchBvbrcProducts(runId);
      addAll(genes, classifyProducts(products));
      addAll(targets, detectTargets(products));
    } catch {
      warnings.push("BV-BRC browser fetch was blocked or unavailable; dynamic Node backend is recommended for BV-BRC IDs.");
    }
  } else {
    const content = request.content ?? "";
    const stats = fastaHeadersAndStats(content);
    bases = stats.bases;
    contigs = stats.contigs;
    ambiguousBaseFraction = stats.ambiguousBaseFraction;
    addAll(genes, classifyProducts(stats.headers));
    addAll(targets, detectTargets(stats.headers));
    source = genes.size ? "Annotated FASTA headers (static parser)" : "Raw FASTA (static k-mer detector)";

    if (!genes.size && bases >= MIN_KMER_BASES) {
      addAll(genes, await detectGenesFromDna(content));
    }
    if (bases >= MIN_CORE_ECOLI_BASES && speciesSupported) {
      addAll(genes, CORE_ECOLI_GENES);
    }
  }

  if (speciesSupported) {
    addAll(targets, new Set(["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]));
  } else {
    warnings.push(`Unsupported species: ${request.species}. Only ${config.supported_species} is configured.`);
  }

  const qcWarnings: string[] = [];
  let qcStatus: "pass" | "warn" | "fail" = "pass";
  if (request.mode === "fasta" && bases > 0 && bases < MIN_KMER_BASES && !genes.size) {
    qcStatus = "warn";
    qcWarnings.push("Short raw FASTA has no marker evidence; static demo returns no-call unless annotations are present.");
  }
  if (ambiguousBaseFraction > config.quality.max_ambiguous_fraction) {
    qcStatus = "warn";
    qcWarnings.push(`High ambiguous base fraction: ${(ambiguousBaseFraction * 100).toFixed(2)}%`);
  }

  const sha = await sha256Short(checksumInput);
  const predictions = buildStaticPredictions(genes, targets, speciesSupported, config);
  if (!speciesSupported) {
    for (const prediction of predictions) {
      prediction.decision = "no_call";
      prediction.calibrated_confidence = null;
      prediction.evidence_category = "Unsupported species scope";
      prediction.reason_codes.push("unsupported_species");
    }
  }

  return {
    run_id: runId,
    species: request.species,
    source,
    detected_genes: [...genes].sort(),
    kmer_count: genes.size,
    qc: {
      genome_id: runId,
      qc_status: qcStatus,
      sequence_length: bases,
      contig_count: contigs,
      ambiguous_base_fraction: Number(ambiguousBaseFraction.toFixed(5)),
      sha256: sha,
      warnings: qcWarnings
    },
    markers: buildMarkers(genes),
    predictions,
    warnings,
    disclaimer: config.safety.lab_confirmation_message
  };
}

function fetchRelativeJson<T>(path: string): Promise<T> {
  return fetch(path).then(async (res) => {
    if (!res.ok) throw new Error(`Static asset missing: ${path}`);
    return (await res.json()) as T;
  });
}

function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function addAll<T>(target: Set<T>, source: Set<T>): void {
  for (const item of source) target.add(item);
}

function classifyProducts(products: string[]): Set<string> {
  const found = new Set<string>();
  for (const product of products) {
    for (const [name, pattern] of GENE_PATTERNS) {
      if (pattern.test(product)) found.add(name);
    }
  }
  return found;
}

function detectTargets(products: string[]): Set<string> {
  const found = new Set<string>();
  for (const [drug, patterns] of Object.entries(TARGET_PATTERNS)) {
    if (products.some((product) => patterns.some((pattern) => pattern.test(product)))) {
      found.add(drug);
    }
  }
  return found;
}

function parseAmrfinderTsv(text: string): string[] {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (!lines.length) return [];
  const header = lines[0]?.split("\t") ?? [];
  const wanted = [
    "Gene symbol", "Sequence name", "Element type", "Element subtype", "Class",
    "Subclass", "Method", "Name of closest sequence", "HMM description", "Mutation name"
  ];
  return lines.slice(1).map((line) => {
    const cells = line.split("\t");
    if (header.length < 2) return line;
    return wanted
      .map((name) => {
        const idx = header.indexOf(name);
        return idx >= 0 ? cells[idx] ?? "" : "";
      })
      .filter(Boolean)
      .join(" ");
  });
}

function fastaHeadersAndStats(text: string): { headers: string[]; bases: number; contigs: number; ambiguousBaseFraction: number } {
  const headers: string[] = [];
  let bases = 0;
  let ambiguous = 0;
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith(">")) {
      headers.push(trimmed.slice(1).trim());
      continue;
    }
    for (const char of trimmed.toUpperCase()) {
      if ("ACGT".includes(char)) {
        bases += 1;
      } else if ("NRYSWKM".includes(char)) {
        bases += 1;
        ambiguous += 1;
      }
    }
  }
  return {
    headers,
    bases,
    contigs: headers.length || (bases ? 1 : 0),
    ambiguousBaseFraction: bases ? ambiguous / bases : 0
  };
}

async function fetchBvbrcProducts(genomeId: string): Promise<string[]> {
  const terms = [
    "resistance", "lactamase", "efflux", "aminoglycoside", "integron", "gyrase",
    "topoisomerase", "quinolone", "ribosomal", "penicillin-binding", "tetracycline", "sulfonamide"
  ];
  const clauses = terms.map((term) => `eq(product,*${encodeURIComponent(term)}*)`).join(",");
  const url = `https://www.bv-brc.org/api/genome_feature/?and(eq(genome_id,${encodeURIComponent(genomeId)}),or(${clauses}))&select(gene,product)&limit=2000&http_accept=application/json`;
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error("BV-BRC request failed");
  const rows = (await res.json()) as Array<{ product?: string }>;
  return rows.map((row) => row.product ?? "").filter(Boolean);
}

async function loadKmerIndex(): Promise<KmerIndex | null> {
  kmerIndexPromise ??= fetchRelativeJson<KmerIndex>("static/kmer_index.json").catch(() => null);
  return kmerIndexPromise;
}

async function detectGenesFromDna(text: string): Promise<Set<string>> {
  const index = await loadKmerIndex();
  const detected = new Set<string>();
  if (!index) return detected;
  const query = genomeKmerSet(text, index.k);
  for (const [family, sketches] of Object.entries(index.families)) {
    let best = 0;
    for (const sketch of sketches) {
      if (!sketch.length) continue;
      let present = 0;
      for (const code of sketch) {
        if (query.has(String(code))) present += 1;
      }
      best = Math.max(best, present / sketch.length);
      if (best >= 0.99) break;
    }
    if (best >= CONTAINMENT_THRESHOLD) detected.add(family);
  }
  return detected;
}

function genomeKmerSet(text: string, k: number): Set<string> {
  const codes = new Set<string>();
  const mask = (1n << BigInt(2 * k)) - 1n;
  const top = BigInt(2 * (k - 1));
  for (const block of text.split(/^>.*$/m)) {
    const seq = block.replace(/\s+/g, "").toUpperCase();
    let fwd = 0n;
    let rev = 0n;
    let valid = 0;
    for (const base of seq) {
      const code = BASE_CODE[base];
      if (code === undefined) {
        fwd = 0n;
        rev = 0n;
        valid = 0;
        continue;
      }
      fwd = ((fwd << 2n) | code) & mask;
      rev = (rev >> 2n) | ((3n - code) << top);
      valid += 1;
      if (valid >= k) codes.add((fwd <= rev ? fwd : rev).toString());
    }
  }
  return codes;
}

function buildStaticPredictions(genes: Set<string>, targets: Set<string>, speciesSupported: boolean, config: AppConfig): AntibioticPrediction[] {
  const drugs = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"];
  return drugs.map((drug) => {
    const matched = [...(KNOWN_MARKERS[drug] ?? new Set<string>())].filter((marker) => genes.has(marker)).sort();
    const probability = staticProbability(drug, genes, matched);
    const drugConfig = config.antibiotics.find((item) => item.name.toLowerCase() === drug);
    const failThreshold = drugConfig?.fail_threshold ?? 0.72;
    const workThreshold = drugConfig?.work_threshold ?? 0.28;
    const targetOk = targets.has(drug) || speciesSupported;
    const reasonCodes: string[] = [];
    let decision: "likely_to_fail" | "likely_to_work" | "no_call";

    if (!genes.size) {
      decision = "no_call";
      reasonCodes.push("no_marker_features_detected");
    } else if (probability >= failThreshold) {
      decision = "likely_to_fail";
      reasonCodes.push("resistance_probability_above_fail_threshold");
    } else if (probability <= workThreshold && targetOk) {
      decision = "likely_to_work";
      reasonCodes.push("resistance_probability_below_work_threshold");
    } else {
      decision = "no_call";
      reasonCodes.push("probability_in_no_call_region");
    }

    if (decision === "likely_to_work" && !targetOk) {
      decision = "no_call";
      reasonCodes.push("target_not_confirmed");
    }

    const confidence = decision === "no_call" ? null : Number(Math.max(probability, 1 - probability).toFixed(3));
    const evidence = matched.length
      ? "known resistance gene or mutation detected"
      : decision === "no_call"
        ? "weak or uncertain statistical evidence"
        : "no known resistance signal found and target gate passed";

    return {
      antibiotic: titleCase(drug),
      decision,
      resistance_probability: Number(probability.toFixed(3)),
      calibrated_confidence: confidence,
      evidence_category: evidence,
      supporting_markers: matched,
      target_status: targetOk ? "present_by_supported_e_coli_scope" : "target_unknown",
      ood_status: genes.size ? "in_distribution" : "no_markers_detected",
      reason_codes: reasonCodes,
      explanation: explain(drug, decision, probability, matched),
      lab_confirmation_required: true,
      drug_class: DRUG_CLASS[drug],
      target: DRUG_TARGET[drug]
    };
  });
}

function staticProbability(drug: string, genes: Set<string>, matched: string[]): number {
  if (!genes.size) return 0.5;
  if (matched.length) {
    const strongBoost: Record<string, string[]> = {
      ceftriaxone: ["blaCTX-M", "blaCMY"],
      ampicillin: ["blaTEM", "blaSHV", "blaCTX-M", "blaCMY"],
      tetracycline: ["tetA", "tetB", "tetM"],
      ciprofloxacin: ["qnrA", "qnrB", "qnrS", "qnrD", "aac6Ib", "oqxA", "oqxB"]
    };
    const strong = matched.some((marker) => strongBoost[drug]?.includes(marker));
    return Math.min(0.97, (strong ? 0.82 : 0.74) + matched.length * 0.045);
  }

  const burden = Math.min(genes.size, 8);
  const broadResistance = ["integraseI", "qacE", "sul1", "acrAB", "marA"].some((marker) => genes.has(marker));
  const baseline = STATIC_BASELINE[drug] ?? 0.2;
  const association = baseline + burden * 0.018 + (broadResistance ? 0.08 : 0);
  return Math.min(0.62, Math.max(0.06, association));
}

function explain(drug: string, decision: string, probability: number, matched: string[]): string {
  const name = titleCase(drug);
  const pct = `${Math.round(probability * 100)}%`;
  if (decision === "likely_to_fail") {
    const markerText = matched.length ? `; markers ${matched.join(", ")}` : "";
    return `${name}: likely to fail (resistance probability ${pct}${markerText}).`;
  }
  if (decision === "likely_to_work") {
    return `${name}: likely to work (resistance probability ${pct}, no strong resistance signal).`;
  }
  return `${name}: no-call (resistance probability ${pct} in the uncertain band or no marker evidence).`;
}

function buildMarkers(genes: Set<string>): Marker[] {
  return [...genes].sort().map((gene) => {
    const drug = Object.entries(KNOWN_MARKERS).find(([, markers]) => markers.has(gene))?.[0] ?? "";
    return {
      symbol: gene,
      element_type: "AMR",
      drug_class: DRUG_CLASS[drug] ?? "resistance",
      evidence_level: "static_reference_kmer_or_annotation",
      identity: null,
      coverage: null
    };
  });
}

async function sha256Short(input: string): Promise<string> {
  try {
    const data = new TextEncoder().encode(input);
    const hash = await crypto.subtle.digest("SHA-256", data);
    return [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, "0")).join("").slice(0, 16);
  } catch {
    let hash = 0;
    for (let i = 0; i < input.length; i += 1) hash = ((hash << 5) - hash + input.charCodeAt(i)) | 0;
    return Math.abs(hash).toString(16).padStart(8, "0");
  }
}
