import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadConfig, rootDir } from "../server/config.js";

type Primitive = string | number | boolean | null | undefined;
type Row = Record<string, Primitive>;

interface FetchOptions {
  taxonId: string;
  maxRows: number;
  pageSize: number;
  antibiotics: string[] | "all";
  outDir: string;
}

const apiBase = "https://www.bv-brc.org/api";

const amrFields = [
  "genome_id",
  "genome_name",
  "taxon_id",
  "antibiotic",
  "resistant_phenotype",
  "evidence",
  "measurement",
  "measurement_value",
  "measurement_unit",
  "testing_standard",
  "testing_standard_year",
  "laboratory_typing_method",
  "laboratory_typing_platform",
  "source"
];

const genomeFields = [
  "genome_id",
  "genome_name",
  "species",
  "taxon_id",
  "genome_length",
  "contigs",
  "genome_quality",
  "genome_quality_flags",
  "assembly_accession",
  "cgmlst_hc10",
  "cgmlst_hc50",
  "cgmlst_hc100",
  "collection_year",
  "isolation_country",
  "host_common_name",
  "isolation_source"
];

function parseArgs(): FetchOptions {
  const args = new Map<string, string>();
  for (let index = 2; index < process.argv.length; index += 2) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
    if (key?.startsWith("--") && value) args.set(key.slice(2), value);
  }

  return {
    taxonId: args.get("taxon-id") ?? "562",
    maxRows: Number(args.get("max-rows") ?? args.get("limit") ?? "2000"),
    pageSize: Number(args.get("page-size") ?? "5000"),
    antibiotics: parseAntibiotics(args.get("antibiotics")),
    outDir: args.get("out-dir") ?? join(rootDir, "data", "bvbrc")
  };
}

function parseAntibiotics(value: string | undefined): string[] | "all" {
  if (!value) {
    return loadConfig().antibiotics.map((antibiotic) => antibiotic.name.toLowerCase());
  }
  if (value.toLowerCase() === "all") return "all";
  return value
    .split(",")
    .map((antibiotic) => antibiotic.trim().toLowerCase())
    .filter(Boolean);
}

function rqlValue(value: string): string {
  return encodeURIComponent(value);
}

function buildQuery(parts: string[]): string {
  return parts.join("&");
}

async function bvbrcQuery<T extends Row>(dataType: string, query: string): Promise<T[]> {
  const url = `${apiBase}/${dataType}/?${query}&http_accept=application/json`;
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`BV-BRC ${dataType} query failed with ${response.status}: ${await response.text()}`);
  }
  return await response.json() as T[];
}

async function bvbrcCursorQuery<T extends Row>(
  dataType: string,
  baseParts: string[],
  maxRows: number,
  pageSize: number
): Promise<T[]> {
  const rows: T[] = [];
  let cursor = "*";

  while (rows.length < maxRows) {
    const remaining = maxRows - rows.length;
    const currentPageSize = Math.min(pageSize, remaining);
    const query = buildQuery([
      ...baseParts,
      `limit(${currentPageSize})`,
      `cursor(${rqlValue(cursor)})`
    ]);
    const url = `${apiBase}/${dataType}/?${query}&http_accept=application/json`;
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new Error(`BV-BRC ${dataType} cursor query failed with ${response.status}: ${await response.text()}`);
    }

    const pageRows = await response.json() as T[];
    const nextCursor = response.headers.get("x-cursor-mark");
    rows.push(...pageRows);
    console.log(`Fetched ${rows.length.toLocaleString()} ${dataType} rows`);

    if (pageRows.length === 0 || !nextCursor || nextCursor === cursor) {
      break;
    }

    cursor = nextCursor;
  }

  return rows;
}

function csvEscape(value: Primitive): string {
  const normalized = Array.isArray(value) ? value.join(";") : String(value ?? "");
  return `"${normalized.replaceAll('"', '""')}"`;
}

function toCsv(rows: Row[], fields: string[]): string {
  return [
    fields.map(csvEscape).join(","),
    ...rows.map((row) => fields.map((field) => csvEscape(row[field])).join(","))
  ].join("\n");
}

function normalizeLabel(value: Primitive): "resistant" | "susceptible" | null {
  const text = String(value ?? "").toLowerCase();
  if (text === "resistant" || text === "non-susceptible") return "resistant";
  if (text === "susceptible") return "susceptible";
  return null;
}

function summarize(rows: Row[]): Row[] {
  const summary = new Map<string, { antibiotic: string; resistant: number; susceptible: number; total: number; unique_genomes: Set<string> }>();

  for (const row of rows) {
    const antibiotic = String(row.antibiotic ?? "");
    const label = normalizeLabel(row.resistant_phenotype);
    if (!antibiotic || !label) continue;

    const current = summary.get(antibiotic) ?? {
      antibiotic,
      resistant: 0,
      susceptible: 0,
      total: 0,
      unique_genomes: new Set<string>()
    };
    current[label] += 1;
    current.total += 1;
    current.unique_genomes.add(String(row.genome_id ?? ""));
    summary.set(antibiotic, current);
  }

  return [...summary.values()]
    .map((item) => ({
      antibiotic: item.antibiotic,
      resistant: item.resistant,
      susceptible: item.susceptible,
      total: item.total,
      unique_genomes: item.unique_genomes.size
    }))
    .sort((a, b) => Number(b.total) - Number(a.total));
}

function chunk<T>(items: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
}

async function fetchGenomeMetadata(genomeIds: string[]): Promise<Row[]> {
  const batches = chunk(genomeIds, 100);
  const rows: Row[] = [];

  for (const batch of batches) {
    const query = buildQuery([
      `in(genome_id,(${batch.map(rqlValue).join(",")}))`,
      `select(${genomeFields.join(",")})`,
      `limit(${batch.length})`
    ]);
    rows.push(...await bvbrcQuery<Row>("genome", query));
  }

  return rows;
}

async function main(): Promise<void> {
  const config = loadConfig();
  const options = parseArgs();
  const antibiotics = options.antibiotics;
  const antibioticFilter = antibiotics === "all"
    ? []
    : [`in(antibiotic,(${antibiotics.map(rqlValue).join(",")}))`];

  const amrQueryParts = [
    `eq(taxon_id,${rqlValue(options.taxonId)})`,
    `eq(evidence,${rqlValue("Laboratory Method")})`,
    ...antibioticFilter,
    `in(resistant_phenotype,(${["Resistant", "Susceptible"].map(rqlValue).join(",")}))`,
    `select(${amrFields.join(",")})`,
    "sort(+genome_id,+antibiotic)"
  ];

  const amrRows = await bvbrcCursorQuery<Row>("genome_amr", amrQueryParts, options.maxRows, options.pageSize);
  const genomeIds = [...new Set(amrRows.map((row) => String(row.genome_id)).filter(Boolean))].sort();
  const genomeRows = await fetchGenomeMetadata(genomeIds);
  const genomeById = new Map(genomeRows.map((row) => [String(row.genome_id), row]));

  const joinedRows = amrRows.map((row) => {
    const genome = genomeById.get(String(row.genome_id)) ?? {};
    return {
      ...row,
      label: normalizeLabel(row.resistant_phenotype),
      species: genome.species,
      genome_length: genome.genome_length,
      contigs: genome.contigs,
      genome_quality: genome.genome_quality,
      genome_quality_flags: genome.genome_quality_flags,
      assembly_accession: genome.assembly_accession,
      genetic_group: genome.cgmlst_hc100 ?? genome.cgmlst_hc50 ?? genome.cgmlst_hc10 ?? row.genome_id,
      cgmlst_hc10: genome.cgmlst_hc10,
      cgmlst_hc50: genome.cgmlst_hc50,
      cgmlst_hc100: genome.cgmlst_hc100,
      collection_year: genome.collection_year,
      isolation_country: genome.isolation_country,
      host_common_name: genome.host_common_name,
      isolation_source: genome.isolation_source
    };
  });

  const summaryRows = summarize(joinedRows);
  const datasetFields = [
    ...amrFields,
    "label",
    "species",
    "genome_length",
    "contigs",
    "genome_quality",
    "genome_quality_flags",
    "assembly_accession",
    "genetic_group",
    "cgmlst_hc10",
    "cgmlst_hc50",
    "cgmlst_hc100",
    "collection_year",
    "isolation_country",
    "host_common_name",
    "isolation_source"
  ];

  await mkdir(options.outDir, { recursive: true });
  await writeFile(join(options.outDir, "amr_labels_raw.json"), JSON.stringify(amrRows, null, 2));
  await writeFile(join(options.outDir, "genome_metadata.json"), JSON.stringify(genomeRows, null, 2));
  await writeFile(join(options.outDir, "training_dataset.json"), JSON.stringify(joinedRows, null, 2));
  await writeFile(join(options.outDir, "training_dataset.csv"), toCsv(joinedRows, datasetFields));
  await writeFile(join(options.outDir, "summary_by_antibiotic.csv"), toCsv(summaryRows, ["antibiotic", "resistant", "susceptible", "total", "unique_genomes"]));
  await writeFile(join(options.outDir, "manifest.json"), JSON.stringify({
    source: "BV-BRC public Data API",
    api_base: apiBase,
    taxon_id: options.taxonId,
    supported_species: config.supported_species,
    requested_antibiotics: antibiotics,
    evidence_filter: "Laboratory Method",
    phenotype_filter: ["Resistant", "Susceptible"],
    max_rows: options.maxRows,
    page_size: options.pageSize,
    amr_rows: amrRows.length,
    unique_genomes: genomeIds.length,
    generated_at: new Date().toISOString()
  }, null, 2));

  console.table(summaryRows);
  console.log(`Saved BV-BRC training data to ${options.outDir}`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
