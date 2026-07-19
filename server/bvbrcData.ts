import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { rootDir } from "./config.js";

type CsvRow = Record<string, string>;

const dataDir = join(rootDir, "data", "bvbrc");

function parseCsv(content: string): CsvRow[] {
  const lines = content.trim().split(/\r?\n/);
  const headerLine = lines.shift();
  if (!headerLine) return [];
  const headers = parseCsvLine(headerLine);

  return lines.map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === '"' && next === '"') {
      current += '"';
      index += 1;
      continue;
    }

    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
      continue;
    }

    current += char;
  }

  values.push(current);
  return values;
}

export async function loadBvbrcTrainingDashboard(): Promise<{
  manifest: Record<string, unknown>;
  summary: CsvRow[];
  sampleRows: CsvRow[];
}> {
  const [manifestRaw, summaryRaw, datasetRaw] = await Promise.all([
    readFile(join(dataDir, "manifest.json"), "utf8"),
    readFile(join(dataDir, "summary_by_antibiotic.csv"), "utf8"),
    readFile(join(dataDir, "training_dataset.csv"), "utf8")
  ]);

  return {
    manifest: JSON.parse(manifestRaw) as Record<string, unknown>,
    summary: parseCsv(summaryRaw),
    sampleRows: parseCsv(datasetRaw).slice(0, 12)
  };
}
