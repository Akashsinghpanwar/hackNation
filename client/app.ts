import type { AnalysisResult, AppConfig } from "../shared/types.js";

const state: { config?: AppConfig; result?: AnalysisResult; selectedFile?: { fileName: string; content: string } } = {};

interface BvbrcTrainingDashboard {
  manifest: Record<string, unknown>;
  summary: Array<Record<string, string>>;
  sampleRows: Array<Record<string, string>>;
}

const $ = <T extends HTMLElement>(selector: string): T => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element ${selector}`);
  return element;
};

function decisionLabel(decision: string): string {
  return decision.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function decisionClass(decision: string): string {
  if (decision === "likely_to_fail") return "danger";
  if (decision === "likely_to_work") return "success";
  return "warn";
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  const data = await response.json() as T | { error: string };
  const hasError = typeof data === "object" && data !== null && "error" in data;
  if (!response.ok) throw new Error(hasError ? String(data.error) : "Request failed");
  return data as T;
}

async function init(): Promise<void> {
  state.config = await fetchJson<AppConfig>("/api/config");
  const speciesSelect = $("#species") as HTMLSelectElement;
  speciesSelect.innerHTML = `
    <option>${state.config.supported_species}</option>
    <option>Unsupported species</option>
  `;
  $("#speciesScope").textContent = state.config.supported_species;
  $("#modelVersion").textContent = state.config.model_version;
  $("#confirmation").textContent = state.config.safety.lab_confirmation_message;
  $("#drugList").innerHTML = state.config.antibiotics.map((drug) => `<li>${drug.name}<span>${drug.target}</span></li>`).join("");

  bindEvents();
  await renderBvbrcTrainingData();
  await renderMetrics();
}

function bindEvents(): void {
  const fileInput = $("#fastaFile") as HTMLInputElement;
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    state.selectedFile = { fileName: file.name, content: await file.text() };
    $("#fileStatus").textContent = `${file.name} loaded`;
  });

  document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((button) => {
    button.addEventListener("click", async () => {
      const sample = button.dataset.sample;
      if (!sample) return;
      const payload = await fetchJson<{ fileName: string; content: string }>(`/api/demo-sample/${sample}`);
      state.selectedFile = payload;
      $("#fileStatus").textContent = `${payload.fileName} loaded`;
      ($("#sequencePreview") as HTMLTextAreaElement).value = payload.content;
    });
  });

  $("#analyseButton").addEventListener("click", runAnalysis);

  document.querySelectorAll<HTMLElement>("[data-nav]").forEach((link) => {
    link.addEventListener("click", () => {
      const target = link.dataset.nav;
      if (!target) return;
      document.querySelectorAll<HTMLElement>("[data-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === target));
      document.querySelectorAll<HTMLElement>("[data-nav]").forEach((nav) => nav.classList.toggle("active", nav.dataset.nav === target));
    });
  });
}

async function runAnalysis(): Promise<void> {
  const preview = ($("#sequencePreview") as HTMLTextAreaElement).value.trim();
  const selected = state.selectedFile ?? (preview ? { fileName: "manual_input.fasta", content: preview } : undefined);
  if (!selected) {
    setStatus("Load a FASTA file, choose a demo sample, or paste sequence text.", true);
    return;
  }

  setStatus("Running FASTA guard -> marker scan -> model lane -> Safety Governor", false);
  try {
    const species = ($("#species") as HTMLSelectElement).value;
    const result = await fetchJson<AnalysisResult>("/api/analyse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ species, fileName: selected.fileName, content: selected.content })
    });
    state.result = result;
    renderResult(result);
    setStatus("Analysis complete. Firewall matrix updated.", false);
    document.querySelector<HTMLElement>('[data-nav="results"]')?.click();
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Analysis failed.", true);
  }
}

function setStatus(message: string, isError: boolean): void {
  const status = $("#runStatus");
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function renderResult(result: AnalysisResult): void {
  $("#runId").textContent = result.run_id;
  $("#qcStatus").textContent = result.qc.qc_status.toUpperCase();
  $("#markerCount").textContent = String(result.markers.length);
  $("#failCount").textContent = String(result.predictions.filter((prediction) => prediction.decision === "likely_to_fail").length);
  $("#workCount").textContent = String(result.predictions.filter((prediction) => prediction.decision === "likely_to_work").length);
  $("#noCallCount").textContent = String(result.predictions.filter((prediction) => prediction.decision === "no_call").length);

  $("#resultTable").innerHTML = result.predictions.map((prediction) => `
    <tr>
      <td>${prediction.antibiotic}</td>
      <td><span class="badge ${decisionClass(prediction.decision)}">${decisionLabel(prediction.decision)}</span></td>
      <td>${prediction.calibrated_confidence === null ? "NA" : `${Math.round(prediction.calibrated_confidence * 100)}%`}</td>
      <td>${prediction.evidence_category}</td>
      <td>${prediction.supporting_markers.join(", ") || "None"}</td>
      <td>${prediction.target_status}</td>
    </tr>
  `).join("");

  $("#evidenceDeck").innerHTML = result.predictions.map((prediction) => `
    <article class="evidence-card">
      <div>
        <h3>${prediction.antibiotic}</h3>
        <span class="badge ${decisionClass(prediction.decision)}">${decisionLabel(prediction.decision)}</span>
      </div>
      <p>${prediction.explanation}</p>
      <dl>
        <dt>Resistance probability</dt><dd>${prediction.resistance_probability}</dd>
        <dt>OOD status</dt><dd>${prediction.ood_status}</dd>
        <dt>Reason codes</dt><dd>${prediction.reason_codes.join(", ")}</dd>
      </dl>
    </article>
  `).join("");

  $("#downloadJson").onclick = () => downloadFile(`${result.run_id}.json`, JSON.stringify(result, null, 2), "application/json");
  $("#downloadCsv").onclick = () => downloadFile(`${result.run_id}.csv`, toCsv(result), "text/csv");
}

async function renderMetrics(): Promise<void> {
  const payload = await fetchJson<{ metrics: Array<Record<string, number | string>>; reliability_points: Array<Record<string, number | string>> }>("/api/metrics");
  $("#metricsTable").innerHTML = payload.metrics.map((row) => `
    <tr>
      <td>${row.antibiotic}</td>
      <td>${formatMetric(row.balanced_accuracy)}</td>
      <td>${formatMetric(row.resistant_recall)}</td>
      <td>${formatMetric(row.susceptible_recall)}</td>
      <td>${formatMetric(row.pr_auc)}</td>
      <td>${formatMetric(row.brier)}</td>
      <td>${formatMetric(row.no_call_rate)}</td>
    </tr>
  `).join("");

  const points = payload.reliability_points;
  const polyline = points.map((point) => {
    const x = 50 + Number(point.mean_confidence) * 420;
    const y = 260 - Number(point.observed_accuracy) * 220;
    return `${x},${y}`;
  }).join(" ");
  $("#reliabilityPolyline").setAttribute("points", polyline);
}

async function renderBvbrcTrainingData(): Promise<void> {
  try {
    const payload = await fetchJson<BvbrcTrainingDashboard>("/api/bvbrc/training-dashboard");
    $("#bvRows").textContent = formatNumber(payload.manifest.amr_rows);
    $("#bvGenomes").textContent = formatNumber(payload.manifest.unique_genomes);
    $("#bvTaxon").textContent = String(payload.manifest.taxon_id ?? "--");
    $("#bvEvidence").textContent = String(payload.manifest.evidence_filter ?? "--");
    $("#bvGenerated").textContent = `Generated ${String(payload.manifest.generated_at ?? "")}`;

    $("#bvSummaryTable").innerHTML = payload.summary.map((row) => `
      <tr>
        <td>${row.antibiotic}</td>
        <td>${row.resistant}</td>
        <td>${row.susceptible}</td>
        <td>${row.total}</td>
        <td>${row.unique_genomes}</td>
      </tr>
    `).join("");

    $("#bvSampleTable").innerHTML = payload.sampleRows.map((row) => `
      <tr>
        <td>${row.genome_id}<br><small>${row.genome_name}</small></td>
        <td>${row.antibiotic}</td>
        <td><span class="badge ${row.label === "resistant" ? "danger" : "success"}">${row.label}</span></td>
        <td>${row.genome_quality || "NA"}</td>
        <td>${row.genetic_group || row.genome_id}</td>
        <td>${row.laboratory_typing_method || "NA"}</td>
      </tr>
    `).join("");
  } catch (error) {
    $("#bvGenerated").textContent = error instanceof Error ? error.message : "BV-BRC training data unavailable.";
    $("#bvSummaryTable").innerHTML = "";
    $("#bvSampleTable").innerHTML = "";
  }
}

function formatMetric(value: unknown): string {
  return typeof value === "number" ? value.toFixed(2) : String(value);
}

function formatNumber(value: unknown): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toLocaleString() : String(value ?? "--");
}

function toCsv(result: AnalysisResult): string {
  const header = ["antibiotic", "decision", "confidence", "resistance_probability", "evidence", "markers", "reason_codes"];
  const rows = result.predictions.map((prediction) => [
    prediction.antibiotic,
    prediction.decision,
    String(prediction.calibrated_confidence),
    String(prediction.resistance_probability),
    prediction.evidence_category,
    prediction.supporting_markers.join("|"),
    prediction.reason_codes.join("|")
  ]);
  return [header, ...rows].map((row) => row.map((cell) => `"${cell.replaceAll('"', '""')}"`).join(",")).join("\n");
}

function downloadFile(fileName: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
}

init().catch((error: unknown) => {
  setStatus(error instanceof Error ? error.message : "Interface failed to initialize.", true);
});
