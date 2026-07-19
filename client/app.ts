import type { AnalysisResult, AppConfig } from "../shared/types.js";

interface MetricRow {
  antibiotic: string; auroc: number | null; pr_auc: number | null;
  balanced_accuracy: number | null; resistant_recall: number | null;
  susceptible_recall: number | null; brier: number | null;
  no_call_rate: number | null; answered_accuracy: number | null;
}
interface GenRow {
  antibiotic: string; overall_auroc: number | null;
  group_mean_auroc: number | null; group_min_auroc: number | null; n_groups: number;
}
interface RelPoint { mean_confidence: number; observed_accuracy: number; count: number; }
interface MetricsPayload {
  metrics: MetricRow[];
  reliability: Record<string, RelPoint[]>;
  reliability_points: RelPoint[];
  generalization: GenRow[];
}
interface BvbrcDashboard {
  manifest: Record<string, unknown>;
  summary: Array<Record<string, string>>;
  sampleRows: Array<Record<string, string>>;
}

const state: { config?: AppConfig; result?: AnalysisResult; selectedFile?: { fileName: string; content: string }; mode: "fasta" | "bvbrc" } = { mode: "fasta" };

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`Missing element ${sel}`);
  return el;
};

const decisionLabel = (d: string): string => d.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
const decisionKind = (d: string): string => (d === "likely_to_fail" ? "fail" : d === "likely_to_work" ? "work" : "nocall");
const pct = (v: number | null | undefined): string => (v === null || v === undefined ? "NA" : `${Math.round(v * 100)}%`);
const fixed = (v: unknown): string => (typeof v === "number" ? v.toFixed(3) : String(v ?? "NA"));

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  const data = (await res.json()) as T | { error: string };
  const hasErr = typeof data === "object" && data !== null && "error" in data;
  if (!res.ok) throw new Error(hasErr ? String((data as { error: string }).error) : "Request failed");
  return data as T;
}

function setStatus(message: string, isError = false): void {
  const s = $("#runStatus");
  s.textContent = message;
  s.classList.toggle("error", isError);
}

async function init(): Promise<void> {
  state.config = await fetchJson<AppConfig>("/api/config");
  const species = $("#species") as HTMLSelectElement;
  species.innerHTML = `<option>${state.config.supported_species}</option><option>Unsupported species</option>`;
  $("#speciesScope").textContent = state.config.supported_species;
  $("#modelVersion").textContent = state.config.model_version;
  $("#confirmation").textContent = state.config.safety.lab_confirmation_message;
  $("#detectorStatus").textContent = "k-mer detector + 4 LightGBM models";
  $("#drugList").innerHTML = state.config.antibiotics
    .map((d) => `<li>${d.name}<span>${d.target}</span></li>`)
    .join("");
  bindEvents();
  await renderBvbrc();
  await renderMetrics();
}

function bindEvents(): void {
  // tab navigation
  document.querySelectorAll<HTMLElement>("[data-nav]").forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.nav;
      document.querySelectorAll<HTMLElement>("[data-panel]").forEach((p) => p.classList.toggle("active", p.dataset.panel === target));
      document.querySelectorAll<HTMLElement>("[data-nav]").forEach((t) => t.classList.toggle("active", t === tab));
    });
  });

  // mode segment
  document.querySelectorAll<HTMLElement>("[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode as "fasta" | "bvbrc";
      document.querySelectorAll<HTMLElement>("[data-mode]").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll<HTMLElement>("[data-modepanel]").forEach((p) => (p.hidden = p.dataset.modepanel !== state.mode));
    });
  });

  const fileInput = $("#fastaFile") as HTMLInputElement;
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    state.selectedFile = { fileName: file.name, content: await file.text() };
    $("#fileStatus").textContent = `${file.name} loaded`;
  });

  document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sample = btn.dataset.sample;
      if (!sample) return;
      try {
        const payload = await fetchJson<{ fileName: string; content: string }>(`/api/demo-sample/${sample}`);
        state.selectedFile = payload;
        $("#fileStatus").textContent = `${payload.fileName} loaded`;
        ($("#sequencePreview") as HTMLTextAreaElement).value = payload.content;
      } catch (e) {
        setStatus(e instanceof Error ? e.message : "Sample load failed", true);
      }
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-gid]").forEach((btn) => {
    btn.addEventListener("click", () => {
      ($("#genomeId") as HTMLInputElement).value = btn.dataset.gid ?? "";
    });
  });

  $("#analyseButton").addEventListener("click", runFasta);
  $("#analyseBvbrcButton").addEventListener("click", runBvbrc);
}

async function runFasta(): Promise<void> {
  const preview = ($("#sequencePreview") as HTMLTextAreaElement).value.trim();
  const selected = state.selectedFile ?? (preview ? { fileName: "manual_input.fasta", content: preview } : undefined);
  if (!selected) {
    setStatus("Load a FASTA file, choose a demo sample, or paste sequence text.", true);
    return;
  }
  setStatus("Scanning genome → k-mer detect → LightGBM → target gate…");
  try {
    const species = ($("#species") as HTMLSelectElement).value;
    const result = await fetchJson<AnalysisResult>("/api/analyse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ species, fileName: selected.fileName, content: selected.content })
    });
    showResult(result);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : "Analysis failed.", true);
  }
}

async function runBvbrc(): Promise<void> {
  const gid = ($("#genomeId") as HTMLInputElement).value.trim();
  if (!/^\d+\.\d+$/.test(gid)) {
    setStatus("Enter a BV-BRC genome id like 562.12960.", true);
    return;
  }
  setStatus(`Fetching BV-BRC ${gid} → real model inference…`);
  try {
    const result = await fetchJson<AnalysisResult>("/api/analyse-bvbrc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ genome_id: gid })
    });
    showResult(result);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : "BV-BRC analysis failed.", true);
  }
}

function showResult(result: AnalysisResult): void {
  state.result = result;
  renderResult(result);
  setStatus("Analysis complete. Firewall matrix updated.");
  document.querySelector<HTMLElement>('[data-nav="results"]')?.click();
}

function renderResult(result: AnalysisResult): void {
  const preds = result.predictions;
  $("#kpiSource").textContent = result.source ?? "--";
  $("#kpiGenes").textContent = String(result.detected_genes?.length ?? result.markers.length);
  $("#failCount").textContent = String(preds.filter((p) => p.decision === "likely_to_fail").length);
  $("#workCount").textContent = String(preds.filter((p) => p.decision === "likely_to_work").length);
  $("#noCallCount").textContent = String(preds.filter((p) => p.decision === "no_call").length);

  // probability bar chart
  $("#barChart").innerHTML = preds
    .map((p) => {
      const prob = p.resistance_probability ?? 0;
      const kind = decisionKind(p.decision);
      return `
      <div class="bar-row">
        <div class="name">${p.antibiotic}<br><span class="mono">${p.drug_class ?? ""}</span></div>
        <div class="bar-track">
          <div class="bar-fill ${kind}" style="width:${Math.round(prob * 100)}%"></div>
          <div class="bar-thresh" style="left:28%"></div>
          <div class="bar-thresh" style="left:72%"></div>
        </div>
        <div class="bar-meta"><span class="tag ${kind}">${decisionLabel(p.decision)}</span><br><span class="mono">P(res) ${pct(p.resistance_probability)}</span></div>
      </div>`;
    })
    .join("");

  // evidence deck
  $("#evidenceDeck").innerHTML = preds
    .map((p) => {
      const kind = decisionKind(p.decision);
      return `
      <article class="ev ${kind}">
        <span class="tag ${kind}">${decisionLabel(p.decision)}</span>
        <h3>${p.antibiotic}</h3>
        <div class="sub">${p.target ?? ""}</div>
        <p>${p.explanation}</p>
        <dl>
          <dt>Confidence</dt><dd>${pct(p.calibrated_confidence)}</dd>
          <dt>Markers</dt><dd>${p.supporting_markers.join(", ") || "none"}</dd>
          <dt>Target gate</dt><dd>${p.target_status}</dd>
          <dt>Evidence</dt><dd>${p.evidence_category}</dd>
        </dl>
      </article>`;
    })
    .join("");

  // detected gene chips
  const genes = result.detected_genes ?? result.markers.map((m) => m.symbol);
  $("#geneChips").innerHTML = genes.length
    ? genes.map((g) => `<span class="chip">${g}</span>`).join("")
    : `<span class="chip none">No acquired resistance genes detected</span>`;

  ($("#downloadJson") as HTMLButtonElement).onclick = () =>
    download(`${result.run_id}.json`, JSON.stringify(result, null, 2), "application/json");
  ($("#downloadCsv") as HTMLButtonElement).onclick = () => download(`${result.run_id}.csv`, toCsv(result), "text/csv");
}

async function renderMetrics(): Promise<void> {
  let payload: MetricsPayload;
  try {
    payload = await fetchJson<MetricsPayload>("/api/metrics");
  } catch {
    return;
  }

  $("#metricKpis").innerHTML = payload.metrics
    .map((m) => `<div class="kpi"><span>${m.antibiotic} AUROC</span><strong>${fixed(m.auroc)}</strong></div>`)
    .join("");

  $("#metricsTable").innerHTML = payload.metrics
    .map((m) => `
      <tr>
        <td>${m.antibiotic}</td>
        <td>${fixed(m.auroc)}</td>
        <td>${fixed(m.pr_auc)}</td>
        <td>${fixed(m.balanced_accuracy)}</td>
        <td>${fixed(m.resistant_recall)}</td>
        <td>${fixed(m.susceptible_recall)}</td>
        <td>${fixed(m.brier)}</td>
        <td>${pct(m.no_call_rate)}</td>
      </tr>`)
    .join("");

  // generalization: overall vs within-group AUROC
  $("#genChart").innerHTML = payload.generalization
    .map((g) => {
      const overall = g.overall_auroc ?? 0;
      const mean = g.group_mean_auroc ?? 0;
      const gap = overall - mean > 0.15 ? "fail" : "work";
      return `
      <div class="bar-row">
        <div class="name">${g.antibiotic}<br><span class="mono">${g.n_groups} groups</span></div>
        <div class="bar-track">
          <div class="bar-fill work" style="width:${Math.round(overall * 100)}%"></div>
          <div class="bar-thresh" style="left:${Math.round(mean * 100)}%"></div>
          <div class="bar-thresh" style="left:50%"></div>
        </div>
        <div class="bar-meta"><span class="mono">overall ${fixed(g.overall_auroc)}</span><br><span class="tag ${gap}">group ${fixed(g.group_mean_auroc)}</span></div>
      </div>`;
    })
    .join("");

  drawReliability(payload.reliability["ampicillin"] ?? payload.reliability_points ?? []);
}

function drawReliability(points: RelPoint[]): void {
  const poly = points.map((p) => `${50 + p.mean_confidence * 420},${260 - p.observed_accuracy * 220}`).join(" ");
  $("#reliabilityPolyline").setAttribute("points", poly);
  const svg = $("#relSvg");
  svg.querySelectorAll("circle").forEach((c) => c.remove());
  const ns = "http://www.w3.org/2000/svg";
  for (const p of points) {
    const c = document.createElementNS(ns, "circle");
    c.setAttribute("cx", String(50 + p.mean_confidence * 420));
    c.setAttribute("cy", String(260 - p.observed_accuracy * 220));
    c.setAttribute("r", "3.5");
    svg.appendChild(c);
  }
}

async function renderBvbrc(): Promise<void> {
  try {
    const payload = await fetchJson<BvbrcDashboard>("/api/bvbrc/training-dashboard");
    $("#bvRows").textContent = numFmt(payload.manifest.amr_rows);
    $("#bvGenomes").textContent = numFmt(payload.manifest.unique_genomes);
    $("#bvTaxon").textContent = String(payload.manifest.taxon_id ?? "--");
    $("#bvEvidence").textContent = String(payload.manifest.evidence_filter ?? "--");
    $("#bvGenerated").textContent = `generated ${String(payload.manifest.generated_at ?? "")}`;
    $("#bvSummaryTable").innerHTML = payload.summary
      .map((r) => `<tr><td>${r.antibiotic}</td><td>${r.resistant}</td><td>${r.susceptible}</td><td>${r.total}</td><td>${r.unique_genomes}</td></tr>`)
      .join("");
    $("#bvSampleTable").innerHTML = payload.sampleRows
      .map((r) => `<tr><td>${r.genome_id}<br><small>${r.genome_name ?? ""}</small></td><td>${r.antibiotic}</td><td><span class="badge ${r.label === "resistant" ? "danger" : "success"}">${r.label}</span></td><td>${r.genome_quality || "NA"}</td><td>${r.genetic_group || r.genome_id}</td><td>${r.laboratory_typing_method || "NA"}</td></tr>`)
      .join("");
  } catch (e) {
    $("#bvGenerated").textContent = e instanceof Error ? e.message : "BV-BRC training data unavailable.";
  }
}

function numFmt(v: unknown): string {
  const n = Number(v);
  return Number.isFinite(n) ? n.toLocaleString() : String(v ?? "--");
}

function toCsv(result: AnalysisResult): string {
  const header = ["antibiotic", "decision", "confidence", "resistance_probability", "evidence", "markers", "reason_codes"];
  const rows = result.predictions.map((p) => [
    p.antibiotic, p.decision, String(p.calibrated_confidence), String(p.resistance_probability),
    p.evidence_category, p.supporting_markers.join("|"), p.reason_codes.join("|")
  ]);
  return [header, ...rows].map((r) => r.map((c) => `"${c.replaceAll('"', '""')}"`).join(",")).join("\n");
}

function download(fileName: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
}

init().catch((e: unknown) => setStatus(e instanceof Error ? e.message : "Interface failed to initialize.", true));
