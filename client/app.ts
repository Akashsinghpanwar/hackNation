import type { AnalysisResult, AppConfig } from "../shared/types.js";
import { loadStaticConfig, loadStaticMetrics, loadStaticSample, runStaticAnalysis } from "./staticFallback.js";

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
type InputMode = "fasta" | "tsv" | "bvbrc";

const state: {
  config?: AppConfig;
  result?: AnalysisResult;
  selectedFile?: { fileName: string; content: string };
  selectedTsv?: { fileName: string; content: string };
  mode: InputMode;
  usingStaticApi: boolean;
} = { mode: "fasta", usingStaticApi: false };

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`Missing element ${sel}`);
  return el;
};

const decisionLabel = (d: string): string => d.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
const decisionKind = (d: string): string => (d === "likely_to_fail" ? "fail" : d === "likely_to_work" ? "work" : "nocall");
const pct = (v: number | null | undefined): string => (v === null || v === undefined ? "NA" : `${Math.round(v * 100)}%`);
const fixed = (v: unknown): string => (typeof v === "number" ? v.toFixed(3) : String(v ?? "NA"));
const num = (v: number | null | undefined): string => (typeof v === "number" ? v.toLocaleString() : "--");

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
  try {
    state.config = await fetchJson<AppConfig>("/api/config");
  } catch {
    state.usingStaticApi = true;
    state.config = await loadStaticConfig();
    $("#detectorStatus").textContent = "Static Pages demo: browser marker detector and rule-based confidence fallback";
  }
  const species = $("#species") as HTMLSelectElement;
  species.innerHTML = `<option>${state.config.supported_species}</option><option>Unsupported species</option>`;
  $("#speciesScope").textContent = state.config.supported_species;
  $("#modelVersion").textContent = state.config.model_version;
  $("#confirmation").textContent = state.config.safety.lab_confirmation_message;
  if (!state.usingStaticApi) {
    $("#detectorStatus").textContent = "Python inference bridge, k-mer detector, and 4 calibrated LightGBM models";
  }
  $("#drugList").innerHTML = state.config.antibiotics
    .map((d) => `<li>${d.name}<span>${d.target}</span></li>`)
    .join("");
  bindEvents();
  renderWorkflow();
  await renderMetrics();
  await checkAiStatus();
}

// ---------- Workflow animation ----------
interface WfStep { icon: string; title: string; what: string; why: string; }
const WF_STEPS: WfStep[] = [
  {
    icon: "🧬", title: "Genome Intake",
    what: "Accept a genome as annotated FASTA, an AMRFinderPlus TSV, or a live BV-BRC genome ID.",
    why: "Sequence is available days before a culture-based susceptibility result — that head start is the entire point of the tool."
  },
  {
    icon: "🧪", title: "Quality Control",
    what: "Count bases and contigs, measure the ambiguous-base fraction, and compute a SHA-256 fingerprint of the input.",
    why: "Low-quality input silently corrupts predictions; the hash makes every run reproducible and auditable."
  },
  {
    icon: "🔍", title: "Marker Detection",
    what: "Scan for known AMR genes using a MinHash 21-mer containment index against the NCBI AMR reference (or parse AMRFinder hits directly).",
    why: "Alignment-free k-mer matching finds resistance genes fast, even on large raw assemblies with no annotation."
  },
  {
    icon: "📊", title: "Feature Extraction",
    what: "Turn detected genes and gene families into the numeric feature vector each drug model expects.",
    why: "The models reason over structured presence/absence features — not raw text — so this step is the bridge from biology to ML."
  },
  {
    icon: "🌲", title: "LightGBM Inference",
    what: "Run four per-drug gradient-boosted models for ampicillin, ciprofloxacin, ceftriaxone, and tetracycline to estimate P(resistant).",
    why: "Learned gene-interaction patterns beat single-gene rules, especially for partial or novel marker combinations."
  },
  {
    icon: "🎯", title: "Calibration + Target Gate",
    what: "Pass raw scores through CalibratedClassifierCV, then verify the drug's molecular target is actually present.",
    why: "Calibrated probabilities can be read as real risk; the target gate blocks biologically implausible calls."
  },
  {
    icon: "⚖️", title: "Confidence Engine",
    what: "A pluggable engine (threshold or entropy) maps probability to Likely-to-fail / Likely-to-work / No-call.",
    why: "When evidence is weak or conflicting the system abstains instead of guessing — safety over coverage."
  },
  {
    icon: "🗣️", title: "Report + Multimodal",
    what: "Render the dashboard, then generate a GPT clinical summary and optional text-to-speech voice.",
    why: "A prediction only helps if a clinician or student can read, hear, and act on it — always with a lab-confirmation caveat."
  }
];

let wfTimer: number | null = null;
let wfIndex = -1;

function renderWorkflow(): void {
  const track = document.querySelector<HTMLElement>("#wfTrack");
  if (!track) return;
  track.innerHTML = WF_STEPS.map((s, i) => `
    ${i > 0 ? `<div class="wf-link" data-link="${i}"><span class="wf-pulse"></span></div>` : ""}
    <div class="wf-node" data-step="${i}">
      <div class="wf-dot">${s.icon}</div>
      <div class="wf-label">${s.title}</div>
    </div>`).join("");
}

function wfStepShow(i: number): void {
  wfIndex = i;
  document.querySelectorAll<HTMLElement>(".wf-node").forEach((n, idx) => {
    n.classList.toggle("active", idx === i);
    n.classList.toggle("done", idx < i);
  });
  document.querySelectorAll<HTMLElement>(".wf-link").forEach((l) => {
    l.classList.toggle("fill", Number(l.dataset.link) <= i);
  });
  const s = WF_STEPS[i];
  if (!s) return;
  $("#loaderCaption").textContent = `${s.title} — ${s.what}`;
  ($("#wfProgressFill") as HTMLElement).style.width = `${((i + 1) / WF_STEPS.length) * 100}%`;
}

let loaderStartedAt = 0;
let loaderTickTimer: number | null = null;

function startLoader(title: string): void {
  if (wfTimer !== null) window.clearTimeout(wfTimer);
  if (loaderTickTimer !== null) { window.clearInterval(loaderTickTimer); loaderTickTimer = null; }
  $("#loaderTitle").textContent = title;
  wfIndex = -1;
  document.querySelectorAll<HTMLElement>(".wf-node").forEach((n) => n.classList.remove("active", "done"));
  document.querySelectorAll<HTMLElement>(".wf-link").forEach((l) => l.classList.remove("fill"));
  ($("#wfProgressFill") as HTMLElement).style.width = "0%";
  $("#loaderCaption").textContent = "Starting...";
  ($("#loaderOverlay") as HTMLElement).hidden = false;
  document.body.style.overflow = "hidden";
  loaderStartedAt = Date.now();

  // The fixed step-by-step animation below only covers the first ~6s. Real
  // inference (especially the k-mer scan on a raw genome, on free-tier
  // hosting with throttled CPU) commonly runs 20-60s+. Previously the
  // progress bar hit 100% and the caption froze the instant the last
  // animated step showed, then sat completely still for the rest of the
  // real wait — indistinguishable from a hang. Cap the animated portion at
  // 92% and keep the caption ticking with elapsed time until finishLoader()
  // actually fires, so there's always visible, honest progress.
  const advance = (): void => {
    if (wfIndex >= WF_STEPS.length - 1) {
      wfTimer = null;
      startElapsedTicker();
      return;
    }
    wfStepShow(wfIndex + 1);
    ($("#wfProgressFill") as HTMLElement).style.width = `${Math.min(92, ((wfIndex + 1) / WF_STEPS.length) * 100)}%`;
    wfTimer = window.setTimeout(advance, 780);
  };
  advance();
}

function startElapsedTicker(): void {
  const tick = (): void => {
    const elapsed = Math.round((Date.now() - loaderStartedAt) / 1000);
    const note = elapsed >= 15 ? " (free-tier hosting can take up to a minute on a cold start)" : "";
    $("#loaderCaption").textContent = `Still running model inference — ${elapsed}s elapsed${note}`;
  };
  tick();
  loaderTickTimer = window.setInterval(tick, 1000);
}

function finishLoader(): void {
  if (wfTimer !== null) { window.clearTimeout(wfTimer); wfTimer = null; }
  if (loaderTickTimer !== null) { window.clearInterval(loaderTickTimer); loaderTickTimer = null; }
  document.querySelectorAll<HTMLElement>(".wf-node").forEach((n) => {
    n.classList.remove("active");
    n.classList.add("done");
  });
  document.querySelectorAll<HTMLElement>(".wf-link").forEach((l) => l.classList.add("fill"));
  ($("#wfProgressFill") as HTMLElement).style.width = "100%";
  const totalElapsed = Math.round((Date.now() - loaderStartedAt) / 1000);
  $("#loaderCaption").textContent = `Pipeline complete (${totalElapsed}s).`;
  window.setTimeout(() => {
    ($("#loaderOverlay") as HTMLElement).hidden = true;
    document.body.style.overflow = "";
  }, 550);
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
      state.mode = btn.dataset.mode as InputMode;
      document.querySelectorAll<HTMLElement>("[data-mode]").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll<HTMLElement>("[data-modepanel]").forEach((p) => (p.hidden = p.dataset.modepanel !== state.mode));
    });
  });

  const fileInput = $("#fastaFile") as HTMLInputElement;
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const content = await file.text();
    state.selectedFile = { fileName: file.name, content };
    $("#fileStatus").textContent = `${file.name} loaded`;
    ($("#sequencePreview") as HTMLTextAreaElement).value = content.slice(0, 100_000);
  });

  const tsvInput = $("#tsvFile") as HTMLInputElement;
  tsvInput.addEventListener("change", async () => {
    const file = tsvInput.files?.[0];
    if (!file) return;
    const content = await file.text();
    state.selectedTsv = { fileName: file.name, content };
    $("#tsvStatus").textContent = `${file.name} loaded`;
    ($("#tsvPreview") as HTMLTextAreaElement).value = content.slice(0, 100_000);
  });

  document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sample = btn.dataset.sample;
      if (!sample) return;
      try {
        const payload = await loadDemoSample(sample);
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
  $("#analyseTsvButton").addEventListener("click", runTsv);
  $("#analyseBvbrcButton").addEventListener("click", runBvbrc);
  $("#btnNarrative").addEventListener("click", runNarrative);
  $("#btnSpeak").addEventListener("click", runSpeak);
}

async function checkAiStatus(): Promise<void> {
  try {
    const { configured } = await fetchJson<{ configured: boolean }>("/api/ai-status");
    $("#aiStatus").textContent = configured ? "OpenAI connected" : "set OPENAI_API_KEY in .env";
    ($("#btnNarrative") as HTMLButtonElement).disabled = !configured;
    if (!configured) ($("#btnSpeak") as HTMLButtonElement).disabled = true;
  } catch {
    $("#aiStatus").textContent = "AI status unavailable";
  }
}

let lastNarrative = "";

async function runNarrative(): Promise<void> {
  if (!state.result) {
    setStatus("Run an analysis first, then generate the summary.", true);
    return;
  }
  const box = $("#aiNarrative");
  box.innerHTML = `<p class="hint">Generating clinical summary with GPT...</p>`;
  try {
    const { narrative } = await fetchJson<{ narrative: string }>("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result: state.result })
    });
    lastNarrative = narrative;
    const html = narrative
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
    box.innerHTML = html;
    ($("#btnSpeak") as HTMLButtonElement).disabled = false;
  } catch (e) {
    box.innerHTML = `<p class="hint error">${e instanceof Error ? e.message : "Summary failed."}</p>`;
  }
}

async function runSpeak(): Promise<void> {
  if (!lastNarrative) {
    setStatus("Generate the clinical summary first, then read it aloud.", true);
    return;
  }
  const btn = $("#btnSpeak") as HTMLButtonElement;
  const audio = $("#aiAudio") as HTMLAudioElement;
  btn.disabled = true;
  btn.textContent = "Synthesizing voice...";
  try {
    const { audio_b64 } = await fetchJson<{ audio_b64: string }>("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: lastNarrative })
    });
    audio.src = `data:audio/mp3;base64,${audio_b64}`;
    audio.hidden = false;
    await audio.play().catch(() => undefined);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : "Voice synthesis failed.", true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Read aloud";
  }
}

async function runFasta(): Promise<void> {
  const preview = ($("#sequencePreview") as HTMLTextAreaElement).value.trim();
  const loadedPreview = state.selectedFile?.content.slice(0, 100_000).trim();
  const selected = preview && preview !== loadedPreview
    ? { fileName: state.selectedFile?.fileName ?? "manual_input.fasta", content: preview }
    : state.selectedFile ?? (preview ? { fileName: "manual_input.fasta", content: preview } : undefined);
  if (!selected) {
    setStatus("Load a FASTA file, choose a demo sample, or paste sequence text.", true);
    return;
  }
  setStatus("Analyzing FASTA through feature extraction and calibrated models...");
  startLoader("Analyzing genome (FASTA)...");
  try {
    const species = ($("#species") as HTMLSelectElement).value;
    const result = await analyseWithFallback({
      mode: "fasta",
      species,
      fileName: selected.fileName,
      content: selected.content
    });
    showResult(result);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : "Analysis failed.", true);
  } finally {
    finishLoader();
  }
}

async function runTsv(): Promise<void> {
  const preview = ($("#tsvPreview") as HTMLTextAreaElement).value.trim();
  const loadedPreview = state.selectedTsv?.content.slice(0, 100_000).trim();
  const selected = preview && preview !== loadedPreview
    ? { fileName: state.selectedTsv?.fileName ?? "amrfinder_input.tsv", content: preview }
    : state.selectedTsv ?? (preview ? { fileName: "amrfinder_input.tsv", content: preview } : undefined);
  if (!selected) {
    setStatus("Load or paste an AMRFinderPlus TSV file.", true);
    return;
  }
  setStatus("Parsing AMRFinderPlus TSV and running calibrated models...");
  startLoader("Analyzing AMRFinder TSV...");
  try {
    const species = ($("#species") as HTMLSelectElement).value;
    const result = await analyseWithFallback({
      mode: "tsv",
      species,
      fileName: selected.fileName,
      content: selected.content
    });
    showResult(result);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : "TSV analysis failed.", true);
  } finally {
    finishLoader();
  }
}

async function runBvbrc(): Promise<void> {
  const gid = ($("#genomeId") as HTMLInputElement).value.trim();
  if (!/^\d+\.\d+$/.test(gid)) {
    setStatus("Enter a BV-BRC genome id like 562.12960.", true);
    return;
  }
  setStatus(`Fetching BV-BRC ${gid} and running model inference...`);
  startLoader(`Fetching BV-BRC ${gid}...`);
  try {
    const result = await analyseWithFallback({
      mode: "bvbrc",
      species: ($("#species") as HTMLSelectElement).value,
      genome_id: gid
    });
    showResult(result);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : "BV-BRC analysis failed.", true);
  } finally {
    finishLoader();
  }
}

async function loadDemoSample(sample: string): Promise<{ fileName: string; content: string }> {
  try {
    return await fetchJson<{ fileName: string; content: string }>(`/api/demo-sample/${sample}`);
  } catch {
    return loadStaticSample(sample);
  }
}

async function analyseWithFallback(request: {
  mode: InputMode;
  species: string;
  fileName?: string;
  content?: string;
  genome_id?: string;
}): Promise<AnalysisResult> {
  const config = state.config ?? await loadStaticConfig();
  if (request.mode === "bvbrc") {
    try {
      return await fetchJson<AnalysisResult>("/api/analyse-bvbrc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ species: request.species, genome_id: request.genome_id })
      });
    } catch (error) {
      if (!state.usingStaticApi) throw error;
      return runStaticAnalysis({ mode: "bvbrc", species: request.species, genome_id: request.genome_id }, config);
    }
  }

  try {
    return await fetchJson<AnalysisResult>("/api/analyse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    });
  } catch (error) {
    if (!state.usingStaticApi) throw error;
    return runStaticAnalysis({
      mode: request.mode === "tsv" ? "tsv" : "fasta",
      species: request.species,
      fileName: request.fileName,
      content: request.content
    }, config);
  }
}

function showResult(result: AnalysisResult): void {
  state.result = result;
  renderResult(result);
  setStatus("Analysis complete. Results updated.");
  document.querySelector<HTMLElement>('[data-nav="results"]')?.click();
}

function renderResult(result: AnalysisResult): void {
  const preds = result.predictions;
  $("#kpiSource").textContent = result.source ?? "--";
  $("#kpiGenes").textContent = String(result.detected_genes?.length ?? result.markers.length);
  $("#failCount").textContent = String(preds.filter((p) => p.decision === "likely_to_fail").length);
  $("#workCount").textContent = String(preds.filter((p) => p.decision === "likely_to_work").length);
  $("#noCallCount").textContent = String(preds.filter((p) => p.decision === "no_call").length);

  const qc = result.qc;
  $("#qcRunId").textContent = result.run_id || qc.genome_id || "--";
  $("#qcBases").textContent = num(qc.sequence_length);
  $("#qcContigs").textContent = num(qc.contig_count);
  $("#qcAmbiguous").textContent = pct(qc.ambiguous_base_fraction);
  $("#qcHash").textContent = qc.sha256 || "--";
  const qcStatus = $("#qcStatus");
  qcStatus.textContent = qc.qc_status;
  qcStatus.className = `status-badge ${qc.qc_status}`;
  const warnings = [...(result.warnings ?? []), ...(qc.warnings ?? [])];
  $("#warningList").innerHTML = warnings.length
    ? warnings.map((w) => `<div class="warning-item">${w}</div>`).join("")
    : `<div class="warning-item ok">No QC warnings returned.</div>`;

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

  $("#resultTable").innerHTML = preds
    .map((p) => {
      const kind = decisionKind(p.decision);
      return `
      <tr>
        <td><strong>${p.antibiotic}</strong><br><small>${p.drug_class ?? ""}</small></td>
        <td><span class="tag ${kind}">${decisionLabel(p.decision)}</span></td>
        <td>${pct(p.resistance_probability)}</td>
        <td>${pct(p.calibrated_confidence)}</td>
        <td>${p.target_status}</td>
        <td>${p.evidence_category}</td>
        <td>${p.reason_codes.join(", ") || "none"}</td>
      </tr>`;
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
    try {
      payload = await loadStaticMetrics() as MetricsPayload;
    } catch {
      return;
    }
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
