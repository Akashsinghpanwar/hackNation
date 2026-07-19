// Multimodal layer: OpenAI text narrative + gpt-image report card on top of the
// genomic AMR predictions. Uses global fetch (Node >= 18) — no SDK dependency.
// The API key is read only from process.env.OPENAI_API_KEY (never hardcoded).

const API = "https://api.openai.com/v1";

interface Prediction {
  antibiotic: string;
  decision: string;
  resistance_probability: number | null;
  calibrated_confidence: number | null;
  supporting_markers: string[];
  target_status: string;
}
export interface AnalysisLike {
  run_id?: string;
  species?: string;
  source?: string;
  detected_genes?: string[];
  confidence_engine?: string;
  predictions: Prediction[];
  disclaimer?: string;
}

export function aiConfigured(): boolean {
  return Boolean(process.env.OPENAI_API_KEY);
}

function apiKey(): string {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY is not set. Add it to .env to enable AI features.");
  return key;
}

function summarize(result: AnalysisLike): string {
  const lines = result.predictions
    .map((p) => `- ${p.antibiotic}: ${p.decision} (P(resistant)=${p.resistance_probability ?? "NA"}, confidence=${p.calibrated_confidence ?? "NA"}, markers: ${p.supporting_markers.join(", ") || "none"})`)
    .join("\n");
  return [
    `Sample: ${result.run_id ?? "genome"} (${result.species ?? "Escherichia coli"})`,
    `Evidence source: ${result.source ?? "unknown"}`,
    `Detected AMR genes: ${(result.detected_genes ?? []).join(", ") || "none"}`,
    `Predictions:\n${lines}`
  ].join("\n");
}

async function postJson(path: string, body: unknown, timeoutMs: number): Promise<Record<string, unknown>> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey()}` },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs)
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`OpenAI ${path} error ${res.status}: ${text.slice(0, 400)}`);
  return JSON.parse(text) as Record<string, unknown>;
}

/** Plain-language clinical narrative of the AMR report (text modality). */
export async function generateNarrative(result: AnalysisLike): Promise<string> {
  const data = await postJson("/chat/completions", {
    model: process.env.OPENAI_TEXT_MODEL ?? "gpt-4o-mini",
    temperature: 0.3,
    max_tokens: 420,
    messages: [
      {
        role: "system",
        content:
          "You are a clinical-microbiology decision-support assistant for a RESEARCH prototype. " +
          "Explain antibiotic-resistance predictions made from a bacterial genome in clear, cautious language " +
          "for a clinician or student. Explain which detected genes drive which prediction, and what a 'no-call' means. " +
          "Never give a definitive treatment recommendation. Always state the results are predictions that MUST be " +
          "confirmed by standard laboratory antibiotic susceptibility testing. Keep it around 150 words in short paragraphs."
      },
      { role: "user", content: `Write a plain-language clinical summary of this AMR prediction report:\n\n${summarize(result)}` }
    ]
  }, 45_000);
  const choices = data.choices as Array<{ message?: { content?: string } }> | undefined;
  return choices?.[0]?.message?.content?.trim() ?? "";
}

/** Generated "resistance report card" infographic (image modality). Returns base64 PNG. */
export async function generateReportImage(result: AnalysisLike): Promise<string> {
  const fails = result.predictions.filter((p) => p.decision === "likely_to_fail").map((p) => p.antibiotic);
  const works = result.predictions.filter((p) => p.decision === "likely_to_work").map((p) => p.antibiotic);
  const nocalls = result.predictions.filter((p) => p.decision === "no_call").map((p) => p.antibiotic);
  const genes = (result.detected_genes ?? []).slice(0, 6);
  const prompt =
    "A clean, modern medical infographic 'AMR Resistance Report Card' for an E. coli bacterial genome. " +
    "Flat vector poster style, dark slate background with teal accents, no photorealism, highly legible short labels only. " +
    "Top: a stylized bacterium next to a DNA double helix and a shield icon. " +
    `A red panel titled 'LIKELY TO FAIL' listing: ${fails.join(", ") || "none"}. ` +
    `A green panel titled 'LIKELY TO WORK' listing: ${works.join(", ") || "none"}. ` +
    `A grey panel titled 'NO CALL' listing: ${nocalls.join(", ") || "none"}. ` +
    `A small row of gene chips: ${genes.join(", ") || "none"}. ` +
    "Minimal, professional, dashboard aesthetic. Do not include long sentences or paragraphs.";
  const data = await postJson("/images/generations", {
    model: process.env.OPENAI_IMAGE_MODEL ?? "gpt-image-1",
    prompt,
    size: "1024x1024"
  }, 120_000);
  const arr = data.data as Array<{ b64_json?: string }> | undefined;
  const b64 = arr?.[0]?.b64_json;
  if (!b64) throw new Error("No image data returned from OpenAI.");
  return b64;
}
