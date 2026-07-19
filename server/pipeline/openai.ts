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

/** Text-to-speech of the clinical narrative (audio modality). Returns base64 MP3. */
export async function generateSpeech(text: string): Promise<string> {
  const input = text.trim().slice(0, 4000);
  if (!input) throw new Error("Nothing to read — generate the summary first.");
  const res = await fetch(`${API}/audio/speech`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey()}` },
    body: JSON.stringify({
      model: process.env.OPENAI_TTS_MODEL ?? "gpt-4o-mini-tts",
      voice: process.env.OPENAI_TTS_VOICE ?? "alloy",
      input,
      response_format: "mp3"
    }),
    signal: AbortSignal.timeout(60_000)
  });
  if (!res.ok) throw new Error(`OpenAI /audio/speech error ${res.status}: ${(await res.text()).slice(0, 400)}`);
  const buf = Buffer.from(await res.arrayBuffer());
  return buf.toString("base64");
}
