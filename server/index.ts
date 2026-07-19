import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { extname, join, normalize } from "node:path";
import { runInference } from "./pipeline/pythonBridge.js";
import { buildMetricsPayload } from "./pipeline/metrics.js";
import { loadConfig, rootDir } from "./config.js";
import { loadBvbrcTrainingDashboard } from "./bvbrcData.js";

const port = Number(process.env.PORT ?? 3000);
const publicDir = join(rootDir, "public");

const mimeTypes: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".fasta": "text/plain; charset=utf-8"
};

function sendJson(response: ServerResponse, statusCode: number, payload: unknown): void {
  response.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

async function readRequestJson<T>(request: IncomingMessage): Promise<T> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as T;
}

async function serveStatic(pathName: string, response: ServerResponse): Promise<void> {
  if (pathName === "/assets/app.js") {
    const clientBundle = join(rootDir, "dist", "client", "app.js");
    response.writeHead(200, { "Content-Type": "text/javascript; charset=utf-8" });
    createReadStream(clientBundle).pipe(response);
    return;
  }

  const requested = pathName === "/" ? "/index.html" : pathName;
  const filePath = normalize(join(publicDir, requested));

  if (!filePath.startsWith(publicDir)) {
    sendJson(response, 403, { error: "Forbidden" });
    return;
  }

  try {
    const fileStat = await stat(filePath);
    if (!fileStat.isFile()) throw new Error("Not a file");
    response.writeHead(200, { "Content-Type": mimeTypes[extname(filePath)] ?? "application/octet-stream" });
    createReadStream(filePath).pipe(response);
  } catch {
    sendJson(response, 404, { error: "Not found" });
  }
}

async function handleRequest(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);

  if (request.method === "GET" && url.pathname === "/api/config") {
    sendJson(response, 200, loadConfig());
    return;
  }

  if (request.method === "GET" && url.pathname === "/api/metrics") {
    sendJson(response, 200, buildMetricsPayload());
    return;
  }

  if (request.method === "GET" && url.pathname === "/api/bvbrc/training-dashboard") {
    try {
      sendJson(response, 200, await loadBvbrcTrainingDashboard());
    } catch (error) {
      sendJson(response, 404, {
        error: error instanceof Error
          ? `BV-BRC training data is not available yet: ${error.message}`
          : "BV-BRC training data is not available yet."
      });
    }
    return;
  }

  if (request.method === "GET" && url.pathname.startsWith("/api/demo-sample/")) {
    const sample = url.pathname.split("/").pop() ?? "";
    const allowed = new Set([
      "mdr_ecoli.fasta", "cipro_resistant_ecoli.fasta", "susceptible_ecoli.fasta",
      "likely_fail_sample.fasta", "likely_work_sample.fasta", "no_call_sample.fasta"
    ]);
    if (!allowed.has(sample)) {
      sendJson(response, 404, { error: "Unknown demo sample" });
      return;
    }
    const content = await readFile(join(rootDir, "demo_samples", sample), "utf8");
    sendJson(response, 200, { fileName: sample, content });
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/analyse") {
    try {
      const body = await readRequestJson<{ species: string; fileName: string; content: string; mode?: string }>(request);
      const mode = body.mode === "tsv" ? "tsv" : "fasta";
      const result = await runInference({ mode, species: body.species, fileName: body.fileName, content: body.content });
      sendJson(response, 200, result);
    } catch (error) {
      sendJson(response, 500, { error: error instanceof Error ? error.message : "Unknown error" });
    }
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/analyse-bvbrc") {
    try {
      const body = await readRequestJson<{ species?: string; genome_id: string }>(request);
      if (!body.genome_id || !/^\d+\.\d+$/.test(body.genome_id.trim())) {
        sendJson(response, 400, { error: "Provide a BV-BRC genome id like 562.12960." });
        return;
      }
      const result = await runInference({ mode: "bvbrc", species: body.species ?? "Escherichia coli", genome_id: body.genome_id.trim() });
      sendJson(response, 200, result);
    } catch (error) {
      sendJson(response, 500, { error: error instanceof Error ? error.message : "Unknown error" });
    }
    return;
  }

  await serveStatic(url.pathname, response);
}

createServer((request, response) => {
  handleRequest(request, response).catch((error: unknown) => {
    sendJson(response, 500, { error: error instanceof Error ? error.message : "Unknown server error" });
  });
}).listen(port, () => {
  console.log(`AMRShield Sentinel running at http://localhost:${port}`);
});
