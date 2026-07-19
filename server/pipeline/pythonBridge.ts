import { spawn } from "node:child_process";
import { join } from "node:path";
import { rootDir } from "../config.js";

const pythonBin = process.env.PYTHON_BIN ?? (process.platform === "win32" ? "python" : "python3");
const cliPath = join(rootDir, "scripts", "predict_cli.py");

export interface InferenceRequest {
  mode: "fasta" | "tsv" | "bvbrc";
  species?: string;
  fileName?: string;
  content?: string;
  genome_id?: string;
}

/**
 * Run the standalone Python inference CLI (real LightGBM models + k-mer detector
 * + BV-BRC fetch) by writing one JSON request to stdin and reading one JSON reply.
 */
export function runInference(request: InferenceRequest): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonBin, [cliPath], { cwd: rootDir });
    let stdout = "";
    let stderr = "";

    const timer = setTimeout(() => {
      child.kill();
      reject(new Error("Inference timed out after 90s."));
    }, 90_000);

    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString("utf8")));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString("utf8")));

    child.on("error", (error) => {
      clearTimeout(timer);
      reject(new Error(`Failed to launch Python (${pythonBin}): ${error.message}`));
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      let parsed: Record<string, unknown> | undefined;
      try {
        parsed = JSON.parse(stdout) as Record<string, unknown>;
      } catch {
        parsed = undefined;
      }
      if (parsed && typeof parsed.error === "string") {
        reject(new Error(parsed.error));
        return;
      }
      if (code !== 0 || !parsed) {
        reject(new Error(stderr.trim() || `Inference exited with code ${code}`));
        return;
      }
      resolve(parsed);
    });

    child.stdin.write(JSON.stringify(request));
    child.stdin.end();
  });
}
