import { spawn, ChildProcessWithoutNullStreams } from "node:child_process";
import { join } from "node:path";
import { rootDir } from "../config.js";

const pythonBin = process.env.PYTHON_BIN ?? (process.platform === "win32" ? "python" : "python3");
const cliPath = join(rootDir, "scripts", "predict_cli.py");
const REQUEST_TIMEOUT_MS = 90_000;

export interface InferenceRequest {
  mode: "fasta" | "tsv" | "bvbrc";
  species?: string;
  fileName?: string;
  content?: string;
  genome_id?: string;
}

interface PendingRequest {
  resolve: (value: Record<string, unknown>) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

// One long-lived Python process (models/numpy/sklearn loaded once) instead of
// spawning + re-importing everything per request — the latter was pushing
// memory-constrained hosts (Render free tier, 512MB) into OOM on every call.
let child: ChildProcessWithoutNullStreams | null = null;
let stdoutBuffer = "";
const queue: PendingRequest[] = [];

function ensureChild(): ChildProcessWithoutNullStreams {
  if (child) return child;

  const proc = spawn(pythonBin, [cliPath, "--serve"], { cwd: rootDir });
  child = proc;
  stdoutBuffer = "";

  proc.stdout.on("data", (chunk: Buffer) => {
    stdoutBuffer += chunk.toString("utf8");
    let newlineIndex: number;
    while ((newlineIndex = stdoutBuffer.indexOf("\n")) !== -1) {
      const line = stdoutBuffer.slice(0, newlineIndex).trim();
      stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
      if (!line) continue;

      let parsed: Record<string, unknown> | undefined;
      try {
        parsed = JSON.parse(line) as Record<string, unknown>;
      } catch {
        continue; // ignore stray non-JSON output (e.g. library warnings that slipped through)
      }
      if (parsed.ready === true) continue; // startup handshake line, not a response

      const pending = queue.shift();
      if (!pending) continue;
      clearTimeout(pending.timer);
      if (typeof parsed.error === "string") {
        pending.reject(new Error(parsed.error));
      } else {
        pending.resolve(parsed);
      }
    }
  });

  let stderrTail = "";
  proc.stderr.on("data", (chunk: Buffer) => {
    stderrTail = (stderrTail + chunk.toString("utf8")).slice(-4000);
  });

  // Without this, a write to a dead child's stdin (EPIPE) is an unhandled
  // 'error' event on the stream and crashes the entire Node process, taking
  // every other in-flight request down with it. onDown() below still runs via
  // the child's own 'close' event, so this handler only needs to swallow the
  // stream error itself.
  proc.stdin.on("error", () => {});

  const onDown = (reason: string) => {
    child = null;
    while (queue.length) {
      const pending = queue.shift()!;
      clearTimeout(pending.timer);
      pending.reject(new Error(`Python inference process ${reason}${stderrTail ? `: ${stderrTail}` : ""}`));
    }
  };
  proc.on("error", (error) => onDown(`failed to start (${error.message})`));
  proc.on("close", (code) => onDown(`exited with code ${code}`));

  return proc;
}

/**
 * Send one request to the persistent Python inference process and resolve with
 * its JSON reply. Requests are queued and answered strictly in order, matching
 * the child's one-line-in/one-line-out stdin/stdout protocol (scripts/predict_cli.py --serve).
 */
export function runInference(request: InferenceRequest): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const proc = ensureChild();

    const timer = setTimeout(() => {
      const idx = queue.findIndex((p) => p.resolve === resolve);
      if (idx !== -1) queue.splice(idx, 1);
      reject(new Error("Inference timed out after 90s."));
    }, REQUEST_TIMEOUT_MS);

    queue.push({ resolve, reject, timer });

    proc.stdin.write(JSON.stringify(request) + "\n", (error) => {
      if (error) {
        const idx = queue.findIndex((p) => p.resolve === resolve);
        if (idx !== -1) queue.splice(idx, 1);
        clearTimeout(timer);
        reject(new Error(`Failed to write to Python process: ${error.message}`));
      }
    });
  });
}
