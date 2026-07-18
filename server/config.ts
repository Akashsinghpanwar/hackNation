import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { AppConfig } from "../shared/types.js";

export const rootDir = process.cwd();

export function loadConfig(): AppConfig {
  const raw = readFileSync(join(rootDir, "configs", "app_config.json"), "utf8");
  return JSON.parse(raw) as AppConfig;
}

export function loadDemoMetrics(): unknown {
  const raw = readFileSync(join(rootDir, "artifacts", "demo_metrics.json"), "utf8");
  return JSON.parse(raw);
}
