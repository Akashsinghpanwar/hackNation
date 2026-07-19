import { readFileSync } from "node:fs";
import { join } from "node:path";
import { rootDir } from "../config.js";

interface ReliabilityBin {
  mean_prob: number;
  frac_positive: number;
  count: number;
}
interface DrugMetric {
  auroc: number | null;
  pr_auc: number | null;
  brier_score: number | null;
  no_call_rate: number | null;
  answered_accuracy: number | null;
  balanced_accuracy_called: number | null;
  resistant_recall_called: number | null;
  susceptible_recall_called: number | null;
  reliability?: ReliabilityBin[];
  generalization?: {
    n_groups_evaluated: number;
    auroc_mean: number | null;
    auroc_median: number | null;
    auroc_min: number | null;
  } | null;
}

/** Transform artifacts/demo_metrics.json into the shape the dashboard consumes. */
export function buildMetricsPayload(): unknown {
  const raw = JSON.parse(
    readFileSync(join(rootDir, "artifacts", "demo_metrics.json"), "utf8")
  ) as Record<string, DrugMetric>;

  const metrics = Object.entries(raw).map(([antibiotic, m]) => ({
    antibiotic: antibiotic.charAt(0).toUpperCase() + antibiotic.slice(1),
    auroc: m.auroc,
    pr_auc: m.pr_auc,
    balanced_accuracy: m.balanced_accuracy_called,
    resistant_recall: m.resistant_recall_called,
    susceptible_recall: m.susceptible_recall_called,
    brier: m.brier_score,
    no_call_rate: m.no_call_rate,
    answered_accuracy: m.answered_accuracy
  }));

  const reliability: Record<string, Array<{ mean_confidence: number; observed_accuracy: number; count: number }>> = {};
  for (const [antibiotic, m] of Object.entries(raw)) {
    reliability[antibiotic] = (m.reliability ?? []).map((bin) => ({
      mean_confidence: bin.mean_prob,
      observed_accuracy: bin.frac_positive,
      count: bin.count
    }));
  }

  const generalization = Object.entries(raw)
    .filter(([, m]) => m.generalization)
    .map(([antibiotic, m]) => ({
      antibiotic: antibiotic.charAt(0).toUpperCase() + antibiotic.slice(1),
      overall_auroc: m.auroc,
      group_mean_auroc: m.generalization?.auroc_mean ?? null,
      group_min_auroc: m.generalization?.auroc_min ?? null,
      n_groups: m.generalization?.n_groups_evaluated ?? 0
    }));

  // primary reliability points for the plot: use ampicillin if present, else first drug
  const firstKey = Object.keys(reliability)[0] ?? "";
  const reliability_points = reliability["ampicillin"] ?? reliability[firstKey] ?? [];

  return { metrics, reliability, reliability_points, generalization };
}
