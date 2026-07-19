"""Evaluate trained models and write Streamlit metrics."""
from __future__ import annotations

import csv
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "bvbrc"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

ANTIBIOTICS = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]
FAIL_THRESHOLD = 0.72
WORK_THRESHOLD = 0.28

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def load_labels() -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with open(DATA_DIR / "labels.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[row["genome_id"]] = {ab: row.get(ab, "") for ab in ANTIBIOTICS}
    return labels


def main() -> None:
    with open(DATA_DIR / "splits.json", encoding="utf-8") as handle:
        splits = json.load(handle)
    labels = load_labels()
    test_ids = set(splits["test"])
    metrics: dict[str, dict] = {}

    for antibiotic in ANTIBIOTICS:
        model_path = MODELS_DIR / f"{antibiotic}_model.pkl"
        if not model_path.exists():
            print(f"No model for {antibiotic}")
            continue

        with open(model_path, "rb") as handle:
            package = pickle.load(handle)
        model = package["model"]
        feature_cols: list[str] = package["feature_cols"]

        x_rows: list[list[float]] = []
        y_rows: list[int] = []
        with open(DATA_DIR / "feature_matrix.csv", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                genome_id = row["genome_id"]
                if genome_id not in test_ids:
                    continue
                label = labels.get(genome_id, {}).get(antibiotic, "")
                if label not in ("resistant", "susceptible"):
                    continue
                x_rows.append([float(row.get(col, 0)) for col in feature_cols])
                y_rows.append(1 if label == "resistant" else 0)

        if len(y_rows) < 5:
            print(f"{antibiotic}: too few test samples")
            continue

        x_test = np.array(x_rows)
        y_test = np.array(y_rows)
        probs = model.predict_proba(x_test)[:, 1]
        decisions = np.array(
            [
                "likely_to_fail"
                if prob >= FAIL_THRESHOLD
                else ("likely_to_work" if prob <= WORK_THRESHOLD else "no_call")
                for prob in probs
            ]
        )
        called_mask = decisions != "no_call"
        no_call_rate = float((~called_mask).mean())

        if called_mask.sum() > 1 and len(np.unique(y_test[called_mask])) > 1:
            y_true_called = y_test[called_mask]
            y_pred_called = np.array([1 if d == "likely_to_fail" else 0 for d in decisions[called_mask]])
            bal_acc = balanced_accuracy_score(y_true_called, y_pred_called)
            f1 = f1_score(y_true_called, y_pred_called, zero_division=0)
            res_rec = recall_score(y_true_called, y_pred_called, pos_label=1, zero_division=0)
            sus_rec = recall_score(y_true_called, y_pred_called, pos_label=0, zero_division=0)
            answered_accuracy = float((y_true_called == y_pred_called).mean())
        else:
            bal_acc = f1 = res_rec = sus_rec = answered_accuracy = None

        auroc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else None
        pr_auc = float(average_precision_score(y_test, probs)) if len(np.unique(y_test)) > 1 else None
        brier = float(brier_score_loss(y_test, probs))

        reliability_bins = []
        for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (probs >= lower) & (probs < upper)
            if mask.sum() == 0:
                continue
            reliability_bins.append(
                {
                    "bin_center": round(float((lower + upper) / 2), 3),
                    "mean_prob": round(float(probs[mask].mean()), 4),
                    "frac_positive": round(float(y_test[mask].mean()), 4),
                    "count": int(mask.sum()),
                }
            )

        metrics[antibiotic] = {
            "n_test": len(y_rows),
            "n_resistant": int(y_test.sum()),
            "n_susceptible": int((y_test == 0).sum()),
            "no_call_rate": round(no_call_rate, 4),
            "balanced_accuracy_called": round(bal_acc, 4) if bal_acc is not None else None,
            "answered_accuracy": round(answered_accuracy, 4) if answered_accuracy is not None else None,
            "f1_called": round(f1, 4) if f1 is not None else None,
            "resistant_recall_called": round(res_rec, 4) if res_rec is not None else None,
            "susceptible_recall_called": round(sus_rec, 4) if sus_rec is not None else None,
            "auroc": round(auroc, 4) if auroc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "brier_score": round(brier, 4),
            "reliability": reliability_bins,
            "fail_threshold": FAIL_THRESHOLD,
            "work_threshold": WORK_THRESHOLD,
        }

        print(f"\n{antibiotic}:")
        print(f"  n_test={len(y_rows)} resistant={int(y_test.sum())} no_call={no_call_rate:.1%}")
        print(f"  AUROC={auroc} PR-AUC={pr_auc} Brier={brier:.4f} BalAcc={bal_acc}")

    output_path = ARTIFACTS_DIR / "demo_metrics.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
