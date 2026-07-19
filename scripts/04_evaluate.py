"""
Evaluate trained models. Compares old vs new, writes demo_metrics.json.
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss,
    f1_score, precision_recall_curve, roc_auc_score, recall_score,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "bvbrc"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

ANTIBIOTICS = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]
FAIL_THRESHOLD = 0.72
WORK_THRESHOLD = 0.28


def load_labels() -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with open(DATA_DIR / "labels.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["genome_id"]] = {ab: row.get(ab, "") for ab in ANTIBIOTICS}
    return labels


def load_gg_map() -> dict[str, str]:
    gg = {}
    with open(DATA_DIR / "feature_matrix.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gg[row["genome_id"]] = row.get("genetic_group", "") or ""
    return gg


def main() -> None:
    with open(DATA_DIR / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    labels = load_labels()
    gg_map = load_gg_map()
    test_ids: set[str] = set(splits["test"])

    metrics: dict[str, dict] = {}

    for ab in ANTIBIOTICS:
        model_path = MODELS_DIR / f"{ab}_model.pkl"
        if not model_path.exists():
            print(f"  No model for {ab}")
            continue

        with open(model_path, "rb") as f:
            pkg = pickle.load(f)
        model = pkg["model"]
        feature_cols: list[str] = pkg["feature_cols"]
        gg_enc: dict[str, float] = pkg.get("gg_encoding", {})
        use_gg: bool = pkg.get("use_genetic_group", bool(gg_enc))
        global_mean = gg_enc.get("_missing", 0.5)

        X_test_rows, y_test_rows = [], []
        with open(DATA_DIR / "feature_matrix.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gid = row["genome_id"]
                if gid not in test_ids:
                    continue
                label = labels.get(gid, {}).get(ab, "")
                if label not in ("resistant", "susceptible"):
                    continue
                num_feat = [float(row.get(c, 0)) for c in feature_cols]
                if use_gg:
                    gg_val = gg_enc.get(gg_map.get(gid, "") or "_missing", global_mean)
                    X_test_rows.append(num_feat + [gg_val])
                else:
                    X_test_rows.append(num_feat)
                y_test_rows.append(1 if label == "resistant" else 0)

        if len(y_test_rows) < 5:
            print(f"  {ab}: too few test samples")
            continue

        X_test = np.array(X_test_rows)
        y_test = np.array(y_test_rows)
        probs = model.predict_proba(X_test)[:, 1]

        decisions = [
            "likely_to_fail" if p >= FAIL_THRESHOLD
            else ("likely_to_work" if p <= WORK_THRESHOLD else "no_call")
            for p in probs
        ]
        no_call_mask = np.array([d == "no_call" for d in decisions])
        no_call_rate = float(no_call_mask.mean())
        called_mask = ~no_call_mask

        if called_mask.sum() > 1 and len(np.unique(y_test[called_mask])) > 1:
            y_pred_called = np.array([1 if d == "likely_to_fail" else 0
                                       for d, m in zip(decisions, called_mask) if m])
            y_true_called = y_test[called_mask]
            bal_acc = balanced_accuracy_score(y_true_called, y_pred_called)
            f1 = f1_score(y_true_called, y_pred_called, zero_division=0)
            res_rec = recall_score(y_true_called, y_pred_called, pos_label=1, zero_division=0)
            sus_rec = recall_score(y_true_called, y_pred_called, pos_label=0, zero_division=0)
        else:
            bal_acc = f1 = res_rec = sus_rec = None

        try:
            auroc = float(roc_auc_score(y_test, probs))
        except Exception:
            auroc = None

        try:
            precision, recall, _ = precision_recall_curve(y_test, probs)
            pr_auc = float(np.trapz(precision[::-1], recall[::-1]))
        except Exception:
            pr_auc = None

        brier = float(brier_score_loss(y_test, probs))

        # Reliability bins
        bin_edges = np.linspace(0, 1, 11)
        reliability_bins = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            reliability_bins.append({
                "bin_center": round(float((lo + hi) / 2), 3),
                "mean_prob": round(float(probs[mask].mean()), 4),
                "frac_positive": round(float(y_test[mask].mean()), 4),
                "count": int(mask.sum()),
            })

        metrics[ab] = {
            "n_test": len(y_test_rows),
            "n_resistant": int(y_test.sum()),
            "n_susceptible": int((y_test == 0).sum()),
            "no_call_rate": round(no_call_rate, 4),
            "balanced_accuracy_called": round(bal_acc, 4) if bal_acc is not None else None,
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

        print(f"\n{ab}:")
        print(f"  n_test={len(y_test_rows)}  resistant={y_test.sum()}  no_call={no_call_rate:.1%}")
        print(f"  AUROC={auroc:.4f}  Brier={brier:.4f}  BalAcc={bal_acc}")
        print(f"  ResRecall={res_rec:.4f}  SusRecall={sus_rec:.4f}")

    out = ARTIFACTS_DIR / "demo_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
