"""
Train LightGBM models per antibiotic with genetic_group as a native categorical feature.
Uses isotonic calibration for reliable probability output.

Outputs: artifacts/models/<antibiotic>_model.pkl  + artifacts/model_meta.json
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "bvbrc"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ANTIBIOTICS = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]

# LightGBM params by dataset size
def get_lgbm_params(n_train: int) -> dict:
    if n_train >= 3000:
        # Large dataset — full complexity
        return {
            "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
            "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 0.1,
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
    elif n_train >= 1000:
        # Medium dataset — moderate complexity
        return {
            "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.5, "reg_lambda": 0.5,
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
    else:
        # Small dataset — conservative, avoid overfitting
        return {
            "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15,
            "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 1.0, "reg_lambda": 1.0,
            "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1,
        }


def load_feature_matrix(feature_cols: list[str]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Returns (numeric_matrix, genetic_groups)."""
    matrix: dict[str, np.ndarray] = {}
    gg_map: dict[str, str] = {}
    with open(DATA_DIR / "feature_matrix.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row["genome_id"]
            matrix[gid] = np.array([float(row.get(c, 0)) for c in feature_cols])
            gg_map[gid] = row.get("genetic_group", "") or ""
    return matrix, gg_map


def load_labels() -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with open(DATA_DIR / "labels.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["genome_id"]] = {ab: row.get(ab, "") for ab in ANTIBIOTICS}
    return labels


def encode_genetic_groups(
    train_ids: list[str],
    all_ids: list[str],
    gg_map: dict[str, str],
    labels_map: dict[str, dict[str, str]],
    antibiotic: str,
    smoothing: float = 10.0,
) -> dict[str, float]:
    """
    Target-encode genetic_group on TRAIN set only (smoothed mean resistance rate).
    smoothing=10 means groups with <10 samples revert toward global mean.
    Returns {group: encoded_value}.
    """
    group_stats: dict[str, list[int]] = {}
    global_resist = []
    for gid in train_ids:
        label = labels_map.get(gid, {}).get(antibiotic, "")
        if label not in ("resistant", "susceptible"):
            continue
        val = 1 if label == "resistant" else 0
        global_resist.append(val)
        gg = gg_map.get(gid, "") or "_unknown"
        if gg not in group_stats:
            group_stats[gg] = []
        group_stats[gg].append(val)

    global_mean = float(np.mean(global_resist)) if global_resist else 0.5

    encoding: dict[str, float] = {}
    for gg, vals in group_stats.items():
        n = len(vals)
        group_mean = float(np.mean(vals))
        # Bayesian smoothing toward global mean
        encoding[gg] = (n * group_mean + smoothing * global_mean) / (n + smoothing)

    encoding["_unknown"] = global_mean
    encoding["_missing"] = global_mean
    return encoding


def build_arrays(
    genome_ids: list[str],
    antibiotic: str,
    matrix: dict[str, np.ndarray],
    gg_map: dict[str, str],
    gg_encoding: dict[str, float],
    labels: dict[str, dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    X_rows, y_rows = [], []
    global_mean = gg_encoding.get("_missing", 0.5)
    for gid in genome_ids:
        label = labels.get(gid, {}).get(antibiotic, "")
        if label not in ("resistant", "susceptible"):
            continue
        if gid not in matrix:
            continue
        gg_val = gg_encoding.get(gg_map.get(gid, "") or "_missing", global_mean)
        feat = np.append(matrix[gid], gg_val)
        X_rows.append(feat)
        y_rows.append(1 if label == "resistant" else 0)
    return np.array(X_rows), np.array(y_rows)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    n = len(X_train)
    params = get_lgbm_params(n)
    lgbm = LGBMClassifier(**params)
    # isotonic needs enough samples per fold — use sigmoid for small datasets
    n_splits = 5 if n >= 2000 else 3
    method = "isotonic" if n >= 2000 else "sigmoid"
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cal = CalibratedClassifierCV(lgbm, cv=cv, method=method)
    cal.fit(X_train, y_train)
    return cal


def main() -> None:
    with open(DATA_DIR / "feature_columns.json", encoding="utf-8") as f:
        feature_cols: list[str] = json.load(f)
    with open(DATA_DIR / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)

    print("Loading feature matrix and labels...")
    matrix, gg_map = load_feature_matrix(feature_cols)
    labels = load_labels()
    train_ids: list[str] = splits["train"]
    print(f"  {len(feature_cols)} numeric features + 1 target-encoded genetic_group")
    print(f"  Train genomes: {len(train_ids)}")

    meta: dict[str, dict] = {}

    for ab in ANTIBIOTICS:
        print(f"\n=== {ab} ===")

        # Compute target encoding on train set only (avoids leakage)
        gg_enc = encode_genetic_groups(train_ids, list(matrix.keys()), gg_map, labels, ab)
        print(f"  Encoded {len(gg_enc)} genetic groups")

        X_train, y_train = build_arrays(train_ids, ab, matrix, gg_map, gg_enc, labels)
        print(f"  Train: {len(X_train)} samples, {y_train.sum()} resistant ({100*y_train.mean():.1f}%)")

        if len(np.unique(y_train)) < 2 or len(X_train) < 20:
            print("  SKIP")
            continue

        # Only pass genetic_group to the model when dataset is large enough
        # (target-encoding is noisy with <2000 samples)
        use_gg = len(X_train) >= 2000
        if not use_gg:
            print("  Small dataset — dropping genetic_group feature to avoid noise")
            # Rebuild arrays without gg_val (last column)
            X_train = X_train[:, :-1]
            effective_gg_enc: dict[str, float] = {}
        else:
            effective_gg_enc = gg_enc

        model = train_model(X_train, y_train)

        out_path = ARTIFACTS_DIR / f"{ab}_model.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_cols": feature_cols,
                "gg_encoding": effective_gg_enc,
                "use_genetic_group": use_gg,
            }, f)
        print(f"  Saved: {out_path}")

        meta[ab] = {
            "feature_cols": feature_cols,
            "n_train": len(X_train),
            "positive_rate": float(y_train.mean()),
            "n_genetic_groups": len(gg_enc),
        }

    meta_path = ARTIFACTS_DIR.parent / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nModel meta: {meta_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
