"""
Train one calibrated LightGBM marker model per antibiotic.

The genetic relatedness group is used for grouped calibration only. It is not
used as a predictive feature, which avoids leakage from related genomes.

Outputs:
  artifacts/models/<antibiotic>_model.pkl
  artifacts/model_meta.json
"""
from __future__ import annotations

import csv
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "bvbrc"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ANTIBIOTICS = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def get_lgbm_params(n_train: int) -> dict:
    if n_train >= 3000:
        return {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    if n_train >= 1000:
        return {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    return {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def load_feature_matrix(feature_cols: list[str]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    matrix: dict[str, np.ndarray] = {}
    group_map: dict[str, str] = {}
    with open(DATA_DIR / "feature_matrix.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            genome_id = row["genome_id"]
            matrix[genome_id] = np.array([float(row.get(col, 0)) for col in feature_cols])
            group_map[genome_id] = row.get("genetic_group", "") or f"_nogroup_{genome_id}"
    return matrix, group_map


def load_labels() -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with open(DATA_DIR / "labels.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[row["genome_id"]] = {ab: row.get(ab, "") for ab in ANTIBIOTICS}
    return labels


def build_arrays(
    genome_ids: list[str],
    antibiotic: str,
    matrix: dict[str, np.ndarray],
    group_map: dict[str, str],
    labels: dict[str, dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    groups: list[str] = []

    for genome_id in genome_ids:
        label = labels.get(genome_id, {}).get(antibiotic, "")
        if label not in ("resistant", "susceptible") or genome_id not in matrix:
            continue
        x_rows.append(matrix[genome_id])
        y_rows.append(1 if label == "resistant" else 0)
        groups.append(group_map.get(genome_id, "") or f"_nogroup_{genome_id}")

    return np.array(x_rows), np.array(y_rows), np.array(groups)


def train_model(x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray) -> CalibratedClassifierCV:
    n_train = len(x_train)
    unique_groups = len(set(groups.tolist()))
    n_splits = min(5 if n_train >= 2000 else 3, unique_groups)
    n_splits = max(2, n_splits)
    method = "isotonic" if n_train >= 2000 else "sigmoid"
    model = LGBMClassifier(**get_lgbm_params(n_train))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = list(splitter.split(x_train, y_train, groups))
    calibrated = CalibratedClassifierCV(model, cv=cv_splits, method=method)
    calibrated.fit(x_train, y_train)
    return calibrated


def main() -> None:
    with open(DATA_DIR / "feature_columns.json", encoding="utf-8") as handle:
        feature_cols: list[str] = json.load(handle)
    with open(DATA_DIR / "splits.json", encoding="utf-8") as handle:
        splits = json.load(handle)

    matrix, group_map = load_feature_matrix(feature_cols)
    labels = load_labels()
    train_ids: list[str] = splits["train"]

    print("Loading feature matrix and labels...")
    print(f"  {len(feature_cols)} numeric features")
    print(f"  Train genomes: {len(train_ids)}")

    meta: dict[str, dict] = {}

    for antibiotic in ANTIBIOTICS:
        print(f"\n=== {antibiotic} ===")
        x_train, y_train, groups = build_arrays(train_ids, antibiotic, matrix, group_map, labels)
        print(
            f"  Train: {len(x_train)} samples, "
            f"{int(y_train.sum())} resistant ({100 * y_train.mean():.1f}%)"
        )
        print(f"  Calibration groups: {len(set(groups.tolist()))}")

        if len(np.unique(y_train)) < 2 or len(x_train) < 20:
            print("  SKIP: not enough class diversity")
            continue

        model = train_model(x_train, y_train, groups)
        out_path = ARTIFACTS_DIR / f"{antibiotic}_model.pkl"
        with open(out_path, "wb") as handle:
            pickle.dump(
                {
                    "model": model,
                    "feature_cols": feature_cols,
                    "gg_encoding": {},
                    "use_genetic_group": False,
                    "model_family": "LightGBM + StratifiedGroupKFold probability calibration",
                },
                handle,
            )
        print(f"  Saved: {out_path}")

        meta[antibiotic] = {
            "feature_cols": feature_cols,
            "n_train": int(len(x_train)),
            "positive_rate": float(y_train.mean()),
            "n_genetic_groups": int(len(set(groups.tolist()))),
            "calibration": "CalibratedClassifierCV with StratifiedGroupKFold",
            "uses_genetic_group_as_feature": False,
            "model_family": "LightGBM",
        }

    meta_path = ARTIFACTS_DIR.parent / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"\nModel meta: {meta_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
