"""
data/ is gitignored, so a fresh clone/deploy is missing data/bvbrc/feature_columns.json,
which streamlit_app.py's successor (predict_cli.py) needs to build the model feature
vector. All four committed models share identical feature_cols (see artifacts/model_meta.json),
so it's fully reconstructable without rerunning the data pipeline. Idempotent: no-ops if
the file already exists.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "artifacts" / "model_meta.json"
OUT_PATH = ROOT / "data" / "bvbrc" / "feature_columns.json"


def main() -> None:
    if OUT_PATH.exists():
        print(f"{OUT_PATH} already present, skipping.")
        return

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    cols_per_antibiotic = {ab: m["feature_cols"] for ab, m in meta.items()}
    distinct = {tuple(c) for c in cols_per_antibiotic.values()}
    if len(distinct) != 1:
        raise SystemExit(
            "feature_cols differ across antibiotic models; cannot safely auto-reconstruct. "
            "Run the full data pipeline (scripts/01_fetch_features.py, 02_build_matrices.py) instead."
        )

    cols = next(iter(distinct))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(list(cols), indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(cols)} feature columns).")


if __name__ == "__main__":
    main()
