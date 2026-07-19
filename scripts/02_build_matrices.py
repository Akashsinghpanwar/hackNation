"""
Build binary feature matrix + metadata features + label matrices.
Applies cluster-based train/test split using cgmlst_hc50.

Outputs:
  data/bvbrc/feature_matrix.csv
  data/bvbrc/labels.csv
  data/bvbrc/splits.json
  data/bvbrc/feature_columns.json
  data/bvbrc/genome_meta.csv
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "bvbrc"
RAW_FEATURES = DATA_DIR / "raw_features.jsonl"
TRAINING_CSV = DATA_DIR / "training_dataset.csv"

ANTIBIOTICS = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]

GENE_FAMILIES: list[tuple[str, str]] = [
    ("blaCTX-M",  r"CTX-M"),
    ("blaTEM",    r"\bTEM\b|TEM family"),
    ("blaSHV",    r"\bSHV\b|SHV family"),
    ("blaOXA",    r"OXA family|Class D beta-lactamase"),
    ("blaCMY",    r"\bCMY\b"),
    ("blaEC",     r"BlaEC family"),
    ("blaAmpC",   r"Class C beta-lactamase|AmpC"),
    ("betaLactamResProtein", r"beta-lactam resistance protein"),
    ("tetA",      r"Tet\(A\)|TetA\b"),
    ("tetB",      r"Tet\(B\)|TetB\b"),
    ("tetC",      r"Tet\(C\)|TetC\b"),
    ("tetM",      r"Tet\(M\)|TetM\b"),
    ("tetG",      r"Tet\(G\)|TetG\b"),
    ("tetR",      r"Tetracycline resistance regulatory|TetR\b"),
    ("qnrA",      r"qnrA|quinolone resistance.*A"),
    ("qnrB",      r"qnrB|quinolone resistance.*B"),
    ("qnrS",      r"qnrS|quinolone resistance.*S"),
    ("qnrD",      r"qnrD"),
    ("aac6Ib",    r"AAC\(6'\).*Ib|aac\(6'\).*Ib"),
    ("oqxA",      r"OqxA|oqxA"),
    ("oqxB",      r"OqxB|oqxB"),
    ("aadA",      r"aadA|ANT\(3''\)"),
    ("aac3",      r"AAC\(3\)"),
    ("ant2",      r"ANT\(2''\)"),
    ("aph3",      r"APH\(3'\)|aminoglycoside 3'-phosphotransferase"),
    ("sul1",      r"type-2.*sulfonamide|Sulfonamide resistance|dihydropteroate synthase type-2"),
    ("dhfr",      r"dihydrofolate reductase|DHFR"),
    ("trimethoprimRes", r"trimethoprim"),
    ("marA",      r"MarA\b"),
    ("marR",      r"MarR\b"),
    ("marB",      r"MarB\b"),
    ("acrAB",     r"AcrA|AcrB|AcrAB"),
    ("tolC",      r"TolC\b"),
    ("mdtEFG",    r"MdtG|MdtH|MdtE|MdtF"),
    ("emrAB",     r"EmrA|EmrB|EmrR"),
    ("qacE",      r"QacE"),
    ("integraseI",r"integron integrase|class 1 integron"),
    ("catA",      r"chloramphenicol acetyltransferase|cat[ABC]\b"),
    ("mcr",       r"\bmcr-"),
]
_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in GENE_FAMILIES]


def classify_product(product: str) -> list[str]:
    return [name for name, pat in _COMPILED if pat.search(product)]


def load_raw_features() -> dict[str, set[str]]:
    genome_genes: dict[str, set[str]] = defaultdict(set)
    with open(RAW_FEATURES, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                gid = obj["genome_id"]
                for product in obj.get("products", []):
                    for gene in classify_product(product):
                        genome_genes[gid].add(gene)
            except Exception:
                pass
    return dict(genome_genes)


def load_metadata_labels_clusters() -> tuple[
    dict[str, dict],
    dict[str, dict[str, str]],
    dict[str, str],
]:
    """Returns (meta, labels, clusters) where meta has genome_length, contigs, genetic_group."""
    meta: dict[str, dict] = {}
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    clusters: dict[str, str] = {}

    with open(TRAINING_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row.get("genome_id", "").strip()
            if not gid:
                continue

            # metadata (take first occurrence)
            if gid not in meta:
                meta[gid] = {
                    "genome_length": row.get("genome_length", "") or "",
                    "contigs":       row.get("contigs", "") or "",
                    "genetic_group": row.get("cgmlst_hc50", "") or row.get("genetic_group", "") or "",
                }

            # labels
            ab = row.get("antibiotic", "").strip().lower()
            label = row.get("label", "").strip().lower()
            if ab and label in ("resistant", "susceptible"):
                labels[gid][ab] = label

            # cluster
            cluster = row.get("cgmlst_hc50", "").strip()
            if gid not in clusters and cluster:
                clusters[gid] = cluster

    return meta, dict(labels), clusters


def cluster_split(
    genome_ids: list[str],
    clusters: dict[str, str],
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    cluster_to_genomes: dict[str, list[str]] = defaultdict(list)
    for gid in genome_ids:
        c = clusters.get(gid, f"_nocl_{gid}")
        cluster_to_genomes[c].append(gid)

    cluster_names = sorted(cluster_to_genomes.keys())
    rng.shuffle(cluster_names)

    n_test_target = int(len(genome_ids) * test_fraction)
    test_genomes, train_genomes = [], []
    placed_test = 0
    for c in cluster_names:
        members = cluster_to_genomes[c]
        if placed_test < n_test_target:
            test_genomes.extend(members)
            placed_test += len(members)
        else:
            train_genomes.extend(members)

    return train_genomes, test_genomes


def compute_global_stats(
    all_genome_ids: list[str],
    meta: dict[str, dict],
) -> dict[str, float]:
    """Compute mean/std for genome_length and contigs for standardization."""
    lengths = [float(meta[g]["genome_length"]) for g in all_genome_ids
               if meta.get(g, {}).get("genome_length")]
    contigs = [float(meta[g]["contigs"]) for g in all_genome_ids
               if meta.get(g, {}).get("contigs")]
    return {
        "length_mean": float(np.mean(lengths)) if lengths else 5_000_000,
        "length_std":  float(np.std(lengths))  if lengths else 500_000,
        "contigs_mean": float(np.mean(contigs)) if contigs else 100,
        "contigs_std":  float(np.std(contigs))  if contigs else 50,
    }


def main() -> None:
    print("Loading raw gene features...")
    genome_genes = load_raw_features()
    print(f"  {len(genome_genes)} genomes with feature data")

    print("Loading metadata, labels, clusters...")
    meta, labels, clusters = load_metadata_labels_clusters()
    all_genome_ids = sorted(set(labels.keys()) | set(genome_genes.keys()))
    print(f"  {len(all_genome_ids)} total genome IDs")

    # ── Gene feature columns (remove always-0/always-1) ──────────────────────
    present_families: set[str] = set()
    for genes in genome_genes.values():
        present_families.update(genes)
    gene_cols = [name for name, _ in GENE_FAMILIES if name in present_families]
    extra = sorted(present_families - set(gene_cols))
    gene_cols += extra

    # Compute prevalence and drop near-constant features
    prevalences = {}
    total = len(all_genome_ids)
    for col in gene_cols:
        count = sum(1 for g in all_genome_ids if col in genome_genes.get(g, set()))
        prevalences[col] = count / total

    useful_gene_cols = [c for c in gene_cols if 0.02 <= prevalences[c] <= 0.98]
    print(f"  Gene features: {len(gene_cols)} total, {len(useful_gene_cols)} useful (2%-98% prevalence)")

    # ── Additional engineered features ───────────────────────────────────────
    # Resistance gene burden (total useful AMR gene families detected)
    # Genetic group (categorical — stored separately for LightGBM)
    # Genome length + contigs (normalised)

    stats = compute_global_stats(all_genome_ids, meta)

    feature_cols = useful_gene_cols + ["amr_gene_burden", "genome_length_z", "contigs_z"]
    print(f"  Total feature columns: {len(feature_cols)}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    print("Building feature matrix...")
    rows = []
    for gid in all_genome_ids:
        genes = genome_genes.get(gid, set())
        m = meta.get(gid, {})
        row = {"genome_id": gid}
        for col in useful_gene_cols:
            row[col] = 1 if col in genes else 0
        # burden: count of useful resistance genes present
        row["amr_gene_burden"] = int(sum(1 for c in useful_gene_cols if c in genes))
        # normalised assembly stats
        try:
            row["genome_length_z"] = round(
                (float(m.get("genome_length") or stats["length_mean"]) - stats["length_mean"])
                / max(stats["length_std"], 1), 4)
        except Exception:
            row["genome_length_z"] = 0.0
        try:
            row["contigs_z"] = round(
                (float(m.get("contigs") or stats["contigs_mean"]) - stats["contigs_mean"])
                / max(stats["contigs_std"], 1), 4)
        except Exception:
            row["contigs_z"] = 0.0
        # genetic group (stored as string for LightGBM categorical)
        row["genetic_group"] = m.get("genetic_group", "") or ""
        rows.append(row)

    feature_matrix_path = DATA_DIR / "feature_matrix.csv"
    all_cols = ["genome_id"] + feature_cols + ["genetic_group"]
    with open(feature_matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {feature_matrix_path}")

    # ── Label matrix ─────────────────────────────────────────────────────────
    print("Building label matrix...")
    label_rows = [{"genome_id": gid, **{ab: labels.get(gid, {}).get(ab, "") for ab in ANTIBIOTICS}}
                  for gid in all_genome_ids]
    labels_path = DATA_DIR / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["genome_id"] + ANTIBIOTICS)
        writer.writeheader()
        writer.writerows(label_rows)
    print(f"  Saved: {labels_path}")

    # ── Cluster split ─────────────────────────────────────────────────────────
    print("Computing cluster-based train/test split...")
    genomes_with_labels = [g for g in all_genome_ids if any(labels.get(g, {}).values())]
    train_ids, test_ids = cluster_split(genomes_with_labels, clusters)
    print(f"  Train: {len(train_ids)}   Test: {len(test_ids)}")

    splits_path = DATA_DIR / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_ids, "test": test_ids}, f)
    print(f"  Saved: {splits_path}")

    # Save feature columns (numerical only — genetic_group handled separately)
    cols_path = DATA_DIR / "feature_columns.json"
    meta_path = DATA_DIR / "feature_meta.json"
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"norm_stats": stats, "useful_gene_cols": useful_gene_cols,
                   "all_feature_cols": feature_cols}, f, indent=2)
    print(f"  Saved: {cols_path}, {meta_path}")

    # Print feature prevalence
    print("\nFeature prevalence (useful gene cols):")
    for col in useful_gene_cols:
        pct = 100 * prevalences[col]
        print(f"  {col:<25} {pct:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
