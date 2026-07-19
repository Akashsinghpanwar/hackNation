"""
Fetch AMR gene features for all training genomes via BV-BRC genome_feature API.
Outputs: data/bvbrc/raw_features.jsonl  (one JSON line per genome)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "bvbrc"
TRAINING_CSV = DATA_DIR / "training_dataset.csv"
OUT_FILE = DATA_DIR / "raw_features.jsonl"

BATCH_SIZE = 50
API_BASE = "https://www.bv-brc.org/api"

# Product keyword patterns that indicate resistance-relevant genes
# Using BV-BRC's PATRIC product annotation vocabulary
# Use only a few broad keywords to keep URL short (BV-BRC has URL length limits)
# Local classification via regex in script 02 handles fine-grained gene families
RESISTANCE_KEYWORDS = [
    "resistance",
    "lactamase",
    "efflux",
    "aminoglycoside",
    "integron",
]


def _build_product_filter() -> str:
    clauses = [f"eq(product,*{urllib.parse.quote(kw)}*)" for kw in RESISTANCE_KEYWORDS]
    return "or(" + ",".join(clauses) + ")"


PRODUCT_FILTER = _build_product_filter()


def fetch_batch(genome_ids: list[str]) -> dict[str, list[str]]:
    """Fetch resistance-related gene products for a batch of genome IDs."""
    ids_str = ",".join(genome_ids)
    url = (
        f"{API_BASE}/genome_feature/"
        f"?and(in(genome_id,({ids_str})),{PRODUCT_FILTER})"
        f"&select(genome_id,gene,product)"
        f"&limit(5000)"
        f"&http_accept=application/json"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())

    by_genome: dict[str, list[str]] = defaultdict(list)
    for row in data:
        gid = row.get("genome_id", "")
        product = row.get("product", "") or ""
        if gid and product:
            by_genome[gid].append(product)
    return dict(by_genome)


def load_genome_ids() -> list[str]:
    import csv
    seen: set[str] = set()
    with open(TRAINING_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row.get("genome_id", "").strip()
            if gid:
                seen.add(gid)
    return sorted(seen)


def main() -> None:
    print("Loading genome IDs from training dataset...")
    genome_ids = load_genome_ids()
    print(f"  {len(genome_ids)} unique genomes found")

    # Resume support: load already-fetched genome_ids
    fetched: set[str] = set()
    if OUT_FILE.exists():
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    fetched.add(obj["genome_id"])
                except Exception:
                    pass
        print(f"  Resuming: {len(fetched)} already fetched")

    remaining = [g for g in genome_ids if g not in fetched]
    print(f"  {len(remaining)} remaining to fetch")

    batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    total = len(batches)

    with open(OUT_FILE, "a", encoding="utf-8") as out:
        for i, batch in enumerate(batches):
            sys.stdout.write(f"\r  Batch {i+1}/{total} ({(i+1)*BATCH_SIZE}/{len(remaining)})...")
            sys.stdout.flush()
            try:
                results = fetch_batch(batch)
                # Ensure every genome in the batch gets a record (even if 0 features)
                for gid in batch:
                    record = {
                        "genome_id": gid,
                        "products": results.get(gid, []),
                    }
                    out.write(json.dumps(record) + "\n")
                out.flush()
                time.sleep(0.2)  # polite pacing
            except Exception as exc:
                print(f"\n  ERROR batch {i+1}: {exc}")
                time.sleep(2)
                # Try again once
                try:
                    results = fetch_batch(batch)
                    for gid in batch:
                        record = {"genome_id": gid, "products": results.get(gid, [])}
                        out.write(json.dumps(record) + "\n")
                    out.flush()
                except Exception as exc2:
                    print(f"  Retry failed: {exc2} — writing empty records")
                    for gid in batch:
                        out.write(json.dumps({"genome_id": gid, "products": []}) + "\n")
                    out.flush()

    print(f"\nDone. Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
