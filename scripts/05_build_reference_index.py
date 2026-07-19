"""
Build a compact k-mer index from the public NCBI AMRFinderPlus reference gene
database (AMR_CDS.fa) so the app can detect AMR genes directly from a raw
nucleotide FASTA -- no AMRFinderPlus / BLAST install required.

Method: for every reference allele whose gene symbol maps to one of the model's
marker families, we compute a bottom-H MinHash sketch of its canonical 21-mers.
At inference time a query genome is reduced to a set of canonical 21-mer codes and
each allele's sketch containment is measured. High containment == gene present.
This is the same containment idea used by Mash / sourmash, in pure Python.

Input :  data/amr_reference/AMR_CDS.fa   (download once from NCBI, public domain)
Output:  data/amr_reference/kmer_index.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data" / "amr_reference"
REF_FASTA = REF_DIR / "AMR_CDS.fa"
# Index is written to artifacts/ (committed) so a fresh clone runs without rebuilding.
OUT_PATH = ROOT / "artifacts" / "kmer_index.json"

K = 21
SKETCH_SIZE = 60            # bottom-H MinHash values kept per allele
MAX_ALLELES_PER_FAMILY = 400

# Reference gene-symbol  ->  model marker family (matches data/bvbrc/feature_columns.json)
SYMBOL_MAP: list[tuple[str, str]] = [
    ("blaCTX-M", r"^blaCTX-M"),
    ("blaTEM",   r"^blaTEM"),
    ("blaOXA",   r"^blaOXA"),
    ("blaCMY",   r"^blaCMY"),
    ("blaEC",    r"^blaEC"),
    ("tetA",     r"^tet\(A\)"),
    ("tetB",     r"^tet\(B\)"),
    ("tetC",     r"^tet\(C\)"),
    ("tetR",     r"^tet\(R\)|^tetR"),
    ("aac6Ib",   r"^aac\(6'\)-Ib"),
    ("aac3",     r"^aac\(3\)"),
    ("aadA",     r"^aadA|^ant\(3''\)"),
    ("aph3",     r"^aph\(3'\)"),
    ("sul1",     r"^sul1"),
    ("qacE",     r"^qacE|^qacEdelta"),
    ("integraseI", r"^intI1|^int1\b"),
    ("qnrA",     r"^qnrA"),
    ("qnrB",     r"^qnrB"),
    ("qnrS",     r"^qnrS"),
    ("qnrD",     r"^qnrD"),
    ("mcr",      r"^mcr-"),
    ("catA",     r"^catA|^catI\b"),
    ("dfrA",     r"^dfrA"),
]
COMPILED_SYMBOL_MAP = [(fam, re.compile(pat, re.IGNORECASE)) for fam, pat in SYMBOL_MAP]

_BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def map_symbol(symbol: str) -> str | None:
    for family, pattern in COMPILED_SYMBOL_MAP:
        if pattern.search(symbol):
            return family
    return None


def canonical_kmer_codes(seq: str, k: int = K) -> set[int]:
    """Return the set of canonical (min of forward/revcomp) 2-bit-encoded k-mers."""
    seq = seq.upper()
    codes: set[int] = set()
    mask = (1 << (2 * k)) - 1
    top_shift = 2 * (k - 1)
    fwd = 0
    rev = 0
    valid = 0
    for base in seq:
        code = _BASE.get(base)
        if code is None:            # N or other -> reset the rolling window
            valid = 0
            fwd = 0
            rev = 0
            continue
        fwd = ((fwd << 2) | code) & mask
        rev = (rev >> 2) | ((3 - code) << top_shift)
        valid += 1
        if valid >= k:
            codes.add(fwd if fwd <= rev else rev)
    return codes


def sketch(codes: set[int], size: int = SKETCH_SIZE) -> list[int]:
    """Bottom-H MinHash: keep the H smallest canonical codes (a uniform sample)."""
    if len(codes) <= size:
        return sorted(codes)
    return sorted(codes)[:size]


def iter_reference(path: Path):
    """Yield (gene_symbol, sequence) from AMR_CDS.fa."""
    header = None
    chunks: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                fields = line[1:].split("|")
                header = fields[4] if len(fields) > 4 else line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def main() -> None:
    if not REF_FASTA.exists():
        raise SystemExit(
            f"Reference FASTA not found: {REF_FASTA}\n"
            "Download it first:\n"
            "  curl -o data/amr_reference/AMR_CDS.fa "
            "https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/"
            "AMRFinderPlus/database/latest/AMR_CDS.fa"
        )

    families: dict[str, list[list[int]]] = {}
    counts: dict[str, int] = {}

    for symbol, seq in iter_reference(REF_FASTA):
        family = map_symbol(symbol)
        if family is None or len(seq) < K:
            continue
        if len(families.get(family, [])) >= MAX_ALLELES_PER_FAMILY:
            counts[family] = counts.get(family, 0) + 1
            continue
        codes = canonical_kmer_codes(seq)
        if not codes:
            continue
        families.setdefault(family, []).append(sketch(codes))
        counts[family] = counts.get(family, 0) + 1

    index = {"k": K, "sketch_size": SKETCH_SIZE, "families": families}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(index, handle)

    print(f"Built k-mer index: {OUT_PATH}")
    print(f"  k={K}, sketch_size={SKETCH_SIZE}")
    print(f"  {len(families)} marker families indexed:")
    for family in sorted(counts):
        print(f"    {family:<14} {len(families[family]):>4} allele sketches  ({counts[family]} refs matched)")


if __name__ == "__main__":
    main()
