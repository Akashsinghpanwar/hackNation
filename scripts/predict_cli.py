"""
Standalone AMRShield inference CLI (no Streamlit).

Reads one JSON request from stdin, writes one JSON AnalysisResult to stdout.
Called by the Node.js server (server/pipeline/pythonBridge.ts).

Request modes:
  {"mode":"fasta","content":"<fasta text>","fileName":"x.fasta","species":"Escherichia coli"}
  {"mode":"tsv","content":"<AMRFinderPlus TSV>","species":"..."}
  {"mode":"bvbrc","genome_id":"562.12960","species":"..."}

Output: JSON matching shared/types.ts AnalysisResult (+ detected_genes, source, kmer_count).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import sys
import urllib.parse
import urllib.request
import warnings
from math import log2
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "bvbrc"
MODELS_DIR = ROOT / "artifacts" / "models"
FEATURE_COLS_PATH = DATA_DIR / "feature_columns.json"
FEATURE_META_PATH = DATA_DIR / "feature_meta.json"
KMER_INDEX_PATH = ROOT / "artifacts" / "kmer_index.json"

ANTIBIOTICS = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]
FAIL_THRESHOLD = 0.72
WORK_THRESHOLD = 0.28
CONTAINMENT_THRESHOLD = 0.5
K_MIN_SEQUENCE_BASES = 5_000
MIN_GENOME_BASES = 500_000
CORE_ECOLI_GENES = {"blaEC", "blaAmpC", "marA", "marR", "marB"}
SUPPORTED_SPECIES = "Escherichia coli"
DISCLAIMER = "Research prototype only. Confirm every result with standard laboratory susceptibility testing."
_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

GENE_FAMILIES: list[tuple[str, str]] = [
    ("blaCTX-M", r"CTX-M"), ("blaTEM", r"\bTEM\b|TEM family"), ("blaSHV", r"\bSHV\b|SHV family"),
    ("blaOXA", r"OXA family|Class D beta-lactamase"), ("blaCMY", r"\bCMY\b"), ("blaEC", r"BlaEC family"),
    ("blaAmpC", r"Class C beta-lactamase|AmpC"), ("betaLactamResProtein", r"beta-lactam resistance protein"),
    ("tetA", r"Tet\(A\)|TetA\b"), ("tetB", r"Tet\(B\)|TetB\b"), ("tetC", r"Tet\(C\)|TetC\b"),
    ("tetM", r"Tet\(M\)|TetM\b"), ("tetG", r"Tet\(G\)|TetG\b"), ("tetR", r"Tetracycline resistance regulatory|TetR\b"),
    ("qnrA", r"qnrA|quinolone resistance.*A"), ("qnrB", r"qnrB|quinolone resistance.*B"),
    ("qnrS", r"qnrS|quinolone resistance.*S"), ("qnrD", r"qnrD"), ("aac6Ib", r"AAC\(6'\).*Ib|aac\(6'\).*Ib"),
    ("oqxA", r"OqxA|oqxA"), ("oqxB", r"OqxB|oqxB"), ("aadA", r"aadA|ANT\(3''\)"), ("aac3", r"AAC\(3\)"),
    ("ant2", r"ANT\(2''\)"), ("aph3", r"APH\(3'\)|aminoglycoside 3'-phosphotransferase"),
    ("sul1", r"type-2.*sulfonamide|Sulfonamide resistance|dihydropteroate synthase type-2"),
    ("dhfr", r"dihydrofolate reductase|DHFR"), ("trimethoprimRes", r"trimethoprim"),
    ("marA", r"MarA\b"), ("marR", r"MarR\b"), ("marB", r"MarB\b"), ("acrAB", r"AcrA|AcrB|AcrAB"),
    ("tolC", r"TolC\b"), ("mdtEFG", r"MdtG|MdtH|MdtE|MdtF"), ("emrAB", r"EmrA|EmrB|EmrR"),
    ("qacE", r"QacE"), ("integraseI", r"integron integrase|class 1 integron"),
    ("catA", r"chloramphenicol acetyltransferase|cat[ABC]\b"), ("mcr", r"\bmcr-"),
]
TARGET_PATTERNS: dict[str, list[str]] = {
    "ampicillin": [r"penicillin-binding protein", r"\bPBP\b", r"\bftsI\b", r"\bmrdA\b", r"\bmrcA\b", r"\bmrcB\b"],
    "ceftriaxone": [r"penicillin-binding protein", r"\bPBP\b", r"\bftsI\b", r"\bmrdA\b", r"\bmrcA\b", r"\bmrcB\b"],
    "ciprofloxacin": [r"DNA gyrase", r"\bgyrA\b", r"\bgyrB\b", r"topoisomerase IV", r"\bparC\b", r"\bparE\b"],
    "tetracycline": [r"30S ribosomal", r"16S ribosomal", r"ribosomal protein S", r"\brrs[A-H]?\b"],
}
KNOWN_MARKERS: dict[str, set[str]] = {
    "ampicillin": {"blaTEM", "blaCTX-M", "blaSHV", "blaOXA", "blaCMY", "blaAmpC", "betaLactamResProtein"},
    "ciprofloxacin": {"qnrA", "qnrB", "qnrS", "qnrD", "aac6Ib", "oqxA", "oqxB"},
    "ceftriaxone": {"blaCTX-M", "blaCMY", "blaSHV", "blaOXA"},
    "tetracycline": {"tetA", "tetB", "tetC", "tetM", "tetG", "tetR"},
}
DRUG_CLASS = {
    "ampicillin": "Beta-lactam", "ciprofloxacin": "Fluoroquinolone",
    "ceftriaxone": "Cephalosporin", "tetracycline": "Tetracycline",
}
DRUG_TARGET = {
    "ampicillin": "Penicillin-binding proteins", "ciprofloxacin": "DNA gyrase / topoisomerase IV",
    "ceftriaxone": "Penicillin-binding proteins", "tetracycline": "30S ribosomal subunit",
}

_GENE_PAT = [(n, re.compile(p, re.IGNORECASE)) for n, p in GENE_FAMILIES]
_TARGET_PAT = {ab: [re.compile(p, re.IGNORECASE) for p in pats] for ab, pats in TARGET_PATTERNS.items()}


# ---------- loaders ----------
def load_models() -> dict:
    models = {}
    for ab in ANTIBIOTICS:
        path = MODELS_DIR / f"{ab}_model.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                models[ab] = pickle.load(fh)
    return models


def load_feature_cols() -> list[str]:
    with open(FEATURE_COLS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_kmer_index() -> dict | None:
    if not KMER_INDEX_PATH.exists():
        return None
    with open(KMER_INDEX_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_norm_stats() -> dict:
    if not FEATURE_META_PATH.exists():
        return {}
    with open(FEATURE_META_PATH, encoding="utf-8") as fh:
        return json.load(fh).get("norm_stats", {})


# ---------- gene / target detection ----------
def classify_products(products: list[str]) -> set[str]:
    found: set[str] = set()
    for product in products:
        for name, pat in _GENE_PAT:
            if pat.search(product):
                found.add(name)
    return found


def detect_targets(products: list[str]) -> set[str]:
    targets: set[str] = set()
    for ab, pats in _TARGET_PAT.items():
        if any(p.search(prod) for prod in products for p in pats):
            targets.add(ab)
    return targets


def fasta_headers_and_stats(text: str) -> tuple[list[str], int, int, float]:
    headers: list[str] = []
    bases = 0
    ambiguous = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(">"):
            headers.append(s[1:].strip())
            continue
        up = s.upper()
        for ch in up:
            if ch in ("A", "C", "G", "T"):
                bases += 1
            elif ch in ("N", "R", "Y", "S", "W", "K", "M"):
                bases += 1
                ambiguous += 1
    contigs = len(headers)
    amb_frac = (ambiguous / bases) if bases else 0.0
    return headers, bases, contigs, amb_frac


# ---------- k-mer detector ----------
def genome_kmer_set(text: str, k: int) -> set[int]:
    codes: set[int] = set()
    mask = (1 << (2 * k)) - 1
    top = 2 * (k - 1)
    for block in re.split(r"^>.*$", text, flags=re.MULTILINE):
        seq = "".join(block.split()).upper()
        fwd = rev = valid = 0
        for base in seq:
            c = _BASE_CODE.get(base)
            if c is None:
                fwd = rev = valid = 0
                continue
            fwd = ((fwd << 2) | c) & mask
            rev = (rev >> 2) | ((3 - c) << top)
            valid += 1
            if valid >= k:
                codes.add(fwd if fwd <= rev else rev)
    return codes


def detect_genes_from_dna(text: str) -> dict[str, float]:
    index = load_kmer_index()
    if index is None:
        return {}
    k = int(index["k"])
    families: dict[str, list[list[int]]] = index["families"]
    query = genome_kmer_set(text, k)
    detected: dict[str, float] = {}
    for family, sketches in families.items():
        best = 0.0
        for sk in sketches:
            if not sk:
                continue
            present = sum(1 for code in sk if code in query)
            ratio = present / len(sk)
            if ratio > best:
                best = ratio
            if best >= 0.99:
                break
        if best >= CONTAINMENT_THRESHOLD:
            detected[family] = round(best, 3)
    return detected


# ---------- BV-BRC ----------
def fetch_bvbrc_products(genome_id: str) -> list[str]:
    eid = urllib.parse.quote(genome_id.strip())
    terms = ["resistance", "lactamase", "efflux", "aminoglycoside", "integron", "gyrase",
             "topoisomerase", "quinolone", "ribosomal", "penicillin-binding", "tetracycline", "sulfonamide"]
    clauses = ",".join(f"eq(product,*{urllib.parse.quote(t)}*)" for t in terms)
    url = (f"https://www.bv-brc.org/api/genome_feature/?and(eq(genome_id,{eid}),or({clauses}))"
           "&select(gene,product)&limit(2000)&http_accept=application/json")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return [row.get("product", "") for row in data if row.get("product")]


def parse_amrfinder_tsv(text: str) -> list[str]:
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    products = []
    for row in rows:
        vals = [row.get(k, "") for k in ("Gene symbol", "Sequence name", "Element type", "Element subtype",
                                         "Class", "Subclass", "Method", "Name of closest sequence",
                                         "HMM description", "Mutation name")]
        products.append(" ".join(v for v in vals if v))
    return products


# ---------- feature vector + predict ----------
def build_feature_vector(genes: set[str], feature_cols: list[str],
                         genome_length_z: float, contigs_z: float) -> np.ndarray:
    gene_cols = [c for c in feature_cols if c not in ("amr_gene_burden", "genome_length_z", "contigs_z")]
    values = []
    for col in feature_cols:
        if col == "amr_gene_burden":
            values.append(float(sum(1 for g in gene_cols if g in genes)))
        elif col == "genome_length_z":
            values.append(float(genome_length_z))
        elif col == "contigs_z":
            values.append(float(contigs_z))
        else:
            values.append(1.0 if col in genes else 0.0)
    return np.array(values)


def target_status(ab: str, targets: set[str], species_supported: bool) -> tuple[str, bool]:
    if ab in targets:
        return "target_detected_in_annotations", True
    if species_supported:
        return "present_by_supported_e_coli_scope", True
    return "target_unknown", False


# ---------- confidence engine (pluggable) ----------
# A confidence engine turns a model probability into a tri-state decision + a
# confidence score. Swap engines via configs/app_config.json ("decision_policy":
# {"engine": ...}) or the CONFIDENCE_ENGINE env var. New engine = one subclass +
# one entry in ENGINES; nothing else in the pipeline changes.
CONFIG_PATH = ROOT / "configs" / "app_config.json"


def load_decision_policy() -> dict:
    policy = {"engine": "threshold", "fail_threshold": FAIL_THRESHOLD, "work_threshold": WORK_THRESHOLD}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            policy.update(json.load(fh).get("decision_policy", {}))
    except Exception:
        pass
    policy["engine"] = os.getenv("CONFIDENCE_ENGINE", policy.get("engine", "threshold"))
    return policy


class ThresholdEngine:
    """Default: fixed fail/work thresholds + symmetric max-probability confidence."""
    name = "threshold"

    def __init__(self, fail_threshold: float, work_threshold: float):
        self.fail = fail_threshold
        self.work = work_threshold

    def confidence(self, prob: float) -> float:
        return max(prob, 1.0 - prob)

    def decide(self, prob: float, target_ok: bool) -> dict:
        reasons: list[str] = []
        if prob >= self.fail:
            decision = "likely_to_fail"
            reasons.append("resistance_probability_above_fail_threshold")
        elif prob <= self.work:
            decision = "likely_to_work"
            reasons.append("resistance_probability_below_work_threshold")
        else:
            decision = "no_call"
            reasons.append("probability_in_no_call_region")
        if decision == "likely_to_work" and not target_ok:
            decision = "no_call"
            reasons.append("target_not_confirmed")
        return {"decision": decision, "confidence": self.confidence(prob), "reason_codes": reasons}


class EntropyEngine(ThresholdEngine):
    """Same decision boundaries, but confidence = 1 - binary entropy of the probability."""
    name = "entropy"

    def confidence(self, prob: float) -> float:
        p = min(max(prob, 1e-9), 1.0 - 1e-9)
        return round(1.0 + (p * log2(p) + (1.0 - p) * log2(1.0 - p)), 6)  # 1 - H(p)


ENGINES: dict[str, type[ThresholdEngine]] = {"threshold": ThresholdEngine, "entropy": EntropyEngine}


def get_engine(policy: dict) -> ThresholdEngine:
    cls = ENGINES.get(policy.get("engine", "threshold"), ThresholdEngine)
    return cls(float(policy.get("fail_threshold", FAIL_THRESHOLD)),
               float(policy.get("work_threshold", WORK_THRESHOLD)))


def predict(genes: set[str], targets: set[str], models: dict, feature_cols: list[str],
            species_supported: bool, genome_length_z: float, contigs_z: float,
            engine: ThresholdEngine) -> list[dict]:
    base_vector = build_feature_vector(genes, feature_cols, genome_length_z, contigs_z)
    predictions = []
    for ab in ANTIBIOTICS:
        pkg = models.get(ab)
        if not pkg:
            predictions.append(_empty_pred(ab, "No trained model available", "missing_model"))
            continue
        model_cols = pkg["feature_cols"]
        vector = np.array([base_vector[feature_cols.index(c)] if c in feature_cols else 0.0 for c in model_cols])
        if pkg.get("use_genetic_group"):
            vector = np.append(vector, pkg.get("gg_encoding", {}).get("_missing", 0.5))
        prob = float(pkg["model"].predict_proba(vector.reshape(1, -1))[0, 1])
        status, target_ok = target_status(ab, targets, species_supported)
        matched = sorted(KNOWN_MARKERS[ab] & genes)
        verdict = engine.decide(prob, target_ok)
        decision = verdict["decision"]
        confidence = verdict["confidence"]
        reason = list(verdict["reason_codes"])
        if matched:
            evidence = "known resistance gene or mutation detected"
        elif decision == "no_call":
            evidence = "weak or uncertain statistical evidence"
        else:
            evidence = "no known resistance signal found and target gate passed"
        predictions.append({
            "antibiotic": ab.capitalize(),
            "decision": decision,
            "resistance_probability": round(prob, 3),
            "calibrated_confidence": round(confidence, 3) if decision != "no_call" else None,
            "evidence_category": evidence,
            "supporting_markers": matched,
            "target_status": status,
            "ood_status": "in_distribution" if genes else "no_markers_detected",
            "reason_codes": reason,
            "explanation": _explain(ab, decision, prob, matched),
            "lab_confirmation_required": True,
            "drug_class": DRUG_CLASS[ab],
            "target": DRUG_TARGET[ab],
        })
    return predictions


def _empty_pred(ab: str, evidence: str, reason: str) -> dict:
    return {
        "antibiotic": ab.capitalize(), "decision": "no_call", "resistance_probability": None,
        "calibrated_confidence": None, "evidence_category": evidence, "supporting_markers": [],
        "target_status": "not_evaluated", "ood_status": "unknown", "reason_codes": [reason],
        "explanation": f"{ab.capitalize()}: {evidence}.", "lab_confirmation_required": True,
        "drug_class": DRUG_CLASS.get(ab, ""), "target": DRUG_TARGET.get(ab, ""),
    }


def _explain(ab: str, decision: str, prob: float, matched: list[str]) -> str:
    name = ab.capitalize()
    pct = f"{prob * 100:.0f}%"
    if decision == "likely_to_fail":
        why = f"resistance probability {pct}" + (f"; markers {', '.join(matched)}" if matched else "")
        return f"{name}: likely to fail ({why})."
    if decision == "likely_to_work":
        return f"{name}: likely to work (resistance probability {pct}, no strong resistance signal)."
    return f"{name}: no-call (resistance probability {pct} in the uncertain band)."


# ---------- request handling ----------
def build_markers(genes: set[str]) -> list[dict]:
    markers = []
    for g in sorted(genes):
        drug_class = next((ab for ab, s in KNOWN_MARKERS.items() if g in s), "")
        markers.append({
            "symbol": g, "element_type": "AMR", "drug_class": DRUG_CLASS.get(drug_class, "resistance"),
            "evidence_level": "reference_kmer_or_annotation", "identity": None, "coverage": None,
        })
    return markers


def handle(req: dict) -> dict:
    mode = req.get("mode", "fasta")
    species = req.get("species", SUPPORTED_SPECIES)
    species_supported = species == SUPPORTED_SPECIES
    models = load_models()
    feature_cols = load_feature_cols()
    stats = load_norm_stats()
    engine = get_engine(load_decision_policy())

    genes: set[str] = set()
    targets: set[str] = set()
    source = ""
    genome_length_z = 0.0
    contigs_z = 0.0
    bases = 0
    contigs = 0
    amb_frac = 0.0
    checksum = ""
    kmer_count = 0
    run_id = req.get("fileName", "run")

    if mode == "bvbrc":
        gid = req.get("genome_id", "").strip()
        products = fetch_bvbrc_products(gid)
        genes = classify_products(products)
        targets = detect_targets(products)
        if species_supported:
            targets |= set(ANTIBIOTICS)
        source = f"BV-BRC genome_feature: {gid}"
        run_id = gid or "bvbrc"
        checksum = hashlib.sha256(gid.encode()).hexdigest()[:16]
    elif mode == "tsv":
        products = parse_amrfinder_tsv(req.get("content", ""))
        genes = classify_products(products)
        targets = detect_targets(products)
        source = "AMRFinderPlus TSV"
        checksum = hashlib.sha256(req.get("content", "").encode()).hexdigest()[:16]
    else:  # fasta
        content = req.get("content", "")
        headers, bases, contigs, amb_frac = fasta_headers_and_stats(content)
        checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        header_genes = classify_products(headers)
        if header_genes:
            genes = header_genes
            targets = detect_targets(headers)
            source = "Annotated FASTA headers"
        elif bases >= K_MIN_SEQUENCE_BASES and load_kmer_index() is not None:
            genes = set(detect_genes_from_dna(content))
            kmer_count = 1
            source = "Built-in k-mer detector (NCBI AMR reference)"
            if bases >= MIN_GENOME_BASES:
                genes |= CORE_ECOLI_GENES
                targets = set(ANTIBIOTICS)
        else:
            source = "Raw FASTA (insufficient sequence / no detector)"
        if bases and stats:
            genome_length_z = (bases - float(stats.get("length_mean", 0))) / max(float(stats.get("length_std", 1)), 1)
            contigs_z = (contigs - float(stats.get("contigs_mean", 0))) / max(float(stats.get("contigs_std", 1)), 1)

    predictions = predict(genes, targets, models, feature_cols, species_supported,
                          genome_length_z, contigs_z, engine)

    warnings_list: list[str] = []
    if not species_supported:
        warnings_list.append(f"Unsupported species: {species}. Only {SUPPORTED_SPECIES} is configured.")
        for p in predictions:
            p["decision"] = "no_call"
            p["evidence_category"] = "Unsupported species scope"
            p["reason_codes"].append("unsupported_species")

    qc_status = "pass"
    if amb_frac > 0.08:
        qc_status = "warn"
        warnings_list.append(f"High ambiguous base fraction: {amb_frac:.2%}")
    if mode == "fasta" and bases and bases < K_MIN_SEQUENCE_BASES:
        qc_status = "warn"

    return {
        "run_id": run_id,
        "species": species,
        "source": source,
        "confidence_engine": engine.name,
        "detected_genes": sorted(genes),
        "kmer_count": kmer_count,
        "qc": {
            "genome_id": run_id,
            "qc_status": qc_status,
            "sequence_length": bases,
            "contig_count": contigs,
            "ambiguous_base_fraction": round(amb_frac, 5),
            "sha256": checksum,
            "warnings": warnings_list,
        },
        "markers": build_markers(genes),
        "predictions": predictions,
        "warnings": warnings_list,
        "disclaimer": DISCLAIMER,
    }


def main() -> None:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        result = handle(req)
        sys.stdout.write(json.dumps(result))
    except Exception as exc:  # surfaced to Node as a 500
        sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)


def serve() -> None:
    """
    Long-lived mode: load models/kmer index/feature cols ONCE, then handle one
    JSON request per stdin line for the life of the process, writing one JSON
    response per stdout line. Spawning a fresh interpreter (and re-importing
    numpy/scikit-learn/lightgbm, re-unpickling 4 models) per request is what
    was pushing a memory-constrained host (Render free tier, 512MB) into OOM —
    this mode is called once by pythonBridge.ts and reused across requests.
    """
    global load_models, load_feature_cols, load_kmer_index, load_norm_stats, load_decision_policy

    models = load_models()
    feature_cols = load_feature_cols()
    kmer_index = load_kmer_index()
    stats = load_norm_stats()
    policy = load_decision_policy()

    load_models = lambda: models
    load_feature_cols = lambda: feature_cols
    load_kmer_index = lambda: kmer_index
    load_norm_stats = lambda: stats
    load_decision_policy = lambda: policy

    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            result = handle(req)
            sys.stdout.write(json.dumps(result) + "\n")
        except Exception as exc:
            sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
    else:
        main()
