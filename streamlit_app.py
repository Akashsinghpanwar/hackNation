"""AMRShield Sentinel - Streamlit demo focused on the hackathon requirements."""
from __future__ import annotations

import csv
import json
import pickle
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "bvbrc"
MODELS_DIR = ROOT / "artifacts" / "models"
METRICS_PATH = ROOT / "artifacts" / "demo_metrics.json"
FEATURE_COLS_PATH = DATA_DIR / "feature_columns.json"
FEATURE_META_PATH = DATA_DIR / "feature_meta.json"
ANTIBIOTIC_SUMMARY_PATH = DATA_DIR / "summary_by_antibiotic.csv"
DEMO_SAMPLES_DIR = ROOT / "demo_samples"
REF_DIR = ROOT / "data" / "amr_reference"
KMER_INDEX_PATH = ROOT / "artifacts" / "kmer_index.json"

ANTIBIOTICS = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]
FAIL_THRESHOLD = 0.72
WORK_THRESHOLD = 0.28
CONTAINMENT_THRESHOLD = 0.5
K_MIN_SEQUENCE_BASES = 5_000     # enough real DNA to run the k-mer detector
MIN_GENOME_BASES = 500_000       # below this we do not assume a full E. coli assembly
# Chromosomal E. coli genes present in ~all isolates but absent from the acquired-AMR
# reference DB. For a real E. coli assembly they are effectively always present, so we
# set them by default to keep inference features consistent with training.
CORE_ECOLI_GENES = {"blaEC", "blaAmpC", "marA", "marR", "marB"}
DISCLAIMER = "Research prototype only. Confirm every result with standard laboratory susceptibility testing."
_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

Decision = Literal["likely_to_fail", "likely_to_work", "no_call"]

GENE_FAMILIES: list[tuple[str, str]] = [
    ("blaCTX-M", r"CTX-M"),
    ("blaTEM", r"\bTEM\b|TEM family"),
    ("blaSHV", r"\bSHV\b|SHV family"),
    ("blaOXA", r"OXA family|Class D beta-lactamase"),
    ("blaCMY", r"\bCMY\b"),
    ("blaEC", r"BlaEC family"),
    ("blaAmpC", r"Class C beta-lactamase|AmpC"),
    ("betaLactamResProtein", r"beta-lactam resistance protein"),
    ("tetA", r"Tet\(A\)|TetA\b"),
    ("tetB", r"Tet\(B\)|TetB\b"),
    ("tetC", r"Tet\(C\)|TetC\b"),
    ("tetM", r"Tet\(M\)|TetM\b"),
    ("tetG", r"Tet\(G\)|TetG\b"),
    ("tetR", r"Tetracycline resistance regulatory|TetR\b"),
    ("qnrA", r"qnrA|quinolone resistance.*A"),
    ("qnrB", r"qnrB|quinolone resistance.*B"),
    ("qnrS", r"qnrS|quinolone resistance.*S"),
    ("qnrD", r"qnrD"),
    ("aac6Ib", r"AAC\(6'\).*Ib|aac\(6'\).*Ib"),
    ("oqxA", r"OqxA|oqxA"),
    ("oqxB", r"OqxB|oqxB"),
    ("aadA", r"aadA|ANT\(3''\)"),
    ("aac3", r"AAC\(3\)"),
    ("ant2", r"ANT\(2''\)"),
    ("aph3", r"APH\(3'\)|aminoglycoside 3'-phosphotransferase"),
    ("sul1", r"type-2.*sulfonamide|Sulfonamide resistance|dihydropteroate synthase type-2"),
    ("dhfr", r"dihydrofolate reductase|DHFR"),
    ("trimethoprimRes", r"trimethoprim"),
    ("marA", r"MarA\b"),
    ("marR", r"MarR\b"),
    ("marB", r"MarB\b"),
    ("acrAB", r"AcrA|AcrB|AcrAB"),
    ("tolC", r"TolC\b"),
    ("mdtEFG", r"MdtG|MdtH|MdtE|MdtF"),
    ("emrAB", r"EmrA|EmrB|EmrR"),
    ("qacE", r"QacE"),
    ("integraseI", r"integron integrase|class 1 integron"),
    ("catA", r"chloramphenicol acetyltransferase|cat[ABC]\b"),
    ("mcr", r"\bmcr-"),
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

DRUG_INFO = {
    "ampicillin": {"drug_class": "Beta-lactam", "target": "Penicillin-binding proteins"},
    "ciprofloxacin": {"drug_class": "Fluoroquinolone", "target": "DNA gyrase / topoisomerase IV"},
    "ceftriaxone": {"drug_class": "Cephalosporin", "target": "Penicillin-binding proteins"},
    "tetracycline": {"drug_class": "Tetracycline", "target": "30S ribosomal subunit"},
}

COMPILED_GENE_PATTERNS = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in GENE_FAMILIES]
COMPILED_TARGET_PATTERNS = {
    antibiotic: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for antibiotic, patterns in TARGET_PATTERNS.items()
}


@dataclass
class FeatureEvidence:
    source: str
    genes: set[str]
    targets: set[str]
    products: list[str]
    amrfinder_rows: list[dict[str, str]]
    species_supported: bool = True
    base_count: int = 0
    contig_count: int = 0
    genome_length_z: float = 0.0
    contigs_z: float = 0.0
    requires_annotation: bool = False
    annotation_warning: str = ""


def inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --bg: #0b0f19;
  --bg2: #0e1524;
  --panel: #131b2e;
  --panel2: #172136;
  --line: #24304a;
  --ink: #e8eefc;
  --muted: #93a1bd;
  --brand: #10a37f;      /* ChatGPT green */
  --brand2: #0e8f6f;
  --blue: #4b8fe3;
  --orange: #f0913f;
  --green: #29c58b;
  --red: #f0616b;
  --gray: #8894ab;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 88% -8%, rgba(16,163,127,.16), transparent 60%),
    radial-gradient(900px 500px at 5% 110%, rgba(75,143,227,.10), transparent 55%),
    linear-gradient(180deg, var(--bg), var(--bg2));
  color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: #0c1220 !important;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: #cdd8ee !important; }
.block-container { padding-top: 1.1rem; max-width: 1560px; }
h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: .2px; }
p, label, .stMarkdown, span, li { color: var(--ink); }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--ink) !important; }

/* ---------- Top hero bar ---------- */
.hero {
  display:flex; align-items:center; justify-content:space-between;
  gap: 18px; padding: 16px 22px; margin: 2px 0 6px;
  background: linear-gradient(120deg, rgba(16,163,127,.14), rgba(19,27,46,.9) 40%, rgba(19,27,46,.9));
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: 0 12px 34px rgba(0,0,0,.35);
  position: relative; overflow: hidden;
}
.hero::after {
  content:""; position:absolute; right:-40px; top:-60px; width:320px; height:320px;
  background: radial-gradient(circle at center, rgba(16,163,127,.22), transparent 62%);
  pointer-events:none;
}
.hero-left { display:flex; align-items:center; gap:14px; z-index:1; }
.hero-title { margin:0; font-size:1.5rem; font-weight:800; color:#fff !important; line-height:1.1; }
.hero-sub { margin:3px 0 0; color:var(--muted) !important; font-size:.86rem; }
.hero-badges { display:flex; align-items:center; gap:10px; z-index:1; }
.badge {
  display:flex; align-items:center; gap:8px;
  background: rgba(255,255,255,.04);
  border: 1px solid var(--line);
  border-radius: 999px; padding: 7px 14px 7px 10px;
  font-weight:700; font-size:.82rem; color:#dfe8fb;
}
.badge small { color: var(--muted); font-weight:600; font-size:.68rem; display:block; line-height:1; }
.badge b { line-height:1.1; }

/* ---------- Horizontal top navigation (styled st.radio) ---------- */
div[data-testid="stRadio"] > div[role="radiogroup"] {
  flex-direction: row; flex-wrap: wrap; gap: 8px; align-items:center;
}
div[data-testid="stRadio"] label {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 999px; padding: 8px 18px; margin: 0 !important;
  cursor: pointer; transition: all .15s ease;
}
div[data-testid="stRadio"] label:hover { border-color: var(--brand); background: var(--panel2); }
div[data-testid="stRadio"] label > div:first-child { display: none; }   /* hide the radio dot */
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p { font-weight:700; font-size:.92rem; }
div[data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(90deg, var(--brand), var(--brand2));
  border-color: var(--brand);
  box-shadow: 0 8px 20px rgba(16,163,127,.35);
}
div[data-testid="stRadio"] label:has(input:checked) p { color:#fff !important; }

/* ---------- File uploader ---------- */
div[data-testid="stFileUploader"] {
  background: var(--panel); border: 1.5px dashed #33507a; border-radius: 12px; padding: 12px;
}
div[data-testid="stFileUploader"] * { color: var(--ink) !important; }
div[data-testid="stFileUploader"] button {
  background: var(--brand); color:#fff; border-radius: 8px; border: 1px solid var(--brand);
}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
  background: var(--panel2) !important; color: var(--ink) !important; border:1px solid var(--line) !important;
}
.stButton > button {
  background: linear-gradient(90deg, var(--brand), var(--brand2)); color:#fff;
  border:1px solid var(--brand); border-radius:10px; font-weight:700;
}
.stButton > button:hover { filter:brightness(1.08); }

/* ---------- Panels / cards ---------- */
.banner {
  background: linear-gradient(120deg, rgba(16,163,127,.16), rgba(19,27,46,.6));
  border: 1px solid var(--line); border-radius: 14px; padding: 16px 20px; margin-bottom: 14px;
}
.banner h1 { margin: 0; color:#fff !important; font-size:1.25rem; }
.banner p { color: var(--muted) !important; margin: 5px 0 0; }
.warning {
  background: rgba(240,145,63,.10); border: 1px solid rgba(240,145,63,.42);
  border-radius: 12px; padding: 11px 14px; color: #f6c99a;
}
.panel {
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 16px; margin: 10px 0; box-shadow: 0 10px 26px rgba(0,0,0,.28);
}
.kpi {
  background: linear-gradient(180deg, var(--panel2), var(--panel));
  border: 1px solid var(--line); border-top: 3px solid var(--brand);
  border-radius: 14px; padding: 13px 15px; min-height: 96px;
}
.kpi .label { color: var(--muted); font-size: .74rem; text-transform: uppercase; font-weight: 800; letter-spacing:.6px; }
.kpi .value { color: #fff; font-size: 1.7rem; font-weight: 800; margin-top: 4px; }
.kpi .sub { color: var(--muted); font-size: .8rem; margin-top: 3px; }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 15px; margin: 8px 0; box-shadow: 0 8px 20px rgba(0,0,0,.28);
}
.card h3 { margin: 8px 0 2px; }
.fail { border-left: 5px solid var(--red); }
.work { border-left: 5px solid var(--green); }
.nocall { border-left: 5px solid var(--gray); }
.pill {
  display:inline-block; padding: 4px 11px; border-radius: 999px;
  font-weight: 800; font-size:.7rem; letter-spacing: .4px;
}
.pill.fail { background:rgba(240,97,107,.14); color:var(--red); border:1px solid rgba(240,97,107,.5); }
.pill.work { background:rgba(41,197,139,.14); color:var(--green); border:1px solid rgba(41,197,139,.5); }
.pill.nocall { background:rgba(136,148,171,.14); color:#aeb9cf; border:1px solid rgba(136,148,171,.5); }
.chip {
  display:inline-block; border:1px solid var(--line); background: var(--panel2);
  color:#a9c6f0; border-radius:6px; padding:2px 8px; margin:2px;
  font-family: ui-monospace, monospace; font-size:.78rem;
}
.hit { border-color:rgba(240,97,107,.5); background:rgba(240,97,107,.12); color:#ff9aa1; }
.muted { color: var(--muted); font-size:.88rem; }
hr { border-color: var(--line) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
</style>
""",
        unsafe_allow_html=True,
    )


def _logo_shield_svg() -> str:
    return (
        '<svg width="40" height="40" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#19c79a"/><stop offset="1" stop-color="#0e8f6f"/></linearGradient></defs>'
        '<path d="M24 3 L42 10 V24 C42 35 34 42 24 45 C14 42 6 35 6 24 V10 Z" fill="url(#g)" opacity="0.95"/>'
        '<path d="M17 15 C31 20 17 28 31 33" stroke="#0b0f19" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
        '<path d="M31 15 C17 20 31 28 17 33" stroke="#0b0f19" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
        '<line x1="20" y1="18" x2="28" y2="18" stroke="#0b0f19" stroke-width="1.6"/>'
        '<line x1="21" y1="24" x2="27" y2="24" stroke="#0b0f19" stroke-width="1.6"/>'
        '<line x1="20" y1="30" x2="28" y2="30" stroke="#0b0f19" stroke-width="1.6"/>'
        "</svg>"
    )


def _logo_openai_svg() -> str:
    return (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="#10a37f" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M12 2.2c-2.1 0-3.9 1.3-4.6 3.2A4.9 4.9 0 0 0 4.2 13a4.9 4.9 0 0 0 .6 5.7c.9 1.9 3 3 5.1 2.7 '
        "1 .9 2.3 1.4 3.7 1.4 2.1 0 3.9-1.3 4.6-3.2a4.9 4.9 0 0 0 3.2-4.4c0-1.1-.3-2.1-.9-3a4.9 4.9 0 0 0-.6-5.7"
        "c-.9-1.9-3-3-5.1-2.7-1-.9-2.3-1.4-3.7-1.4Zm0 1.7c.9 0 1.7.3 2.4.9l-4 2.3a1 1 0 0 0-.5.9v4.7l-1.7-1V7.4"
        'c0-1.9 1.6-3.5 3.8-3.5Z" opacity="0.95"/></svg>'
    )


def _logo_hacknation_svg() -> str:
    return (
        '<svg width="18" height="20" viewBox="0 0 24 26" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M12 1 L22 6.5 V17.5 L12 23 L2 17.5 V6.5 Z" fill="none" stroke="#4b8fe3" stroke-width="1.8"/>'
        '<path d="M8.5 8 V17 M15.5 8 V17 M8.5 12.5 H15.5" stroke="#4b8fe3" stroke-width="1.8" stroke-linecap="round"/>'
        "</svg>"
    )


def render_topbar() -> None:
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-left">
    {_logo_shield_svg()}
    <div>
      <p class="hero-title">AMRShield&nbsp;Sentinel</p>
      <p class="hero-sub">Genome-based antibiotic-response prediction for <i>E. coli</i> &middot; predict before the lab result</p>
    </div>
  </div>
  <div class="hero-badges">
    <div class="badge">{_logo_hacknation_svg()}<span><small>Challenge&nbsp;06</small><b>HackNation</b></span></div>
    <div class="badge">{_logo_openai_svg()}<span><small>Built&nbsp;with</small><b>OpenAI</b></span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_models() -> dict:
    models = {}
    for antibiotic in ANTIBIOTICS:
        path = MODELS_DIR / f"{antibiotic}_model.pkl"
        if path.exists():
            with open(path, "rb") as handle:
                models[antibiotic] = pickle.load(handle)
    return models


@st.cache_resource
def load_kmer_index() -> dict | None:
    """Load the prebuilt AMR reference k-mer index (from scripts/05_build_reference_index.py)."""
    if not KMER_INDEX_PATH.exists():
        return None
    with open(KMER_INDEX_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def kmer_detector_available() -> bool:
    return KMER_INDEX_PATH.exists()


def genome_kmer_set(text: str, k: int) -> set[int]:
    """Canonical 2-bit-encoded k-mer set for a FASTA genome (both strands, per contig)."""
    codes: set[int] = set()
    mask = (1 << (2 * k)) - 1
    top_shift = 2 * (k - 1)
    for block in re.split(r"^>.*$", text, flags=re.MULTILINE):
        seq = "".join(block.split()).upper()
        fwd = rev = valid = 0
        for base in seq:
            code = _BASE_CODE.get(base)
            if code is None:
                fwd = rev = valid = 0
                continue
            fwd = ((fwd << 2) | code) & mask
            rev = (rev >> 2) | ((3 - code) << top_shift)
            valid += 1
            if valid >= k:
                codes.add(fwd if fwd <= rev else rev)
    return codes


def detect_amr_genes_from_dna(text: str) -> tuple[dict[str, float], int]:
    """Detect AMR marker families in a raw nucleotide FASTA via reference k-mer containment.

    Returns ({family: containment_score}, unique_kmer_count).
    """
    index = load_kmer_index()
    if index is None:
        return {}, 0
    k = int(index["k"])
    families: dict[str, list[list[int]]] = index["families"]
    query = genome_kmer_set(text, k)
    detected: dict[str, float] = {}
    for family, sketches in families.items():
        best = 0.0
        for sketch in sketches:
            if not sketch:
                continue
            present = sum(1 for code in sketch if code in query)
            ratio = present / len(sketch)
            if ratio > best:
                best = ratio
            if best >= 0.99:
                break
        if best >= CONTAINMENT_THRESHOLD:
            detected[family] = round(best, 3)
    return detected, len(query)


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_feature_cols() -> list[str]:
    with open(FEATURE_COLS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_feature_meta() -> dict:
    if not FEATURE_META_PATH.exists():
        return {}
    with open(FEATURE_META_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_manifest() -> dict:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_antibiotic_summary() -> list[dict]:
    if not ANTIBIOTIC_SUMMARY_PATH.exists():
        return []
    with open(ANTIBIOTIC_SUMMARY_PATH, newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "antibiotic": row["antibiotic"],
                    "resistant": int(row["resistant"]),
                    "susceptible": int(row["susceptible"]),
                    "total": int(row["total"]),
                    "unique_genomes": int(row["unique_genomes"]),
                }
            )
    return rows


def classify_products(products: list[str]) -> set[str]:
    found: set[str] = set()
    for product in products:
        for name, pattern in COMPILED_GENE_PATTERNS:
            if pattern.search(product):
                found.add(name)
    return found


def detect_targets(products: list[str]) -> set[str]:
    targets: set[str] = set()
    for antibiotic, patterns in COMPILED_TARGET_PATTERNS.items():
        if any(pattern.search(product) for product in products for pattern in patterns):
            targets.add(antibiotic)
    return targets


def fasta_stats(text: str) -> tuple[list[str], int, int, float, float]:
    headers: list[str] = []
    base_count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            headers.append(stripped[1:].strip())
            continue
        base_count += sum(1 for char in stripped.upper() if char in {"A", "C", "G", "T", "N"})

    stats = load_feature_meta().get("norm_stats", {})
    length_mean = float(stats.get("length_mean", 0.0) or 0.0)
    length_std = float(stats.get("length_std", 1.0) or 1.0)
    contigs_mean = float(stats.get("contigs_mean", 0.0) or 0.0)
    contigs_std = float(stats.get("contigs_std", 1.0) or 1.0)
    contig_count = len(headers)
    genome_length_z = (base_count - length_mean) / length_std if length_std else 0.0
    contigs_z = (contig_count - contigs_mean) / contigs_std if contigs_std else 0.0
    return headers, base_count, contig_count, genome_length_z, contigs_z


def headers_look_annotated(headers: list[str]) -> bool:
    annotation_pattern = re.compile(
        r"product=|gene=|protein|lactamase|resistance|efflux|gyrase|topoisomerase|ribosomal|"
        r"penicillin-binding|aminoglycoside|tetracycline|Escherichia|fig\|",
        re.IGNORECASE,
    )
    return any(annotation_pattern.search(header) for header in headers)


def parse_annotated_fasta(text: str, require_annotation: bool = False) -> FeatureEvidence:
    headers, base_count, contig_count, genome_length_z, contigs_z = fasta_stats(text)
    genes = classify_products(headers)
    targets = detect_targets(headers)
    missing_annotation = require_annotation and not genes and not targets and not headers_look_annotated(headers)
    warning = (
        "Raw FASTA sequence was loaded, but no gene/product annotations were detected. "
        "This model needs AMR marker features from AMRFinderPlus or annotated FASTA headers."
        if missing_annotation
        else ""
    )
    return FeatureEvidence(
        source="Annotated FASTA headers",
        genes=genes,
        targets=targets,
        products=headers,
        amrfinder_rows=[],
        base_count=base_count,
        contig_count=contig_count,
        genome_length_z=genome_length_z,
        contigs_z=contigs_z,
        requires_annotation=missing_annotation,
        annotation_warning=warning,
    )


def parse_amrfinder_tsv(text: str) -> FeatureEvidence:
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    products: list[str] = []
    for row in rows:
        values = [
            row.get("Gene symbol", ""),
            row.get("Sequence name", ""),
            row.get("Element type", ""),
            row.get("Element subtype", ""),
            row.get("Class", ""),
            row.get("Subclass", ""),
            row.get("Method", ""),
            row.get("Name of closest sequence", ""),
            row.get("HMM description", ""),
            row.get("Mutation name", ""),
        ]
        products.append(" ".join(value for value in values if value))
    return FeatureEvidence(
        source="AMRFinderPlus TSV",
        genes=classify_products(products),
        targets=detect_targets(products),
        products=products,
        amrfinder_rows=rows,
    )


def amrfinder_available() -> bool:
    return shutil.which("amrfinder") is not None


def run_amrfinder_on_fasta(fasta_text: str) -> FeatureEvidence:
    executable = shutil.which("amrfinder")
    if not executable:
        raise RuntimeError("AMRFinderPlus is not installed or not on PATH.")
    with tempfile.TemporaryDirectory() as tmp:
        fasta_path = Path(tmp) / "input.fna"
        out_path = Path(tmp) / "amrfinder.tsv"
        fasta_path.write_text(fasta_text, encoding="utf-8")
        command = [executable, "-n", str(fasta_path), "-O", "Escherichia", "-o", str(out_path)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
        return parse_amrfinder_tsv(out_path.read_text(encoding="utf-8"))


def fetch_bvbrc_products(genome_id: str) -> FeatureEvidence:
    encoded_id = urllib.parse.quote(genome_id.strip())
    # IMPORTANT: BV-BRC RQL wildcard terms must be single words. A multi-word term
    # (e.g. "DNA gyrase") url-encodes to contain %20 and silently breaks the whole
    # or() clause, returning almost no rows. Use single tokens only.
    product_terms = [
        "resistance",
        "lactamase",
        "efflux",
        "aminoglycoside",
        "integron",
        "gyrase",
        "topoisomerase",
        "quinolone",
        "ribosomal",
        "penicillin-binding",
        "tetracycline",
        "sulfonamide",
    ]
    clauses = ",".join(f"eq(product,*{urllib.parse.quote(term)}*)" for term in product_terms)
    url = (
        "https://www.bv-brc.org/api/genome_feature/"
        f"?and(eq(genome_id,{encoded_id}),or({clauses}))"
        "&select(gene,product)&limit(2000)&http_accept=application/json"
    )
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read())
    products = [row.get("product", "") for row in data if row.get("product")]
    return FeatureEvidence(
        source=f"BV-BRC genome_feature: {genome_id}",
        genes=classify_products(products),
        targets=detect_targets(products),
        products=products,
        amrfinder_rows=[],
    )


def build_feature_vector(evidence: FeatureEvidence, feature_cols: list[str]) -> np.ndarray:
    values = []
    gene_feature_cols = [col for col in feature_cols if col not in {"amr_gene_burden", "genome_length_z", "contigs_z"}]
    for col in feature_cols:
        if col == "amr_gene_burden":
            values.append(float(sum(1 for gene in gene_feature_cols if gene in evidence.genes)))
        elif col == "genome_length_z":
            values.append(float(getattr(evidence, "genome_length_z", 0.0)))
        elif col == "contigs_z":
            values.append(float(getattr(evidence, "contigs_z", 0.0)))
        else:
            values.append(1.0 if col in evidence.genes else 0.0)
    return np.array(values)


def target_status(antibiotic: str, evidence: FeatureEvidence) -> tuple[str, bool]:
    if antibiotic in evidence.targets:
        return "target_detected_in_annotations", True
    if evidence.species_supported:
        return "present_by_supported_e_coli_scope", True
    return "target_unknown", False


def predict(evidence: FeatureEvidence, models: dict, feature_cols: list[str]) -> list[dict]:
    if getattr(evidence, "requires_annotation", False):
        return [
            {
                "antibiotic": antibiotic,
                "decision": "no_call",
                "prob": None,
                "confidence": None,
                "evidence_category": "raw FASTA requires AMRFinderPlus or annotated gene features",
                "markers": [],
                "target_status": "not_evaluated",
                "reason_codes": ["missing_amrfinderplus_annotation"],
            }
            for antibiotic in ANTIBIOTICS
        ]

    base_vector = build_feature_vector(evidence, feature_cols)
    predictions = []
    for antibiotic in ANTIBIOTICS:
        package = models.get(antibiotic)
        if not package:
            predictions.append(
                {
                    "antibiotic": antibiotic,
                    "decision": "no_call",
                    "prob": None,
                    "confidence": None,
                    "evidence_category": "No trained model available",
                    "markers": [],
                    "target_status": "not_evaluated",
                    "reason_codes": ["missing_model"],
                }
            )
            continue

        model_cols = package["feature_cols"]
        vector = np.array([base_vector[feature_cols.index(col)] if col in feature_cols else 0.0 for col in model_cols])
        if package.get("use_genetic_group"):
            vector = np.append(vector, package.get("gg_encoding", {}).get("_missing", 0.5))

        probability = float(package["model"].predict_proba(vector.reshape(1, -1))[0, 1])
        confidence = max(probability, 1.0 - probability)
        status, target_ok = target_status(antibiotic, evidence)
        matched = sorted(KNOWN_MARKERS[antibiotic] & evidence.genes)
        reason_codes: list[str] = []

        if probability >= FAIL_THRESHOLD:
            decision: Decision = "likely_to_fail"
            reason_codes.append("resistance_probability_above_fail_threshold")
        elif probability <= WORK_THRESHOLD:
            decision = "likely_to_work"
            reason_codes.append("resistance_probability_below_work_threshold")
        else:
            decision = "no_call"
            reason_codes.append("probability_in_no_call_region")

        if decision == "likely_to_work" and not target_ok:
            decision = "no_call"
            reason_codes.append("target_not_confirmed")

        if matched:
            evidence_category = "known resistance gene or mutation detected"
        elif decision == "no_call":
            evidence_category = "weak or uncertain statistical evidence"
        else:
            evidence_category = "no known resistance signal found and target gate passed"

        predictions.append(
            {
                "antibiotic": antibiotic,
                "decision": decision,
                "prob": probability,
                "confidence": confidence if decision != "no_call" else None,
                "evidence_category": evidence_category,
                "markers": matched,
                "target_status": status,
                "reason_codes": reason_codes,
            }
        )
    return predictions


def decision_label(decision: str) -> str:
    return {
        "likely_to_fail": "LIKELY TO FAIL",
        "likely_to_work": "LIKELY TO WORK",
        "no_call": "NO CALL",
    }.get(decision, "NO CALL")


def decision_color(decision: str) -> str:
    return {
        "likely_to_fail": "#c43f3f",
        "likely_to_work": "#2e8b57",
        "no_call": "#7b8494",
    }.get(decision, "#7b8494")


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.1%}"


def plotly_go():
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        return None


def tableau_layout(fig, height: int = 360) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#172033"},
        margin={"l": 48, "r": 22, "t": 52, "b": 44},
        legend={"orientation": "h", "y": -0.18},
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="#e7ebf2", zerolinecolor="#d7dde8")
    fig.update_yaxes(gridcolor="#e7ebf2", zerolinecolor="#d7dde8")


def kpi_card(label: str, value: str, sub: str = "", accent: str = "#2f6db3") -> None:
    st.markdown(
        f"""
<div class="kpi" style="border-top-color:{accent}">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_prediction_overview(predictions: list[dict], evidence: FeatureEvidence) -> None:
    fail_count = sum(1 for item in predictions if item["decision"] == "likely_to_fail")
    work_count = sum(1 for item in predictions if item["decision"] == "likely_to_work")
    no_call_count = sum(1 for item in predictions if item["decision"] == "no_call")
    max_risk = max((item["prob"] or 0.0 for item in predictions), default=0.0)
    cols = st.columns(4)
    with cols[0]:
        kpi_card("Detected marker families", str(len(evidence.genes)), f"Source: {evidence.source}", "#2f6db3")
    with cols[1]:
        kpi_card("Likely to fail", str(fail_count), "High resistance probability", "#c43f3f")
    with cols[2]:
        kpi_card("Likely to work", str(work_count), "Target gate must pass", "#2e8b57")
    with cols[3]:
        kpi_card("No-call", str(no_call_count), f"Max P(resistant): {max_risk:.1%}", "#7b8494")


def plot_prediction_probability(predictions: list[dict]) -> None:
    go = plotly_go()
    if go is None:
        st.info("Install plotly for dashboard charts.")
        return
    rows = sorted(predictions, key=lambda item: item["prob"] or -1, reverse=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[item["prob"] or 0.0 for item in rows],
            y=[item["antibiotic"].capitalize() for item in rows],
            orientation="h",
            marker={"color": [decision_color(item["decision"]) for item in rows]},
            text=[decision_label(item["decision"]) for item in rows],
            textposition="auto",
            hovertemplate="%{y}<br>P(resistant): %{x:.1%}<extra></extra>",
        )
    )
    fig.add_vline(x=FAIL_THRESHOLD, line_dash="dash", line_color="#c43f3f", annotation_text="fail")
    fig.add_vline(x=WORK_THRESHOLD, line_dash="dash", line_color="#2e8b57", annotation_text="work")
    fig.update_layout(title="Resistance Probability by Antibiotic", xaxis_title="P(resistant)", yaxis_title="")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    tableau_layout(fig, height=330)
    st.plotly_chart(fig, use_container_width=True)


def plot_marker_heatmap(evidence: FeatureEvidence, feature_cols: list[str]) -> None:
    go = plotly_go()
    if go is None:
        return
    marker_cols = [col for col in feature_cols if col not in {"amr_gene_burden", "genome_length_z", "contigs_z"}]
    values = [1 if col in evidence.genes else 0 for col in marker_cols]
    fig = go.Figure(
        data=go.Heatmap(
            z=[values],
            x=marker_cols,
            y=["sample"],
            colorscale=[[0, "#edf1f7"], [1, "#2f6db3"]],
            showscale=False,
            hovertemplate="%{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Detected AMR Marker Matrix", xaxis_title="", yaxis_title="")
    tableau_layout(fig, height=230)
    st.plotly_chart(fig, use_container_width=True)


def plot_metric_comparison(metrics: dict) -> None:
    go = plotly_go()
    if go is None:
        st.info("Install plotly for dashboard charts.")
        return
    antibiotics = [name.capitalize() for name in metrics]
    fig = go.Figure()
    series = [
        ("AUROC", "auroc", "#2f6db3"),
        ("PR-AUC", "pr_auc", "#e8743b"),
        ("Balanced accuracy", "balanced_accuracy_called", "#2e8b57"),
    ]
    for label, key, color in series:
        fig.add_trace(
            go.Bar(
                name=label,
                x=antibiotics,
                y=[metrics[name].get(key) for name in metrics],
                marker={"color": color},
                hovertemplate=f"{label}: " + "%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(title="Held-out Model Quality", yaxis_title="Score", barmode="group")
    fig.update_yaxes(range=[0, 1])
    tableau_layout(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)


def plot_no_call_brier(metrics: dict) -> None:
    go = plotly_go()
    if go is None:
        return
    antibiotics = [name.capitalize() for name in metrics]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="No-call rate",
            x=antibiotics,
            y=[metrics[name].get("no_call_rate") for name in metrics],
            marker={"color": "#7b8494"},
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Brier score",
            x=antibiotics,
            y=[metrics[name].get("brier_score") for name in metrics],
            mode="lines+markers",
            marker={"color": "#e8743b"},
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Abstention and Calibration Error",
        yaxis={"title": "No-call rate", "tickformat": ".0%", "range": [0, 0.35]},
        yaxis2={"title": "Brier", "overlaying": "y", "side": "right", "range": [0, 0.18]},
    )
    tableau_layout(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)


def plot_reliability_curves(metrics: dict) -> None:
    go = plotly_go()
    if go is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line={"dash": "dash", "color": "#7b8494"}))
    colors = ["#2f6db3", "#e8743b", "#2e8b57", "#6f5fb8"]
    for color, (antibiotic, metric) in zip(colors, metrics.items()):
        bins = metric.get("reliability", [])
        if not bins:
            continue
        fig.add_trace(
            go.Scatter(
                x=[item["mean_prob"] for item in bins],
                y=[item["frac_positive"] for item in bins],
                mode="lines+markers",
                name=antibiotic.capitalize(),
                line={"color": color},
            )
        )
    fig.update_layout(title="Reliability Curves", xaxis_title="Mean predicted resistance", yaxis_title="Observed resistance")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    tableau_layout(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)


def plot_antibiotic_summary(summary: list[dict]) -> None:
    go = plotly_go()
    if go is None or not summary:
        return
    top_rows = sorted(summary, key=lambda row: row["total"], reverse=True)[:14]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Susceptible",
            y=[row["antibiotic"] for row in top_rows],
            x=[row["susceptible"] for row in top_rows],
            orientation="h",
            marker={"color": "#2f6db3"},
        )
    )
    fig.add_trace(
        go.Bar(
            name="Resistant",
            y=[row["antibiotic"] for row in top_rows],
            x=[row["resistant"] for row in top_rows],
            orientation="h",
            marker={"color": "#e8743b"},
        )
    )
    fig.update_layout(title="BV-BRC AST Label Volume", xaxis_title="Rows", yaxis_title="", barmode="stack")
    tableau_layout(fig, height=560)
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_cards(predictions: list[dict]) -> None:
    columns = st.columns(4)
    for column, item in zip(columns, predictions):
        decision = item["decision"]
        css = "fail" if decision == "likely_to_fail" else "work" if decision == "likely_to_work" else "nocall"
        probability = item["prob"]
        probability_text = "NA" if probability is None else f"{probability:.1%}"
        confidence = item["confidence"]
        confidence_text = "abstained" if confidence is None else f"{confidence:.1%}"
        markers = item["markers"] or []
        marker_html = " ".join(f'<span class="chip hit">{marker}</span>' for marker in markers) or "<span class='muted'>None</span>"
        info = DRUG_INFO[item["antibiotic"]]
        with column:
            st.markdown(
                f"""
<div class="card {css}">
  <span class="pill {css}">{decision_label(decision)}</span>
  <h3>{item['antibiotic'].capitalize()}</h3>
  <div class="muted">{info['drug_class']} | Target: {info['target']}</div>
  <p>P(resistant): <b>{probability_text}</b><br/>Confidence: <b>{confidence_text}</b></p>
  <div class="muted">Evidence</div>
  <div>{marker_html}</div>
  <p class="muted">Target gate: {item['target_status']}<br/>Category: {item['evidence_category']}</p>
</div>
""",
                unsafe_allow_html=True,
            )


def render_source_details(evidence: FeatureEvidence) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Input Evidence")
    st.write(f"Source: `{evidence.source}`")
    st.write(f"FASTA stats: `{getattr(evidence, 'contig_count', 0):,}` contigs / `{getattr(evidence, 'base_count', 0):,}` bases")
    st.write(f"Detected AMR feature families: `{len(evidence.genes)}`")
    if getattr(evidence, "annotation_warning", ""):
        st.warning(evidence.annotation_warning)
        st.info("The ML model is loaded, but raw DNA FASTA must be converted into AMR marker features before prediction.")
        st.code(
            "amrfinder -n your_genome.fna -O Escherichia -o your_genome.amrfinder.tsv\n"
            "# Then upload your_genome.amrfinder.tsv in the AMRFinderPlus TSV tab.",
            language="bash",
        )
    if evidence.genes:
        st.markdown(" ".join(f'<span class="chip">{gene}</span>' for gene in sorted(evidence.genes)), unsafe_allow_html=True)
    st.write(f"Detected target families: `{', '.join(sorted(evidence.targets)) or 'none from annotations'}`")
    if evidence.amrfinder_rows:
        with st.expander("AMRFinderPlus rows"):
            st.dataframe(evidence.amrfinder_rows, use_container_width=True)
    elif evidence.products:
        with st.expander("Raw annotation products"):
            st.write(evidence.products[:300])
    st.markdown("</div>", unsafe_allow_html=True)


def set_evidence(evidence: FeatureEvidence) -> None:
    st.session_state["evidence"] = evidence


def get_evidence() -> FeatureEvidence | None:
    value = st.session_state.get("evidence")
    return value if isinstance(value, FeatureEvidence) else None


def clear_evidence() -> None:
    st.session_state.pop("evidence", None)


def decode_upload(upload) -> str:
    return upload.getvalue().decode("utf-8", errors="replace")


def build_evidence_from_dna(fasta_text: str, source: str) -> FeatureEvidence:
    """Raw nucleotide FASTA -> AMR marker features using the reference k-mer detector."""
    headers, base_count, contig_count, genome_length_z, contigs_z = fasta_stats(fasta_text)
    detected, n_kmers = detect_amr_genes_from_dna(fasta_text)
    genes = set(detected)
    is_genome = base_count >= MIN_GENOME_BASES
    if is_genome:
        # A real E. coli assembly: add core chromosomal genes always present in the species.
        genes |= CORE_ECOLI_GENES
    targets = set(ANTIBIOTICS) if is_genome else detect_targets(headers)
    warning = ""
    if not is_genome:
        warning = (
            f"Only {base_count:,} bases found. This looks like a gene/fragment file, not a full "
            "genome assembly, so species-level core genes were not assumed."
        )
    return FeatureEvidence(
        source=f"Built-in k-mer detector (NCBI AMR reference): {source}",
        genes=genes,
        targets=targets,
        products=[f"{fam} (containment {score:.0%})" for fam, score in sorted(detected.items())],
        amrfinder_rows=[],
        base_count=base_count,
        contig_count=contig_count,
        genome_length_z=genome_length_z,
        contigs_z=contigs_z,
        requires_annotation=False,
        annotation_warning=warning,
    )


def process_fasta_text(fasta_text: str, source: str, run_amrfinder: bool) -> FeatureEvidence:
    if run_amrfinder:
        evidence = run_amrfinder_on_fasta(fasta_text)
        evidence.source = f"AMRFinderPlus from FASTA: {source}"
        return evidence

    # If headers already NAME specific AMR genes, trust them directly.
    headers, base_count, _, _, _ = fasta_stats(fasta_text)
    if classify_products(headers):
        evidence = parse_annotated_fasta(fasta_text, require_annotation=False)
        evidence.source = f"Annotated FASTA headers: {source}"
        return evidence

    # Otherwise, if there is real nucleotide sequence, detect AMR genes from the DNA
    # itself with the reference k-mer index. (A species name in a contig header does
    # not count as gene annotation.)
    if kmer_detector_available() and base_count >= K_MIN_SEQUENCE_BASES:
        return build_evidence_from_dna(fasta_text, source)

    # Annotated-looking headers but no detectable genes and no usable sequence.
    if headers_look_annotated(headers):
        evidence = parse_annotated_fasta(fasta_text, require_annotation=False)
        evidence.source = f"Annotated FASTA headers: {source}"
        return evidence

    # No detector index available -> fall back to the safe annotation-required path.
    evidence = parse_annotated_fasta(fasta_text, require_annotation=True)
    evidence.source = f"Annotated FASTA fallback: {source}"
    return evidence


def render_input_panel() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Genome Input")
    mode = st.radio(
        "Input source",
        ["FASTA upload", "AMRFinderPlus TSV", "BV-BRC Genome ID", "Demo sample"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "FASTA upload":
        amrfinder_ok = amrfinder_available()
        kmer_ok = kmer_detector_available()
        st.caption(
            "Upload accepts .fa, .fasta, .fna, .ffn, .faa, or .txt. "
            "Raw nucleotide assemblies are supported - no external tools required."
        )
        if kmer_ok:
            st.info("Built-in AMR gene detector (NCBI reference k-mer index): ready")
        else:
            st.warning("k-mer index missing. Run: python scripts/05_build_reference_index.py")
        run_amrfinder = st.checkbox(
            "Prefer AMRFinderPlus if installed",
            value=amrfinder_ok,
            disabled=not amrfinder_ok,
            help="AMRFinderPlus on PATH: " + ("yes" if amrfinder_ok else "no"),
        )
        upload = st.file_uploader(
            "Drop or browse FASTA",
            type=["fa", "fasta", "fna", "ffn", "faa", "txt"],
            key="fasta_upload",
            help="Raw assembled genome FASTA is scanned by the built-in reference k-mer detector.",
        )
        pasted = st.text_area("Or paste FASTA text", height=130, placeholder=">contig_1\nATGCATGCATGC...")
        source = upload.name if upload else "pasted FASTA"
        fasta_text = decode_upload(upload) if upload else pasted.strip()
        if fasta_text:
            try:
                with st.spinner("Scanning genome for AMR genes (k-mer detection)..."):
                    evidence = process_fasta_text(fasta_text, source, run_amrfinder)
                set_evidence(evidence)
                st.success(f"Loaded FASTA input: {source}")
                if getattr(evidence, "requires_annotation", False):
                    st.warning(evidence.annotation_warning)
                elif evidence.genes:
                    st.info(f"Detected {len(evidence.genes)} AMR marker families: " + ", ".join(sorted(evidence.genes)))
                else:
                    st.info("No acquired resistance genes detected; prediction relies on absence of markers and the target gate.")
            except Exception as exc:
                st.error(f"FASTA analysis failed: {exc}")

    elif mode == "AMRFinderPlus TSV":
        st.caption("Best path when AMRFinderPlus was run outside Streamlit.")
        tsv = st.file_uploader("Drop or browse AMRFinderPlus TSV", type=["tsv", "txt", "tab"], key="amrfinder_tsv")
        pasted_tsv = st.text_area("Or paste AMRFinderPlus TSV", height=140)
        tsv_text = decode_upload(tsv) if tsv else pasted_tsv.strip()
        if tsv_text:
            evidence = parse_amrfinder_tsv(tsv_text)
            evidence.source = f"AMRFinderPlus TSV: {tsv.name if tsv else 'pasted TSV'}"
            set_evidence(evidence)
            st.success("Parsed AMRFinderPlus TSV.")

    elif mode == "BV-BRC Genome ID":
        st.caption("Live demo path using BV-BRC public genome_feature annotations.")
        genome_id = st.text_input("BV-BRC genome ID", placeholder="562.12345")
        if st.button("Fetch BV-BRC annotations", type="primary", use_container_width=True) and genome_id.strip():
            try:
                with st.spinner("Querying BV-BRC genome_feature API..."):
                    set_evidence(fetch_bvbrc_products(genome_id))
                st.success("Fetched BV-BRC annotations.")
            except Exception as exc:
                st.error(f"BV-BRC API error: {exc}")

    else:
        demos = {
            "MDR E. coli": "mdr_ecoli.fasta",
            "Cipro resistant": "cipro_resistant_ecoli.fasta",
            "Susceptible-like": "susceptible_ecoli.fasta",
            "No-call sample": "no_call_sample.fasta",
        }
        selected = st.selectbox("Demo sample", list(demos))
        if st.button("Load demo sample", type="primary", use_container_width=True):
            text = (DEMO_SAMPLES_DIR / demos[selected]).read_text(encoding="utf-8")
            evidence = parse_annotated_fasta(text, require_annotation=True)
            evidence.source = f"Demo annotated FASTA: {demos[selected]}"
            set_evidence(evidence)
            st.success("Demo sample loaded.")

    if get_evidence() is not None and st.button("Clear current sample"):
        clear_evidence()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_decision_table(predictions: list[dict]) -> None:
    st.dataframe(
        [
            {
                "antibiotic": item["antibiotic"],
                "decision": item["decision"],
                "probability_resistant": item["prob"],
                "confidence": item["confidence"],
                "evidence_category": item["evidence_category"],
                "supporting_markers": ", ".join(item["markers"]),
                "target_status": item["target_status"],
                "reason_codes": ", ".join(item["reason_codes"]),
                "lab_confirmation_required": True,
            }
            for item in predictions
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_prediction_dashboard(evidence: FeatureEvidence, models: dict, feature_cols: list[str]) -> None:
    predictions = predict(evidence, models, feature_cols)
    render_prediction_overview(predictions, evidence)
    if getattr(evidence, "requires_annotation", False):
        st.markdown(
            """
<div class="panel">
  <h3>Prediction blocked: annotation required</h3>
  <p class="muted">This uploaded FASTA contains sequence bases, but no detected gene/product names.
  Upload AMRFinderPlus TSV or install AMRFinderPlus so the app can extract AMR marker features.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        render_decision_table(predictions)
        render_source_details(evidence)
        st.markdown(f'<div class="warning">{DISCLAIMER}</div>', unsafe_allow_html=True)
        return

    left, right = st.columns([1.12, 0.88])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        plot_prediction_probability(predictions)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        plot_marker_heatmap(evidence, feature_cols)
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Decision Report")
    render_prediction_cards(predictions)
    render_decision_table(predictions)
    render_source_details(evidence)
    st.markdown(f'<div class="warning">{DISCLAIMER}</div>', unsafe_allow_html=True)


def page_predict(models: dict, feature_cols: list[str]) -> None:
    st.markdown(f'<div class="warning">{DISCLAIMER}</div>', unsafe_allow_html=True)

    render_input_panel()
    evidence = get_evidence()
    if evidence is None:
        st.markdown(
            """
<div class="panel">
  <h3>Dashboard waiting for genome input</h3>
  <p class="muted">Upload a FASTA, upload/paste AMRFinderPlus TSV, fetch a BV-BRC genome ID, or load a demo sample.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    st.divider()
    render_prediction_dashboard(evidence, models, feature_cols)


def page_metrics(metrics: dict) -> None:
    st.header("Model Metrics")
    if not metrics:
        st.warning("No metrics file found.")
        return

    cols = st.columns(4)
    for idx, (antibiotic, metric) in enumerate(metrics.items()):
        with cols[idx % 4]:
            kpi_card(
                antibiotic.capitalize(),
                f"{metric.get('auroc', 0):.3f}",
                f"PR-AUC {metric.get('pr_auc', 0):.3f} | n={metric.get('n_test')}",
                "#2f6db3",
            )

    left, right = st.columns(2)
    with left:
        plot_metric_comparison(metrics)
    with right:
        plot_no_call_brier(metrics)
    plot_reliability_curves(metrics)

    rows = []
    for antibiotic, metric in metrics.items():
        rows.append(
            {
                "Antibiotic": antibiotic.capitalize(),
                "Test samples": metric.get("n_test"),
                "AUROC": metric.get("auroc"),
                "PR-AUC": metric.get("pr_auc"),
                "Brier": metric.get("brier_score"),
                "No-call rate": metric.get("no_call_rate"),
                "Answered accuracy": metric.get("answered_accuracy"),
                "Balanced accuracy": metric.get("balanced_accuracy_called"),
                "Resistant recall": metric.get("resistant_recall_called"),
                "Susceptible recall": metric.get("susceptible_recall_called"),
            }
        )
    st.subheader("Metric Table")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Evaluation uses the held-out grouped split from `data/bvbrc/splits.json`.")

    render_generalization(metrics)


def plot_generalization(metrics: dict) -> None:
    go = plotly_go()
    if go is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 5], y=[0, 0], mode="lines", showlegend=False, line={"color": "rgba(0,0,0,0)"}))
    colors = {"likely": "#2f6db3"}
    x_labels, overall_vals, mean_vals, min_vals = [], [], [], []
    for antibiotic, metric in metrics.items():
        gen = metric.get("generalization")
        if not gen:
            continue
        x_labels.append(antibiotic.capitalize())
        overall_vals.append(metric.get("auroc"))
        mean_vals.append(gen.get("auroc_mean"))
        min_vals.append(gen.get("auroc_min"))
    if not x_labels:
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Overall AUROC", x=x_labels, y=overall_vals, marker={"color": "#2f6db3"}))
    fig.add_trace(go.Bar(name="Per-group mean AUROC", x=x_labels, y=mean_vals, marker={"color": "#e8743b"}))
    fig.add_trace(go.Bar(name="Worst-group AUROC", x=x_labels, y=min_vals, marker={"color": "#c43f3f"}))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#7b8494", annotation_text="random")
    fig.update_layout(title="Generalization: Overall vs Within-Genetic-Group AUROC", barmode="group", yaxis_title="AUROC")
    fig.update_yaxes(range=[0, 1])
    tableau_layout(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


def render_generalization(metrics: dict) -> None:
    has_any = any(metric.get("generalization") for metric in metrics.values())
    if not has_any:
        return
    st.subheader("Generalization by Genetic Group")
    st.caption(
        "Held-out genomes are grouped by cgMLST-derived lineage. Overall AUROC can be inflated by "
        "differences BETWEEN lineages; within-group AUROC shows how well the model ranks resistance "
        "WITHIN a related group. A large gap means the score leans on lineage prevalence, not sequence signal."
    )
    plot_generalization(metrics)

    summary_rows = []
    for antibiotic, metric in metrics.items():
        gen = metric.get("generalization")
        if not gen:
            continue
        summary_rows.append(
            {
                "Antibiotic": antibiotic.capitalize(),
                "Overall AUROC": metric.get("auroc"),
                "Groups evaluated": gen.get("n_groups_evaluated"),
                "Per-group AUROC (mean)": gen.get("auroc_mean"),
                "Per-group AUROC (median)": gen.get("auroc_median"),
                "Worst group AUROC": gen.get("auroc_min"),
                "Per-group bal. acc (mean)": gen.get("balanced_accuracy_mean"),
            }
        )
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    for antibiotic, metric in metrics.items():
        by_group = metric.get("by_group")
        if not by_group:
            continue
        with st.expander(f"{antibiotic.capitalize()} - per-group detail ({len(by_group)} groups, >= {metric['generalization']['min_group_size']} genomes each)"):
            st.dataframe(
                [
                    {
                        "Genetic group": g["group"],
                        "n": g["n"],
                        "Resistant rate": g["resistant_rate"],
                        "AUROC": g["auroc"],
                        "Balanced acc": g["balanced_accuracy_called"],
                        "No-call rate": g["no_call_rate"],
                    }
                    for g in by_group
                ],
                use_container_width=True,
                hide_index=True,
            )
    st.info(
        "Honest reading: within-lineage AUROC for ciprofloxacin is weak because its main mechanism "
        "(gyrA/parC point mutations) is not captured by acquired-gene presence features. The no-call "
        "policy and calibration exist precisely so the system abstains instead of overstating confidence."
    )


def page_data() -> None:
    st.header("BV-BRC Training Data")
    manifest = load_manifest()
    if not manifest:
        st.warning("No BV-BRC manifest found.")
        return
    summary = load_antibiotic_summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("AMR rows", f"{manifest.get('amr_rows', 0):,}", "BV-BRC AST labels", "#2f6db3")
    with col2:
        kpi_card("Unique genomes", f"{manifest.get('unique_genomes', 0):,}", "E. coli only", "#2e8b57")
    with col3:
        kpi_card("Antibiotics", f"{len(summary):,}", "downloaded label table", "#e8743b")
    with col4:
        kpi_card("Taxon", str(manifest.get("taxon_id", "--")), manifest.get("evidence_filter", "--"), "#7b8494")

    left, right = st.columns([1.15, 0.85])
    with left:
        plot_antibiotic_summary(summary)
    with right:
        feature_meta = load_feature_meta()
        useful = feature_meta.get("useful_gene_cols", [])
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Model Feature Columns")
        st.write(f"{len(useful)} marker families plus engineered burden/assembly features.")
        if useful:
            st.markdown(" ".join(f'<span class="chip">{gene}</span>' for gene in useful), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if summary:
        st.subheader("Antibiotic Label Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with st.expander("Manifest"):
        st.json(manifest)


def page_safety() -> None:
    st.header("Safety Scope")
    st.markdown(f'<div class="warning">{DISCLAIMER}</div>', unsafe_allow_html=True)
    st.write("This prototype predicts and explains resistance signals that may already exist.")
    st.write("It does not design, modify, optimize, strengthen, or synthesize organisms.")
    st.write("It does not prescribe antibiotics or replace laboratory susceptibility testing.")
    st.write("Supported scope: E. coli and four configured antibiotics.")
    st.write("Likely-to-work outputs require a target gate; uncertain cases return no-call.")


NAV_PAGES = ["Predict", "Model Metrics", "Training Data", "Safety Scope"]


def render_topnav() -> str:
    page = st.radio("Navigation", NAV_PAGES, horizontal=True, label_visibility="collapsed", key="topnav")
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    return page


def render_status_sidebar(models: dict, metrics: dict) -> None:
    with st.sidebar:
        st.markdown("### System status")
        detector = "ready" if kmer_detector_available() else "missing"
        amr = "yes" if amrfinder_available() else "no"
        st.write(f"Raw-FASTA gene detector: **{detector}**")
        st.write(f"AMRFinderPlus on PATH: **{amr}**")
        st.divider()
        st.markdown("### Models")
        for antibiotic in ANTIBIOTICS:
            loaded = "loaded" if antibiotic in models else "missing"
            auroc = metrics.get(antibiotic, {}).get("auroc", "NA")
            st.write(f"{antibiotic.capitalize()}: {loaded} | AUROC {auroc}")
        st.divider()
        st.caption("BV-BRC public data | LightGBM + calibration | research prototype")


def main() -> None:
    st.set_page_config(page_title="AMRShield Sentinel", layout="wide", initial_sidebar_state="collapsed")
    inject_css()
    models = load_models()
    metrics = load_metrics()
    feature_cols = load_feature_cols()

    render_topbar()
    page = render_topnav()
    render_status_sidebar(models, metrics)

    if page == "Predict":
        page_predict(models, feature_cols)
    elif page == "Model Metrics":
        page_metrics(metrics)
    elif page == "Training Data":
        page_data()
    else:
        page_safety()


if __name__ == "__main__":
    main()
