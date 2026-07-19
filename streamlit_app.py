"""AMRShield Sentinel — Streamlit prediction UI backed by trained models."""
from __future__ import annotations

import json
import pickle
import re
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Literal

import numpy as np
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "artifacts" / "models"
METRICS_PATH = ROOT / "artifacts" / "demo_metrics.json"
CONFIG_PATH = ROOT / "configs" / "app_config.json"
FEATURE_COLS_PATH = ROOT / "data" / "bvbrc" / "feature_columns.json"
DEMO_SAMPLES_DIR = ROOT / "demo_samples"

ANTIBIOTICS = ["ampicillin", "ciprofloxacin", "ceftriaxone", "tetracycline"]
FAIL_THRESHOLD = 0.72
WORK_THRESHOLD = 0.28

_DEMO_FILES = {
    "MDR E. coli\n(blaTEM + blaCTX-M + tetA)":    "mdr_ecoli.fasta",
    "Cipro-Resistant\n(qnrS + aac6Ib)":            "cipro_resistant_ecoli.fasta",
    "Susceptible E. coli\n(no resistance genes)":  "susceptible_ecoli.fasta",
}

_AB_MARKERS = {
    "ampicillin":    {"blaTEM", "blaCTX-M", "blaSHV", "blaOXA", "blaCMY", "blaAmpC"},
    "ciprofloxacin": {"qnrA", "qnrB", "qnrS", "qnrD", "aac6Ib", "oqxA", "oqxB"},
    "ceftriaxone":   {"blaCTX-M", "blaCMY", "blaSHV", "blaOXA"},
    "tetracycline":  {"tetA", "tetB", "tetC", "tetM", "tetG"},
}

_AB_INFO = {
    "ampicillin":    {"emoji": "💊", "class": "Beta-lactam",     "target": "Cell wall (PBPs)"},
    "ciprofloxacin": {"emoji": "🔬", "class": "Fluoroquinolone", "target": "DNA gyrase / Topo IV"},
    "ceftriaxone":   {"emoji": "💉", "class": "Cephalosporin",   "target": "Cell wall (PBPs)"},
    "tetracycline":  {"emoji": "🧪", "class": "Tetracycline",    "target": "30S ribosome"},
}

DISCLAIMER = "Research prototype only — confirm every result with lab susceptibility testing before clinical use."

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27 !important; border-right: 1px solid #2d3148; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
h1, h2, h3 { color: #ffffff !important; }
p, label, .stMarkdown { color: #c8ccd8 !important; }

/* Banner */
.sentinel-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 50%, #1a1040 100%);
    border: 1px solid #2a4a7f;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.sentinel-banner h1 { font-size: 2rem; margin: 0; color: #fff !important; }
.sentinel-banner p { margin: 4px 0 0; color: #7ba7d4 !important; font-size: 0.95rem; }

/* Prediction cards */
.pred-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    transition: all 0.2s;
}
.pred-card:hover { border-color: #4a5580; transform: translateY(-2px); }
.pred-card-fail  { border-left: 4px solid #ff4b4b; }
.pred-card-work  { border-left: 4px solid #21c354; }
.pred-card-nocall{ border-left: 4px solid #888; }

/* Decision pills */
.pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}
.pill-fail   { background: #ff4b4b22; color: #ff6b6b; border: 1px solid #ff4b4b; }
.pill-work   { background: #21c35422; color: #21c354; border: 1px solid #21c354; }
.pill-nocall { background: #88888822; color: #aaaaaa; border: 1px solid #888888; }

/* Prob bar */
.prob-bar-wrap { background: #2d3148; border-radius: 6px; height: 8px; margin: 8px 0; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease; }

/* Gene chip */
.gene-chip {
    display: inline-block;
    background: #1e2a45;
    border: 1px solid #2a4a7f;
    color: #7ba7d4 !important;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78rem;
    margin: 2px;
    font-family: monospace;
}
.gene-chip-hit {
    background: #2a1e1e;
    border-color: #7f2a2a;
    color: #ff8888 !important;
}

/* Metric box */
.metric-box {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-box .val { font-size: 1.8rem; font-weight: 700; color: #fff; }
.metric-box .lbl { font-size: 0.78rem; color: #888; margin-top: 4px; }

/* Info box */
.info-box {
    background: #0d1a2e;
    border: 1px solid #1a3a5f;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
}
.info-box .title { color: #7ba7d4 !important; font-weight: 600; font-size: 0.9rem; margin-bottom: 8px; }

/* Step box */
.step-box {
    background: #141720;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 20px;
    margin: 12px 0;
    display: flex;
    gap: 16px;
    align-items: flex-start;
}
.step-num {
    background: #1e3a5f;
    color: #7ba7d4;
    border-radius: 50%;
    width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem; flex-shrink: 0;
}

/* Table override */
[data-testid="stDataFrame"] { background: #1a1d27 !important; }

/* Divider */
hr { border-color: #2d3148 !important; }

/* Warning / disclaimer */
.disclaimer {
    background: #1f1a00;
    border: 1px solid #5c4a00;
    border-radius: 8px;
    padding: 12px 16px;
    color: #f0c040 !important;
    font-size: 0.85rem;
}

/* Tab styling */
[data-testid="stTabs"] button { color: #888 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #fff !important; border-bottom-color: #4a7fd4 !important; }
</style>
""", unsafe_allow_html=True)


# ── gene-family regex ──────────────────────────────────────────────────────────
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
_COMPILED = [(n, re.compile(p, re.IGNORECASE)) for n, p in GENE_FAMILIES]


# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models() -> dict:
    models = {}
    for ab in ANTIBIOTICS:
        path = MODELS_DIR / f"{ab}_model.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[ab] = pickle.load(f)
    return models

@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_feature_cols() -> list[str]:
    with open(FEATURE_COLS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── logic ─────────────────────────────────────────────────────────────────────
def classify_products(products: list[str]) -> set[str]:
    found: set[str] = set()
    for p in products:
        for name, pat in _COMPILED:
            if pat.search(p):
                found.add(name)
    return found

def fasta_to_genes(text: str) -> set[str]:
    headers = [line[1:] for line in text.splitlines() if line.startswith(">")]
    return classify_products(headers)

def fetch_bvbrc_genes(genome_id: str) -> tuple[set[str], list[str]]:
    ids = urllib.parse.quote(genome_id)
    url = (
        f"https://www.bv-brc.org/api/genome_feature/"
        f"?and(eq(genome_id,{ids}),or(eq(product,*resistance*),eq(product,*lactamase*)"
        f",eq(product,*efflux*),eq(product,*aminoglycoside*),eq(product,*integron*)))"
        f"&select(product)&limit(500)&http_accept=application/json"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    products = [row.get("product", "") for row in data if row.get("product")]
    return classify_products(products), products

def build_feature_vector(genes: set[str], feature_cols: list[str]) -> np.ndarray:
    return np.array([1.0 if c in genes else 0.0 for c in feature_cols])

def predict(vec: np.ndarray, models: dict, feature_cols: list[str]) -> list[dict]:
    results = []
    for ab in ANTIBIOTICS:
        pkg = models.get(ab)
        if not pkg:
            results.append({"antibiotic": ab, "decision": "no_call", "prob": None})
            continue
        cols = pkg["feature_cols"]
        v = np.array([vec[feature_cols.index(c)] if c in feature_cols else 0.0 for c in cols])
        prob = float(pkg["model"].predict_proba(v.reshape(1, -1))[0, 1])
        if prob >= FAIL_THRESHOLD:
            decision = "likely_to_fail"
        elif prob <= WORK_THRESHOLD:
            decision = "likely_to_work"
        else:
            decision = "no_call"
        results.append({"antibiotic": ab, "decision": decision, "prob": prob})
    return results


# ── UI components ─────────────────────────────────────────────────────────────
def prob_bar(prob: float, decision: str) -> str:
    color = {"likely_to_fail": "#ff4b4b", "likely_to_work": "#21c354", "no_call": "#888888"}.get(decision, "#888")
    pct = int(prob * 100)
    return (
        f'<div class="prob-bar-wrap">'
        f'<div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
    )

def gene_chips(all_genes: set[str], hit_genes: set[str]) -> str:
    chips = ""
    for g in sorted(all_genes):
        cls = "gene-chip-hit" if g in hit_genes else "gene-chip"
        chips += f'<span class="{cls}">{g}</span>'
    return chips

def render_prediction_cards(predictions: list[dict], genes: set[str]) -> None:
    cols = st.columns(4)
    for col, pred in zip(cols, predictions):
        ab = pred["antibiotic"]
        decision = pred["decision"]
        prob = pred.get("prob") or 0.0
        info = _AB_INFO[ab]
        markers = _AB_MARKERS[ab]
        matched = sorted(markers & genes)

        card_cls = {"likely_to_fail": "pred-card-fail", "likely_to_work": "pred-card-work", "no_call": "pred-card-nocall"}.get(decision, "")
        pill_cls = {"likely_to_fail": "pill-fail", "likely_to_work": "pill-work", "no_call": "pill-nocall"}.get(decision, "")
        pill_label = {"likely_to_fail": "LIKELY TO FAIL", "likely_to_work": "LIKELY TO WORK", "no_call": "NO CALL"}.get(decision, "NO CALL")

        conf_str = f"{prob:.0%}" if decision != "no_call" else "—"
        marker_html = "".join(f'<span class="gene-chip gene-chip-hit">{m}</span>' for m in matched) if matched else '<span style="color:#666;font-size:0.8rem">None detected</span>'
        ev_text = "Known resistance gene detected" if matched else ("Model abstains — borderline probability" if decision == "no_call" else "Statistical signal (no direct gene marker)")

        with col:
            st.markdown(f"""
<div class="pred-card {card_cls}">
  <div style="font-size:1.3rem">{info['emoji']}</div>
  <div style="font-weight:700;color:#fff;margin:4px 0">{ab.capitalize()}</div>
  <div style="font-size:0.75rem;color:#666;margin-bottom:12px">{info['class']} · {info['target']}</div>
  <span class="pill {pill_cls}">{pill_label}</span>
  <div style="color:#ccc;font-size:0.85rem;margin:4px 0">P(resistant) = <strong style="color:#fff">{prob:.1%}</strong></div>
  <div style="color:#ccc;font-size:0.85rem">Confidence = <strong style="color:#fff">{conf_str}</strong></div>
  {prob_bar(prob, decision)}
  <div style="margin-top:12px;font-size:0.75rem;color:#888">RESISTANCE MARKERS</div>
  <div style="margin-top:4px">{marker_html}</div>
  <div style="margin-top:10px;font-size:0.75rem;color:#666;font-style:italic">{ev_text}</div>
</div>
""", unsafe_allow_html=True)


# ── pages ─────────────────────────────────────────────────────────────────────
def page_predict(models: dict, feature_cols: list[str]) -> None:
    st.markdown("""
<div class="sentinel-banner">
  <div style="font-size:2.5rem">🛡️</div>
  <div>
    <h1>AMRShield Sentinel</h1>
    <p>AI-powered antibiotic resistance prediction from bacterial genome · E. coli · Research use only</p>
  </div>
</div>
""", unsafe_allow_html=True)

    tab_fasta, tab_id, tab_howit = st.tabs(["📂  Upload FASTA", "🔍  BV-BRC Genome ID", "❓  How It Works"])

    genes: set[str] | None = None
    products_raw: list[str] = []
    source_label = ""

    # ── Tab 1: FASTA ──────────────────────────────────────────────────────────
    with tab_fasta:
        st.markdown("""
<div class="info-box">
  <div class="title">What FASTA format is supported?</div>
  Upload an <strong>annotated gene FASTA</strong> — headers must contain gene product names
  (e.g. output from BV-BRC, RAST, or Prokka). Raw assembly contigs with headers like
  <code>&gt;NODE_1_length_...</code> don't work because they have no gene names. Use the
  BV-BRC Genome ID tab instead for raw assemblies.
</div>
""", unsafe_allow_html=True)

        st.markdown("**Quick demo — click to load:**")
        demo_cols = st.columns(3)
        fasta_text: str | None = None
        fasta_source = ""
        for col, (label, fname) in zip(demo_cols, _DEMO_FILES.items()):
            short_label = label.replace("\n", " ")
            if col.button(short_label, use_container_width=True):
                fasta_text = (DEMO_SAMPLES_DIR / fname).read_text(encoding="utf-8")
                fasta_source = f"Demo: {fname}"
        st.divider()
        uploaded = st.file_uploader("Or upload your own annotated FASTA (.fa / .fasta / .fna)", type=["fa","fasta","fna"])
        if uploaded and not fasta_text:
            fasta_text = uploaded.read().decode("utf-8", errors="replace")
            fasta_source = uploaded.name
        if fasta_text:
            genes = fasta_to_genes(fasta_text)
            source_label = fasta_source
            st.success(f"Parsed — detected **{len(genes)}** AMR gene-family signals from headers.")

    # ── Tab 2: BV-BRC ID ─────────────────────────────────────────────────────
    with tab_id:
        st.markdown("""
<div class="info-box">
  <div class="title">BV-BRC Genome ID</div>
  Enter a BV-BRC genome ID (format: <code>taxon.number</code>, e.g. <code>562.100000</code>).
  The system fetches resistance-related gene annotations live from the BV-BRC API.
</div>
""", unsafe_allow_html=True)
        gid = st.text_input("Genome ID", placeholder="e.g. 562.100000")
        if st.button("🔍  Fetch & Predict", type="primary") and gid.strip():
            with st.spinner("Querying BV-BRC API for resistance gene annotations…"):
                try:
                    genes, products_raw = fetch_bvbrc_genes(gid.strip())
                    source_label = f"BV-BRC: {gid.strip()}"
                    st.success(f"Fetched — {len(products_raw)} resistance-related features → **{len(genes)}** gene families detected.")
                except Exception as exc:
                    st.error(f"BV-BRC API error: {exc}")

    # ── Tab 3: How It Works ───────────────────────────────────────────────────
    with tab_howit:
        st.markdown("""
<div class="step-box">
  <div class="step-num">1</div>
  <div>
    <strong style="color:#fff">Genome Input</strong><br/>
    <span style="color:#888;font-size:0.9rem">
      You provide either an annotated gene FASTA file or a BV-BRC genome ID.
      The system extracts which resistance-related gene products are present in the genome.
    </span>
  </div>
</div>
<div class="step-box">
  <div class="step-num">2</div>
  <div>
    <strong style="color:#fff">Feature Extraction (33 binary features)</strong><br/>
    <span style="color:#888;font-size:0.9rem">
      Each gene is classified into one of 33 AMR gene families (blaTEM, blaCTX-M, tetA, qnrS, aadA, etc.).
      Result: a binary vector — <code>1</code> if gene family present, <code>0</code> if absent.
      This is the input to the ML model.
    </span>
  </div>
</div>
<div class="step-box">
  <div class="step-num">3</div>
  <div>
    <strong style="color:#fff">Calibrated Logistic Regression (one model per antibiotic)</strong><br/>
    <span style="color:#888;font-size:0.9rem">
      4 separate models — one for each antibiotic. Each model was trained on real lab data
      from BV-BRC (8,725 E. coli genomes, up to 8,625 labeled examples per antibiotic).
      Models are calibrated so probabilities are trustworthy, not just rankings.
    </span>
  </div>
</div>
<div class="step-box">
  <div class="step-num">4</div>
  <div>
    <strong style="color:#fff">Decision Thresholds</strong><br/>
    <span style="color:#888;font-size:0.9rem">
      P(resistant) ≥ 0.72 → <span style="color:#ff6b6b">LIKELY TO FAIL</span><br/>
      P(resistant) ≤ 0.28 → <span style="color:#21c354">LIKELY TO WORK</span><br/>
      0.28 &lt; P &lt; 0.72 → <span style="color:#aaa">NO CALL</span> (send to lab — model is uncertain)
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("**Training data summary:**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown('<div class="metric-box"><div class="val">92,452</div><div class="lbl">Total lab measurements</div></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="metric-box"><div class="val">8,725</div><div class="lbl">Unique E. coli genomes</div></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="metric-box"><div class="val">76</div><div class="lbl">Antibiotics in DB</div></div>', unsafe_allow_html=True)
        with c4: st.markdown('<div class="metric-box"><div class="val">4</div><div class="lbl">We predict</div></div>', unsafe_allow_html=True)

        st.markdown("""
<br/>
<div class="info-box">
  <div class="title">Why only 4 antibiotics?</div>
  The BV-BRC database has lab results for 76 antibiotics across our 8,725 genomes. But
  each genome was only tested against ~10 antibiotics on average — so for rarer antibiotics
  there isn't enough data to train a reliable model. The 4 chosen (Ampicillin, Ciprofloxacin,
  Ceftriaxone, Tetracycline) have the most samples and are clinically critical for E. coli.
</div>
<div class="info-box">
  <div class="title">Why is Ciprofloxacin harder? (AUROC 0.86 vs 0.95)</div>
  Ciprofloxacin resistance is mainly caused by point mutations in the gyrA and parC genes
  (DNA gyrase). These mutations can't be detected from gene product names — the gene is still
  called "DNA gyrase subunit A" whether or not it has the mutation. So the model misses the
  main signal and has to rely on correlates (qnr plasmids, co-resistance patterns), causing
  more NO CALL outputs (27.6%) and lower accuracy.
</div>
""", unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if genes is None:
        return

    vec = build_feature_vector(genes, feature_cols)
    predictions = predict(vec, models, feature_cols)

    st.divider()
    st.markdown(f"### Resistance Predictions  <span style='color:#555;font-size:0.85rem;font-weight:400'>· {source_label}</span>", unsafe_allow_html=True)

    render_prediction_cards(predictions, genes)

    st.divider()
    col_genes, col_raw = st.columns([2, 1])
    with col_genes:
        st.markdown("**Detected AMR Gene Families**")
        if genes:
            all_amr = {g for markers in _AB_MARKERS.values() for g in markers}
            hit_html = gene_chips(genes, genes & all_amr)
            st.markdown(hit_html, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#555">None detected</span>', unsafe_allow_html=True)

    if products_raw:
        with col_raw:
            with st.expander(f"Raw BV-BRC annotations ({len(products_raw)})"):
                for p in products_raw:
                    st.markdown(f'<span style="color:#888;font-size:0.8rem">• {p}</span>', unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)


def page_metrics(metrics: dict) -> None:
    st.markdown("## Model Performance Metrics")
    st.markdown('<p style="color:#666">Evaluated on cluster-held-out test set · cgmlst_hc50 split · 20% of data withheld</p>', unsafe_allow_html=True)

    rows = []
    for ab, m in metrics.items():
        rows.append({
            "Antibiotic": ab.capitalize(),
            "Test samples": m.get("n_test"),
            "AUROC": m.get("auroc"),
            "Balanced Acc": m.get("balanced_accuracy_called"),
            "Resistant Recall": m.get("resistant_recall_called"),
            "Susceptible Recall": m.get("susceptible_recall_called"),
            "Brier Score": m.get("brier_score"),
            "No-call Rate": f"{m.get('no_call_rate', 0):.1%}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("### Calibration Plots")
    st.markdown('<p style="color:#666">A well-calibrated model follows the dashed diagonal — predicted probabilities match actual resistance rates.</p>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (ab, m) in enumerate(metrics.items()):
        bins = m.get("reliability", [])
        if not bins:
            continue
        try:
            import plotly.graph_objects as go
            x = [b["mean_prob"] for b in bins]
            y = [b["frac_positive"] for b in bins]
            sizes = [max(6, b["count"] // 10) for b in bins]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash", color="#444", width=1)))
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Model",
                marker=dict(size=sizes, color="#4a7fd4"), line=dict(color="#4a7fd4", width=2)))
            fig.update_layout(
                title=dict(text=ab.capitalize(), font=dict(color="#fff")),
                paper_bgcolor="#1a1d27", plot_bgcolor="#141720",
                font=dict(color="#888"),
                xaxis=dict(title="Mean predicted P(resistant)", gridcolor="#2d3148", range=[0,1]),
                yaxis=dict(title="Actual fraction resistant", gridcolor="#2d3148", range=[0,1]),
                height=300, margin=dict(l=40,r=20,t=40,b=40),
                showlegend=True,
                legend=dict(font=dict(color="#888"), bgcolor="#1a1d27"),
            )
            cols[i % 2].plotly_chart(fig, use_container_width=True)
        except ImportError:
            cols[i % 2].write(f"{ab}: plotly not installed")


def page_safety() -> None:
    st.markdown("## Safety Scope & Limitations")
    st.markdown(f'<div class="disclaimer" style="margin-bottom:24px">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
  <div class="title">✅ What this system does</div>
  <ul style="color:#ccc;margin:0;padding-left:20px">
    <li>Predicts likely antibiotic resistance from <strong>gene content</strong> of bacterial genomes</li>
    <li>Covers <strong>4 antibiotics</strong> in <strong>E. coli only</strong></li>
    <li>Returns three decisions: <span style="color:#ff6b6b">Likely to Fail</span> · <span style="color:#21c354">Likely to Work</span> · <span style="color:#aaa">No Call</span></li>
    <li>Provides evidence category: known gene vs statistical signal vs abstain</li>
  </ul>
</div>
<div class="info-box">
  <div class="title">❌ What it does NOT do</div>
  <ul style="color:#ccc;margin:0;padding-left:20px">
    <li>Does <strong>not</strong> design, modify, or engineer organisms</li>
    <li>Does <strong>not</strong> replace laboratory susceptibility testing</li>
    <li>Does <strong>not</strong> make clinical prescribing decisions</li>
    <li>Does <strong>not</strong> work for species other than E. coli</li>
    <li>Does <strong>not</strong> detect point mutations (gyrA, parC) from product names</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("**Supported antibiotics and known gene markers:**")
    for ab in ANTIBIOTICS:
        info = _AB_INFO[ab]
        markers = sorted(_AB_MARKERS[ab])
        chips = "".join(f'<span class="gene-chip gene-chip-hit">{m}</span>' for m in markers)
        st.markdown(f"""
<div class="pred-card" style="margin:6px 0">
  <strong style="color:#fff">{info['emoji']} {ab.capitalize()}</strong>
  <span style="color:#555;font-size:0.8rem"> · {info['class']} · Target: {info['target']}</span><br/>
  <div style="margin-top:8px">{chips}</div>
</div>
""", unsafe_allow_html=True)


# ── sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(models: dict, metrics: dict) -> str:
    with st.sidebar:
        st.markdown("### 🛡️ AMRShield Sentinel")
        st.markdown('<p style="color:#666;font-size:0.8rem">Genome Firewall · Hack-Nation Challenge 06</p>', unsafe_allow_html=True)
        st.divider()

        st.markdown("**Models loaded**")
        for ab in ANTIBIOTICS:
            ok = ab in models
            m = metrics.get(ab, {})
            auroc = f"AUROC {m.get('auroc','?')}" if m else ""
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin:4px 0">'
                f'<span>{"✅" if ok else "❌"} {ab.capitalize()}</span>'
                f'<span style="color:#555;font-size:0.75rem">{auroc}</span>'
                f'</div>', unsafe_allow_html=True
            )
        st.divider()

        page = st.radio("Navigate", ["🔬  Predict", "📊  Model Metrics", "🛡️  Safety Scope"], label_visibility="collapsed")
        st.divider()
        st.markdown('<p style="color:#444;font-size:0.75rem">BV-BRC · E. coli · Research use only</p>', unsafe_allow_html=True)
    return page


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="AMRShield Sentinel", page_icon="🛡️", layout="wide")
    inject_css()

    models = load_models()
    metrics = load_metrics()
    feature_cols = load_feature_cols()

    page = render_sidebar(models, metrics)

    if "Predict" in page:
        page_predict(models, feature_cols)
    elif "Metrics" in page:
        page_metrics(metrics)
    else:
        page_safety()


if __name__ == "__main__":
    main()
