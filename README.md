# AMRShield Sentinel

**Hack-Nation Challenge 06 — Genome Firewall**

Predicts antibiotic resistance in *E. coli* from annotated genome features **before** standard lab results arrive. Tri-state output: `Likely to Fail` / `Likely to Work` / `No Call`. Every prediction includes a calibrated probability, supporting gene markers, and a mandatory lab-confirmation disclaimer.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  BV-BRC Public API                                                  │
│  (92,452 AST rows / 8,725 E. coli genomes)                         │
│          │                                                          │
│          ▼                                                          │
│  01_fetch_features.py  ──►  raw_features.jsonl                      │
│  (batch AMR gene annotations, 5 keyword filters)                    │
│          │                                                          │
│          ▼                                                          │
│  02_build_matrices.py  ──►  feature_matrix.csv  +  labels.csv       │
│  (39 regex gene families → 20 useful features                       │
│   + amr_gene_burden, genome_length_z, contigs_z)                   │
│          │                                                          │
│          ▼                                                          │
│  Cluster-based split (cgmlst_hc50)                                  │
│  Train: 6,979 genomes   Test: 1,746 genomes                        │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ML PIPELINE                                  │
│                                                                     │
│  03_train.py                                                        │
│  ┌─────────────────────────────────────┐                           │
│  │  For each antibiotic (×4):          │                           │
│  │                                     │                           │
│  │  genetic_group  ──► Bayesian target │                           │
│  │  (cgMLST lineage)    encoding on    │                           │
│  │                      train only     │                           │
│  │                          │          │                           │
│  │  23 numeric features  ──►│          │                           │
│  │  + 1 encoded gg_val      │          │                           │
│  │                          ▼          │                           │
│  │              LightGBM (adaptive     │                           │
│  │              complexity by n_train) │                           │
│  │                          │          │                           │
│  │              CalibratedClassifierCV │                           │
│  │              (isotonic / sigmoid)   │                           │
│  └─────────────────────────────────────┘                           │
│          │                                                          │
│          ▼                                                          │
│  artifacts/models/<antibiotic>_model.pkl                           │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI                                   │
│                                                                     │
│  Upload annotated FASTA  OR  enter BV-BRC Genome ID                │
│          │                                                          │
│          ▼                                                          │
│  Gene product scan → 39-regex family classification                │
│          │                                                          │
│          ▼                                                          │
│  4 × LightGBM models → P(resistant)                                │
│          │                                                          │
│          ▼                                                          │
│  Tri-state decision (FAIL / WORK / NO CALL)                        │
│  + calibrated probability bar + gene chip annotations              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Performance

Evaluated on **held-out test set** (1,746 genomes, cluster-stratified — related genomes never split across train/test).

| Antibiotic     | AUROC  | Balanced Acc | Resistant Recall | Susceptible Recall | No-Call Rate | Brier Score |
|----------------|--------|-------------|-----------------|-------------------|--------------|-------------|
| Ampicillin     | 0.948  | 93.9%       | 92.4%           | 95.4%             | 6.3%         | 0.068       |
| Ciprofloxacin  | 0.902  | 82.4%       | 67.7%           | 97.1%             | 11.3%        | 0.079       |
| Ceftriaxone    | 0.922  | 92.1%       | 87.3%           | 96.9%             | 5.6%         | 0.071       |
| Tetracycline   | 0.965  | 93.2%       | 97.0%           | 89.4%             | 4.6%         | 0.058       |

> Metrics computed on called predictions only (No-Call excluded from Balanced Acc / Recalls).

**Decision thresholds:** P ≥ 0.72 → `LIKELY TO FAIL` | P ≤ 0.28 → `LIKELY TO WORK` | else → `NO CALL`

---

## Dataset

- **Source:** [BV-BRC](https://www.bv-brc.org/) public antimicrobial susceptibility testing (AST) records
- **Species:** *Escherichia coli* (NCBI taxon 562)
- **Scale:** 92,452 AST rows across 8,725 unique genomes and 76 antibiotics
- **Modelled antibiotics:** Ampicillin, Ciprofloxacin, Ceftriaxone, Tetracycline
- **Train / Test split:** Cluster-based on `cgmlst_hc50` (80/20) — prevents data leakage from related strains

---

## Feature Engineering

### Gene Families (20 binary features after prevalence filter)

| Group | Genes |
|-------|-------|
| β-lactamases | blaCTX-M, blaTEM, blaOXA, blaCMY, blaEC, blaAmpC, betaLactamResProtein |
| Tetracycline | tetA, tetB, tetR |
| Aminoglycosides | aac6Ib, aadA, aac3, aph3 |
| Sulfonamides | sul1 |
| Regulators / Efflux | marA, marR, marB, qacE |
| Integrons | integraseI |

Features with <2% or >98% prevalence dropped as near-constant.

### Engineered Features (3)
- `amr_gene_burden` — count of distinct AMR gene families detected
- `genome_length_z` — z-score normalised assembly length
- `contigs_z` — z-score normalised contig count

### Genetic Group (1 target-encoded feature)
- `genetic_group` — cgMLST-derived lineage identifier
- Bayesian target-encoded on train set only (smoothing=10) to prevent leakage
- Key signal for ciprofloxacin: ST131-like lineage (group 18) is **96% ciprofloxacin resistant** across 793 genomes
- Only used when training set ≥ 2,000 samples

---

## ML Model Details

**Algorithm:** LightGBM + CalibratedClassifierCV (isotonic/sigmoid)

Adaptive complexity based on training set size:

| Training Samples | Trees | Leaves | Calibration | CV Folds |
|-----------------|-------|--------|------------|---------|
| ≥ 3,000         | 400   | 63     | isotonic   | 5       |
| 1,000–2,999     | 200   | 31     | sigmoid    | 3       |
| < 1,000         | 100   | 15     | sigmoid    | 3       |

Calibration ensures probabilities are well-calibrated (Brier scores 0.058–0.079).

---

## Repository Structure

```
hackNation/
├── scripts/
│   ├── 01_fetch_features.py    # Fetch AMR gene annotations from BV-BRC API
│   ├── 02_build_matrices.py    # Build feature matrix + cluster-based split
│   ├── 03_train.py             # Train LightGBM models per antibiotic
│   └── 04_evaluate.py          # Evaluate models, write demo_metrics.json
│
├── streamlit_app.py            # Dark-themed Streamlit prediction UI
│
├── artifacts/
│   ├── demo_metrics.json       # Final test-set evaluation metrics
│   ├── model_meta.json         # Training metadata per model
│   └── models/
│       ├── ampicillin_model.pkl
│       ├── ciprofloxacin_model.pkl
│       ├── ceftriaxone_model.pkl
│       └── tetracycline_model.pkl
│
├── configs/
│   └── app_config.json         # Thresholds, antibiotic targets, safety config
│
├── demo_samples/
│   ├── mdr_ecoli.fasta         # Multi-drug resistant E. coli demo
│   ├── cipro_resistant_ecoli.fasta  # Ciprofloxacin-resistant demo
│   └── susceptible_ecoli.fasta     # Susceptible (low-resistance) demo
│
├── data/bvbrc/                 # (gitignored — too large)
│   ├── training_dataset.csv    # 92,452 AST rows
│   ├── raw_features.jsonl      # BV-BRC gene annotations
│   ├── feature_matrix.csv      # Processed feature matrix (8,725 genomes × 24 cols)
│   └── labels.csv              # Per-genome antibiotic labels
│
├── client/                     # TypeScript browser client (original prototype)
├── server/                     # Node.js API server (original prototype)
├── public/                     # Web UI HTML/CSS (original prototype)
└── shared/                     # Shared TypeScript types
```

---

## Setup

### Python Pipeline (LightGBM models + Streamlit UI)

```bash
pip install lightgbm scikit-learn numpy streamlit plotly requests
```

### Node.js Web Prototype (original TypeScript UI)

```bash
npm install
npm run dev       # Build and run at http://localhost:3000
```

---

## Running the Pipeline

> **Note:** `data/` is gitignored. To reproduce from scratch, run steps 1–4. Otherwise, the pre-trained models in `artifacts/models/` are already committed and the Streamlit app works immediately.

### Step 1 — Fetch gene annotations from BV-BRC (~2–4 hours for 8,725 genomes)

```bash
python scripts/01_fetch_features.py
```

Queries the BV-BRC `genome_feature` API in batches of 50 genomes. Saves `data/bvbrc/raw_features.jsonl`. Supports resume — safe to interrupt and restart.

### Step 2 — Build feature matrix

```bash
python scripts/02_build_matrices.py
```

Classifies gene product names via 39 regex patterns into gene families. Outputs `feature_matrix.csv`, `labels.csv`, `splits.json`.

### Step 3 — Train models (~5–15 min)

```bash
python scripts/03_train.py
```

Trains one LightGBM + calibration model per antibiotic. Saves `artifacts/models/*.pkl`.

### Step 4 — Evaluate

```bash
python scripts/04_evaluate.py
```

Computes AUROC, Balanced Accuracy, Recalls, Brier score. Saves `artifacts/demo_metrics.json`.

### Step 5 — Launch Streamlit UI

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Streamlit UI Features

- **Upload FASTA tab** — Upload annotated gene FASTA or use 3 built-in demo samples
- **BV-BRC Genome ID tab** — Enter any public E. coli genome ID to query live
- **Prediction cards** — Color-coded (red/green/grey) per antibiotic with probability bar and detected gene chips
- **Model Metrics tab** — AUROC table + calibration plots for all 4 models
- **Safety & Scope tab** — Explicit out-of-scope use cases and lab confirmation requirement

> The FASTA upload requires **annotated** gene product headers (BV-BRC / Prokka / RAST output style, e.g. `>gene|product=blaTEM-1 beta-lactamase`). Raw assembly FASTA from sequencers (`>NODE_1_length_...`) contains no gene names — use the demo buttons or the BV-BRC ID tab instead.

---

## Safety & Scope

| In scope | Out of scope |
|----------|-------------|
| E. coli reconstructed genomes | Other species |
| Research decision support | Clinical prescribing |
| Genome annotation FASTA | Raw sequencer output |
| Known resistance mechanisms | Novel resistance mutations |

**Every prediction includes:** "Research prototype only. Confirm every result with standard laboratory susceptibility testing."

---

## Challenge Context

**Hack-Nation Challenge 06 — Genome Firewall**

The challenge asks: can AI predict whether an antibiotic will work for a bacterial infection **before** the 24–72 hour lab culture result? AMRShield Sentinel addresses this by:

1. Extracting resistance gene families from a sequenced genome annotation
2. Incorporating bacterial lineage (genetic group) — critical because ST131-like *E. coli* clones carry gyrA point mutations invisible to gene-name scanning
3. Returning calibrated tri-state decisions with explicit uncertainty (No-Call zone) rather than forcing a binary guess

Data source: [BV-BRC](https://www.bv-brc.org/) — USDA/NIAID-funded public pathogen genomics database.

---

## License

Research prototype. Not for clinical use. All training data sourced from public BV-BRC records.
