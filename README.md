# AMRShield Sentinel

Research Streamlit prototype for predicting antibiotic response in E. coli from genome-derived AMR marker features.

Output is tri-state: `likely_to_fail`, `likely_to_work`, or `no_call`. Every result includes probability, confidence, evidence category, target-gate status, and the required lab-confirmation warning.

> Research prototype only. Confirm every result with standard laboratory susceptibility testing.

## Architecture

```mermaid
flowchart TD
    A[Assembled E. coli FASTA] --> B{Raw DNA or annotated?}
    B -->|raw nucleotide| C[Built-in k-mer detector vs NCBI AMR reference]
    B -->|AMRFinderPlus installed| C2[Run amrfinder -n FASTA -O Escherichia]
    B -->|annotated headers| C3[Parse gene/product headers]
    D[Upload precomputed AMRFinderPlus TSV] --> E[Map to marker families]
    F[BV-BRC genome ID demo path] --> G[Fetch genome_feature products]
    C --> E
    C2 --> E
    C3 --> E
    G --> E
    E --> H[Map genes/mutations to 20 AMR marker families]
    H --> I[Build 23 numeric model features]
    I --> J[Target gate per antibiotic]
    I --> K[Calibrated LightGBM models]
    J --> L[Decision report]
    K --> L
    L --> M[Streamlit UI: prediction, metrics, data, safety]
```

## Raw FASTA to Genes (Built-in Detector)

A raw genome assembly (just `>contig` headers and DNA) carries no gene names, so it
cannot be predicted without an annotation step. Instead of requiring an AMRFinderPlus
install, the app ships a **self-contained nucleotide AMR gene detector**:

1. `scripts/05_build_reference_index.py` downloads the public-domain NCBI AMRFinderPlus
   reference gene database (`AMR_CDS.fa`, ~9,700 real allele sequences) and builds a
   compact **MinHash k-mer index** (`artifacts/kmer_index.json`, ~1.5 MB, committed).
2. At upload time the genome is reduced to a set of canonical 21-mers (both strands),
   and each reference allele's sketch **containment** is measured. Containment >= 0.5
   marks a gene family as present. This is the same idea used by Mash / sourmash, in
   pure Python - no BLAST, no external tools, works offline after the one-time build.
3. Detected acquired genes plus the species' core chromosomal genes become the same
   20-family feature vector the LightGBM models were trained on.

Validated on a real 4.6 Mbp *E. coli* K-12 assembly: ~3.5 s scan, zero false-positive
acquired genes (only chromosomal `blaEC`/`ampC`), while a genome carrying real
`blaTEM`, `blaCTX-M`, `tetA`, `aadA`, `sul1`, `qnrS` sequences is called
`likely_to_fail` for all four drugs with the correct supporting markers.

## What Is Fixed

- Raw nucleotide FASTA uploads now work directly via the built-in k-mer AMR detector - no AMRFinderPlus or BLAST install required.
- AMRFinderPlus is still used automatically when `amrfinder` is available on `PATH`, and precomputed AMRFinderPlus TSV upload is also supported.
- A species name in a contig header is no longer mistaken for gene annotation (routing fix).
- The previous `genetic_group` predictive feature was removed to avoid lineage leakage.
- Calibration now uses `StratifiedGroupKFold` splits based on the grouped training data.
- PR-AUC now uses `average_precision_score`, not the previous broken trapezoid calculation.
- Target gates are deterministic per antibiotic target family before a `likely_to_work` result is allowed.
- The app text now matches the actual LightGBM model family and no-call policy.

## Dataset

- Source: BV-BRC public AST records and genome feature annotations.
- Species: E. coli, NCBI taxon 562.
- Scale: 92,452 AST rows, 8,725 unique genomes, 76 antibiotics in the downloaded label table.
- Modeled antibiotics: ampicillin, ciprofloxacin, ceftriaxone, tetracycline.
- Train/test split: grouped 80/20 split using `cgmlst_hc50`, with 6,979 train genomes and 1,746 test genomes.

Training is done as one binary model per antibiotic. The raw BV-BRC table has 92,452 AST rows, but each antibiotic model only uses genomes that have a resistant/susceptible label for that antibiotic after cleaning and de-duplication.

| Antibiotic | Train genomes | Train resistant | Train susceptible | Test genomes | Test resistant | Test susceptible | Total labeled genomes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ampicillin | 5,105 | 2,933 | 2,172 | 1,392 | 729 | 663 | 6,497 |
| Ciprofloxacin | 5,888 | 1,638 | 4,250 | 1,437 | 215 | 1,222 | 7,325 |
| Ceftriaxone | 803 | 399 | 404 | 161 | 55 | 106 | 964 |
| Tetracycline | 703 | 414 | 289 | 174 | 103 | 71 | 877 |

The committed training features are BV-BRC `genome_feature` product-name markers. For a stricter AMRFinderPlus rebuild, run AMRFinderPlus over the same genome assemblies and feed the TSV through the same marker parser used by `streamlit_app.py`.

## End-to-End Process

1. Download BV-BRC AST labels and genome metadata.
   - Input: public BV-BRC E. coli AST records.
   - Output: `data/bvbrc/training_dataset.csv` and metadata files.
   - Scale used here: 92,452 AST rows across 8,725 genomes.

2. Fetch genome feature annotations.
   - Script: `python scripts/01_fetch_features.py`
   - Input: genome IDs from the BV-BRC dataset.
   - Output: `data/bvbrc/raw_features.jsonl`
   - Purpose: collect gene/product annotations used to detect AMR marker families.

3. Build model matrices.
   - Script: `python scripts/02_build_matrices.py`
   - Output: `feature_matrix.csv`, `labels.csv`, `splits.json`, `feature_columns.json`, `feature_meta.json`
   - Feature output: 20 binary AMR marker columns plus `amr_gene_burden`, `genome_length_z`, and `contigs_z`.
   - Split policy: grouped train/test split by `cgmlst_hc50`, so related genomes do not cross train/test.

4. Train calibrated models.
   - Script: `python scripts/03_train.py`
   - Model: one LightGBM classifier per antibiotic.
   - Calibration: `CalibratedClassifierCV` with `StratifiedGroupKFold`.
   - Leakage control: `genetic_group` is used only for grouping, not as a predictive feature.
   - Output: `artifacts/models/<antibiotic>_model.pkl` and `artifacts/model_meta.json`.

5. Evaluate held-out test data.
   - Script: `python scripts/04_evaluate.py`
   - Metrics: AUROC, PR-AUC, Brier score, no-call rate, answered accuracy, balanced accuracy, resistant recall, susceptible recall.
   - Output: `artifacts/demo_metrics.json`.

6. Serve Streamlit dashboard.
   - Script: `streamlit run streamlit_app.py`
   - Inputs: AMRFinderPlus TSV, FASTA with AMRFinderPlus when installed, BV-BRC genome ID, or demo annotated FASTA.
   - Output: tri-state antibiotic report plus Tableau-style charts and safety disclaimer.
   - Important: raw contig FASTA without gene/product annotations cannot be converted into AMR marker features by this app unless AMRFinderPlus is installed. In that case the app returns `no_call` instead of producing misleading repeated predictions.

## Feature Output Format

The model feature vector is ordered by `data/bvbrc/feature_columns.json`.

Current columns:

- 20 binary AMR marker features after prevalence filtering.
- `amr_gene_burden`: count of detected marker families.
- `genome_length_z`: normalized genome length when available, `0.0` for uploaded inference files.
- `contigs_z`: normalized contig count when available, `0.0` for uploaded inference files.

Runtime prediction output rows contain:

- `antibiotic`
- `decision`
- `prob`
- `confidence`
- `evidence_category`
- `markers`
- `target_status`
- `reason_codes`

## Model Performance

Held-out grouped test evaluation from `artifacts/demo_metrics.json`:

| Antibiotic | Test n | AUROC | PR-AUC | Brier | No-call | Answered accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ampicillin | 1,392 | 0.9593 | 0.9577 | 0.0570 | 0.0338 | 0.9457 | 0.9470 |
| Ciprofloxacin | 1,437 | 0.8631 | 0.6655 | 0.0989 | 0.2192 | 0.9278 | 0.8305 |
| Ceftriaxone | 161 | 0.9274 | 0.8724 | 0.0682 | 0.0435 | 0.9221 | 0.9212 |
| Tetracycline | 174 | 0.9626 | 0.9676 | 0.0601 | 0.0460 | 0.9398 | 0.9395 |

Decision thresholds:

- `P(resistant) >= 0.72`: `likely_to_fail`
- `P(resistant) <= 0.28`: `likely_to_work`
- otherwise: `no_call`

## Run Locally

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The committed `artifacts/kmer_index.json` already lets raw FASTA uploads work out of
the box. To rebuild it from the current NCBI reference:

```bash
curl -o data/amr_reference/AMR_CDS.fa \
  https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/AMR_CDS.fa
python scripts/05_build_reference_index.py
```

Optional, only if you prefer the gold-standard annotation for assembled FASTA:

```bash
amrfinder -n sample.fna -O Escherichia -o sample.amrfinder.tsv
```

Rebuild model artifacts:

```bash
python scripts/02_build_matrices.py
python scripts/03_train.py
python scripts/04_evaluate.py
```

Launch the app:

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`.

## Streamlit Inputs

- Assembled FASTA: runs AMRFinderPlus if `amrfinder` is installed.
- AMRFinderPlus TSV: upload precomputed AMRFinderPlus output.
- BV-BRC Genome ID: fetches public genome feature annotations for demo/prototype use.
- Demo annotated FASTA: built-in synthetic examples for quick UI testing.

If every raw FASTA file appears to produce the same result, that means no AMR marker annotations were available. Use one of these paths:

1. Install AMRFinderPlus and let the app run it.
2. Run AMRFinderPlus outside the app and upload the TSV.
3. Upload an annotated FASTA whose headers contain gene/product names.

## Repository Map

```text
hackNation/
  scripts/
    01_fetch_features.py
    02_build_matrices.py
    03_train.py
    04_evaluate.py
  streamlit_app.py
  requirements.txt
  artifacts/
    demo_metrics.json
    model_meta.json
    models/
  data/bvbrc/
    feature_matrix.csv
    labels.csv
    splits.json
    feature_columns.json
  demo_samples/
  client/ server/ public/ shared/   # original TypeScript prototype, not used by Streamlit path
```

## Safety Scope

In scope: defensive E. coli AMR response prediction, explainable markers, calibrated uncertainty, no-call behavior, and lab-confirmation messaging.

Out of scope: clinical prescribing, organism design, resistance optimization, unsupported species, or replacing standard AST.
