# AMRShield Sentinel

Old-school TypeScript + Node.js website prototype for the Genome Firewall hackathon.

AMRShield Sentinel accepts one reconstructed FASTA genome from one supported bacterial species and returns evidence-backed tri-state antibiotic response predictions:

- `Likely to fail`
- `Likely to work`
- `No-call`

Every result includes calibrated confidence, reason codes, supporting markers, target-gate status, and the required instruction to confirm with standard laboratory testing.

## Run Locally

```powershell
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## Scripts

```text
npm run build   Compile TypeScript
npm run start   Run compiled Node server
npm run dev     Build and run local demo server
npm run test    Compile and run pipeline self-tests
```

## Repository Structure

```text
client/                     Browser TypeScript for the interface
server/                     Node.js API and static server
server/pipeline/            FASTA guard, annotation adapter, predictor, Safety Governor
shared/                     Shared TypeScript types
public/                     Old-school website HTML/CSS/assets
configs/                    Species, antibiotic, threshold, and safety config
artifacts/                  Demo metrics and future trained artifacts
demo_samples/               FASTA files for live demos
```

## Prototype Boundaries

- Reconstructed genomes only.
- One configured species only.
- No sample collection, species identification, genome reconstruction, organism design, or prescribing.
- Predictions are research decision support and must be confirmed by standard laboratory testing.

## Current Demo Behavior

The demo adapter scans uploaded FASTA text for marker tokens used in the sample files, such as `blaTEM-1`, `gyrA_S83L`, and `tetA`. This keeps the website runnable immediately. Replace `demoAnnotateMarkers` with real AMRFinderPlus TSV parsing before serious evaluation.
