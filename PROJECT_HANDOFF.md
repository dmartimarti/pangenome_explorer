# Pangenome Explorer - Project Handoff

## Scope
This document is a portable handoff for continuing development in a different folder or a new Copilot chat instance.

## Repository
- Name: NCBI_pangenome
- Subproject folder: pangenome_explorer
- Primary app entrypoint: app.py

## Current Pipeline Status
The app currently supports a 3-phase workflow:
1. Phase 1: Panaroo CSV + FASTA processing and translation
2. Phase 2: ProtT5 embedding generation
3. Phase 3: UMAP visualization and analysis

## Key Implemented Improvements

### Translation and Data Loading
- Identifier matching was hardened for Panaroo group IDs and locus tags.
- FASTA ID mode detection exists (group-based vs locus-based).
- CSV-to-FASTA mapping path was optimized by filtering identifiers.
- Translation supports parallel execution for large FASTA task lists.
- Compact representation fields are present:
  - Strain_Names
  - Strain_Count
- Summary and validation logic supports compact mode and avoids empty-data crashes.

### Embedding Performance and Memory
- Embedding input deduplicates protein sequences before model inference.
- Chunked embedding flow is implemented (chunk_size + batch_size).
- Device cache cleanup is included between chunks.
- Mapping embeddings back to all rows uses DataFrame merge on Sequence_AA.
- Sidebar controls include batch size and chunk size.

### Checkpoint System (High Priority Item Completed)
A checkpoint system is implemented with restore support in the sidebar.

- Restore UI:
  - Upload checkpoint ZIP
  - Restore from checkpoint button
- Save UI:
  - Prepare Phase 1 checkpoint + download
  - Prepare Phase 2 checkpoint + download

Checkpoint storage format:
- ZIP container with manifest + payload files
- JSON for metadata and stats
- CSV for tabular data
- NPZ for embeddings matrix in Phase 2

Validation and provenance:
- Versioned checkpoints via checkpoint_version
- Required-file checks by phase
- Embedding-row consistency checks for Phase 2
- Source provenance in manifest (filenames, SHA256, file sizes, FASTA mode)

## Checkpoint File Layout

### Phase 1 checkpoint ZIP
- manifest.json
- phase1_master_df.csv
- phase1_summary_stats.json
- phase1_validation_stats.json
- phase1_strain_columns.json

### Phase 2 checkpoint ZIP
- manifest.json
- phase2_master_df.csv
- phase2_embeddings.npz
- phase1_summary_stats.json
- phase1_validation_stats.json
- phase1_strain_columns.json

## Main Files to Continue Work
- app.py
- src/data_loading.py
- src/embedding.py
- src/checkpoint.py
- src/visualization.py
- src/analysis.py
- config.py
- README.md

## Model and Runtime Notes
- Current model: Rostlab/prot_t5_xl_uniref50
- Device target in config: mps (Apple Silicon)
- Precision in config: float16

## Known Remaining Priorities
- Optional support for smaller ProtT5 variants (base vs XL)
- Optional compact vs expanded output toggle in UI
- Timing report per major step (CSV, mapping, translation, embedding)
- Additional visualization features (for example 3D UMAP)

## Recommended Resume Prompt (Copy/Paste)
Use this in a new Copilot session in another folder:

"Read PROJECT_HANDOFF.md and continue this project from current status. Prioritize remaining high-impact tasks: model-size toggle, compact vs expanded representation toggle, and per-step timing report. Preserve current checkpoint compatibility (checkpoint_version=1) unless migration support is added."

## Quick Start Commands

```bash
conda activate pangenome
streamlit run app.py
```

Syntax check command:

```bash
python3 -m py_compile app.py src/*.py config.py && echo "syntax ok"
```
