# Bacterial Pangenome Analysis with ProtT5

A Streamlit web application for analyzing bacterial pangenomes using Large Protein Language Models. This tool integrates Panaroo gene presence/absence data with ProtT5 protein embeddings for functional landscape visualization using UMAP.

## Features

- **Phase 1: Data Processing**
  - Parse Panaroo gene_presence_absence.csv files
  - Translate nucleotide FASTA to amino acid sequences
  - Data validation and quality control

- **Phase 2: Protein Embeddings**
  - ProtT5-XL-U50 model integration
  - Automatic accelerator detection (CUDA, MPS, CPU fallback)
  - Batch processing for memory efficiency

- **Phase 3: Visualization**
  - UMAP dimensionality reduction
  - Interactive Plotly visualizations
  - Multiple coloring and clustering options

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pangenome_explorer
```

2. Create and activate a conda environment:

**Option A: Quick setup using environment file**
```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate pangenome
```

**Option B: Manual setup**
```bash
# Create a new conda environment with Python 3.9+
conda create -n pangenome python=3.9

# Activate the environment
conda activate pangenome
```

3. Install dependencies (skip if using Option A above):
```bash
# Install core scientific packages from conda-forge
conda install -c conda-forge streamlit pandas numpy scikit-learn plotly biopython

# Install PyTorch (automatically detects Apple Silicon MPS support)
conda install pytorch -c pytorch

# Install additional packages via pip (not available in conda)
pip install transformers umap-learn
pip install sentencepiece
# Alternative: Install all via pip in the conda environment
# pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

**Note**: The conda environment ensures better dependency management and cross-platform compatibility (macOS and Windows + WSL2). Always activate the environment before running the application:
```bash
conda activate pangenome
```

## Usage

1. **Upload Files**: Use the sidebar to upload your Panaroo CSV and reference FASTA files
2. **Process Data**: Click "Process Data" to parse and translate sequences
3. **Generate Embeddings**: Click "Generate ProtT5 Embeddings" to compute protein representations
4. **Visualize**: Click "Compute UMAP" to create the functional landscape visualization

## Checkpoint System

The app supports resumable checkpoints so you can stop after Phase 1 or Phase 2 and restore later.

### Why ZIP + JSON + CSV + NPZ?

- **ZIP bundle** keeps each checkpoint as one portable file.
- **JSON manifest** stores metadata, provenance, and versioning.
- **CSV tables** remain human-readable and easy to inspect.
- **NPZ embeddings** (Phase 2) stores numeric arrays compactly and efficiently.

This hybrid format is more robust than plain CSV-only or JSON-only checkpoints.

### Checkpoint Types

- **Phase 1 checkpoint** (after translation)
  - `manifest.json`
  - `phase1_master_df.csv`
  - `phase1_summary_stats.json`
  - `phase1_validation_stats.json`
  - `phase1_strain_columns.json`

- **Phase 2 checkpoint** (after embeddings)
  - `manifest.json`
  - `phase2_master_df.csv`
  - `phase2_embeddings.npz`
  - `phase1_summary_stats.json`
  - `phase1_validation_stats.json`
  - `phase1_strain_columns.json`

### Provenance and Validation

Each checkpoint includes source-run provenance in `manifest.json`, including:

- Panaroo filename, SHA256 hash, and file size
- FASTA filename, SHA256 hash, and file size
- FASTA ID mode
- Processing and embedding metadata
- Checkpoint version and phase

On restore, the app validates:

- checkpoint version compatibility
- required files for the selected phase
- row consistency between Phase 2 table and embedding matrix

### Example Workflow

1. Start the app and run **Phase 1** (`Process Data`).
2. In the sidebar, click **Prepare Phase 1 Checkpoint**.
3. Click **Download Phase 1 Checkpoint** and save the `.zip` file.
4. Later (or on another machine), open the app.
5. In **Restore Session**, upload the checkpoint `.zip`.
6. Click **Restore from Checkpoint** to resume from the saved state.
7. Continue with **Generate ProtT5 Embeddings** and optionally **Compute UMAP**.

For long runs, repeat the same process after embeddings:

1. Click **Prepare Phase 2 Checkpoint**.
2. Download the Phase 2 `.zip`.
3. Restore from this file in future sessions to skip both Phase 1 and Phase 2.

## System Requirements

- Python 3.8+
- 64GB RAM recommended for large datasets
- Apple Silicon Mac, Linux, or Windows with WSL2
- Optional GPU acceleration via CUDA (NVIDIA) or MPS (Apple Silicon)
- ~10GB free disk space for model downloads

## File Structure

```
pangenome_explorer/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration parameters
├── requirements.txt      # Python dependencies (pip)
├── environment.yml       # Conda environment specification
├── src/                  # Source modules
│   ├── data_loading.py   # Panaroo/FASTA parsing
│   ├── checkpoint.py     # Checkpoint system for pipeline resumption
│   ├── embedding.py      # ProtT5 model integration
│   ├── analysis.py       # UMAP and clustering
│   └── visualization.py  # Plotly visualizations
├── cache/                # Streamlit cache directory
└── data/                 # Sample data
    ├── gene_presence_absence.csv
    ├── pan_genome_reference.fa
    └── pangenome_phase1_checkpoint.zip

```

## Configuration

Key parameters can be adjusted in `config.py`:

- `BATCH_SIZE`: Number of sequences per embedding batch (default: 32)
- `MAX_SEQUENCE_LENGTH`: Maximum protein length in amino acids (default: 1000)
- `UMAP_PARAMS`: UMAP visualization parameters
- `DEVICE`: Runtime backend preference (`auto`, `cuda`, `mps`, `cpu`)

## Changelog

### v0.4.0

- Added automatic runtime accelerator detection with seamless CUDA, MPS, and CPU fallback.
- Added backend status reporting in the app so users can see whether GPU acceleration is active.
- Added resumable Phase 1 and Phase 2 checkpoints using ZIP bundles with manifest, table, JSON, and NPZ assets.
- Improved Phase 1 and Phase 2 scalability by filtering FASTA matching, parallelizing translation, avoiding redundant embeddings for identical proteins, and reducing embedding-memory overhead.
- Improved UMAP performance and memory efficiency with multi-core execution, optional PCA pre-reduction, and a low-memory mode.
- Expanded downstream exploration with a searchable protein browser, detail panel, sequence export tools, and external links for BLASTP and InterPro.
- Added stronger validation and logging to make long runs easier to diagnose and resume.

## TODO List

### High Priority
- [x] **Checkpoint System**: Implement intermediate file saving/loading for pipeline resumption
  - [x] Save processed data after Phase 1 (translation) as checkpoint
  - [x] Save embeddings after Phase 2 as checkpoint  
  - [x] Allow loading from checkpoints to skip previous steps
  - [x] Add checkpoint validation and version compatibility

- [ ] **Performance Optimizations**
  - [x] Implement chunked processing for very large datasets (>10K genes)
  - [x] Add progress bars for long-running operations
  - [x] Optimize memory usage during embedding generation
  - [ ] Add option to use smaller ProtT5 models (base vs XL)

### Medium Priority
- [ ] **Enhanced Visualizations**
  - [ ] Add 3D UMAP visualization option

- [ ] **Scalable Processing Modes**
  - [x] Add progress bars for long-running operations
  - [x] Optimize Phase 1 mapping by filtering to FASTA identifiers
  - [x] Add parallel translation for large FASTA inputs
  - [x] Avoid redundant embeddings by embedding unique protein sequences only
  - [x] Improve embedding memory usage (preallocated output array + cleanup)
  - [ ] Add user toggle for compact (unique protein) vs expanded (per-strain) representation
  - [ ] Add per-step timing report (CSV load, mapping, translation, embedding)
  - [ ] Add option to use smaller ProtT5 models (base vs XL)
  - [ ] Statistical comparison between strains/conditions  
  - [ ] Integration with ortholog databases (COG, KEGG)
  - [ ] Phylogenetic tree integration

### Low Priority
- [ ] **User Experience**
  - [ ] Add data upload validation and format checking
  - [ ] Implement tutorial/guided walkthrough
- [ ] **Protein Browser and Exports**
  - [x] Full-dataset searchable protein browser with pagination
  - [x] Adjustable rows-per-page controls
  - [x] Protein detail panel with amino-acid sequence display
  - [x] BLASTP launch button for selected sequence
  - [x] InterPro sequence-search launch button
  - [x] Download selected protein as FASTA
  - [x] Export filtered proteins as CSV/FASTA
  - [x] Copy selected sequence to clipboard
  - [ ] Add configurable FASTA header templates for exports
  - [ ] Add multi-protein selection and export from current filtered page

- [ ] **Data Export/Import**
  - [ ] Export results to various formats (Excel, HDF5, Parquet)
  - [ ] Import pre-computed embeddings from other tools
  - [ ] Batch processing mode for multiple datasets
  - [ ] Add unit tests and CI/CD pipeline
  - [ ] Docker containerization
  - [ ] Add logging and debugging modes
  - [x] Add robust data upload validation and format checking

### Future Enhancements
- [ ] **Multi-modal Analysis**
  - [ ] Integration with genomic context (synteny analysis)
  - [ ] Protein structure prediction integration (AlphaFold)
  - [ ] Metabolic pathway reconstruction
  - [ ] Comparative genomics with reference databases
  - [x] Add logging and debugging diagnostics for pipeline stages
- [ ] **Scalability** 
  - [ ] Cloud deployment options (AWS, GCP, Azure)
  - [ ] Distributed computing for large-scale analysis
  - [ ] Database backend for large datasets
  - [ ] REST API for programmatic access

## Troubleshooting

**Memory Issues**: Reduce `BATCH_SIZE` in config.py or use smaller datasets
**Model Loading Errors**: Verify that PyTorch can detect CUDA on WSL2/NVIDIA or MPS on Apple Silicon; the app will automatically fall back to CPU with a warning
**Translation Errors**: Check FASTA file format and ensure nucleotide sequences
**Environment Issues**: Make sure the conda environment is activated (`conda activate pangenome`)
**Package Conflicts**: If using mixed conda/pip installation, try installing everything via pip in the conda environment

## License

This project is licensed under MIT License.

## Citation

If you use this tool in your research, please cite:
- ProtT5: Elnaggar, A. et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell. 2022.
- UMAP: McInnes, L. et al. UMAP: Uniform Manifold Approximation and Projection. J Open Source Softw. 2018.
- Panaroo: Tonkin-Hill, G. et al. Producing polished prokaryotic pangenomes with the Panaroo pipeline. Genome Biol. 2020.