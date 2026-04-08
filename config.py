"""Configuration file for the pangenome analysis application."""

# Model Configuration
PROT_T5_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
# Runtime backend preference: "auto" chooses CUDA > MPS > CPU.
# You can override with "cuda", "mps", or "cpu".
DEVICE = "auto"
MODEL_PRECISION = "float16"

# Processing Parameters
BATCH_SIZE = 32  # Optimal for 64GB Apple Silicon RAM
MAX_SEQUENCE_LENGTH = 1000  # ProtT5 tokenizer safe limit (amino acids)
MAX_TOKENIZER_LENGTH = 1024  # Maximum tokens for ProtT5

# UMAP Parameters
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "cosine",
    "random_state": 42
}

# Data Processing
PROTEIN_TRANSLATION_TABLE = 1  # Standard genetic code
REMOVE_STOP_CODONS = True
FILTER_INCOMPLETE_SEQUENCES = True

# UI Configuration
DEFAULT_COLOR_BY = "Strain_Name"
PLOT_HEIGHT = 600
PLOT_WIDTH = 800
MARKER_SIZE = 6
MARKER_OPACITY = 0.7

# Cache Configuration
CACHE_DIR = "cache"
ENABLE_MEMORY_CLEANUP = True