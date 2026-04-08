"""
Bacterial Pangenome Analysis Web Application
===========================================

A Streamlit application for analyzing bacterial pangenomes using Large Protein Language Models.
Integrates Panaroo gene presence/absence data with ProtT5 embeddings for functional landscape visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
sys.path.append('src')
from data_loading import (load_panaroo_csv, parse_fasta_and_translate, 
                         validate_data_consistency, compute_summary_statistics,
                         detect_fasta_id_mode, extract_fasta_identifiers)
from embedding import (
    load_prot_t5_model,
    generate_protein_embeddings_chunked_with_progress,
    check_model_availability,
    estimate_model_download_size,
    get_runtime_backend_info,
)
from checkpoint import (
    CheckpointError,
    build_source_metadata,
    create_phase1_checkpoint_bytes,
    create_phase2_checkpoint_bytes,
    load_checkpoint_bytes,
)
from analysis import compute_umap_embedding, add_umap_coordinates, analyze_umap_clusters
from visualization import (create_umap_scatter_plot, create_summary_statistics_plot, 
                          create_enhanced_summary_plots, save_plots_to_files,
                          display_data_table, create_validation_plot)
from config import MAX_SEQUENCE_LENGTH, BATCH_SIZE, DEFAULT_COLOR_BY, PROT_T5_MODEL_NAME, UMAP_PCA_COMPONENTS


def main():
    """Main application function."""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Pangenome Explorer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("🧬 Bacterial Pangenome Analysis with ProtT5")
    st.markdown("*Functional landscape visualization using protein language models*")
    
    # Initialize session state
    initialize_session_state()

    backend_info = get_runtime_backend_info()
    render_backend_status(backend_info)
    
    # Create layout
    create_sidebar()
    
    # Main content area with tabs
    create_main_content()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'embeddings_generated' not in st.session_state:
        st.session_state.embeddings_generated = False
    
    if 'umap_computed' not in st.session_state:
        st.session_state.umap_computed = False
    
    if 'master_df' not in st.session_state:
        st.session_state.master_df = None
    
    if 'summary_stats' not in st.session_state:
        st.session_state.summary_stats = None
    
    if 'validation_stats' not in st.session_state:
        st.session_state.validation_stats = None

    if 'source_metadata' not in st.session_state:
        st.session_state.source_metadata = {}

    if 'processing_metadata' not in st.session_state:
        st.session_state.processing_metadata = {}

    if 'embedding_metadata' not in st.session_state:
        st.session_state.embedding_metadata = {}

    if 'checkpoint_manifest' not in st.session_state:
        st.session_state.checkpoint_manifest = None

    if 'phase1_checkpoint_bytes' not in st.session_state:
        st.session_state.phase1_checkpoint_bytes = None

    if 'phase2_checkpoint_bytes' not in st.session_state:
        st.session_state.phase2_checkpoint_bytes = None


def render_backend_status(backend_info: Dict[str, str]) -> None:
    """Display runtime accelerator status and warnings."""
    selected_device = backend_info.get('selected_device', 'cpu')
    gpu_name = backend_info.get('gpu_name', '')
    warning = backend_info.get('warning', '')

    if selected_device == 'cuda':
        st.success(f"Runtime backend: CUDA ({gpu_name})")
    elif selected_device == 'mps':
        st.success("Runtime backend: MPS (Apple Silicon GPU)")
    else:
        st.warning("Runtime backend: CPU (CUDA and MPS not available)")

    if warning:
        st.warning(warning)


def create_sidebar():
    """Create the sidebar with file upload and parameter controls."""
    
    with st.sidebar:
        st.header("📂 Data Input")

        st.subheader("♻️ Restore Session")
        checkpoint_file = st.file_uploader(
            "Checkpoint (.zip)",
            type=['zip'],
            help="Load a previously saved Phase 1 or Phase 2 checkpoint"
        )
        if checkpoint_file is not None:
            if st.button("Restore from Checkpoint"):
                restore_from_checkpoint(checkpoint_file)
        
        # File uploads
        st.subheader("Upload Files")
        panaroo_file = st.file_uploader(
            "**Panaroo Gene Presence/Absence CSV**",
            type=['csv'],
            help="Upload the gene_presence_absence.csv file from Panaroo analysis"
        )
        
        fasta_file = st.file_uploader(
            "**Reference Nucleotide FASTA**", 
            type=['fasta', 'fa', 'fas'],
            help="Upload the reference genome nucleotide FASTA file"
        )
        
        # Processing parameters
        st.subheader("⚙️ Processing Parameters")
        
        max_length = st.slider(
            "Max Sequence Length (AA)",
            min_value=100,
            max_value=2000,
            value=MAX_SEQUENCE_LENGTH,
            step=50,
            help=f"Filter out proteins longer than this threshold. Default: {MAX_SEQUENCE_LENGTH} AA"
        )
        
        batch_size = st.select_slider(
            "Embedding Batch Size",
            options=[8, 16, 32, 64, 128],
            value=BATCH_SIZE,
            help=f"Number of sequences to process per batch. Default: {BATCH_SIZE}"
        )

        chunk_size = st.select_slider(
            "Embedding Chunk Size",
            options=[500, 1000, 2000, 5000, 10000, 20000],
            value=5000,
            help="Process this many unique proteins at a time. Smaller chunks use less memory."
        )
        
        # UMAP parameters
        st.subheader("🗺️ UMAP Parameters")
        
        n_neighbors = st.slider(
            "Number of Neighbors",
            min_value=5,
            max_value=50,
            value=15,
            help="Controls local vs global structure in UMAP"
        )
        
        min_dist = st.slider(
            "Minimum Distance", 
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.2f",
            help="Controls how tightly points cluster in UMAP"
        )

        pca_components = st.select_slider(
            "PCA pre-reduction dimensions",
            options=[0, 30, 50, 100, 150, 200],
            value=UMAP_PCA_COMPONENTS,
            help="Reduce embedding dimensions with PCA before UMAP. 0 = disabled. "
                 "Recommended: 50. Greatly reduces memory and speeds up computation "
                 "with minimal quality loss for ProtT5 1024-dim embeddings.",
        )

        low_memory = st.checkbox(
            "UMAP low-memory mode",
            value=False,
            help="Slower but uses significantly less RAM. Recommended for systems "
                 "with less than 32 GB or datasets with more than 50 000 proteins.",
        )

        # Process data button
        st.subheader("🚀 Processing")
        
        if panaroo_file and fasta_file:
            if st.button("Process Data", type="primary"):
                process_uploaded_files(panaroo_file, fasta_file, max_length)
        else:
            st.info("Please upload both CSV and FASTA files to proceed")
        
        # Model loading button
        if st.session_state.data_loaded and not st.session_state.embeddings_generated:
            if st.button("Generate ProtT5 Embeddings", type="primary"):
                generate_embeddings(batch_size, chunk_size)
        
        # UMAP computation button
        if st.session_state.embeddings_generated and not st.session_state.umap_computed:
            if st.button("Compute UMAP", type="primary"):
                compute_umap_visualization(n_neighbors, min_dist, pca_components, low_memory)

        if st.session_state.data_loaded:
            st.subheader("💾 Checkpoints")

            if st.button("Prepare Phase 1 Checkpoint"):
                prepare_phase1_checkpoint()

            if st.session_state.phase1_checkpoint_bytes is not None:
                st.download_button(
                    label="Download Phase 1 Checkpoint",
                    data=st.session_state.phase1_checkpoint_bytes,
                    file_name="pangenome_phase1_checkpoint.zip",
                    mime="application/zip"
                )

            if st.session_state.embeddings_generated:
                if st.button("Prepare Phase 2 Checkpoint"):
                    prepare_phase2_checkpoint()

                if st.session_state.phase2_checkpoint_bytes is not None:
                    st.download_button(
                        label="Download Phase 2 Checkpoint",
                        data=st.session_state.phase2_checkpoint_bytes,
                        file_name="pangenome_phase2_checkpoint.zip",
                        mime="application/zip"
                    )


def create_main_content():
    """Create the main content area with tabs."""
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🧬 Embeddings", "🗺️ Visualization", "📈 Analysis"])
    
    with tab1:
        display_data_overview_tab()
    
    with tab2:
        display_embeddings_tab()
    
    with tab3:
        display_visualization_tab()
    
    with tab4:
        display_analysis_tab()


def process_uploaded_files(panaroo_file, fasta_file, max_length: int):
    """Process the uploaded CSV and FASTA files."""
    
    try:
        progress = st.progress(0, text="Starting data processing...")

        with st.spinner("Processing uploaded files..."):
            fasta_contents = fasta_file.getvalue()
            fasta_mode = detect_fasta_id_mode(fasta_contents)
            required_identifiers = extract_fasta_identifiers(fasta_contents)

            # Load Panaroo CSV
            progress.progress(10, text="Loading Panaroo CSV...")
            st.info("Loading Panaroo CSV...")
            panaroo_contents = panaroo_file.getvalue()
            include_locus_tag_mapping = fasta_mode != "group"
            panaroo_df, locus_mapping, strain_columns = load_panaroo_csv(
                panaroo_contents,
                include_locus_tag_mapping=include_locus_tag_mapping,
                required_identifiers=required_identifiers
            )
            if fasta_mode == "group":
                st.info("Detected group-based FASTA headers; using optimized group mapping path")
            
            # Parse and translate FASTA
            progress.progress(45, text="Translating nucleotide sequences...")
            st.info("Translating nucleotide sequences...")
            sequences_df = parse_fasta_and_translate(fasta_contents, locus_mapping)
            
            # Compute enhanced summary statistics
            progress.progress(75, text="Computing summary statistics...")
            st.info("Computing summary statistics...")
            summary_stats = compute_summary_statistics(sequences_df, panaroo_df, strain_columns)
            
            # Validate data
            progress.progress(90, text="Validating processed data...")
            st.info("Validating data consistency...")
            validation_stats = validate_data_consistency(sequences_df, panaroo_df, strain_columns)
            
            # Store in session state
            st.session_state.master_df = sequences_df
            st.session_state.summary_stats = summary_stats
            st.session_state.validation_stats = validation_stats
            st.session_state.panaroo_df = panaroo_df
            st.session_state.strain_columns = strain_columns
            st.session_state.source_metadata = build_source_metadata(
                panaroo_name=panaroo_file.name,
                panaroo_content=panaroo_contents,
                fasta_name=fasta_file.name,
                fasta_content=fasta_contents,
                fasta_mode=fasta_mode
            )
            st.session_state.processing_metadata = {
                'max_sequence_length': int(max_length),
                'required_identifier_count': int(len(required_identifiers)),
                'panaroo_rows': int(len(panaroo_df)),
                'translated_rows': int(len(sequences_df)),
            }
            st.session_state.embedding_metadata = {}
            st.session_state.checkpoint_manifest = None
            st.session_state.phase1_checkpoint_bytes = None
            st.session_state.phase2_checkpoint_bytes = None
            st.session_state.data_loaded = True

        progress.progress(100, text="Data processing complete")
            
        st.success(f"✅ Successfully processed {len(sequences_df)} protein sequences!")
        
        # Display quick summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Proteins", len(sequences_df))
        with col2:
            st.metric("Strains", len(strain_columns))
        with col3:
            st.metric("Unique Genes", sequences_df['Gene_Name'].nunique() if len(sequences_df) > 0 else 0)
        with col4:
            success_rate = summary_stats['translation_stats']['success_rate']
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        st.error(f"❌ Error processing files: {e}")
        st.error("Please check your file formats and try again.")


def generate_embeddings(batch_size: int, chunk_size: int):
    """Generate ProtT5 embeddings for the protein sequences."""
    
    try:
        progress = st.progress(0, text="Preparing ProtT5 embedding workflow...")

        is_available, model_info = check_model_availability()

        tokenizer = None
        model = None

        if not is_available:
            download_size = estimate_model_download_size()
            st.warning(
                f"""
**ProtT5 Model Not Found Locally**

Downloading automatically now.

**Download size**: {download_size}
**Location**: Hugging Face cache (~/.cache/huggingface/)

This is a one-time download and will be cached for future runs.
                """
            )

            progress.progress(20, text="Downloading ProtT5 model to local cache...")
            with st.spinner("Downloading and loading ProtT5 model (first run may take several minutes)..."):
                tokenizer, model = load_prot_t5_model(force_download=True)
            st.success("✅ ProtT5 model downloaded and loaded")
        else:
            st.success(f"✅ ProtT5 model found locally at: {model_info}")
            progress.progress(30, text="Loading ProtT5 model from local cache...")
            with st.spinner("Loading ProtT5 model..."):
                tokenizer, model = load_prot_t5_model(force_download=False)

        progress.progress(60, text="Generating embeddings from protein sequences...")
        with st.spinner("Generating embeddings..."):
            # Get unique sequences to avoid redundant embeddings
            unique_sequences = st.session_state.master_df['Sequence_AA'].unique()
            unique_seq_list = list(unique_sequences)
            
            st.info(f"Generating embeddings for {len(unique_seq_list):,} unique sequences ({len(st.session_state.master_df):,} total rows)...")
            logger.info(f"Embedding input: {len(unique_seq_list):,} unique sequences from {len(st.session_state.master_df):,} total rows")

            batch_progress = st.progress(0, text="Embedding batches: 0/0")

            total_chunks = (len(unique_seq_list) + max(chunk_size, 1) - 1) // max(chunk_size, 1)

            def update_batch_progress(chunk_idx: int, num_chunks: int, batch_idx: int, num_batches: int):
                chunk_fraction = (chunk_idx - 1) / max(num_chunks, 1)
                batch_fraction = batch_idx / max(num_batches, 1)
                overall_fraction = chunk_fraction + (batch_fraction / max(num_chunks, 1))
                batch_percent = int(overall_fraction * 100)

                batch_progress.progress(
                    min(batch_percent, 100),
                    text=(
                        f"Embedding chunks: {chunk_idx}/{num_chunks} | "
                        f"Batches in chunk: {batch_idx}/{num_batches}"
                    )
                )

                overall_percent = 60 + int(overall_fraction * 35)
                progress.progress(min(overall_percent, 95), text="Generating embeddings from protein sequences...")

            # Generate embeddings for unique sequences only
            unique_embeddings = generate_protein_embeddings_chunked_with_progress(
                sequences=unique_seq_list,
                _tokenizer=tokenizer,
                _model=model,
                batch_size=batch_size,
                chunk_size=chunk_size,
                progress_callback=update_batch_progress
            )

            batch_progress.progress(100, text="Embedding batches complete")

            # Create embedding mapping and expand to all rows
            progress.progress(95, text="Mapping embeddings to all strain instances...")
            embedding_df = pd.DataFrame({
                'Sequence_AA': unique_seq_list,
                'Embeddings': [emb for emb in unique_embeddings]
            })
            df_without_embeddings = st.session_state.master_df.drop(columns=['Embeddings'], errors='ignore')
            st.session_state.master_df = df_without_embeddings.merge(
                embedding_df,
                on='Sequence_AA',
                how='left',
                validate='many_to_one'
            )
            st.session_state.embedding_metadata = {
                'batch_size': int(batch_size),
                'chunk_size': int(chunk_size),
                'model_name': PROT_T5_MODEL_NAME
            }
            st.session_state.embedding_metadata['embedded_unique_sequences'] = int(len(unique_seq_list))
            st.session_state.embedding_metadata['total_rows_with_embeddings'] = int(len(st.session_state.master_df))
            st.session_state.embeddings_generated = True
            st.session_state.phase2_checkpoint_bytes = None

        progress.progress(100, text="Embedding generation complete")
            
        st.success("✅ Embeddings generated successfully!")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        st.error(f"Error generating embeddings: {e}")


def prepare_phase1_checkpoint() -> None:
    """Build and cache Phase 1 checkpoint bytes for download."""
    try:
        checkpoint_bytes = create_phase1_checkpoint_bytes(
            master_df=st.session_state.master_df,
            summary_stats=st.session_state.summary_stats,
            validation_stats=st.session_state.validation_stats,
            strain_columns=st.session_state.strain_columns,
            source_metadata=st.session_state.source_metadata,
            processing_metadata=st.session_state.processing_metadata,
        )
        st.session_state.phase1_checkpoint_bytes = checkpoint_bytes
        st.success("✅ Phase 1 checkpoint prepared")
    except Exception as e:
        st.error(f"❌ Failed to build Phase 1 checkpoint: {e}")


def prepare_phase2_checkpoint() -> None:
    """Build and cache Phase 2 checkpoint bytes for download."""
    try:
        checkpoint_bytes = create_phase2_checkpoint_bytes(
            master_df=st.session_state.master_df,
            summary_stats=st.session_state.summary_stats,
            validation_stats=st.session_state.validation_stats,
            strain_columns=st.session_state.strain_columns,
            source_metadata=st.session_state.source_metadata,
            processing_metadata=st.session_state.processing_metadata,
            embedding_metadata=st.session_state.embedding_metadata,
        )
        st.session_state.phase2_checkpoint_bytes = checkpoint_bytes
        st.success("✅ Phase 2 checkpoint prepared")
    except Exception as e:
        st.error(f"❌ Failed to build Phase 2 checkpoint: {e}")


def restore_from_checkpoint(checkpoint_file) -> None:
    """Restore session state from uploaded checkpoint zip."""
    try:
        checkpoint_bytes = checkpoint_file.getvalue()
        payload = load_checkpoint_bytes(checkpoint_bytes)

        st.session_state.master_df = payload['master_df']
        st.session_state.summary_stats = payload['summary_stats']
        st.session_state.validation_stats = payload['validation_stats']
        st.session_state.strain_columns = payload['strain_columns']
        st.session_state.data_loaded = payload['data_loaded']
        st.session_state.embeddings_generated = payload['embeddings_generated']
        st.session_state.umap_computed = payload['umap_computed']
        st.session_state.panaroo_df = None
        st.session_state.checkpoint_manifest = payload['manifest']
        st.session_state.source_metadata = payload['manifest'].get('source', {})
        st.session_state.processing_metadata = payload['manifest'].get('processing', {})
        st.session_state.embedding_metadata = payload['manifest'].get('embedding', {})
        st.session_state.phase1_checkpoint_bytes = None
        st.session_state.phase2_checkpoint_bytes = None

        phase = payload['manifest'].get('phase', 'unknown')
        row_count = len(st.session_state.master_df)
        st.success(f"✅ Restored {phase} checkpoint with {row_count:,} rows")

        source = payload['manifest'].get('source', {})
        if source:
            st.info(
                "Source provenance: "
                f"Panaroo={source.get('panaroo_filename', 'unknown')} "
                f"(SHA256 {str(source.get('panaroo_sha256', ''))[:12]}...), "
                f"FASTA={source.get('fasta_filename', 'unknown')} "
                f"(SHA256 {str(source.get('fasta_sha256', ''))[:12]}...)"
            )

        st.rerun()

    except CheckpointError as e:
        st.error(f"❌ Invalid checkpoint: {e}")
    except Exception as e:
        st.error(f"❌ Could not restore checkpoint: {e}")


def compute_umap_visualization(n_neighbors: int, min_dist: float,
                               pca_components: int = UMAP_PCA_COMPONENTS,
                               low_memory: bool = False):
    """Compute UMAP dimensionality reduction."""

    try:
        progress = st.progress(0, text="Starting UMAP computation...")

        with st.spinner("Computing UMAP dimensionality reduction..."):
            # Extract embeddings as float32 to avoid an implicit copy inside UMAP.
            progress.progress(20, text="Extracting embedding matrix...")
            embeddings = np.vstack(
                st.session_state.master_df['Embeddings'].values
            ).astype(np.float32)

            # Compute UMAP (PCA pre-reduction and n_jobs handled inside)
            pca_msg = f" → PCA {pca_components}d" if pca_components else ""
            progress.progress(55, text=f"Running UMAP{pca_msg}...")
            umap_coords, umap_model = compute_umap_embedding(
                embeddings,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                pca_components=pca_components,
                low_memory=low_memory,
            )
            
            # Add coordinates to dataframe
            progress.progress(80, text="Adding UMAP coordinates to dataset...")
            st.session_state.master_df = add_umap_coordinates(
                st.session_state.master_df, umap_coords
            )
            
            # Analyze clusters
            progress.progress(95, text="Analyzing cluster structure...")
            cluster_analysis = analyze_umap_clusters(
                umap_coords, 
                st.session_state.master_df['Strain_Name'].values
            )
            
            st.session_state.cluster_analysis = cluster_analysis
            st.session_state.umap_computed = True

        progress.progress(100, text="UMAP computation complete")
            
        st.success("✅ UMAP visualization computed!")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error computing UMAP: {e}")
        st.error(f"Error computing UMAP: {e}")


def display_data_overview_tab():
    """Display the data overview tab content with enhanced statistics."""
    
    if not st.session_state.data_loaded:
        st.info("👈 Please upload data files in the sidebar to begin analysis")
        return
    
    st.subheader("📊 Data Processing Summary")
    
    # Display key metrics
    if st.session_state.summary_stats:
        summary_stats = st.session_state.summary_stats
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sequences", len(st.session_state.master_df))
            
        with col2:
            st.metric("Unique Strains", summary_stats['strain_stats']['total_strains'])
            
        with col3:
            st.metric("Unique Genes", summary_stats['gene_stats']['total_unique_genes'])
            
        with col4:
            success_rate = summary_stats['translation_stats']['success_rate']
            st.metric("Translation Success Rate", f"{success_rate:.1%}")
        
        # Additional metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mean_length = summary_stats['length_stats']['mean']
            st.metric("Mean Protein Length", f"{mean_length:.0f} AA")
            
        with col2:
            core_genes = summary_stats['gene_stats']['core_genes']
            st.metric("Core Genes (>95%)", core_genes)
            
        with col3:
            accessory_genes = summary_stats['gene_stats']['accessory_genes']
            st.metric("Accessory Genes (<95%)", accessory_genes)
            
        with col4:
            mean_proteins = summary_stats['strain_stats']['mean_proteins_per_strain']
            st.metric("Mean Proteins/Strain", f"{mean_proteins:.0f}")
    
    # Enhanced summary plots
    st.subheader("📈 Detailed Summary Statistics")
    
    if st.session_state.summary_stats:
        try:
            # Generate enhanced plots
            enhanced_figures = create_enhanced_summary_plots(st.session_state.summary_stats)
            
            if enhanced_figures:
                # Create tabs for different plot categories
                plot_tabs = st.tabs([
                    "📏 Sequence Lengths", 
                    "🧬 Strain Analysis", 
                    "🧮 Gene Analysis", 
                    "📊 Success Rates"
                ])
                
                # Organize plots by category
                length_plots = [(name, fig) for name, fig in enhanced_figures if 'length' in name]
                strain_plots = [(name, fig) for name, fig in enhanced_figures if 'strain' in name]
                gene_plots = [(name, fig) for name, fig in enhanced_figures if any(x in name for x in ['gene', 'core', 'accessory'])]
                success_plots = [(name, fig) for name, fig in enhanced_figures if 'translation' in name]
                
                with plot_tabs[0]:
                    for name, fig in length_plots:
                        st.plotly_chart(fig, use_container_width=True)
                
                with plot_tabs[1]:
                    for name, fig in strain_plots:
                        st.plotly_chart(fig, use_container_width=True)
                
                with plot_tabs[2]:
                    for name, fig in gene_plots:
                        st.plotly_chart(fig, use_container_width=True)
                
                with plot_tabs[3]:
                    for name, fig in success_plots:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Add plot saving functionality
                st.subheader("💾 Save Plots")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    save_formats = st.multiselect(
                        "Select formats to save:",
                        options=['png', 'pdf', 'svg', 'html'],
                        default=['png', 'html']
                    )
                
                with col2:
                    output_dir = st.text_input("Output directory:", value="plots")
                
                if st.button("📥 Save All Plots") and save_formats:
                    try:
                        saved_files = save_plots_to_files(enhanced_figures, save_formats, output_dir)
                        
                        st.success(f"✅ Plots saved successfully!")
                        
                        # Display saved files
                        with st.expander("📁 Saved Files"):
                            for plot_name, files in saved_files.items():
                                st.write(f"**{plot_name}:**")
                                for file_path in files:
                                    st.write(f"  - {file_path}")
                                    
                    except Exception as e:
                        st.error(f"❌ Error saving plots: {e}")
            
            else:
                st.warning("Could not generate enhanced plots")
                
        except Exception as e:
            st.error(f"Error creating enhanced plots: {e}")
            
            # Fallback to basic summary plot
            try:
                summary_fig = create_summary_statistics_plot(st.session_state.master_df)
                st.plotly_chart(summary_fig, use_container_width=True)
            except Exception as e2:
                st.error(f"Error creating any plots: {e2}")
    
    # Display data table
    st.subheader("🔍 Data Preview")
    display_data_table(
        st.session_state.master_df,
        max_rows=100,
        columns_to_show=['Locus_Tag', 'Gene_Name', 'Strain_Count', 'Length', 'Annotation']
    )


def display_embeddings_tab():
    """Display the embeddings tab content."""
    
    if not st.session_state.data_loaded:
        st.info("Please process data first in the Data Overview tab")
        return
    
    if not st.session_state.embeddings_generated:
        st.info("👈 Click 'Generate ProtT5 Embeddings' in the sidebar to compute protein embeddings")
        return
    
    st.subheader("🧬 ProtT5 Protein Embeddings")
    
    # Display embedding statistics
    embeddings = np.vstack(st.session_state.master_df['Embeddings'].values)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Embedding Dimension", embeddings.shape[1])
    with col2:
        st.metric("Total Proteins", embeddings.shape[0])
    with col3:
        st.metric("Mean Vector Norm", f"{np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")
    with col4:
        st.metric("Memory Usage", f"{embeddings.nbytes / 1024**2:.1f} MB")
    
    # PCA preview (optional quick visualization)
    if st.checkbox("Show PCA Preview", help="Quick 2D projection using PCA (faster than UMAP)"):
        try:
            from sklearn.decomposition import PCA
            
            with st.spinner("Computing PCA..."):
                pca = PCA(n_components=2, random_state=42)
                pca_coords = pca.fit_transform(embeddings)
                
                # Create simple PCA plot
                import plotly.express as px
                pca_df = st.session_state.master_df.copy()
                pca_df['PCA_1'] = pca_coords[:, 0]
                pca_df['PCA_2'] = pca_coords[:, 1]
                
                fig = px.scatter(
                    pca_df, 
                    x='PCA_1', 
                    y='PCA_2', 
                    color='Strain_Name',
                    title="PCA Projection of ProtT5 Embeddings",
                    hover_data=['Gene_Name', 'Strain_Name']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
                
        except Exception as e:
            st.error(f"Error computing PCA preview: {e}")


def display_visualization_tab():
    """Display the visualization tab content."""
    
    if not st.session_state.embeddings_generated:
        st.info("Please generate embeddings first")
        return
    
    if not st.session_state.umap_computed:
        st.info("👈 Click 'Compute UMAP' in the sidebar to create the visualization")
        return
    
    st.subheader("🗺️ Interactive UMAP Visualization")
    
    # Color selection
    color_options = ['Strain_Name', 'Gene_Name', 'Length', 'Annotation']
    available_options = [opt for opt in color_options if opt in st.session_state.master_df.columns]
    
    color_by = st.selectbox(
        "Color points by:",
        available_options,
        index=0 if DEFAULT_COLOR_BY not in available_options else available_options.index(DEFAULT_COLOR_BY)
    )
    
    # Create and display UMAP plot
    try:
        umap_fig = create_umap_scatter_plot(
            st.session_state.master_df,
            color_by=color_by,
            hover_data=['Gene_Name', 'Strain_Name', 'Length', 'Annotation']
        )
        st.plotly_chart(umap_fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating UMAP plot: {e}")
    
    # Display cluster analysis
    if hasattr(st.session_state, 'cluster_analysis'):
        st.subheader("🎯 Cluster Analysis")
        
        cluster_info = st.session_state.cluster_analysis
        if 'clustering' in cluster_info:
            clustering = cluster_info['clustering']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Discovered Clusters", clustering.get('best_k', 'N/A'))
            with col2:
                st.metric("Silhouette Score", f"{clustering.get('silhouette_score', 0):.3f}")


def display_analysis_tab():
    """Display the analysis tab content."""
    
    if not st.session_state.umap_computed:
        st.info("Please complete the UMAP visualization first")
        return
    
    st.subheader("📈 Advanced Analysis")
    
    # Analysis options
    analysis_type = st.selectbox(
        "Analysis Type:",
        ["Strain Distribution", "Gene Family Analysis", "Functional Annotation", "Custom Analysis"]
    )
    
    if analysis_type == "Strain Distribution":
        # Strain-based analysis
        strain_counts = st.session_state.master_df['Strain_Name'].value_counts()
        
        import plotly.express as px
        fig = px.bar(
            x=strain_counts.index[:20],  # Top 20 strains
            y=strain_counts.values[:20],
            title="Top 20 Strains by Protein Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Gene Family Analysis":
        # Gene family analysis
        gene_counts = st.session_state.master_df['Gene_Name'].value_counts()
        
        import plotly.express as px
        fig = px.bar(
            x=gene_counts.values[:20],
            y=gene_counts.index[:20],
            orientation='h',
            title="Top 20 Gene Families by Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Functional Annotation":
        # Annotation analysis
        st.info("Functional annotation analysis - implement based on your specific annotation format")
    
    # Download options
    st.subheader("💾 Download Results")
    
    if st.button("Prepare Downloads"):
        # Prepare CSV download
        csv_data = st.session_state.master_df.drop('Embeddings', axis=1)  # Remove embeddings for CSV
        csv = csv_data.to_csv(index=False)
        
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="pangenome_analysis_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()