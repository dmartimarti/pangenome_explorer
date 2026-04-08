"""Interactive visualization utilities for pangenome analysis."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, List, Optional, Tuple
import logging
from urllib.parse import quote_plus
import json

from config import DEFAULT_COLOR_BY, PLOT_HEIGHT, PLOT_WIDTH, MARKER_SIZE, MARKER_OPACITY

logger = logging.getLogger(__name__)


def _build_blastp_url(sequence: str) -> str:
    """Build an NCBI BLASTP URL with the selected protein sequence prefilled."""
    return (
        "https://blast.ncbi.nlm.nih.gov/Blast.cgi?"
        f"PAGE=Proteins&PROGRAM=blastp&PAGE_TYPE=BlastSearch&QUERY={quote_plus(sequence)}"
        "&DATABASE=nr&CMD=request"
    )


def _build_interpro_url() -> str:
    """Return the InterPro sequence search page URL."""
    return "https://www.ebi.ac.uk/interpro/search/sequence/"


def _make_protein_label(row: pd.Series) -> str:
    """Create a compact label for selecting a protein record."""
    gene_name = str(row.get('Gene_Name', 'Unknown'))
    locus_tag = str(row.get('Locus_Tag', 'Unknown'))
    annotation = str(row.get('Annotation', 'Unknown'))
    annotation_short = annotation[:60] + ("..." if len(annotation) > 60 else "")
    return f"{gene_name} | {locus_tag} | {annotation_short}"


def _format_fasta_record(row: pd.Series) -> str:
        """Format one protein row as FASTA."""
        header_parts = [str(row.get('Locus_Tag', 'unknown')), str(row.get('Gene_Name', 'unknown'))]
        if pd.notna(row.get('Annotation')):
                header_parts.append(str(row.get('Annotation')))
        header = " | ".join(header_parts)
        sequence = str(row.get('Sequence_AA', ''))
        wrapped = "\n".join(sequence[i:i + 80] for i in range(0, len(sequence), 80))
        return f">{header}\n{wrapped}"


def _format_fasta_records(df: pd.DataFrame) -> str:
        """Format a dataframe of protein rows as multi-FASTA."""
        return "\n".join(_format_fasta_record(row) for _, row in df.iterrows())


def _render_copy_sequence_button(sequence: str, key: str) -> None:
        """Render a small copy-to-clipboard button for a protein sequence."""
        safe_sequence = json.dumps(sequence)
        button_id = f"copy-seq-{key}"
        status_id = f"copy-status-{key}"
        components.html(
                f"""
                <div style=\"display:flex;align-items:center;gap:0.5rem;margin-top:0.25rem;\">
                    <button id=\"{button_id}\" style=\"padding:0.4rem 0.8rem;border:1px solid #999;border-radius:0.4rem;background:#f5f5f5;cursor:pointer;\">Copy Sequence</button>
                    <span id=\"{status_id}\" style=\"font-size:0.9rem;color:#555;\"></span>
                </div>
                <script>
                    const button = document.getElementById({json.dumps(button_id)});
                    const status = document.getElementById({json.dumps(status_id)});
                    button?.addEventListener('click', async () => {{
                        try {{
                            await navigator.clipboard.writeText({safe_sequence});
                            status.textContent = 'Copied';
                        }} catch (err) {{
                            status.textContent = 'Clipboard blocked by browser';
                        }}
                    }});
                </script>
                """,
                height=45,
        )


def create_umap_scatter_plot(df: pd.DataFrame,
                           x_col: str = "UMAP_1",
                           y_col: str = "UMAP_2", 
                           color_by: str = DEFAULT_COLOR_BY,
                           size_by: Optional[str] = None,
                           hover_data: Optional[List[str]] = None,
                           title: str = "Protein Functional Landscape (UMAP)",
                           width: int = PLOT_WIDTH,
                           height: int = PLOT_HEIGHT) -> go.Figure:
    """
    Create an interactive UMAP scatter plot.
    
    Args:
        df: DataFrame with UMAP coordinates and metadata
        x_col: Column name for X coordinates
        y_col: Column name for Y coordinates
        color_by: Column name for coloring points
        size_by: Optional column name for sizing points
        hover_data: List of columns to show in hover tooltip
        title: Plot title
        width: Plot width
        height: Plot height
    
    Returns:
        Plotly figure
    """
    try:
        # Default hover data
        if hover_data is None:
            hover_data = ['Gene_Name', 'Strain_Name', 'Length', 'Annotation']
        
        # Remove any hover_data columns that don't exist
        available_hover_data = [col for col in hover_data if col in df.columns]
        
        # Create the scatter plot
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_by,
            size=size_by,
            hover_data=available_hover_data,
            title=title,
            width=width,
            height=height,
            opacity=MARKER_OPACITY
        )
        
        # Update marker properties
        fig.update_traces(
            marker=dict(
                size=MARKER_SIZE if size_by is None else None,
                line=dict(width=0.5, color='white'),
                sizemode='diameter' if size_by else None
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(x=0.5, xanchor='center'),
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=1.01
            ),
            margin=dict(r=150)  # Space for legend
        )
        
        # Custom hover template
        hover_template = f"<b>%{{hovertext}}</b><br>"
        hover_template += f"{x_col}: %{{x:.3f}}<br>"
        hover_template += f"{y_col}: %{{y:.3f}}<br>"
        
        for col in available_hover_data:
            if col != color_by:  # Avoid repeating color column
                hover_template += f"{col}: %{{customdata[{available_hover_data.index(col)}]}}<br>"
        
        hover_template += "<extra></extra>"
        
        fig.update_traces(hovertemplate=hover_template)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating UMAP scatter plot: {e}")
        raise


def create_density_plot(df: pd.DataFrame,
                       x_col: str = "UMAP_1",
                       y_col: str = "UMAP_2",
                       title: str = "Protein Density Distribution") -> go.Figure:
    """
    Create a density plot overlay for UMAP visualization.
    
    Args:
        df: DataFrame with UMAP coordinates
        x_col: X coordinate column
        y_col: Y coordinate column
        title: Plot title
    
    Returns:
        Plotly figure with density plot
    """
    try:
        fig = px.density_contour(
            df,
            x=x_col,
            y=y_col,
            title=title
        )
        
        fig.update_layout(
            title=dict(x=0.5, xanchor='center'),
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating density plot: {e}")
        raise


def create_enhanced_summary_plots(summary_stats: Dict[str, Any]) -> List[go.Figure]:
    """
    Create enhanced summary statistics plots with save functionality.
    
    Args:
        summary_stats: Summary statistics dictionary from compute_summary_statistics
    
    Returns:
        List of Plotly figures
    """
    figures = []
    
    try:
        # 1. Sequence Length Distribution
        length_stats = summary_stats.get('length_stats', {})
        if length_stats:
            fig1 = go.Figure()
            
            # Histogram
            fig1.add_trace(go.Histogram(
                x=length_stats['data'],
                nbinsx=50,
                name='Length Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Add vertical lines for statistics
            fig1.add_vline(x=length_stats['mean'], line_dash="dash", 
                          line_color="red", annotation_text=f"Mean: {length_stats['mean']:.0f}")
            fig1.add_vline(x=length_stats['median'], line_dash="dash", 
                          line_color="green", annotation_text=f"Median: {length_stats['median']:.0f}")
            
            fig1.update_layout(
                title="Protein Sequence Length Distribution",
                xaxis_title="Sequence Length (amino acids)",
                yaxis_title="Count",
                showlegend=False
            )
            
            figures.append(("length_distribution", fig1))
        
        # 2. Protein count per strain histogram (all strains)
        strain_stats = summary_stats.get('strain_stats', {})
        if strain_stats and len(strain_stats.get('data', [])) > 0:
            fig2 = go.Figure()

            fig2.add_trace(go.Histogram(
                x=strain_stats['data'],
                nbinsx=40,
                marker_color='lightsalmon',
                name='Strain Protein Counts',
                opacity=0.8
            ))

            fig2.update_layout(
                title="Protein per Strain Histogram",
                xaxis_title="Proteins per Strain",
                yaxis_title="Number of Strains",
                showlegend=False
            )

            figures.append(("protein_per_strain_histogram", fig2))

        # 3. Top strains by protein count
        if (
            strain_stats
            and len(strain_stats.get('labels', [])) > 0
            and len(strain_stats.get('data', [])) > 0
        ):
            fig3 = go.Figure()

            sorted_data = sorted(
                zip(strain_stats['labels'], strain_stats['data']),
                key=lambda x: x[1],
                reverse=True
            )
            labels, counts = zip(*sorted_data[:20])

            fig3.add_trace(go.Bar(
                x=list(labels),
                y=list(counts),
                marker_color='lightcoral',
                name='Protein Count'
            ))

            fig3.update_layout(
                title="Top 20 Strains by Protein Count",
                xaxis_title="Strain",
                yaxis_title="Number of Proteins",
                xaxis_tickangle=-45,
                showlegend=False
            )

            figures.append(("top_strains_by_protein_count", fig3))
        
        # 4. Gene Frequency Distribution
        gene_stats = summary_stats.get('gene_stats', {})
        if gene_stats and len(gene_stats.get('labels', [])) > 0 and len(gene_stats.get('data', [])) > 0:
            fig4 = go.Figure()
            
            # Sort genes by frequency
            sorted_genes = sorted(zip(gene_stats['labels'], gene_stats['data']), 
                                key=lambda x: x[1], reverse=True)
            labels, counts = zip(*sorted_genes[:25])  # Top 25
            
            fig4.add_trace(go.Bar(
                y=list(labels),
                x=list(counts),
                orientation='h',
                marker_color='lightgreen',
                name='Gene Frequency'
            ))
            
            fig4.update_layout(
                title="Most Frequent Genes (Top 25)",
                xaxis_title="Frequency (number of strains)",
                yaxis_title="Gene",
                height=600
            )
            
            figures.append(("gene_frequency", fig4))
        
        # 5. Core vs Accessory Genes
        if gene_stats:
            core_genes = gene_stats.get('core_genes', 0)
            accessory_genes = gene_stats.get('accessory_genes', 0)
            
            fig5 = go.Figure()
            
            fig5.add_trace(go.Pie(
                labels=['Core Genes (>95% strains)', 'Accessory Genes (<95% strains)'],
                values=[core_genes, accessory_genes],
                hole=0.3,
                marker_colors=['darkblue', 'lightsteelblue']
            ))
            
            fig5.update_layout(
                title="Core vs Accessory Gene Distribution",
                annotations=[dict(text=f'Total: {core_genes + accessory_genes}', 
                                x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            
            figures.append(("core_vs_accessory", fig5))
        
        # 6. Translation Success Overview
        trans_stats = summary_stats.get('translation_stats', {})
        if trans_stats:
            fig6 = go.Figure()
            
            categories = ['Present in CSV', 'Successfully Translated', 'Failed Translation']
            values = [
                trans_stats['total_present_in_csv'],
                trans_stats['total_translated'],
                trans_stats['total_present_in_csv'] - trans_stats['total_translated']
            ]
            
            fig6.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=['lightblue', 'lightgreen', 'lightcoral'],
                text=[f'{v:,}' for v in values],
                textposition='auto'
            ))
            
            fig6.update_layout(
                title=f"Translation Success Rate: {trans_stats['success_rate']:.1%}",
                xaxis_title="Processing Stage",
                yaxis_title="Number of Sequences"
            )
            
            figures.append(("translation_success", fig6))
        
        return figures
        
    except Exception as e:
        logger.error(f"Error creating enhanced summary plots: {e}")
        return []


def save_plots_to_files(figures: List[Tuple[str, go.Figure]], 
                       save_formats: List[str] = None,
                       output_dir: str = "plots") -> Dict[str, List[str]]:
    """
    Save plots to files in specified formats.
    
    Args:
        figures: List of (name, figure) tuples
        save_formats: List of formats to save ('png', 'pdf', 'svg', 'html')
        output_dir: Directory to save plots
    
    Returns:
        Dictionary mapping plot names to list of saved file paths
    """
    import os
    from pathlib import Path
    
    if save_formats is None:
        save_formats = ['png', 'html']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_files = {}
    
    for plot_name, fig in figures:
        plot_files = []
        
        for format_type in save_formats:
            filename = f"{plot_name}.{format_type}"
            filepath = output_path / filename
            
            try:
                if format_type == 'png':
                    fig.write_image(str(filepath), width=1200, height=800)
                elif format_type == 'pdf':
                    fig.write_image(str(filepath), width=1200, height=800)
                elif format_type == 'svg':
                    fig.write_image(str(filepath), width=1200, height=800)
                elif format_type == 'html':
                    fig.write_html(str(filepath))
                
                plot_files.append(str(filepath))
                
            except Exception as e:
                logger.error(f"Error saving {filename}: {e}")
        
        saved_files[plot_name] = plot_files
    
    return saved_files


def create_summary_statistics_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create summary statistics visualizations.
    
    Args:
        df: DataFrame with protein data
    
    Returns:
        Plotly subplot figure with multiple statistics
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sequence Length Distribution', 'Proteins per Strain',
                          'Protein per Strain Histogram', 'Top 20 Strains by Protein Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Length distribution histogram
        fig.add_trace(
            go.Histogram(x=df['Length'], nbinsx=30, name='Length Distribution'),
            row=1, col=1
        )
        
        # 2. Proteins per strain bar chart
        strain_counts = df['Strain_Name'].value_counts().head(20)  # Top 20 strains
        fig.add_trace(
            go.Bar(x=strain_counts.index, y=strain_counts.values, name='Strain Counts'),
            row=1, col=2
        )
        
        # 3. Protein-per-strain histogram (all strains)
        strain_distribution = df['Strain_Name'].value_counts().values
        fig.add_trace(
            go.Histogram(x=strain_distribution, nbinsx=40, name='Strain Distribution'),
            row=2, col=1
        )
        
        # 4. Top 20 strains by protein count (renamed)
        fig.add_trace(
            go.Bar(x=strain_counts.index, y=strain_counts.values, name='Top Strains'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Data Summary Statistics",
            title_x=0.5
        )
        
        # Update axes
        fig.update_xaxes(title_text="Sequence Length (AA)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Strain", row=1, col=2)
        fig.update_yaxes(title_text="Protein Count", row=1, col=2)
        fig.update_xaxes(title_text="Proteins per Strain", row=2, col=1)
        fig.update_yaxes(title_text="Number of Strains", row=2, col=1)
        fig.update_xaxes(title_text="Strain", row=2, col=2)
        fig.update_yaxes(title_text="Protein Count", row=2, col=2)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating summary statistics plot: {e}")
        raise


def create_cluster_overlay_plot(df: pd.DataFrame,
                               cluster_labels: np.ndarray,
                               x_col: str = "UMAP_1",
                               y_col: str = "UMAP_2",
                               original_color_by: str = DEFAULT_COLOR_BY) -> go.Figure:
    """
    Create a plot showing both original coloring and cluster overlay.
    
    Args:
        df: DataFrame with coordinates and metadata 
        cluster_labels: Cluster assignments for each point
        x_col: X coordinate column
        y_col: Y coordinate column  
        original_color_by: Original color column
    
    Returns:
        Plotly figure with cluster overlay
    """
    try:
        # Add cluster labels to dataframe copy
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = [f'Cluster {i}' for i in cluster_labels]
        
        # Create subplot with two traces
        fig = go.Figure()
        
        # Add original coloring
        for category in df[original_color_by].unique():
            mask = df[original_color_by] == category
            fig.add_trace(go.Scatter(
                x=df.loc[mask, x_col],
                y=df.loc[mask, y_col],
                mode='markers',
                name=f'{original_color_by}: {category}',
                marker=dict(size=MARKER_SIZE, opacity=0.6),
                visible=True
            ))
        
        # Add cluster overlay (initially hidden)
        for cluster_id in np.unique(cluster_labels):
            if cluster_id >= 0:  # Skip noise points (-1) for DBSCAN
                mask = cluster_labels == cluster_id
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, x_col],
                    y=df.loc[mask, y_col],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(size=MARKER_SIZE, opacity=0.8),
                    visible='legendonly'  # Hidden initially
                ))
        
        fig.update_layout(
            title="Protein Landscape: Original Categories vs Discovered Clusters",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating cluster overlay plot: {e}")
        raise


def create_gene_annotation_plot(df: pd.DataFrame,
                               x_col: str = "UMAP_1", 
                               y_col: str = "UMAP_2",
                               annotation_keywords: List[str] = None) -> go.Figure:
    """
    Create a plot highlighting specific gene annotations.
    
    Args:
        df: DataFrame with coordinates and annotations
        x_col: X coordinate column
        y_col: Y coordinate column
        annotation_keywords: Keywords to highlight in annotations
    
    Returns:
        Plotly figure highlighting specific annotations
    """
    try:
        if annotation_keywords is None:
            annotation_keywords = ['ribosomal', 'ATP', 'transport', 'DNA', 'RNA']
        
        # Create annotation categories
        df_annotated = df.copy()
        df_annotated['Annotation_Category'] = 'Other'
        
        for keyword in annotation_keywords:
            mask = df_annotated['Annotation'].str.contains(keyword, case=False, na=False)
            df_annotated.loc[mask, 'Annotation_Category'] = keyword.title()
        
        # Create plot
        fig = px.scatter(
            df_annotated,
            x=x_col,
            y=y_col,
            color='Annotation_Category',
            hover_data=['Gene_Name', 'Strain_Name', 'Annotation'],
            title="Functional Annotation Landscape"
        )
        
        fig.update_traces(marker=dict(size=MARKER_SIZE, opacity=MARKER_OPACITY))
        fig.update_layout(height=PLOT_HEIGHT, width=PLOT_WIDTH)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating gene annotation plot: {e}")
        raise


def display_data_table(df: pd.DataFrame, 
                      max_rows: int = 100,
                      columns_to_show: List[str] = None) -> None:
    """
    Display a formatted data table in Streamlit.
    
    Args:
        df: DataFrame to display
        max_rows: Maximum number of rows to show
        columns_to_show: Specific columns to display (None shows all)
    """
    try:
        if len(df) == 0:
            st.info("No proteins available to display")
            return

        browser_cols = st.columns([2, 1, 1])
        with browser_cols[0]:
            search_query = st.text_input(
                "Search proteins across the full dataset",
                placeholder="Search by locus tag, gene name, annotation, or strain",
                key="protein_browser_search"
            ).strip()
        with browser_cols[1]:
            page_size = st.selectbox(
                "Rows per page",
                options=[25, 50, 100, 200, 500],
                index=2,
                key="protein_browser_page_size"
            )
        with browser_cols[2]:
            search_mode = st.selectbox(
                "Search field",
                options=["All", "Gene_Name", "Locus_Tag", "Annotation", "Strain_Name", "Strain_Names"],
                index=0,
                key="protein_browser_search_mode"
            )

        if columns_to_show:
            available_cols = [col for col in columns_to_show if col in df.columns]
        else:
            available_cols = [col for col in df.columns if col not in ['Embeddings', 'Sequence_AA']]

        filtered_df = df
        if search_query:
            if search_mode == "All":
                search_columns = [
                    col for col in ['Gene_Name', 'Locus_Tag', 'Annotation', 'Strain_Name', 'Strain_Names']
                    if col in df.columns
                ]
            else:
                search_columns = [search_mode] if search_mode in df.columns else []

            if search_columns:
                mask = pd.Series(False, index=df.index)
                for col in search_columns:
                    mask = mask | df[col].fillna('').astype(str).str.contains(search_query, case=False, regex=False)
                filtered_df = df.loc[mask]

        total_rows = len(filtered_df)
        if total_rows == 0:
            st.warning("No proteins match the current search")
            return

        export_cols = st.columns(3)
        filtered_export_df = filtered_df.copy()
        filtered_export_table = filtered_export_df.drop(columns=['Embeddings'], errors='ignore')
        filtered_export_fasta = _format_fasta_records(filtered_export_df)
        with export_cols[0]:
            st.download_button(
                "Download Filtered Table (CSV)",
                data=filtered_export_table.to_csv(index=False),
                file_name="filtered_proteins.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_cols[1]:
            st.download_button(
                "Download Filtered Sequences (FASTA)",
                data=filtered_export_fasta,
                file_name="filtered_proteins.fasta",
                mime="text/plain",
                use_container_width=True,
            )
        with export_cols[2]:
            st.caption("Exports respect the current search filter across the full dataset.")

        total_pages = max(1, int(np.ceil(total_rows / page_size)))
        page_controls = st.columns([1, 2, 2])
        with page_controls[0]:
            page_number = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key="protein_browser_page"
            )
        with page_controls[1]:
            st.caption(f"Filtered proteins: {total_rows:,}")
        with page_controls[2]:
            start_row = (page_number - 1) * page_size + 1
            end_row = min(page_number * page_size, total_rows)
            st.caption(f"Showing rows {start_row:,} to {end_row:,} of {total_rows:,}")

        page_start = (page_number - 1) * page_size
        page_end = min(page_start + page_size, total_rows)
        page_df = filtered_df.iloc[page_start:page_end].copy()

        display_df = page_df[available_cols].copy()
        display_df.insert(0, 'Row', np.arange(page_start + 1, page_end + 1))

        selected_idx = None
        try:
            event = st.dataframe(
                display_df,
                use_container_width=True,
                height=420,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )
            if event.selection.rows:
                selected_idx = int(event.selection.rows[0])
        except TypeError:
            st.dataframe(display_df, use_container_width=True, height=420, hide_index=True)

        selector_col, _ = st.columns([3, 2])
        with selector_col:
            selected_label = st.selectbox(
                "Inspect protein",
                options=list(range(len(page_df))),
                format_func=lambda idx: _make_protein_label(page_df.iloc[idx]),
                index=selected_idx if selected_idx is not None and 0 <= selected_idx < len(page_df) else 0,
                key=f"protein_browser_select_{page_number}_{page_size}_{search_query}_{search_mode}"
            )

        selected_row = page_df.iloc[int(selected_label)]

        st.subheader("Protein Detail")
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.metric("Gene", str(selected_row.get('Gene_Name', 'Unknown')))
        with meta_cols[1]:
            st.metric("Locus Tag", str(selected_row.get('Locus_Tag', 'Unknown')))
        with meta_cols[2]:
            st.metric("Length", f"{int(selected_row.get('Length', 0))} AA")
        with meta_cols[3]:
            strain_count = int(selected_row.get('Strain_Count', 1)) if pd.notna(selected_row.get('Strain_Count', 1)) else 1
            st.metric("Strains", strain_count)

        st.write(f"**Annotation:** {selected_row.get('Annotation', 'Unknown')}")
        if 'Strain_Names' in selected_row and pd.notna(selected_row.get('Strain_Names')):
            with st.expander("Strains carrying this protein"):
                st.write(str(selected_row.get('Strain_Names')))

        sequence = str(selected_row.get('Sequence_AA', ''))
        st.text_area(
            "Protein Sequence",
            value=sequence,
            height=180,
            key=f"protein_sequence_{selected_row.get('Locus_Tag', 'unknown')}"
        )

        _render_copy_sequence_button(sequence, str(selected_row.get('Locus_Tag', 'unknown')))

        selected_fasta = _format_fasta_record(selected_row)
        action_cols = st.columns(4)
        with action_cols[0]:
            st.link_button("Open in BLASTP", _build_blastp_url(sequence), use_container_width=True)
        with action_cols[1]:
            st.link_button("Open InterPro Sequence Search", _build_interpro_url(), use_container_width=True)
        with action_cols[2]:
            st.download_button(
                "Download Selected FASTA",
                data=selected_fasta,
                file_name=f"{selected_row.get('Locus_Tag', 'protein')}.fasta",
                mime="text/plain",
                use_container_width=True,
            )
        with action_cols[3]:
            st.caption("InterPro does not expose a simple prefilled URL here; paste the sequence shown above into the opened page.")
            
    except Exception as e:
        logger.error(f"Error displaying data table: {e}")
        st.error(f"Error displaying table: {e}")


def create_validation_plot(validation_stats: Dict[str, Any]) -> go.Figure:
    """
    Create visualization for data validation statistics.
    
    Args:
        validation_stats: Dictionary with validation statistics
    
    Returns:
        Plotly figure showing validation metrics
    """
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Processing Success Rate', 'Length Distribution Stats',
                          'Coverage Statistics', 'Data Quality Metrics')
        )
        
        coverage = validation_stats.get('coverage', {})
        length_dist = validation_stats.get('length_distribution', {})
        
        # Processing success rate
        if coverage:
            categories = ['Present in CSV', 'Successfully Translated', 'Failed']
            values = [
                coverage.get('total_present_in_csv', 0),
                coverage.get('total_successfully_translated', 0), 
                coverage.get('total_present_in_csv', 0) - coverage.get('total_successfully_translated', 0)
            ]
            
            fig.add_trace(
                go.Bar(x=categories, y=values, name='Processing'),
                row=1, col=1
            )
        
        # Length distribution stats
        if length_dist:
            stats = ['Min', 'Mean', 'Max']
            values = [length_dist.get('min', 0), length_dist.get('mean', 0), length_dist.get('max', 0)]
            
            fig.add_trace(
                go.Bar(x=stats, y=values, name='Length Stats'),
                row=1, col=2
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Data Validation Summary"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating validation plot: {e}")
        raise