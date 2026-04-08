"""Data loading and processing utilities for pangenome analysis."""

import pandas as pd
import numpy as np
import streamlit as st
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from typing import Dict, List, Tuple, Any, Optional, FrozenSet
import io
import logging
import re
import os
from concurrent.futures import ThreadPoolExecutor

from config import MAX_SEQUENCE_LENGTH, PROTEIN_TRANSLATION_TABLE, REMOVE_STOP_CODONS

logger = logging.getLogger(__name__)


def _translate_record_task(task: Tuple[str, str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """Translate one FASTA record and return one unique protein row with strain metadata."""
    locus_tag, nucleotide_seq_text, metadata_list = task

    best_protein, best_frame = find_best_reading_frame(Seq(nucleotide_seq_text))

    if REMOVE_STOP_CODONS:
        best_protein = best_protein.rstrip('*')

    protein_length = len(best_protein)
    if protein_length == 0 or protein_length > MAX_SEQUENCE_LENGTH:
        return {}

    strain_names = sorted({str(m['Strain_Name']) for m in metadata_list if m.get('Strain_Name')})
    first_metadata = metadata_list[0]

    return {
        'Locus_Tag': locus_tag,
        'Sequence_AA': best_protein,
        'Length': protein_length,
        'Gene_Name': first_metadata['Gene_Name'],
        'Strain_Name': strain_names[0] if strain_names else 'Unknown',
        'Strain_Names': ';'.join(strain_names),
        'Strain_Count': len(strain_names),
        'Annotation': first_metadata['Annotation'],
        'Reading_Frame': best_frame
    }


def _normalize_identifier(value: Any) -> str:
    """Normalize identifiers from CSV/FASTA for robust matching."""
    if value is None:
        return ""
    text = str(value).strip().strip('"').strip("'")
    # FASTA ids are often first token; keep behavior consistent.
    text = text.split()[0] if text else ""
    return text


def detect_fasta_id_mode(fasta_contents: bytes, sample_size: int = 100) -> str:
    """
    Detect whether FASTA record IDs are Panaroo group IDs or locus tags.

    Args:
        fasta_contents: Raw FASTA file contents as bytes
        sample_size: Number of headers to inspect

    Returns:
        "group" if most IDs look like group_*; otherwise "locus"
    """
    try:
        fasta_text = fasta_contents.decode('utf-8', errors='ignore')
        group_like = 0
        total = 0

        for record in SeqIO.parse(io.StringIO(fasta_text), "fasta"):
            rid = _normalize_identifier(record.id).lower()
            if rid.startswith("group_"):
                group_like += 1
            total += 1
            if total >= sample_size:
                break

        if total == 0:
            return "locus"

        return "group" if (group_like / total) >= 0.8 else "locus"
    except Exception:
        return "locus"


def extract_fasta_identifiers(fasta_contents: bytes) -> FrozenSet[str]:
    """Extract normalized FASTA IDs/names for targeted CSV mapping."""
    identifiers = set()
    try:
        fasta_text = fasta_contents.decode('utf-8', errors='ignore')
        for record in SeqIO.parse(io.StringIO(fasta_text), "fasta"):
            rid = _normalize_identifier(record.id)
            rname = _normalize_identifier(record.name)
            if rid:
                identifiers.add(rid)
            if rname:
                identifiers.add(rname)
    except Exception as e:
        logger.warning(f"Could not pre-extract FASTA identifiers: {e}")
    return frozenset(identifiers)


@st.cache_data
def load_panaroo_csv(
    file_contents: bytes,
    include_locus_tag_mapping: bool = True,
    required_identifiers: Optional[FrozenSet[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, str]]], List[str]]:
    """
    Load and parse Panaroo gene_presence_absence.csv file.
    Memory-optimized version for large datasets.
    
    Args:
        file_contents: Raw CSV file contents as bytes
    
    Returns:
        Tuple containing:
        - DataFrame with gene presence/absence data
        - Dictionary mapping locus_tag -> list of {strain, gene_name, annotation}
        - List of strain column names
    """
    try:
        # Use efficient dtypes and chunking for large files
        with st.spinner("Loading Panaroo CSV (optimized for memory)..."):
            df = pd.read_csv(
                io.BytesIO(file_contents),
                dtype='string',
                engine='c',
                low_memory=False
            )
        
        # Validate required columns
        required_cols = ['Gene', 'Annotation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Find the start of strain columns (after 'Colour' or 'Annotation')
        if 'Colour' in df.columns:
            start_idx = df.columns.get_loc('Colour') + 1
        else:
            start_idx = df.columns.get_loc('Annotation') + 1
            
        strain_columns = df.columns[start_idx:].tolist()
        
        # Create locus_tag/group_id -> metadata mapping
        # Map key -> list of (strain, gene_name, annotation) entries.
        locus_mapping = {}
        identifier_filtering = bool(required_identifiers)

        gene_idx = df.columns.get_loc('Gene')
        ann_idx = df.columns.get_loc('Annotation')
        strain_indices = [df.columns.get_loc(c) for c in strain_columns]

        with st.spinner("Creating locus tag mappings..."):
            for row in df.itertuples(index=False, name=None):
                gene_name = _normalize_identifier(row[gene_idx])
                annotation = row[ann_idx]
                if pd.isna(annotation):
                    annotation = 'Unknown'
                annotation = str(annotation)

                if not gene_name:
                    continue

                group_entries: List[Dict[str, str]] = []

                for strain, strain_idx in zip(strain_columns, strain_indices):
                    locus_tag = row[strain_idx]

                    # Skip empty/NaN entries (gene absent in strain)
                    if pd.isna(locus_tag) or str(locus_tag).strip() == '':
                        continue

                    entry = {
                        'Strain_Name': strain,
                        'Gene_Name': gene_name,
                        'Annotation': annotation
                    }
                    group_entries.append(entry)

                    if include_locus_tag_mapping:
                        # Handle multiple locus tags (semicolon/comma separated)
                        tags = [_normalize_identifier(t) for t in re.split(r'[;,]', str(locus_tag))]
                        for tag in tags:
                            if tag and (not identifier_filtering or tag in required_identifiers):
                                if tag not in locus_mapping:
                                    locus_mapping[tag] = []
                                locus_mapping[tag].append(entry)

                # Always map group IDs for group-based FASTA headers.
                if group_entries and (not identifier_filtering or gene_name in required_identifiers):
                    locus_mapping[gene_name] = group_entries
                

        
        logger.info(f"Loaded {len(df)} genes across {len(strain_columns)} strains")
        logger.info(
            "Created mapping for %d keys (locus tags %s, identifier filtering %s)",
            len(locus_mapping),
            "enabled" if include_locus_tag_mapping else "disabled",
            "enabled" if identifier_filtering else "disabled"
        )
        
        return df, locus_mapping, strain_columns
        
    except Exception as e:
        logger.error(f"Error loading Panaroo CSV: {e}")
        raise


@st.cache_data
def parse_fasta_and_translate(fasta_contents: bytes, 
                              locus_mapping: Dict[str, List[Dict[str, str]]]) -> pd.DataFrame:
    """
    Parse FASTA file and translate nucleotide sequences to amino acids.
    Maps each group ID to all strains where it exists.
    
    Args:
        fasta_contents: Raw FASTA file contents as bytes
        locus_mapping: Dictionary mapping locus_tag -> list of {strain_name, gene_name, annotation}
    
    Returns:
        DataFrame with columns: Locus_Tag, Sequence_AA, Length, Gene_Name, Strain_Name, Annotation
    """
    try:
        sequences_data: List[Dict[str, Any]] = []
        translation_tasks: List[Tuple[str, str, List[Dict[str, str]]]] = []
        total_records = 0
        matched_records = 0
        translated_instances = 0
        
        # Convert bytes to text for SeqIO
        fasta_text = fasta_contents.decode('utf-8')
        
        for record in SeqIO.parse(io.StringIO(fasta_text), "fasta"):
            total_records += 1
            locus_tag = _normalize_identifier(record.id)
            alt_tag = _normalize_identifier(record.name)
            
            # Get all metadata entries for this locus tag/group ID
            metadata_list = []
            if locus_tag in locus_mapping:
                metadata_list = locus_mapping[locus_tag]
            elif alt_tag in locus_mapping:
                metadata_list = locus_mapping[alt_tag]
                locus_tag = alt_tag
            
            # Skip if no matches found
            if not metadata_list:
                continue

            translation_tasks.append((locus_tag, str(record.seq), metadata_list))

        # Parallel translation path for large inputs.
        cpu_count = os.cpu_count() or 1
        max_workers = min(8, cpu_count)
        use_parallel = max_workers > 1 and len(translation_tasks) >= 500

        if use_parallel:
            logger.info(
                "Running parallel translation with %d workers for %d FASTA records",
                max_workers,
                len(translation_tasks)
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for translated_row in executor.map(_translate_record_task, translation_tasks):
                    if translated_row:
                        matched_records += 1
                        translated_instances += int(translated_row.get('Strain_Count', 1))
                        sequences_data.append(translated_row)
        else:
            logger.info("Running serial translation for %d FASTA records", len(translation_tasks))
            for task in translation_tasks:
                translated_row = _translate_record_task(task)
                if translated_row:
                    matched_records += 1
                    translated_instances += int(translated_row.get('Strain_Count', 1))
                    sequences_data.append(translated_row)
        
        df_sequences = pd.DataFrame(
            sequences_data,
            columns=['Locus_Tag', 'Sequence_AA', 'Length', 'Gene_Name', 'Strain_Name', 'Strain_Names', 'Strain_Count', 'Annotation', 'Reading_Frame']
        )
        
        logger.info(
            "FASTA records: %d; translated unique proteins: %d; represented strain instances: %d",
            total_records,
            matched_records,
            translated_instances
        )
        logger.info(f"Successfully translated {len(df_sequences)} unique protein sequences")
        
        # Only log length distribution if we have sequences
        if len(df_sequences) > 0:
            logger.info(f"Length distribution: min={df_sequences['Length'].min()}, "
                       f"max={df_sequences['Length'].max()}, "
                       f"mean={df_sequences['Length'].mean():.1f}")
        else:
            logger.warning("No sequences were successfully translated. Check:")
            logger.warning("  - FASTA file format (nucleotide sequences expected)")
            logger.warning("  - Locus tag matching between CSV and FASTA")
            logger.warning("  - Sequence length limits (current max: {} AA)".format(MAX_SEQUENCE_LENGTH))
        
        return df_sequences
        
    except Exception as e:
        logger.error(f"Error parsing FASTA: {e}")
        raise


def find_best_reading_frame(nucleotide_seq: Seq) -> Tuple[str, int]:
    """
    Find the reading frame that produces the longest protein without internal stop codons.
    
    Args:
        nucleotide_seq: Nucleotide sequence
    
    Returns:
        Tuple of (best_protein_sequence, reading_frame)
    """
    best_protein = ""
    best_frame = 0
    
    for frame in range(3):
        # Get sequence in this reading frame
        frame_seq = nucleotide_seq[frame:]
        
        # Translate
        try:
            protein = frame_seq.translate(table=PROTEIN_TRANSLATION_TABLE, to_stop=False)
            protein_str = str(protein)
            
            # Check for internal stop codons (ignore terminal ones)
            protein_no_terminal = protein_str.rstrip('*')
            internal_stops = protein_no_terminal.count('*')
            
            # Prefer longer sequences with fewer internal stops
            score = len(protein_no_terminal) - (internal_stops * 100)  # Heavily penalize internal stops
            best_score = len(best_protein.rstrip('*')) - (best_protein.rstrip('*').count('*') * 100)
            
            if score > best_score:
                best_protein = protein_str
                best_frame = frame
                
        except Exception as e:
            logger.warning(f"Translation error in frame {frame}: {e}")
            continue
    
    return best_protein, best_frame


def compute_summary_statistics(df_sequences: pd.DataFrame, 
                              panaroo_df: pd.DataFrame,
                              strain_columns: List[str]) -> Dict[str, Any]:
    """
    Compute comprehensive summary statistics for the dataset.
    
    Args:
        df_sequences: DataFrame with translated sequences
        panaroo_df: Original Panaroo DataFrame
        strain_columns: List of strain column names
    
    Returns:
        Dictionary with summary statistics for visualization
    """
    stats = {}
    
    # Handle empty DataFrame case
    if len(df_sequences) == 0:
        logger.warning("No sequences available for summary statistics")
        return {
            'length_stats': {'data': [], 'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0, 'q25': 0, 'q75': 0},
            'strain_stats': {'counts': {}, 'data': [], 'labels': [], 'total_strains': 0, 'mean_proteins_per_strain': 0, 'std_proteins_per_strain': 0},
            'gene_stats': {'counts': {}, 'data': [], 'labels': [], 'total_unique_genes': 0, 'mean_gene_frequency': 0, 'core_genes': 0, 'accessory_genes': 0},
            'translation_stats': {'total_genes_in_csv': len(panaroo_df), 'total_possible_combinations': 0, 'total_present_in_csv': 0, 'total_translated': 0, 'total_unique_translated': 0, 'success_rate': 0.0, 'coverage_rate': 0.0}
        }
    
    # Sequence length statistics
    lengths = df_sequences['Length'].values
    stats['length_stats'] = {
        'data': lengths,
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'std': float(np.std(lengths)),
        'q25': float(np.percentile(lengths, 25)),
        'q75': float(np.percentile(lengths, 75))
    }
    
    # Proteins per strain (supports compact unique-protein representation)
    if 'Strain_Names' in df_sequences.columns and len(df_sequences) > 0:
        strain_counter: Dict[str, int] = {}
        for strain_names in df_sequences['Strain_Names'].fillna(''):
            if not strain_names:
                continue
            for s in str(strain_names).split(';'):
                if s:
                    strain_counter[s] = strain_counter.get(s, 0) + 1
        strain_counts = pd.Series(strain_counter).sort_values(ascending=False)
    else:
        strain_counts = df_sequences['Strain_Name'].value_counts()
    stats['strain_stats'] = {
        'counts': strain_counts.to_dict(),
        'data': strain_counts.values,
        'labels': strain_counts.index.tolist(),
        'total_strains': len(strain_counts),
        'mean_proteins_per_strain': float(strain_counts.mean()),
        'std_proteins_per_strain': float(strain_counts.std())
    }
    
    # Gene frequency across strains. In compact mode this must use Strain_Count,
    # not row counts, otherwise every gene appears once and core gene detection fails.
    if 'Strain_Count' in df_sequences.columns:
        gene_counts = df_sequences.groupby('Gene_Name')['Strain_Count'].max().sort_values(ascending=False)
    else:
        gene_counts = df_sequences['Gene_Name'].value_counts()
    stats['gene_stats'] = {
        'counts': gene_counts.to_dict(),
        'data': gene_counts.values,
        'labels': gene_counts.index.tolist(),
        'total_unique_genes': len(gene_counts),
        'mean_gene_frequency': float(gene_counts.mean()),
        'core_genes': len(gene_counts[gene_counts >= len(strain_columns) * 0.95]),  # Present in >95% strains
        'accessory_genes': len(gene_counts[gene_counts < len(strain_columns) * 0.95])
    }
    
    # Reading frame distribution (if available)
    if 'Reading_Frame' in df_sequences.columns:
        frame_counts = df_sequences['Reading_Frame'].value_counts()
        stats['frame_stats'] = {
            'counts': frame_counts.to_dict(),
            'data': frame_counts.values,
            'labels': frame_counts.index.tolist()
        }
    
    # Translation success rates
    total_genes_in_csv = len(panaroo_df)
    total_possible = total_genes_in_csv * len(strain_columns)
    total_present = panaroo_df[strain_columns].notna().sum().sum()
    total_translated = int(df_sequences['Strain_Count'].sum()) if 'Strain_Count' in df_sequences.columns else len(df_sequences)
    total_unique_translated = len(df_sequences)
    
    stats['translation_stats'] = {
        'total_genes_in_csv': total_genes_in_csv,
        'total_possible_combinations': total_possible,
        'total_present_in_csv': int(total_present),
        'total_translated': total_translated,
        'total_unique_translated': total_unique_translated,
        'success_rate': float(total_translated / total_present) if total_present > 0 else 0.0,
        'coverage_rate': float(total_present / total_possible) if total_possible > 0 else 0.0
    }
    
    return stats


def validate_data_consistency(df_sequences: pd.DataFrame, 
                             panaroo_df: pd.DataFrame, 
                             strain_columns: List[str]) -> Dict[str, Any]:
    """
    Validate consistency between parsed sequences and original Panaroo data.
    
    Args:
        df_sequences: DataFrame with translated sequences
        panaroo_df: Original Panaroo DataFrame
        strain_columns: List of strain column names
    
    Returns:
        Dictionary with validation statistics
    """
    if len(df_sequences) == 0:
        total_possible_genes = len(panaroo_df) * len(strain_columns)
        total_present_genes = panaroo_df[strain_columns].notna().sum().sum() if len(strain_columns) > 0 else 0
        return {
            'sequences_per_strain': {},
            'gene_frequency': {},
            'length_distribution': {'min': 0, 'max': 0, 'mean': 0.0, 'std': 0.0},
            'coverage': {
                'total_genes_in_csv': len(panaroo_df),
                'total_strains': len(strain_columns),
                'total_possible_combinations': total_possible_genes,
                'total_present_in_csv': int(total_present_genes),
                'total_successfully_translated': 0,
                'translation_success_rate': 0.0
            }
        }

    stats = {}
    
    # Count sequences per strain (supports compact unique-protein representation)
    if 'Strain_Names' in df_sequences.columns and len(df_sequences) > 0:
        strain_counter: Dict[str, int] = {}
        for strain_names in df_sequences['Strain_Names'].fillna(''):
            if not strain_names:
                continue
            for s in str(strain_names).split(';'):
                if s:
                    strain_counter[s] = strain_counter.get(s, 0) + 1
        stats['sequences_per_strain'] = strain_counter
    else:
        strain_counts = df_sequences['Strain_Name'].value_counts()
        stats['sequences_per_strain'] = strain_counts.to_dict()
    
    # Count gene frequency across strains
    if 'Strain_Count' in df_sequences.columns:
        gene_counts = df_sequences.groupby('Gene_Name')['Strain_Count'].max().sort_values(ascending=False)
    else:
        gene_counts = df_sequences['Gene_Name'].value_counts()
    stats['gene_frequency'] = gene_counts.to_dict()
    
    # Length distribution
    stats['length_distribution'] = {
        'min': int(df_sequences['Length'].min()),
        'max': int(df_sequences['Length'].max()),
        'mean': float(df_sequences['Length'].mean()),
        'std': float(df_sequences['Length'].std())
    }
    
    # Coverage statistics
    total_possible_genes = len(panaroo_df) * len(strain_columns)
    total_present_genes = panaroo_df[strain_columns].notna().sum().sum()
    total_translated = int(df_sequences['Strain_Count'].sum()) if 'Strain_Count' in df_sequences.columns else len(df_sequences)
    
    stats['coverage'] = {
        'total_genes_in_csv': len(panaroo_df),
        'total_strains': len(strain_columns),
        'total_possible_combinations': total_possible_genes,
        'total_present_in_csv': int(total_present_genes),
        'total_successfully_translated': total_translated,
        'translation_success_rate': float(total_translated / total_present_genes) if total_present_genes > 0 else 0.0
    }
    
    return stats