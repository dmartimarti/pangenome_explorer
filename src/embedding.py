"""ProtT5 embedding generation utilities."""

import torch
import numpy as np
import pandas as pd
import streamlit as st
from transformers import T5Tokenizer, T5EncoderModel
from typing import List, Tuple, Callable, Optional
import logging
import gc
from pathlib import Path

from config import (PROT_T5_MODEL_NAME, DEVICE, MODEL_PRECISION, BATCH_SIZE, 
                   MAX_TOKENIZER_LENGTH, ENABLE_MEMORY_CLEANUP)

logger = logging.getLogger(__name__)


def check_model_availability(model_name: str = PROT_T5_MODEL_NAME) -> Tuple[bool, str]:
    """
    Check if ProtT5 model is available locally or needs to be downloaded.
    
    Args:
        model_name: Name of the model to check
    
    Returns:
        Tuple of (is_available, model_path_or_message)
    """
    try:
        from transformers import AutoConfig
        
        # Try to load config without downloading
        config = AutoConfig.from_pretrained(
            model_name,
            local_files_only=True,
            _raise_exceptions_for_missing_entries=False
        )
        
        # If we get here, model is available locally
        cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
        model_path = cache_dir / f"models--{model_name.replace('/', '--')}"
        
        if model_path.exists():
            return True, str(model_path)
        else:
            return False, f"Model cache directory not found at {model_path}"
            
    except Exception as e:
        # Model not available locally
        return False, f"Model not available locally: {e}"


def estimate_model_download_size(model_name: str = PROT_T5_MODEL_NAME) -> str:
    """
    Estimate download size for the model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Estimated download size as string
    """
    # Known sizes for common models
    model_sizes = {
        "Rostlab/prot_t5_xl_uniref50": "~3.0 GB",
        "Rostlab/prot_t5_base_mt_uniref50": "~1.2 GB"
    }
    
    return model_sizes.get(model_name, "~3-5 GB (estimated)")


@st.cache_resource
def load_prot_t5_model(force_download: bool = False) -> Tuple[T5Tokenizer, T5EncoderModel]:
    """
    Load ProtT5 model and tokenizer with Apple Silicon optimization.
    
    Args:
        force_download: Force download even if model exists locally
    
    Returns:
        Tuple of (tokenizer, model)
    """
    try:
        logger.info(f"Loading ProtT5 model: {PROT_T5_MODEL_NAME} (force_download={force_download})")

        # Keep cached function UI-free and explicit about download behavior.
        if not force_download:
            is_available, model_info = check_model_availability()
            if not is_available:
                raise FileNotFoundError(
                    f"Model {PROT_T5_MODEL_NAME} not found locally. "
                    f"Run download flow first. Details: {model_info}"
                )
        
        # Check device availability
        if DEVICE == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
            logger.warning("MPS not available, falling back to CPU")
        else:
            device = DEVICE
            if device == "mps":
                logger.info("Using Apple Silicon MPS acceleration")
        
        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(
            PROT_T5_MODEL_NAME,
            do_lower_case=False,
            legacy=True,
            local_files_only=not force_download
        )
        
        # Load model
        model = T5EncoderModel.from_pretrained(
            PROT_T5_MODEL_NAME,
            local_files_only=not force_download
        )
        
        # Convert to specified precision for memory efficiency
        if MODEL_PRECISION == "float16":
            model = model.half()
            logger.info("Model converted to float16 precision")
        elif MODEL_PRECISION == "bfloat16":
            model = model.to(torch.bfloat16)
            logger.info("Model converted to bfloat16 precision")
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on device: {device}")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Error loading ProtT5 model: {e}")
        raise


def tokenize_protein_sequences(sequences: List[str], tokenizer: T5Tokenizer) -> dict:
    """
    Tokenize protein sequences with required space separation for ProtT5.
    
    Args:
        sequences: List of protein sequences (amino acid strings)
        tokenizer: ProtT5 tokenizer
    
    Returns:
        Tokenized inputs dictionary
    """
    try:
        # Convert sequences to space-separated format required by ProtT5
        spaced_sequences = []
        
        for seq in sequences:
            # Remove any existing stop codons and ensure uppercase
            cleaned_seq = seq.replace('*', '').upper()
            
            # Add spaces between amino acids
            spaced_seq = ' '.join(list(cleaned_seq))
            spaced_sequences.append(spaced_seq)
        
        # Tokenize with padding and truncation
        encoding = tokenizer(
            spaced_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENIZER_LENGTH
        )
        
        return encoding
        
    except Exception as e:
        logger.error(f"Error tokenizing sequences: {e}")
        raise


def compute_mean_pooled_embeddings(input_ids: torch.Tensor, 
                                 attention_mask: torch.Tensor, 
                                 model: T5EncoderModel) -> np.ndarray:
    """
    Compute mean-pooled embeddings from ProtT5 model outputs.
    
    Args:
        input_ids: Tokenized input IDs
        attention_mask: Attention mask for padding tokens
        model: ProtT5 model
    
    Returns:
        Mean-pooled embeddings as numpy array (batch_size, 1024)
    """
    try:
        device = next(model.parameters()).device
        device_type = str(device)
        
        # Move inputs to model device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract last hidden states: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        # Perform mean pooling with attention mask
        # Mask out padding tokens
        masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
        
        # Sum over sequence dimension
        sum_embeddings = masked_embeddings.sum(dim=1)
        
        # Divide by sequence lengths (excluding padding)
        sequence_lengths = attention_mask.sum(dim=1).unsqueeze(-1)
        mean_embeddings = sum_embeddings / sequence_lengths
        
        # Move to CPU and convert to numpy
        embeddings_np = mean_embeddings.detach().to('cpu').numpy()
        
        # Memory cleanup if enabled
        if ENABLE_MEMORY_CLEANUP:
            del input_ids, attention_mask, outputs, last_hidden_state
            del masked_embeddings, sum_embeddings, sequence_lengths, mean_embeddings
            gc.collect()
            if device_type == "mps":
                torch.mps.empty_cache()
            elif device_type.startswith("cuda"):
                torch.cuda.empty_cache()
        
        return embeddings_np
        
    except Exception as e:
        logger.error(f"Error computing embeddings: {e}")
        raise


@st.cache_data
def generate_protein_embeddings(sequences: List[str], 
                              _tokenizer: T5Tokenizer, 
                              _model: T5EncoderModel,
                              batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Generate ProtT5 embeddings for a list of protein sequences.
    
    Args:
        sequences: List of protein sequences
        _tokenizer: ProtT5 tokenizer (prefixed with _ to avoid caching)
        _model: ProtT5 model (prefixed with _ to avoid caching)
        batch_size: Number of sequences to process per batch
    
    Returns:
        Embeddings array of shape (n_sequences, 1024)
    """
    return generate_protein_embeddings_with_progress(
        sequences=sequences,
        _tokenizer=_tokenizer,
        _model=_model,
        batch_size=batch_size,
        progress_callback=None
    )


def generate_protein_embeddings_with_progress(
    sequences: List[str],
    _tokenizer: T5Tokenizer,
    _model: T5EncoderModel,
    batch_size: int = BATCH_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> np.ndarray:
    """
    Generate ProtT5 embeddings with optional per-batch progress callback.

    Args:
        sequences: List of protein sequences
        _tokenizer: ProtT5 tokenizer
        _model: ProtT5 model
        batch_size: Number of sequences to process per batch
        progress_callback: Optional callback(batch_idx, num_batches)

    Returns:
        Embeddings array of shape (n_sequences, 1024)
    """
    try:
        seq_count = len(sequences)
        num_batches = (seq_count + batch_size - 1) // batch_size
        logger.info(
            f"Generating embeddings for {seq_count:,} sequences in {num_batches:,} batches of {batch_size}"
        )
        
        embeddings = None
        
        for i in range(0, len(sequences), batch_size):
            batch_idx = i // batch_size + 1
            logger.info(f"Processing batch {batch_idx}/{num_batches}")
            
            # Get batch sequences
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize batch
            encoding = tokenize_protein_sequences(batch_sequences, _tokenizer)
            
            # Compute embeddings for batch
            batch_embeddings = compute_mean_pooled_embeddings(
                encoding['input_ids'], 
                encoding['attention_mask'], 
                _model
            )

            if embeddings is None:
                embeddings = np.empty((seq_count, batch_embeddings.shape[1]), dtype=batch_embeddings.dtype)

            start_idx = i
            end_idx = i + len(batch_embeddings)
            embeddings[start_idx:end_idx] = batch_embeddings

            del encoding, batch_embeddings
            if ENABLE_MEMORY_CLEANUP:
                gc.collect()

            if progress_callback is not None:
                progress_callback(batch_idx, num_batches)

        if embeddings is None:
            embeddings = np.empty((0, 1024), dtype=np.float32)
        
        logger.info(f"Successfully generated embeddings: shape {embeddings.shape}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def _clean_device_cache(model: T5EncoderModel) -> None:
    """Release accelerator cache to reduce peak memory between chunks."""
    if not ENABLE_MEMORY_CLEANUP:
        return

    device_type = str(next(model.parameters()).device)
    gc.collect()
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type.startswith("cuda"):
        torch.cuda.empty_cache()


def generate_protein_embeddings_chunked_with_progress(
    sequences: List[str],
    _tokenizer: T5Tokenizer,
    _model: T5EncoderModel,
    batch_size: int = BATCH_SIZE,
    chunk_size: int = 5000,
    progress_callback: Optional[Callable[[int, int, int, int], None]] = None
) -> np.ndarray:
    """
    Generate embeddings by processing sequence chunks sequentially.

    This keeps memory stable for large datasets while preserving throughput.

    Args:
        sequences: Protein sequences to embed
        _tokenizer: ProtT5 tokenizer
        _model: ProtT5 model
        batch_size: Per-forward-pass batch size
        chunk_size: Number of sequences per processing chunk
        progress_callback: Optional callback(chunk_idx, total_chunks, batch_idx, total_batches)

    Returns:
        Embeddings array of shape (n_sequences, 1024)
    """
    try:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        seq_count = len(sequences)
        if seq_count == 0:
            return np.empty((0, 1024), dtype=np.float32)

        total_chunks = (seq_count + chunk_size - 1) // chunk_size
        logger.info(
            "Chunked embedding run: %d sequences, chunk_size=%d, total_chunks=%d",
            seq_count,
            chunk_size,
            total_chunks,
        )

        embeddings = None
        write_start = 0

        for chunk_start in range(0, seq_count, chunk_size):
            chunk_idx = chunk_start // chunk_size + 1
            chunk_sequences = sequences[chunk_start:chunk_start + chunk_size]
            chunk_len = len(chunk_sequences)
            chunk_batches = (chunk_len + batch_size - 1) // batch_size

            logger.info(
                "Processing chunk %d/%d (%d sequences, %d batches)",
                chunk_idx,
                total_chunks,
                chunk_len,
                chunk_batches,
            )

            def _chunk_batch_progress(batch_idx: int, num_batches: int) -> None:
                if progress_callback is not None:
                    progress_callback(chunk_idx, total_chunks, batch_idx, num_batches)

            chunk_embeddings = generate_protein_embeddings_with_progress(
                sequences=chunk_sequences,
                _tokenizer=_tokenizer,
                _model=_model,
                batch_size=batch_size,
                progress_callback=_chunk_batch_progress,
            )

            if embeddings is None:
                embeddings = np.empty(
                    (seq_count, chunk_embeddings.shape[1]),
                    dtype=chunk_embeddings.dtype,
                )

            write_end = write_start + len(chunk_embeddings)
            embeddings[write_start:write_end] = chunk_embeddings
            write_start = write_end

            del chunk_embeddings
            _clean_device_cache(_model)

        if embeddings is None:
            embeddings = np.empty((0, 1024), dtype=np.float32)

        return embeddings

    except Exception as e:
        logger.error(f"Error generating chunked embeddings: {e}")
        raise


def validate_embeddings(embeddings: np.ndarray, expected_dim: int = 1024) -> dict:
    """
    Validate the generated embeddings.
    
    Args:
        embeddings: Generated embeddings array
        expected_dim: Expected embedding dimension
    
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'shape': embeddings.shape,
        'expected_dimension': expected_dim,
        'dimension_correct': embeddings.shape[1] == expected_dim,
        'has_nan': np.any(np.isnan(embeddings)),
        'has_inf': np.any(np.isinf(embeddings)),
        'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
        'min_value': float(np.min(embeddings)),
        'max_value': float(np.max(embeddings))
    }
    
    return stats


def add_embeddings_to_dataframe(df: pd.DataFrame, 
                               embeddings: np.ndarray) -> pd.DataFrame:
    """
    Add embedding vectors to the dataframe.
    
    Args:
        df: DataFrame with sequence data
        embeddings: Generated embeddings array
    
    Returns:
        DataFrame with added embedding column
    """
    try:
        if len(df) != len(embeddings):
            raise ValueError(f"Dataframe length {len(df)} doesn't match embeddings length {len(embeddings)}")
        
        # Add embeddings as a new column
        df_with_embeddings = df.copy()
        df_with_embeddings['Embeddings'] = [emb for emb in embeddings]
        
        return df_with_embeddings
        
    except Exception as e:
        logger.error(f"Error adding embeddings to dataframe: {e}")
        raise