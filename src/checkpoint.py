"""Checkpoint save/load utilities for pipeline resumption."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

CHECKPOINT_VERSION = 1


class CheckpointError(Exception):
    """Raised when checkpoint data is invalid or incompatible."""


def compute_sha256(content: bytes) -> str:
    """Compute SHA256 digest for uploaded source files."""
    return hashlib.sha256(content).hexdigest()


def build_source_metadata(
    panaroo_name: str,
    panaroo_content: bytes,
    fasta_name: str,
    fasta_content: bytes,
    fasta_mode: str,
) -> Dict[str, Any]:
    """Create provenance metadata for source inputs used in the run."""
    return {
        "panaroo_filename": panaroo_name,
        "panaroo_sha256": compute_sha256(panaroo_content),
        "panaroo_size_bytes": len(panaroo_content),
        "fasta_filename": fasta_name,
        "fasta_sha256": compute_sha256(fasta_content),
        "fasta_size_bytes": len(fasta_content),
        "fasta_id_mode": fasta_mode,
    }


def _json_default(value: Any) -> Any:
    """Serialize pandas/numpy values safely to JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _to_json_bytes(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, indent=2, default=_json_default).encode("utf-8")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_master_df_types(df: pd.DataFrame) -> pd.DataFrame:
    """Restore expected dataframe dtypes after CSV roundtrip."""
    typed = df.copy()

    for col in ["Length", "Strain_Count", "Reading_Frame"]:
        if col in typed.columns:
            typed[col] = pd.to_numeric(typed[col], errors="coerce").fillna(0).astype(int)

    for col in ["Locus_Tag", "Sequence_AA", "Gene_Name", "Strain_Name", "Strain_Names", "Annotation"]:
        if col in typed.columns:
            typed[col] = typed[col].astype(str)

    return typed


def create_phase1_checkpoint_bytes(
    master_df: pd.DataFrame,
    summary_stats: Dict[str, Any],
    validation_stats: Dict[str, Any],
    strain_columns: List[str],
    source_metadata: Dict[str, Any],
    processing_metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Serialize post-translation state (Phase 1) into a checkpoint zip."""
    processing_metadata = processing_metadata or {}

    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "phase": "phase1",
        "created_at_utc": _now_utc_iso(),
        "source": source_metadata,
        "processing": processing_metadata,
        "counts": {
            "rows": int(len(master_df)),
            "unique_sequences": int(master_df["Sequence_AA"].nunique()) if "Sequence_AA" in master_df.columns else 0,
            "unique_genes": int(master_df["Gene_Name"].nunique()) if "Gene_Name" in master_df.columns else 0,
            "strains": int(len(strain_columns)),
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", _to_json_bytes(manifest))
        zf.writestr("phase1_master_df.csv", master_df.to_csv(index=False).encode("utf-8"))
        zf.writestr("phase1_summary_stats.json", _to_json_bytes(summary_stats))
        zf.writestr("phase1_validation_stats.json", _to_json_bytes(validation_stats))
        zf.writestr("phase1_strain_columns.json", _to_json_bytes({"strain_columns": strain_columns}))

    return buf.getvalue()


def create_phase2_checkpoint_bytes(
    master_df: pd.DataFrame,
    summary_stats: Dict[str, Any],
    validation_stats: Dict[str, Any],
    strain_columns: List[str],
    source_metadata: Dict[str, Any],
    processing_metadata: Optional[Dict[str, Any]] = None,
    embedding_metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Serialize post-embedding state (Phase 2) into a checkpoint zip."""
    if "Embeddings" not in master_df.columns:
        raise CheckpointError("Phase 2 checkpoint requires an 'Embeddings' column")

    processing_metadata = processing_metadata or {}
    embedding_metadata = embedding_metadata or {}

    embeddings = np.vstack(master_df["Embeddings"].values)
    df_no_embeddings = master_df.drop(columns=["Embeddings"]).copy()

    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "phase": "phase2",
        "created_at_utc": _now_utc_iso(),
        "source": source_metadata,
        "processing": processing_metadata,
        "embedding": {
            **embedding_metadata,
            "embedding_dim": int(embeddings.shape[1]),
        },
        "counts": {
            "rows": int(len(master_df)),
            "unique_sequences": int(master_df["Sequence_AA"].nunique()) if "Sequence_AA" in master_df.columns else 0,
            "unique_genes": int(master_df["Gene_Name"].nunique()) if "Gene_Name" in master_df.columns else 0,
            "strains": int(len(strain_columns)),
        },
    }

    emb_buf = io.BytesIO()
    np.savez_compressed(emb_buf, embeddings=embeddings)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", _to_json_bytes(manifest))
        zf.writestr("phase2_master_df.csv", df_no_embeddings.to_csv(index=False).encode("utf-8"))
        zf.writestr("phase2_embeddings.npz", emb_buf.getvalue())
        zf.writestr("phase1_summary_stats.json", _to_json_bytes(summary_stats))
        zf.writestr("phase1_validation_stats.json", _to_json_bytes(validation_stats))
        zf.writestr("phase1_strain_columns.json", _to_json_bytes({"strain_columns": strain_columns}))

    return buf.getvalue()


def load_checkpoint_bytes(checkpoint_bytes: bytes) -> Dict[str, Any]:
    """Load Phase 1 or Phase 2 checkpoint and return restorable session payload."""
    try:
        with zipfile.ZipFile(io.BytesIO(checkpoint_bytes), mode="r") as zf:
            names = set(zf.namelist())
            if "manifest.json" not in names:
                raise CheckpointError("Checkpoint missing manifest.json")

            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            version = int(manifest.get("checkpoint_version", -1))
            if version != CHECKPOINT_VERSION:
                raise CheckpointError(
                    f"Unsupported checkpoint version {version}; expected {CHECKPOINT_VERSION}"
                )

            phase = str(manifest.get("phase", "")).strip().lower()
            if phase not in {"phase1", "phase2"}:
                raise CheckpointError(f"Unsupported checkpoint phase: {phase}")

            required_common = {
                "phase1_summary_stats.json",
                "phase1_validation_stats.json",
                "phase1_strain_columns.json",
            }
            missing_common = required_common.difference(names)
            if missing_common:
                raise CheckpointError(f"Checkpoint missing required files: {sorted(missing_common)}")

            summary_stats = json.loads(zf.read("phase1_summary_stats.json").decode("utf-8"))
            validation_stats = json.loads(zf.read("phase1_validation_stats.json").decode("utf-8"))
            strain_payload = json.loads(zf.read("phase1_strain_columns.json").decode("utf-8"))
            strain_columns = list(strain_payload.get("strain_columns", []))

            if phase == "phase1":
                if "phase1_master_df.csv" not in names:
                    raise CheckpointError("Phase 1 checkpoint missing phase1_master_df.csv")
                master_df = pd.read_csv(io.BytesIO(zf.read("phase1_master_df.csv")))
                master_df = _coerce_master_df_types(master_df)
                embeddings_generated = False
            else:
                needed = {"phase2_master_df.csv", "phase2_embeddings.npz"}
                missing = needed.difference(names)
                if missing:
                    raise CheckpointError(f"Phase 2 checkpoint missing files: {sorted(missing)}")

                master_df = pd.read_csv(io.BytesIO(zf.read("phase2_master_df.csv")))
                master_df = _coerce_master_df_types(master_df)

                emb_data = np.load(io.BytesIO(zf.read("phase2_embeddings.npz")), allow_pickle=False)
                embeddings = emb_data["embeddings"]
                if len(master_df) != len(embeddings):
                    raise CheckpointError(
                        "Embedding row count does not match dataframe row count in checkpoint"
                    )
                master_df["Embeddings"] = [row for row in embeddings]
                embeddings_generated = True

            return {
                "manifest": manifest,
                "master_df": master_df,
                "summary_stats": summary_stats,
                "validation_stats": validation_stats,
                "strain_columns": strain_columns,
                "data_loaded": True,
                "embeddings_generated": embeddings_generated,
                "umap_computed": False,
            }
    except CheckpointError:
        raise
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint: {e}") from e
