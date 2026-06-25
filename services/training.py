"""
services/training.py

Phase 1: Training / Knowledge Preparation.

Converts historical resolved ServiceNow tickets (data/sample_tickets.csv)
into a searchable knowledge base:

    Historical Tickets
      -> Load CSV
      -> Clean Data
      -> Mask PII
      -> Combine Title + Description
      -> Generate Embeddings
      -> Store Metadata
      -> Build FAISS Index
      -> Persist Knowledge Base

The Streamlit "Train Knowledge Base" button in app.py calls
`run_training_pipeline()`, which orchestrates all of the steps below and
returns summary statistics for display.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from services.embeddings import embed_batch
from services.preprocess import combine_title_and_description, sanitize_ticket_text
from services.vector_store import build_index, save_index
from services import storage_service

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path("data/sample_tickets.csv")

REQUIRED_COLUMNS = [
    "ticket_id",
    "short_description",
    "description",
    "business_service",
    "assignment_group",
    "priority",
]


def load_training_data(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Load historical resolved tickets from the training CSV.

    Args:
        csv_path: Path to the CSV file of historical tickets.

    Returns:
        A pandas DataFrame with the raw ticket rows.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data file not found at '{csv_path}'. Make sure "
            "data/sample_tickets.csv exists, or provide a different path."
        )

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Training CSV is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info("Loaded %d raw ticket rows from %s", len(df), csv_path)
    return df


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and PII-mask the raw ticket DataFrame in preparation for
    embedding generation.

    - Drops rows missing a ticket_id or with both title and description empty.
    - Fills missing optional fields with safe empty-string defaults.
    - Applies text cleaning + PII masking to free-text fields.

    Args:
        df: Raw DataFrame as returned by load_training_data().

    Returns:
        A cleaned DataFrame, re-indexed from 0, ready for embedding.
    """
    working = df.copy()

    # Normalize column presence for optional fields.
    for optional_col in ["resolution_notes", "caller", "created_date", "status"]:
        if optional_col not in working.columns:
            working[optional_col] = ""
        working[optional_col] = working[optional_col].fillna("")

    working["short_description"] = working["short_description"].fillna("")
    working["description"] = working["description"].fillna("")
    working["business_service"] = working["business_service"].fillna("Unknown")
    working["assignment_group"] = working["assignment_group"].fillna("Unknown")
    working["priority"] = working["priority"].fillna("3 - Moderate")

    before_count = len(working)
    working = working.dropna(subset=["ticket_id"])
    working = working[
        (working["short_description"].str.strip() != "") |
        (working["description"].str.strip() != "")
    ]
    dropped = before_count - len(working)
    if dropped:
        logger.warning("Dropped %d rows during cleaning (missing id or empty text).", dropped)

    # Apply text cleaning + PII masking to all free-text fields.
    for col in ["short_description", "description", "resolution_notes", "caller"]:
        working[col] = working[col].astype(str).apply(sanitize_ticket_text)

    working = working.reset_index(drop=True)
    logger.info("Cleaned training data: %d rows remain.", len(working))
    return working


def generate_embeddings(df: pd.DataFrame):
    """
    Generate embedding vectors for each ticket by combining its title and
    description into a single normalized text blob.

    Args:
        df: Cleaned DataFrame as returned by clean_training_data().

    Returns:
        A tuple of (combined_texts, embedding_matrix) where combined_texts
        is the list of per-ticket text blobs (str) and embedding_matrix is
        a numpy array of shape (n_tickets, embedding_dimensions).
    """
    combined_texts: List[str] = [
        combine_title_and_description(row["short_description"], row["description"])
        for _, row in df.iterrows()
    ]
    logger.info("Generating embeddings for %d tickets...", len(combined_texts))
    vectors = embed_batch(combined_texts)
    logger.info("Generated embedding matrix of shape %s", vectors.shape)
    return combined_texts, vectors


def build_vector_index(vectors):
    """Build a FAISS index from an embedding matrix (thin pass-through)."""
    return build_index(vectors)


def build_metadata_records(df: pd.DataFrame, combined_texts: List[str]) -> List[Dict[str, Any]]:
    """
    Build the list of metadata records aligned by position with the FAISS
    index / embedding matrix.

    Args:
        df: Cleaned DataFrame.
        combined_texts: The combined title+description text used for
            embedding each row, stored alongside metadata for transparency
            and debugging.

    Returns:
        List of dicts, one per ticket, containing all required metadata
        fields plus the combined embedding text.
    """
    records: List[Dict[str, Any]] = []
    for position, (_, row) in enumerate(df.iterrows()):
        records.append({
            "position": position,
            "ticket_id": str(row["ticket_id"]),
            "title": row["short_description"],
            "description": row["description"],
            "business_service": row["business_service"],
            "assignment_group": row["assignment_group"],
            "priority": row["priority"],
            "resolution_notes": row.get("resolution_notes", ""),
            "embedding_text": combined_texts[position],
        })
    return records


def save_index_and_metadata(index, metadata: List[Dict[str, Any]]) -> None:
    """Persist both the FAISS index and its aligned metadata to disk."""
    save_index(index)
    storage_service.save_ticket_metadata(metadata)


def run_training_pipeline(csv_path: Path = DEFAULT_CSV_PATH) -> Dict[str, Any]:
    """
    Run the full Phase 1 training pipeline end-to-end: load -> clean ->
    embed -> build index -> persist. Used by the "Train Knowledge Base"
    button in the Streamlit UI.

    Args:
        csv_path: Path to the historical tickets CSV.

    Returns:
        A dict of training statistics suitable for display in the UI:
            tickets_indexed, business_service_count, assignment_group_count,
            embedding_dimensions, status.

    Raises:
        FileNotFoundError, ValueError: Propagated from load_training_data()
            if the input CSV is missing or malformed. Callers (the
            Streamlit UI) should catch these and show a friendly error.
    """
    storage_service.ensure_storage_files()

    raw_df = load_training_data(csv_path)
    clean_df = clean_training_data(raw_df)

    if clean_df.empty:
        raise ValueError(
            "No usable rows remained after cleaning the training data. "
            "Check that data/sample_tickets.csv has valid rows."
        )

    combined_texts, vectors = generate_embeddings(clean_df)
    index = build_vector_index(vectors)
    metadata = build_metadata_records(clean_df, combined_texts)
    save_index_and_metadata(index, metadata)

    stats = {
        "tickets_indexed": len(metadata),
        "business_service_count": clean_df["business_service"].nunique(),
        "assignment_group_count": clean_df["assignment_group"].nunique(),
        "embedding_dimensions": int(vectors.shape[1]),
        "status": "READY",
    }
    logger.info("Training pipeline complete: %s", stats)
    return stats
