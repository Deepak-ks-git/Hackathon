"""
services/vector_store.py

FAISS-backed vector store for similarity search over historical ticket vectors.
The vector dimension is read from the actual vectors returned by the GenAI
embedding model, so the app works with any embedding model from /models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_DIR = Path("storage/faiss_index")
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"


def _get_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    return faiss


def build_index(vectors: np.ndarray):
    """Build a FAISS IndexFlatIP from already-normalized vectors."""
    faiss = _get_faiss()

    if vectors.size == 0:
        raise ValueError("Cannot build a FAISS index from an empty vector array.")
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D vector array, got shape {vectors.shape}.")

    dimensions = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dimensions)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    logger.info("Built FAISS IndexFlatIP with %d vectors and %d dimensions.", index.ntotal, dimensions)
    return index


def save_index(index, path: Path = FAISS_INDEX_FILE) -> None:
    faiss = _get_faiss()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s", path)


def load_index(path: Path = FAISS_INDEX_FILE):
    faiss = _get_faiss()
    path = Path(path)
    if not path.exists():
        logger.info("No existing FAISS index found at %s", path)
        return None

    try:
        index = faiss.read_index(str(path))
        logger.info("Loaded FAISS index from %s with %d vectors and %d dimensions.", path, index.ntotal, index.d)
        return index
    except Exception:
        logger.exception("Failed to load FAISS index from %s", path)
        return None


def index_exists(path: Path = FAISS_INDEX_FILE) -> bool:
    return Path(path).exists()


def search(index, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
    if index is None or index.ntotal == 0:
        return []

    query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != index.d:
        raise ValueError(
            f"Query vector dimension {query.shape[1]} does not match FAISS index dimension {index.d}. "
            "Retrain the knowledge base after changing the embedding model."
        )

    k = min(top_k, index.ntotal)
    scores, positions = index.search(query, k)

    results: List[Tuple[int, float]] = []
    for pos, score in zip(positions[0], scores[0]):
        if pos == -1:
            continue
        results.append((int(pos), float(score)))
    return results
