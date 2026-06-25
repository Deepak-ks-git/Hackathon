"""
services/embeddings.py

Embedding service backed by the GenAI Lab OpenAI-compatible API.

The embedding model is NOT loaded from Hugging Face. Instead, the available
embedding model is discovered from GenAI Lab /models and called through
LangChain/OpenAI-compatible embeddings.

Both the training pipeline and inference pipeline use these same functions,
ensuring ticket vectors and query vectors live in the same vector space.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np
from dotenv import load_dotenv

from services.model_registry import get_default_embedding_model

load_dotenv()

logger = logging.getLogger(__name__)

# This value is updated after the first successful call. text-embedding-3-large
# usually returns 3072 dimensions.
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))


def _base_url() -> str:
    return os.getenv("GENAI_BASE_URL", "https://genailab.tcs.in").strip().rstrip("/")


def _api_key() -> str:
    return os.getenv("GENAI_API_KEY", "").strip()


def _disable_ssl_verify() -> bool:
    return os.getenv("DISABLE_SSL_VERIFY", "false").strip().lower() == "true"


def _timeout_seconds() -> float:
    try:
        return float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    except ValueError:
        return 60.0


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def _embed_with_langchain(texts: List[str], model_name: str) -> np.ndarray:
    """Call GenAI Lab embeddings using LangChain OpenAIEmbeddings."""
    if not _api_key():
        raise RuntimeError("GENAI_API_KEY is missing. Set it in .env before training.")

    try:
        import httpx
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai and httpx are required for GenAI embeddings. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    http_client = httpx.Client(
        verify=not _disable_ssl_verify(),
        timeout=_timeout_seconds(),
    )

    embeddings = OpenAIEmbeddings(
        model=model_name,
        api_key=_api_key(),
        base_url=_base_url(),
        http_client=http_client,
        timeout=_timeout_seconds(),
        # Avoid tiktoken/HF-style model lookups for custom enterprise model ids.
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
    )

    logger.info("Calling GenAI embedding model '%s' for %d texts.", model_name, len(texts))
    vectors = embeddings.embed_documents(texts)
    return np.asarray(vectors, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """Generate a single normalized embedding vector for one text."""
    if not text or not text.strip():
        logger.warning("embed_text called with empty text; returning zero vector.")
        return np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)

    vectors = embed_batch([text])
    return vectors[0]


def embed_batch(texts: List[str]) -> np.ndarray:
    """Generate normalized embedding vectors for a batch of texts."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIMENSIONS), dtype=np.float32)

    safe_texts = [text if text and text.strip() else " " for text in texts]
    empty_mask = [not (text and text.strip()) for text in texts]

    model_name = get_default_embedding_model()
    vectors = _embed_with_langchain(safe_texts, model_name)

    for idx, is_empty in enumerate(empty_mask):
        if is_empty:
            vectors[idx] = np.zeros(vectors.shape[1], dtype=np.float32)

    return _normalize(vectors)
