"""
services/model_registry.py

Loads the model list from the GenAI Lab OpenAI-compatible /models API.

Important design rule:
- /models is the single source of truth for available models.
- Chat/classification dropdown must show only chat-capable models.
- Embedding generation must use only embedding-capable models from /models.

The API response shape expected is OpenAI-like:
{
  "object": "list",
  "data": [
    {"id": "genailab-maas-DeepSeek-V3-0324", "object": "model"},
    {"id": "azure/genailab-maas-text-embedding-3-large", "object": "model"}
  ]
}
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import httpx

logger = logging.getLogger(__name__)

# These are used only when the /models API is unavailable. Keep them aligned
# with the GenAI Lab /models response the hackathon team shared.
FALLBACK_ALL_MODELS: List[str] = [
    "genailab-maas-gpt-35-turbo",
    "gemini-3.1-pro-preview",
    "azure/genailab-maas-gpt-4o-mini",
    "azure/genailab-maas-text-embedding-3-large",
    "azure/genailab-maas-whisper",
    "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",
    "azure_ai/genailab-maas-Llama-3.3-70B-Instruct",
    "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8",
    "genailab-maas-gpt-4o",
    "azure_ai/genailab-maas-Phi-4-reasoning",
    "azure/genailab-maas-gpt-4.1-mini",
    "azure/genailab-maas-gpt-4.1-nano",
    "azure_ai/genailab-maas-DeepSeek-R1",
    "azure_ai/Llama-3.3-70B-Instruct_Mass",
    "azure/genailab-maas-gpt-4.1",
    "azure/genailab-maas-gpt-5-mini",
    "genailab-maas-DeepSeek-V3-0324",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "genailab-maas-gpt-5.0",
    "genailab-maas-gpt-5.1",
    "genailab-maas-gpt-5.2",
    "genailab-maas-gpt-5.4",
    "genailab-maas-gpt-5.4-mini",
    "genailab-maas-gpt-5.4-nano",
    "azure_ai/genailab-maas-kimi-k2.5",
    "genailab-maas-gpt-5.3-codex",
    "genailab-maas-gpt-5.2-codex",
]


def _normalize_model_list(raw_models: List[str]) -> List[str]:
    """De-duplicate while preserving API order."""
    seen: set[str] = set()
    normalized: List[str] = []
    for name in raw_models:
        if not isinstance(name, str):
            continue
        clean_name = name.strip()
        if clean_name and clean_name not in seen:
            seen.add(clean_name)
            normalized.append(clean_name)
    return normalized


def _base_url() -> str:
    return os.getenv("GENAI_BASE_URL", "https://genailab.tcs.in").strip().rstrip("/")


def _models_url() -> str:
    # Optional override for unusual deployments.
    configured = os.getenv("GENAI_MODELS_URL", "").strip()
    if configured:
        return configured
    return f"{_base_url()}/models"


def _request_timeout() -> float:
    try:
        return float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0


def _disable_ssl_verify() -> bool:
    return os.getenv("DISABLE_SSL_VERIFY", "false").strip().lower() == "true"


def _extract_model_ids(payload: Dict) -> List[str]:
    """Extract model ids from OpenAI-compatible /models payload."""
    model_ids: List[str] = []

    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                model_ids.append(str(item["id"]))
            elif isinstance(item, str):
                model_ids.append(item)

    # Backward-compatible shapes, in case the lab proxy changes.
    models = payload.get("models")
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and item.get("id"):
                model_ids.append(str(item["id"]))
            elif isinstance(item, str):
                model_ids.append(item)

    return _normalize_model_list(model_ids)


@lru_cache(maxsize=1)
def fetch_all_models() -> List[str]:
    """
    Fetch all models from GenAI Lab /models. Falls back to known models when
    the endpoint is unavailable, so the demo UI never breaks.
    """
    url = _models_url()
    headers = {}
    api_key = os.getenv("GENAI_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=_request_timeout(), verify=not _disable_ssl_verify()) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            model_ids = _extract_model_ids(response.json())
    except Exception:  # noqa: BLE001
        logger.warning("Could not fetch model list from %s; using fallback list.", url, exc_info=True)
        model_ids = FALLBACK_ALL_MODELS

    normalized = _normalize_model_list(model_ids)
    logger.info("Loaded %d total models.", len(normalized))
    return normalized


def is_embedding_model(model_id: str) -> bool:
    text = model_id.lower()
    return "embedding" in text or "embed" in text


def is_non_chat_model(model_id: str) -> bool:
    text = model_id.lower()
    return (
        is_embedding_model(model_id)
        or "whisper" in text
        or "speech" in text
        or "audio" in text
        or "tts" in text
    )


def fetch_embedding_models() -> List[str]:
    """Return only embedding models listed by /models."""
    embedding_models = [model for model in fetch_all_models() if is_embedding_model(model)]
    return _normalize_model_list(embedding_models)


def fetch_available_models(timeout_seconds: float | None = None) -> List[str]:
    """
    Return only chat/classification models for the Streamlit dropdown.

    The timeout_seconds argument is kept for backward compatibility with the
    old call sites; model fetching now uses REQUEST_TIMEOUT_SECONDS from env.
    """
    del timeout_seconds
    chat_models = [model for model in fetch_all_models() if not is_non_chat_model(model)]
    if not chat_models:
        logger.warning("No chat models found from /models; falling back to non-embedding known models.")
        chat_models = [model for model in FALLBACK_ALL_MODELS if not is_non_chat_model(model)]
    return _normalize_model_list(chat_models)


def get_default_model(available_models: List[str]) -> str:
    if not available_models:
        return ""

    configured_default = os.getenv("DEFAULT_MODEL", "").strip()
    if configured_default and configured_default in available_models:
        return configured_default

    # Prefer the hackathon DeepSeek V3 model if present.
    preferred = "genailab-maas-DeepSeek-V3-0324"
    if preferred in available_models:
        return preferred

    return available_models[0]


def get_fallback_model(available_models: List[str] | None = None) -> str:
    models = available_models or fetch_available_models()
    configured = os.getenv("FALLBACK_MODEL", "").strip()
    if configured and configured in models:
        return configured

    for preferred in [
        "azure/genailab-maas-gpt-4o-mini",
        "genailab-maas-gpt-4o",
        "genailab-maas-DeepSeek-V3-0324",
    ]:
        if preferred in models:
            return preferred

    return models[0] if models else ""


def get_default_embedding_model() -> str:
    configured = os.getenv("EMBEDDING_MODEL", "").strip()
    embedding_models = fetch_embedding_models()

    if configured and configured in embedding_models:
        return configured

    preferred = "azure/genailab-maas-text-embedding-3-large"
    if preferred in embedding_models:
        return preferred

    if embedding_models:
        return embedding_models[0]

    raise RuntimeError(
        "No embedding model was found in GenAI Lab /models. The /models API must include "
        "an embedding model such as 'azure/genailab-maas-text-embedding-3-large'."
    )


def validate_chat_model(model_name: str) -> Tuple[bool, str]:
    """Return (is_valid, replacement_model)."""
    models = fetch_available_models()
    if model_name in models:
        return True, model_name
    replacement = get_default_model(models)
    return False, replacement
