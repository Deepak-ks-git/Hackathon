"""
services/llm_service.py

Wraps the hackathon GenAI Lab endpoint using LangChain's ChatOpenAI client
(the GenAI Lab is OpenAI-API-compatible).

Key behaviors required by the project spec:
- Reads configuration exclusively from environment variables (.env), never
  hardcodes API keys.
- If GENAI_API_KEY is missing/empty, the app must still run: this module
  returns deterministic MOCK responses instead of calling out to a real
  endpoint.
- If a real call to the selected/default model fails, or the LLM returns a
  response that fails confidence validation upstream, callers can retry
  with FALLBACK_MODEL via `call_llm(..., use_fallback=True)`.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Model names must come from GenAI Lab /models. We validate in call_llm_with_model.
from services.model_registry import validate_chat_model, get_fallback_model


@dataclass
class LLMResponse:
    """Normalized response from any LLM call (real or mocked)."""
    content: str
    model_used: str
    is_mock: bool
    error: Optional[str] = None


def _get_env_config() -> dict:
    """Read all LLM-related environment variables into a single dict."""
    return {
        "base_url": os.getenv("GENAI_BASE_URL", "").strip(),
        "api_key": os.getenv("GENAI_API_KEY", "").strip(),
        "default_model": os.getenv("DEFAULT_MODEL", "genailab-maas-DeepSeek-V3-0324").strip(),
        "fallback_model": os.getenv("FALLBACK_MODEL", "azure/genailab-maas-gpt-4o-mini").strip(),
        "disable_ssl_verify": os.getenv("DISABLE_SSL_VERIFY", "false").strip().lower() == "true",
        "timeout_seconds": float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
    }


def is_mock_mode() -> bool:
    """
    Return True if the app should use mock LLM responses, i.e. no API key
    has been configured. This lets the entire app run end-to-end (training
    + classification + UI) without any real GenAI Lab access.
    """
    return not _get_env_config()["api_key"]


def _build_chat_client(model_name: str):
    """
    Construct a LangChain ChatOpenAI client pointed at the GenAI Lab
    endpoint for the given model.

    Args:
        model_name: The model identifier to use for this client.

    Returns:
        A configured `ChatOpenAI` instance.
    """
    from langchain_openai import ChatOpenAI
    import httpx as httpx_module

    config = _get_env_config()

    http_client = httpx_module.Client(
        verify=not config["disable_ssl_verify"],
        timeout=config["timeout_seconds"],
    )

    return ChatOpenAI(
        model=model_name,
        api_key=config["api_key"],
        base_url=config["base_url"] or None,
        http_client=http_client,
        timeout=config["timeout_seconds"],
        temperature=0.1,
    )


def _mock_classification_response(prompt: str) -> str:
    """
    Generate a deterministic, structurally-valid mock JSON classification
    response so the app remains fully demoable without GenAI Lab access.

    The mock applies very simple keyword heuristics over ONLY the new
    ticket's text (extracted from the "NEW TICKET TO CLASSIFY:" section of
    the prompt, if present) so the demo still feels responsive to different
    ticket content. Matching against the full prompt would also catch
    keywords from the retrieved similar-tickets context or the "known
    valid business services" list, causing misclassification.
    """
    # Extract just the new-ticket section of the prompt if it's present
    # (this is how build_classification_prompt() in classify.py formats it).
    # Otherwise fall back to using the whole prompt.
    marker = "new ticket to classify:"
    lower_prompt = prompt.lower()
    marker_pos = lower_prompt.find(marker)
    if marker_pos != -1:
        # Stop at the next section header to avoid pulling in retrieved
        # similar tickets / feedback / known categories.
        section_start = marker_pos + len(marker)
        next_section_pos = lower_prompt.find("\n\n", section_start)
        ticket_section = prompt[section_start:next_section_pos] if next_section_pos != -1 else prompt[section_start:]
    else:
        ticket_section = prompt

    text = ticket_section.lower()

    keyword_map = [
        (["wifi", "wireless", "captive portal", "bandwidth"],
         "Guest WiFi Network", "Network Operations Team"),
        (["door lock", "key card", "keycard", "digital key"],
         "Door Lock & Key System", "Security Systems Team"),
        (["pos", "simphony", "terminal", "receipt printer", "payment screen"],
         "Point of Sale System", "POS Support Team"),
        (["opera pms", "pms", "room status", "night audit", "rate plan", "group block"],
         "Property Management System", "PMS Support Team"),
        (["booking engine", "confirmation email", "online booking"],
         "Booking & Reservations Platform", "Reservations Systems Team"),
        (["housekeeping app", "room inspection", "lost and found"],
         "Housekeeping Mobile App", "Mobile Apps Support Team"),
        (["revenue management", "occupancy", "rate recommendation", "competitor rate"],
         "Revenue Management System", "Revenue Systems Team"),
        (["crs", "central reservation", "rooming list", "rate parity"],
         "Central Reservation System", "Reservations Systems Team"),
        (["spa", "activities booking", "kayak"],
         "Spa & Activities Booking", "Guest Services Apps Team"),
        (["finance", "accounts payable", "invoice", "tax calculation"],
         "Back-Office Finance System", "Finance Systems Team"),
    ]

    business_service, assignment_group = "Property Management System", "PMS Support Team"
    matched = False
    for keywords, biz, group in keyword_map:
        if any(kw in text for kw in keywords):
            business_service, assignment_group = biz, group
            matched = True
            break

    priority = "1 - Critical" if any(
        w in text for w in ["down", "frozen", "critical", "unable to check in", "all guests", "everyone"]
    ) else "3 - Moderate" if not any(w in text for w in ["urgent", "high"]) else "2 - High"

    confidence = round(random.uniform(0.55, 0.85) if matched else random.uniform(0.3, 0.55), 2)

    mock_payload = {
        "summary": "Mock summary: ticket text was matched against keyword heuristics "
                    "because no GENAI_API_KEY is configured (mock mode active).",
        "business_service": business_service,
        "assignment_group": assignment_group,
        "priority": priority,
        "confidence": confidence,
        "reasoning": (
            "MOCK MODE (no GENAI_API_KEY configured): this suggestion was produced by "
            "simple keyword matching against the ticket text, not a real LLM call. "
            "Configure GENAI_API_KEY and GENAI_BASE_URL in your .env file to enable "
            "real LLM-based classification."
        ),
    }
    return json.dumps(mock_payload)


def call_llm(
    prompt: str,
    system_prompt: str = "",
    use_fallback: bool = False,
) -> LLMResponse:
    """
    Call the GenAI Lab LLM with the given prompt, or return a mock response
    if no API key is configured.

    Args:
        prompt: The user-role prompt content.
        system_prompt: Optional system-role instructions.
        use_fallback: If True, use FALLBACK_MODEL instead of DEFAULT_MODEL
            (or the explicitly selected model — see classify.py, which
            passes model_name through `call_llm_with_model` instead when a
            specific model has been chosen in the UI).

    Returns:
        An LLMResponse with `.content` containing the raw text response
        (expected to be a JSON string for classification calls).
    """
    config = _get_env_config()
    model_name = config["fallback_model"] if use_fallback else config["default_model"]
    return call_llm_with_model(prompt, model_name, system_prompt=system_prompt)


def call_llm_with_model(
    prompt: str,
    model_name: str,
    system_prompt: str = "",
) -> LLMResponse:
    """
    Call the GenAI Lab LLM using a specific model name (e.g. the model
    selected by the user in the UI dropdown). Falls back to a mock response
    if no API key is configured, and gracefully degrades to FALLBACK_MODEL
    on any request error.

    Args:
        prompt: The user-role prompt content.
        model_name: The exact model identifier to request.
        system_prompt: Optional system-role instructions.

    Returns:
        An LLMResponse describing the outcome (mock, success, or error with
        a fallback attempted).
    """
    is_valid_model, validated_model = validate_chat_model(model_name)
    if not is_valid_model:
        logger.warning(
            "Requested model '%s' is not present in GenAI Lab /models chat list. Using '%s' instead.",
            model_name, validated_model,
        )
        model_name = validated_model

    if is_mock_mode():
        logger.info("GENAI_API_KEY not set; returning mock LLM response for model '%s'.", model_name)
        return LLMResponse(
            content=_mock_classification_response(prompt),
            model_used=f"{model_name} (mock)",
            is_mock=True,
        )

    config = _get_env_config()
    fallback_model = get_fallback_model()
    config["fallback_model"] = fallback_model

    try:
        client = _build_chat_client(model_name)
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))

        result = client.invoke(messages)
        content = result.content if hasattr(result, "content") else str(result)
        logger.info("LLM call succeeded using model '%s'.", model_name)
        return LLMResponse(content=content, model_used=model_name, is_mock=False)

    except Exception as primary_exc:  # noqa: BLE001
        logger.warning(
            "LLM call failed for model '%s': %s. Attempting fallback model '%s'.",
            model_name, primary_exc, config["fallback_model"],
        )

        if model_name == config["fallback_model"]:
            # We already were on the fallback model; nothing left to try.
            logger.error("Fallback model also failed (or was already in use). Returning mock response.")
            return LLMResponse(
                content=_mock_classification_response(prompt),
                model_used=f"{model_name} (mock-after-error)",
                is_mock=True,
                error=str(primary_exc),
            )

        try:
            fallback_client = _build_chat_client(config["fallback_model"])
            messages = []
            if system_prompt:
                messages.append(("system", system_prompt))
            messages.append(("human", prompt))

            result = fallback_client.invoke(messages)
            content = result.content if hasattr(result, "content") else str(result)
            logger.info("Fallback LLM call succeeded using model '%s'.", config["fallback_model"])
            return LLMResponse(
                content=content,
                model_used=config["fallback_model"],
                is_mock=False,
                error=f"Primary model '{model_name}' failed: {primary_exc}",
            )
        except Exception as fallback_exc:  # noqa: BLE001
            logger.error(
                "Both primary ('%s') and fallback ('%s') LLM calls failed. Returning mock response.",
                model_name, config["fallback_model"], exc_info=True,
            )
            return LLMResponse(
                content=_mock_classification_response(prompt),
                model_used="mock (all LLM calls failed)",
                is_mock=True,
                error=f"Primary error: {primary_exc} | Fallback error: {fallback_exc}",
            )
