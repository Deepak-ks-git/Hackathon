"""
services/classify.py

Phase 2: AI Classification / Inference.

    New Ticket
      -> Preprocess Text
      -> Generate Embedding
      -> Retrieve Top 3 Similar Tickets
      -> Retrieve Relevant Feedback Examples
      -> Build LLM Prompt
      -> LLM Classification
      -> Confidence Validation
      -> Fallback Model (if confidence < threshold)
      -> Return Suggestions
      -> (Agent Review / Save Feedback handled by app.py + storage_service)

This module deliberately contains NO traditional ML classifiers
(no scikit-learn, XGBoost, LightGBM, etc.) per the project's architecture
requirement. All classification is performed by an LLM, grounded with
retrieved similar tickets (RAG) and prior user feedback (few-shot).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.embeddings import embed_text
from services.llm_service import call_llm_with_model, LLMResponse
from services.model_registry import get_fallback_model
from services.preprocess import combine_title_and_description, sanitize_ticket_text
from services.vector_store import load_index, search
from services import storage_service

logger = logging.getLogger(__name__)

VALID_PRIORITIES = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"]


@dataclass
class SimilarTicket:
    """A historical ticket retrieved as similar to the new incoming ticket."""
    ticket_id: str
    title: str
    description: str
    business_service: str
    assignment_group: str
    priority: str
    resolution_notes: str
    similarity_score: float


@dataclass
class ClassificationResult:
    """The full structured output of the classification pipeline."""
    summary: str
    business_service: str
    assignment_group: str
    priority: str
    confidence: float
    reasoning: str
    similar_tickets: List[SimilarTicket] = field(default_factory=list)
    model_used: str = ""
    used_fallback: bool = False
    is_mock: bool = False
    warning: Optional[str] = None


def _get_confidence_threshold() -> float:
    try:
        return float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    except ValueError:
        return 0.6


def retrieve_similar_tickets(embedding_text: str, top_k: int = 3) -> List[SimilarTicket]:
    """
    Retrieve the top_k most similar historical tickets to the given
    (already combined + sanitized) embedding text, using the persisted
    FAISS index and aligned metadata.

    Args:
        embedding_text: Combined title+description text for the new ticket.
        top_k: Number of similar tickets to retrieve.

    Returns:
        List of SimilarTicket objects, ordered by descending similarity.
        Returns an empty list if no FAISS index has been trained yet.
    """
    index = load_index()
    if index is None:
        logger.warning("No FAISS index found; retrieval skipped. Train the knowledge base first.")
        return []

    metadata = storage_service.load_ticket_metadata()
    if not metadata:
        logger.warning("FAISS index exists but ticket_metadata.json is empty; retrieval skipped.")
        return []

    query_vector = embed_text(embedding_text)
    raw_results = search(index, query_vector, top_k=top_k)

    similar_tickets: List[SimilarTicket] = []
    for position, score in raw_results:
        if position < 0 or position >= len(metadata):
            continue
        record = metadata[position]
        similar_tickets.append(SimilarTicket(
            ticket_id=record.get("ticket_id", "UNKNOWN"),
            title=record.get("title", ""),
            description=record.get("description", ""),
            business_service=record.get("business_service", ""),
            assignment_group=record.get("assignment_group", ""),
            priority=record.get("priority", ""),
            resolution_notes=record.get("resolution_notes", ""),
            similarity_score=round(score, 4),
        ))
    return similar_tickets


def retrieve_feedback_examples(max_examples: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent feedback records to use as few-shot examples
    in the LLM prompt, teaching the model from past corrections.

    Args:
        max_examples: Maximum number of feedback records to include.

    Returns:
        List of the most recent feedback dicts (may be empty if no
        feedback has been recorded yet).
    """
    feedback = storage_service.load_feedback()
    if not feedback:
        return []
    return feedback[-max_examples:]


def _format_similar_tickets_for_prompt(similar_tickets: List[SimilarTicket]) -> str:
    if not similar_tickets:
        return "No similar historical tickets were found (knowledge base may be untrained)."

    lines = []
    for i, t in enumerate(similar_tickets, start=1):
        lines.append(
            f"{i}. [similarity={t.similarity_score:.2f}] Title: {t.title}\n"
            f"   Description: {t.description}\n"
            f"   -> Business Service: {t.business_service} | "
            f"Assignment Group: {t.assignment_group} | Priority: {t.priority}\n"
            f"   Resolution: {t.resolution_notes}"
        )
    return "\n".join(lines)


def _format_feedback_for_prompt(feedback_examples: List[Dict[str, Any]]) -> str:
    if not feedback_examples:
        return "No prior user feedback is available yet."

    lines = []
    for i, fb in enumerate(feedback_examples, start=1):
        ai_pred = fb.get("ai_prediction", {})
        correction = fb.get("user_correction", {})
        lines.append(
            f"{i}. Ticket text: {fb.get('ticket_text', '')[:300]}\n"
            f"   AI originally suggested: business_service={ai_pred.get('business_service')}, "
            f"assignment_group={ai_pred.get('assignment_group')}, priority={ai_pred.get('priority')}\n"
            f"   Agent corrected to: business_service={correction.get('business_service')}, "
            f"assignment_group={correction.get('assignment_group')}, priority={correction.get('priority')}"
        )
    return "\n".join(lines)


def build_classification_prompt(
    ticket_text: str,
    similar_tickets: List[SimilarTicket],
    feedback_examples: List[Dict[str, Any]],
    known_business_services: List[str],
    known_assignment_groups: List[str],
) -> str:
    """
    Build the full LLM prompt for ticket classification, grounding the
    model with retrieved similar tickets (RAG) and historical feedback
    corrections (few-shot learning).

    Args:
        ticket_text: The new ticket's combined, sanitized title+description.
        similar_tickets: Retrieved similar historical tickets.
        feedback_examples: Recent agent correction examples.
        known_business_services: Valid business service values to constrain
            the model's output to known categories.
        known_assignment_groups: Valid assignment group values.

    Returns:
        A single string prompt ready to send to the LLM.
    """
    biz_services_str = ", ".join(known_business_services) if known_business_services else "(none trained yet)"
    assignment_groups_str = ", ".join(known_assignment_groups) if known_assignment_groups else "(none trained yet)"

    return f"""You are an expert ServiceNow incident triage assistant for a hospitality company's IT support organization.

NEW TICKET TO CLASSIFY:
{ticket_text}

SIMILAR HISTORICAL TICKETS (retrieved via semantic search):
{_format_similar_tickets_for_prompt(similar_tickets)}

RECENT AGENT FEEDBACK / CORRECTIONS (learn from these patterns):
{_format_feedback_for_prompt(feedback_examples)}

KNOWN VALID BUSINESS SERVICES: {biz_services_str}
KNOWN VALID ASSIGNMENT GROUPS: {assignment_groups_str}
VALID PRIORITY VALUES: {", ".join(VALID_PRIORITIES)}

INSTRUCTIONS:
1. Write a concise 1-2 sentence summary of the issue.
2. Choose the single best Business Service from the known valid list above (or the closest match from similar tickets if nothing fits well).
3. Choose the single best Assignment Group from the known valid list above.
4. Choose a Priority from the valid priority values, based on user impact and urgency.
5. Provide a confidence score between 0.0 and 1.0 reflecting how certain you are, considering similarity scores of retrieved tickets and clarity of the new ticket's description.
6. Provide brief reasoning (2-3 sentences) explaining your choices, referencing similar tickets or feedback patterns where relevant.

Respond with ONLY a single valid JSON object (no markdown fences, no extra text) in exactly this shape:
{{
  "summary": "...",
  "business_service": "...",
  "assignment_group": "...",
  "priority": "...",
  "confidence": 0.0,
  "reasoning": "..."
}}"""


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from raw LLM text output, tolerating
    markdown code fences or stray leading/trailing text.

    Args:
        raw_text: Raw string content returned by the LLM.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON object could be extracted.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first '{' and the matching last '}' in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON object from LLM response: {exc}") from exc

    raise ValueError("LLM response did not contain a parseable JSON object.")


def _parse_llm_classification(
    llm_response: LLMResponse,
    fallback_business_service: str,
    fallback_assignment_group: str,
) -> Dict[str, Any]:
    """
    Parse and validate the LLM's JSON classification output, filling in
    safe defaults for any missing/invalid fields rather than raising, so a
    slightly malformed LLM response never crashes the UI.
    """
    try:
        parsed = _extract_json_object(llm_response.content)
    except ValueError:
        logger.warning("Failed to parse LLM JSON output; using safe defaults. Raw output: %r", llm_response.content)
        parsed = {}

    summary = str(parsed.get("summary") or "Unable to generate summary from LLM response.")
    business_service = str(parsed.get("business_service") or fallback_business_service)
    assignment_group = str(parsed.get("assignment_group") or fallback_assignment_group)
    priority = str(parsed.get("priority") or "3 - Moderate")
    if priority not in VALID_PRIORITIES:
        # Try a loose match (e.g. model returns "High" instead of "2 - High")
        loose_match = next((p for p in VALID_PRIORITIES if priority.lower() in p.lower()), None)
        priority = loose_match or "3 - Moderate"

    try:
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    reasoning = str(parsed.get("reasoning") or "No reasoning provided by the model.")

    return {
        "summary": summary,
        "business_service": business_service,
        "assignment_group": assignment_group,
        "priority": priority,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def classify_ticket(
    short_description: str,
    description: str,
    model_name: str,
    top_k: int = 3,
) -> ClassificationResult:
    """
    Run the full Phase 2 classification pipeline on a new ticket.

    Args:
        short_description: New ticket's short description / title.
        description: New ticket's full description.
        model_name: The LLM model identifier selected in the UI to use for
            classification.
        top_k: Number of similar historical tickets to retrieve via RAG.

    Returns:
        A ClassificationResult with suggested fields, confidence,
        reasoning, and the similar tickets used for grounding.
    """
    # 1. Preprocess text (clean + mask PII) and combine title+description.
    clean_title = sanitize_ticket_text(short_description)
    clean_description = sanitize_ticket_text(description)
    ticket_text = combine_title_and_description(clean_title, clean_description)

    if not ticket_text.strip():
        return ClassificationResult(
            summary="",
            business_service="",
            assignment_group="",
            priority="3 - Moderate",
            confidence=0.0,
            reasoning="No ticket text was provided, so no classification could be generated.",
            similar_tickets=[],
            model_used="",
            warning="Please enter a short description and/or description before requesting AI suggestions.",
        )

    # 2 & 3. Generate embedding + retrieve top-k similar tickets (RAG).
    similar_tickets = retrieve_similar_tickets(ticket_text, top_k=top_k)

    # 4. Retrieve relevant feedback examples (few-shot learning from corrections).
    feedback_examples = retrieve_feedback_examples()

    # Known categories, derived from the trained metadata, to constrain LLM output.
    metadata = storage_service.load_ticket_metadata()
    known_business_services = sorted({m.get("business_service", "") for m in metadata if m.get("business_service")})
    known_assignment_groups = sorted({m.get("assignment_group", "") for m in metadata if m.get("assignment_group")})

    fallback_business_service = similar_tickets[0].business_service if similar_tickets else (
        known_business_services[0] if known_business_services else "Unknown"
    )
    fallback_assignment_group = similar_tickets[0].assignment_group if similar_tickets else (
        known_assignment_groups[0] if known_assignment_groups else "Unknown"
    )

    # 5. Build the LLM prompt.
    prompt = build_classification_prompt(
        ticket_text=ticket_text,
        similar_tickets=similar_tickets,
        feedback_examples=feedback_examples,
        known_business_services=known_business_services,
        known_assignment_groups=known_assignment_groups,
    )
    system_prompt = (
        "You are a precise, deterministic JSON-generating assistant for IT incident "
        "triage. Always respond with strictly valid JSON and nothing else."
    )

    # 6. LLM classification call with the selected model.
    llm_response = call_llm_with_model(prompt, model_name=model_name, system_prompt=system_prompt)
    parsed = _parse_llm_classification(llm_response, fallback_business_service, fallback_assignment_group)

    used_fallback = False
    warning = None

    # 7. Confidence validation + fallback model retry if below threshold.
    threshold = _get_confidence_threshold()
    if parsed["confidence"] < threshold and not llm_response.is_mock:
        fallback_model_name = get_fallback_model()
        if fallback_model_name and fallback_model_name != model_name:
            logger.info(
                "Confidence %.2f below threshold %.2f; retrying with fallback model '%s'.",
                parsed["confidence"], threshold, fallback_model_name,
            )
            fallback_response = call_llm_with_model(prompt, model_name=fallback_model_name, system_prompt=system_prompt)
            fallback_parsed = _parse_llm_classification(fallback_response, fallback_business_service, fallback_assignment_group)

            if fallback_parsed["confidence"] >= parsed["confidence"]:
                parsed = fallback_parsed
                llm_response = fallback_response
                used_fallback = True
            else:
                warning = (
                    f"Confidence remained low ({parsed['confidence']:.2f}) even after retrying with "
                    f"the fallback model. Please review this suggestion carefully before accepting it."
                )

    if parsed["confidence"] < threshold and warning is None:
        warning = (
            f"AI confidence ({parsed['confidence']:.2f}) is below the configured threshold "
            f"({threshold:.2f}). Please review this suggestion carefully before accepting it."
        )

    # 8. Return suggestions (agent review + feedback saving handled in app.py).
    return ClassificationResult(
        summary=parsed["summary"],
        business_service=parsed["business_service"],
        assignment_group=parsed["assignment_group"],
        priority=parsed["priority"],
        confidence=parsed["confidence"],
        reasoning=parsed["reasoning"],
        similar_tickets=similar_tickets,
        model_used=llm_response.model_used,
        used_fallback=used_fallback,
        is_mock=llm_response.is_mock,
        warning=warning,
    )
