"""
services/preprocess.py

Text preprocessing utilities used by both the training (Phase 1) and
classification (Phase 2) pipelines:

- Cleaning raw ticket text (whitespace, control characters)
- Masking PII (emails, phone numbers, credit-card-like numbers, IP addresses,
  and simple "Name:" style patterns) before anything is embedded, stored, or
  sent to an LLM.
- Combining title + description into a single normalized text blob used for
  embedding generation.

This module has no external dependencies beyond the Python standard library
so it can be safely imported by every other service without risk of
circular imports or heavy import costs.
"""

from __future__ import annotations

import html
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII masking patterns
# ---------------------------------------------------------------------------
# NOTE: These are intentionally conservative, regex-based heuristics suitable
# for a hackathon-grade demo. They are NOT a substitute for a certified PII
# redaction service in a production environment handling regulated data.

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?<!\d)(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
)
_CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]*?){13,16}(?!\d)")
_IP_ADDRESS_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Matches simple "Name: John Smith" / "Guest Name: Jane Doe" style fields.
_NAME_FIELD_RE = re.compile(
    r"(?i)\b((?:guest|customer|caller|employee)?\s*name)\s*[:\-]\s*[A-Za-z'\-]+(?:\s+[A-Za-z'\-]+){0,2}"
)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def mask_pii(text: str) -> str:
    """
    Mask common PII patterns in free-text ticket content.

    This is applied BEFORE text is embedded, persisted to JSON metadata, or
    sent to any LLM, ensuring sensitive data never leaves the boundary of
    the raw source ticket.

    Args:
        text: Raw input text (title, description, or resolution notes).

    Returns:
        Text with PII patterns replaced by bracketed placeholder tokens.
    """
    if not text:
        return text

    masked = text
    masked = _EMAIL_RE.sub("[EMAIL_REDACTED]", masked)
    masked = _CREDIT_CARD_RE.sub("[CARD_REDACTED]", masked)
    masked = _SSN_RE.sub("[SSN_REDACTED]", masked)
    masked = _IP_ADDRESS_RE.sub("[IP_REDACTED]", masked)
    masked = _PHONE_RE.sub("[PHONE_REDACTED]", masked)
    masked = _NAME_FIELD_RE.sub(lambda m: f"{m.group(1)}: [NAME_REDACTED]", masked)

    return masked


def clean_text(text: str) -> str:
    """
    Normalize raw ticket text: unescape HTML entities, strip control
    characters, collapse excess whitespace, and trim.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned, normalized text. Returns an empty string for falsy input.
    """
    if not text:
        return ""

    cleaned = html.unescape(text)
    cleaned = _CONTROL_CHARS_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def sanitize_ticket_text(text: str) -> str:
    """
    Convenience helper that runs cleaning followed by PII masking, the
    standard sequence applied to any text before it is embedded or stored.

    Args:
        text: Raw text.

    Returns:
        Cleaned and PII-masked text.
    """
    return mask_pii(clean_text(text))


def combine_title_and_description(title: str, description: str) -> str:
    """
    Combine a ticket's short description (title) and full description into
    a single normalized text blob suitable for embedding generation.

    Both inputs are cleaned and PII-masked independently before being
    combined, then joined with a clear separator so the embedding model
    can still distinguish title signal from body signal.

    Args:
        title: Ticket short description / title.
        description: Ticket full description.

    Returns:
        Combined text in the form "Title: ...\\nDescription: ...".
    """
    clean_title = sanitize_ticket_text(title or "")
    clean_description = sanitize_ticket_text(description or "")

    parts = []
    if clean_title:
        parts.append(f"Title: {clean_title}")
    if clean_description:
        parts.append(f"Description: {clean_description}")

    combined = "\n".join(parts)
    if not combined:
        logger.warning("combine_title_and_description received empty title and description")
    return combined
