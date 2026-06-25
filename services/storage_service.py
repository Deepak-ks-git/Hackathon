"""
services/storage_service.py

Centralized JSON-file storage layer. Per project constraints, NO database
(SQLite/SQLAlchemy) is used anywhere in this app — all persistence is plain
JSON on disk under storage/.

Files managed here:
    storage/ticket_metadata.json   - metadata for every embedded historical
                                      ticket, aligned by list position with
                                      the FAISS index.
    storage/feedback.json          - user corrections to AI suggestions,
                                      used as few-shot examples in future
                                      classification prompts.
    storage/submitted_tickets.json - tickets submitted through the UI.

All read/write operations are defensive: missing files are created with
sensible empty defaults, and corrupt/unreadable JSON is logged and treated
as empty rather than crashing the app.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

STORAGE_DIR = Path("storage")
TICKET_METADATA_PATH = STORAGE_DIR / "ticket_metadata.json"
FEEDBACK_PATH = STORAGE_DIR / "feedback.json"
SUBMITTED_TICKETS_PATH = STORAGE_DIR / "submitted_tickets.json"

MAX_FEEDBACK_RECORDS = 1000

# A simple module-level lock guards concurrent writes from within a single
# Streamlit process. This does not provide cross-process file locking, which
# is an acceptable tradeoff for a hackathon-scale, single-instance app.
_write_lock = threading.Lock()


def _read_json(path: Path, default: Any) -> Any:
    """Read JSON from `path`, returning `default` if missing or corrupt."""
    if not path.exists():
        logger.info("%s does not exist yet; using default value.", path)
        return default

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read/parse %s; treating as default.", path)
        return default


def _write_json(path: Path, data: Any) -> None:
    """Write `data` to `path` as pretty-printed JSON, creating dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with _write_lock:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp_path.replace(path)  # atomic on POSIX/Windows for same-volume renames


def ensure_storage_files() -> None:
    """
    Ensure all expected storage files and directories exist, creating them
    with empty defaults if missing. Safe to call on every app startup.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    (STORAGE_DIR / "faiss_index").mkdir(parents=True, exist_ok=True)

    if not TICKET_METADATA_PATH.exists():
        _write_json(TICKET_METADATA_PATH, [])
        logger.info("Created empty %s", TICKET_METADATA_PATH)

    if not FEEDBACK_PATH.exists():
        _write_json(FEEDBACK_PATH, [])
        logger.info("Created empty %s", FEEDBACK_PATH)

    if not SUBMITTED_TICKETS_PATH.exists():
        _write_json(SUBMITTED_TICKETS_PATH, [])
        logger.info("Created empty %s", SUBMITTED_TICKETS_PATH)


# ---------------------------------------------------------------------------
# Ticket metadata (aligned with FAISS index positions)
# ---------------------------------------------------------------------------

def load_ticket_metadata() -> List[Dict[str, Any]]:
    """Load the list of ticket metadata records, aligned by position with FAISS."""
    return _read_json(TICKET_METADATA_PATH, default=[])


def save_ticket_metadata(metadata: List[Dict[str, Any]]) -> None:
    """Overwrite the ticket metadata file with the given list of records."""
    _write_json(TICKET_METADATA_PATH, metadata)
    logger.info("Saved %d ticket metadata records to %s", len(metadata), TICKET_METADATA_PATH)


# ---------------------------------------------------------------------------
# Feedback (user corrections used as few-shot examples)
# ---------------------------------------------------------------------------

def load_feedback() -> List[Dict[str, Any]]:
    """Load all stored feedback records (AI prediction vs. user correction)."""
    return _read_json(FEEDBACK_PATH, default=[])


def save_feedback_record(record: Dict[str, Any]) -> None:
    """
    Append a single feedback record and persist it, trimming to the most
    recent MAX_FEEDBACK_RECORDS entries (oldest dropped first).

    Args:
        record: Dict expected to contain at least:
            - ticket_text
            - ai_prediction (dict)
            - user_correction (dict)
            - timestamp (added automatically if absent)
    """
    record = dict(record)  # avoid mutating caller's dict
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    feedback = load_feedback()
    feedback.append(record)

    if len(feedback) > MAX_FEEDBACK_RECORDS:
        feedback = feedback[-MAX_FEEDBACK_RECORDS:]
        logger.info("Trimmed feedback log to most recent %d records.", MAX_FEEDBACK_RECORDS)

    _write_json(FEEDBACK_PATH, feedback)
    logger.info("Saved new feedback record (total now: %d).", len(feedback))


# ---------------------------------------------------------------------------
# Submitted tickets
# ---------------------------------------------------------------------------

def load_submitted_tickets() -> List[Dict[str, Any]]:
    """Load all tickets that have been submitted through the UI."""
    return _read_json(SUBMITTED_TICKETS_PATH, default=[])


def save_submitted_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append a newly submitted ticket and persist it.

    Args:
        ticket: Dict describing the submitted ticket. A `ticket_id` and
            `submitted_at` field are added automatically if not present.

    Returns:
        The final ticket record as persisted (including generated fields).
    """
    ticket = dict(ticket)
    tickets = load_submitted_tickets()

    if "ticket_id" not in ticket or not ticket["ticket_id"]:
        ticket["ticket_id"] = f"INC{1000 + len(tickets) + 1:07d}"
    ticket.setdefault("submitted_at", datetime.now(timezone.utc).isoformat())

    tickets.append(ticket)
    _write_json(SUBMITTED_TICKETS_PATH, tickets)
    logger.info("Saved submitted ticket %s (total now: %d).", ticket["ticket_id"], len(tickets))
    return ticket
