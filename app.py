"""
app.py

AI-Powered ServiceNow Ticket Creation & Smart Routing Assistant.

A two-column Streamlit application embedded into a ServiceNow-like ticket
creation screen:

- Left panel: ticket creation form (Caller, Short Description, Description,
  Business Service, Assignment Group, Priority, Model selector, and action
  buttons).
- Right panel: AI Suggestion Panel showing AI Summary, recommended Business
  Service / Assignment Group / Priority, Confidence Score, Reasoning, and
  Similar Historical Tickets retrieved via RAG.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import os
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from services import storage_service
from services.classify import ClassificationResult, classify_ticket
from services.llm_service import is_mock_mode
from services.model_registry import fetch_available_models, get_default_model
from services.training import run_training_pipeline
from services.vector_store import index_exists

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

st.set_page_config(
    page_title="AI Ticket Router",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRIORITY_OPTIONS = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"]

CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1400px;
    }
    .ticket-header {
        background: linear-gradient(90deg, #2c3e50 0%, #1a2533 100%);
        padding: 1.1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        color: white;
    }
    .ticket-header h1 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
    }
    .ticket-header p {
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
        color: #b8c4d0;
    }
    .panel-card {
        background-color: #ffffff;
        border: 1px solid #e1e5ea;
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #2c3e50;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.6rem;
        border-bottom: 2px solid #2c79c7;
        padding-bottom: 0.4rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .confidence-high { background-color: #d4edda; color: #155724; }
    .confidence-medium { background-color: #fff3cd; color: #856404; }
    .confidence-low { background-color: #f8d7da; color: #721c24; }
    .similar-ticket {
        background-color: #f8f9fb;
        border-left: 3px solid #2c79c7;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .mock-banner {
        background-color: #fff3cd;
        border: 1px solid #ffe69c;
        color: #856404;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize all Streamlit session_state keys used across the app."""
    defaults = {
        "caller": "",
        "short_description": "",
        "description": "",
        "business_service": "",
        "assignment_group": "",
        "priority": "3 - Moderate",
        "selected_model": "",
        "available_models": [],
        "ai_result": None,           # ClassificationResult | None
        "ai_result_snapshot": None,  # dict snapshot of AI suggestion at suggest-time, for feedback diffing
        "training_stats": None,
        "submission_message": None,
        "kb_ready": index_exists(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_form() -> None:
    """Reset the ticket form fields and AI suggestion state after submission."""
    st.session_state["caller"] = ""
    st.session_state["short_description"] = ""
    st.session_state["description"] = ""
    st.session_state["business_service"] = ""
    st.session_state["assignment_group"] = ""
    st.session_state["priority"] = "3 - Moderate"
    st.session_state["ai_result"] = None
    st.session_state["ai_result_snapshot"] = None


# ---------------------------------------------------------------------------
# Startup: ensure storage files exist + load model list
# ---------------------------------------------------------------------------

storage_service.ensure_storage_files()
init_session_state()

if not st.session_state["available_models"]:
    st.session_state["available_models"] = fetch_available_models()
    if not st.session_state["selected_model"]:
        st.session_state["selected_model"] = get_default_model(st.session_state["available_models"])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="ticket-header">
        <h1>🎫 Create New Incident — AI-Assisted Routing</h1>
        <p>Application Support · AI suggests Business Service, Assignment Group, and Priority from historical tickets</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if is_mock_mode():
    st.markdown(
        '<div class="mock-banner">⚠️ <strong>Mock Mode Active</strong> — '
        "No GENAI_API_KEY found in your environment. AI suggestions are being generated "
        "by lightweight keyword heuristics so you can fully demo the app. Add your GenAI "
        "Lab credentials to <code>.env</code> to enable real LLM classification.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Knowledge base training section
# ---------------------------------------------------------------------------

with st.expander("📚 Knowledge Base (Phase 1: Training)", expanded=not st.session_state["kb_ready"]):
    col_train_btn, col_train_status = st.columns([1, 3])

    with col_train_btn:
        if st.button("🔁 Train Knowledge Base", use_container_width=True):
            with st.spinner("Loading tickets, generating embeddings, and building FAISS index..."):
                try:
                    stats = run_training_pipeline()
                    st.session_state["training_stats"] = stats
                    st.session_state["kb_ready"] = True
                    st.success("Knowledge base trained successfully.")
                except (FileNotFoundError, ValueError) as exc:
                    st.error(f"Training failed: {exc}")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Unexpected error during training")
                    st.error(f"Unexpected error during training: {exc}")

    with col_train_status:
        if not st.session_state["kb_ready"]:
            st.info("No trained knowledge base found yet. Click **Train Knowledge Base** to get started.")

    stats = st.session_state.get("training_stats")
    if stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tickets Indexed", stats["tickets_indexed"])
        m2.metric("Business Services", stats["business_service_count"])
        m3.metric("Assignment Groups", stats["assignment_group_count"])
        m4.metric("Embedding Dims", stats["embedding_dimensions"])
        st.caption(f"Status: **{stats['status']}**")
    elif st.session_state["kb_ready"]:
        st.caption("A trained knowledge base was found on disk from a previous session. "
                    "Re-train any time to refresh it from data/sample_tickets.csv.")


if st.session_state.get("submission_message"):
    st.success(st.session_state["submission_message"])
    st.session_state["submission_message"] = None


# ---------------------------------------------------------------------------
# Main two-column layout: Ticket Form (left) | AI Suggestion Panel (right)
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([1.1, 1])

# ----------------------------- LEFT PANEL: TICKET FORM ----------------------
with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📝 Incident Details</div>', unsafe_allow_html=True)

    st.session_state["caller"] = st.text_input(
        "Caller", value=st.session_state["caller"], placeholder="e.g. Maria Garcia"
    )
    st.session_state["short_description"] = st.text_input(
        "Short Description",
        value=st.session_state["short_description"],
        placeholder="e.g. Guest WiFi down on 3rd floor",
    )
    st.session_state["description"] = st.text_area(
        "Description",
        value=st.session_state["description"],
        height=140,
        placeholder="Describe the issue in as much detail as possible...",
    )

    field_col1, field_col2 = st.columns(2)
    with field_col1:
        st.session_state["business_service"] = st.text_input(
            "Business Service", value=st.session_state["business_service"],
            placeholder="AI will suggest this →",
        )
    with field_col2:
        st.session_state["assignment_group"] = st.text_input(
            "Assignment Group", value=st.session_state["assignment_group"],
            placeholder="AI will suggest this →",
        )

    priority_index = (
        PRIORITY_OPTIONS.index(st.session_state["priority"])
        if st.session_state["priority"] in PRIORITY_OPTIONS else 2
    )
    st.session_state["priority"] = st.selectbox("Priority", PRIORITY_OPTIONS, index=priority_index)

    available_models = st.session_state["available_models"] or ["(no models available)"]
    model_index = (
        available_models.index(st.session_state["selected_model"])
        if st.session_state["selected_model"] in available_models else 0
    )
    st.session_state["selected_model"] = st.selectbox(
        "AI Model", available_models, index=model_index,
        help="Model used for AI Suggest classification (populated from the GenAI Lab model registry).",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        ai_suggest_clicked = st.button("✨ AI Suggest", use_container_width=True, type="primary")
    with btn_col2:
        apply_clicked = st.button("✅ Apply Suggestions", use_container_width=True)
    with btn_col3:
        submit_clicked = st.button("📤 Submit Ticket", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- AI SUGGEST ACTION -----------------------------
if ai_suggest_clicked:
    if not st.session_state["short_description"].strip() and not st.session_state["description"].strip():
        st.warning("Please enter a Short Description or Description before requesting AI suggestions.")
    elif not st.session_state["kb_ready"]:
        st.warning("Please train the knowledge base first (see 'Knowledge Base' section above) for the best results. "
                    "Proceeding without historical grounding.")
        with st.spinner("Generating AI suggestions..."):
            result = classify_ticket(
                short_description=st.session_state["short_description"],
                description=st.session_state["description"],
                model_name=st.session_state["selected_model"],
            )
            st.session_state["ai_result"] = result
            st.session_state["ai_result_snapshot"] = {
                "business_service": result.business_service,
                "assignment_group": result.assignment_group,
                "priority": result.priority,
            }
    else:
        with st.spinner("Generating AI suggestions..."):
            result = classify_ticket(
                short_description=st.session_state["short_description"],
                description=st.session_state["description"],
                model_name=st.session_state["selected_model"],
            )
            st.session_state["ai_result"] = result
            st.session_state["ai_result_snapshot"] = {
                "business_service": result.business_service,
                "assignment_group": result.assignment_group,
                "priority": result.priority,
            }

# ----------------------------- APPLY SUGGESTIONS ACTION ----------------------
if apply_clicked:
    result: ClassificationResult | None = st.session_state.get("ai_result")
    if result is None:
        st.warning("Run **AI Suggest** first to generate suggestions to apply.")
    else:
        st.session_state["business_service"] = result.business_service
        st.session_state["assignment_group"] = result.assignment_group
        st.session_state["priority"] = result.priority if result.priority in PRIORITY_OPTIONS else st.session_state["priority"]
        st.rerun()

# ----------------------------- SUBMIT TICKET ACTION ---------------------------
if submit_clicked:
    if not st.session_state["short_description"].strip():
        st.error("Short Description is required to submit a ticket.")
    elif not st.session_state["business_service"].strip() or not st.session_state["assignment_group"].strip():
        st.error("Business Service and Assignment Group are required to submit a ticket.")
    else:
        result: ClassificationResult | None = st.session_state.get("ai_result")
        snapshot = st.session_state.get("ai_result_snapshot")

        # Feedback loop: if the agent edited fields after an AI suggestion, save the correction.
        if result is not None and snapshot is not None:
            final_values = {
                "business_service": st.session_state["business_service"],
                "assignment_group": st.session_state["assignment_group"],
                "priority": st.session_state["priority"],
            }
            if final_values != snapshot:
                storage_service.save_feedback_record({
                    "ticket_text": f"{st.session_state['short_description']} {st.session_state['description']}".strip(),
                    "ai_prediction": snapshot,
                    "user_correction": final_values,
                })
                logger.info("Saved feedback record: AI suggestion overridden by agent.")

        ticket = storage_service.save_submitted_ticket({
            "caller": st.session_state["caller"],
            "short_description": st.session_state["short_description"],
            "description": st.session_state["description"],
            "business_service": st.session_state["business_service"],
            "assignment_group": st.session_state["assignment_group"],
            "priority": st.session_state["priority"],
            "ai_summary": result.summary if result else "",
            "ai_confidence": result.confidence if result else None,
            "model_used": result.model_used if result else "",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })

        st.session_state["submission_message"] = f"Ticket {ticket['ticket_id']} submitted successfully!"
        reset_form()
        st.rerun()


# ----------------------------- RIGHT PANEL: AI SUGGESTION PANEL --------------
with right_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🤖 AI Suggestion Panel</div>', unsafe_allow_html=True)

    result: ClassificationResult | None = st.session_state.get("ai_result")

    if result is None:
        st.info("Click **AI Suggest** on the left to generate routing suggestions for this ticket.")
    elif result.warning and not result.business_service:
        st.warning(result.warning)
    else:
        if result.is_mock:
            st.caption(f"🧪 Model used: {result.model_used}")
        else:
            badge = "🔁 (fallback model used)" if result.used_fallback else ""
            st.caption(f"🧠 Model used: {result.model_used} {badge}")

        st.markdown("**AI Summary**")
        st.write(result.summary)

        sug_col1, sug_col2 = st.columns(2)
        with sug_col1:
            st.markdown("**Business Service**")
            st.write(result.business_service or "—")
        with sug_col2:
            st.markdown("**Assignment Group**")
            st.write(result.assignment_group or "—")

        sug_col3, sug_col4 = st.columns(2)
        with sug_col3:
            st.markdown("**Priority**")
            st.write(result.priority or "—")
        with sug_col4:
            st.markdown("**Confidence Score**")
            confidence_pct = result.confidence * 100
            badge_class = (
                "confidence-high" if result.confidence >= 0.7
                else "confidence-medium" if result.confidence >= 0.5
                else "confidence-low"
            )
            st.markdown(
                f'<span class="confidence-badge {badge_class}">{confidence_pct:.0f}%</span>',
                unsafe_allow_html=True,
            )

        if result.warning:
            st.warning(result.warning)

        with st.expander("💭 Reasoning", expanded=True):
            st.write(result.reasoning)

        with st.expander(f"📋 Similar Historical Tickets ({len(result.similar_tickets)})", expanded=False):
            if not result.similar_tickets:
                st.caption("No similar tickets found. Train the knowledge base to enable retrieval.")
            else:
                for t in result.similar_tickets:
                    safe_title = html.escape(t.title)
                    safe_biz = html.escape(t.business_service)
                    safe_group = html.escape(t.assignment_group)
                    safe_priority = html.escape(t.priority)
                    safe_ticket_id = html.escape(t.ticket_id)
                    st.markdown(
                        f"""<div class="similar-ticket">
                        <strong>{safe_ticket_id}</strong> · similarity {t.similarity_score:.2f}<br>
                        <em>{safe_title}</em><br>
                        → {safe_biz} / {safe_group} / {safe_priority}
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Recent submitted tickets panel
    with st.expander("🗂️ Recently Submitted Tickets", expanded=False):
        submitted = storage_service.load_submitted_tickets()
        if not submitted:
            st.caption("No tickets submitted yet.")
        else:
            for t in reversed(submitted[-5:]):
                st.markdown(
                    f"**{t.get('ticket_id', 'N/A')}** — {t.get('short_description', '')} "
                    f"({t.get('priority', 'N/A')})"
                )
