"""
ticket_creation_app.py

AI-Powered ServiceNow Ticket Creation App (Service Desk Agent App).

A standalone Streamlit application, independent from the administrator-facing
Knowledge Base Training app. Run this on its own port (e.g. 8502) — it is the
day-to-day tool used by service desk agents to log incidents.

Layout is modeled on a ServiceNow incident intake screen (dark, enterprise
styling). The only AI-facing control exposed to the agent is a single
"AI Suggest" button next to Business Service; everything else about model
selection, scope guardrails, and classification happens behind the scenes.

Model selection is entirely code-driven (see services.model_registry):
    1. DEFAULT_MODEL from .env
    2. Configured fallback model
    3. Preferred / first available model from the GenAI Lab model registry
There is no model dropdown, selector, or AI configuration visible anywhere
in this UI.

This file contains NO backend logic of its own. It only calls into the
existing `services.classify`, `services.storage_service`, and
`services.vector_store` modules, which are shared with `training_app.py`.

Run with:
    streamlit run ticket_creation_app.py --server.port 8502
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from services import storage_service
from services.classify import ClassificationResult, classify_ticket
from services.model_registry import fetch_available_models, get_default_model
from services.vector_store import index_exists

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ticket_creation_app")

st.set_page_config(
    page_title="IHG mySupport — Create Incident",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRIORITY_OPTIONS = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"]
IMPACT_OPTIONS = ["1 - High", "2 - Medium", "3 - Moderate / Limited"]
URGENCY_OPTIONS = ["1 - High", "2 - Medium", "3 - Low"]
STATE_OPTIONS = ["New", "In Progress", "On Hold", "Resolved", "Closed"]
CONTACT_TYPE_OPTIONS = ["Phone", "Email", "Self-Service", "Chat", "Walk-in"]

# Below this confidence, the AI Suggest dialog shows a generic
# "couldn't find a suggestion" message instead of the recommendation
# fields. Matches services.classify._get_confidence_threshold()'s default
# (CONFIDENCE_THRESHOLD env var, defaulting to 0.70).
AI_SUGGEST_CONFIDENCE_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Resolve the model to use internally. This is the ONLY place this app
# decides which model powers AI Suggest. Nothing in the UI ever asks the
# agent to choose a model.
#
# Priority order (handled entirely inside services.model_registry):
#   1. DEFAULT_MODEL from .env
#   2. Configured fallback model
#   3. Preferred model loaded from the GenAI Lab model registry
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _resolve_active_model() -> str:
    available = fetch_available_models()
    return get_default_model(available)


ACTIVE_MODEL = _resolve_active_model()


# ---------------------------------------------------------------------------
# Styling — dark, ServiceNow-inspired theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* Hide Streamlit's default header bar only */
    header[data-testid="stHeader"] { display: none !important; }

    :root {
        --sn-bg: #ffffff;
        --sn-panel: #f7f9fc;
        --sn-field: #ffffff;
        --sn-field-border: #cdd6e3;
        --sn-accent: #2c79c7;
        --sn-text: #1a2533;
        --sn-text-dim: #5b6a80;
        --sn-required: #d6336c;
    }

    .stApp { background-color: var(--sn-bg); }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .sn-topbar {
        background-color: var(--sn-panel);
        border: 1px solid #d7deea;
        border-radius: 6px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .sn-topbar .sn-title {
        color: var(--sn-text);
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0;
    }
    .sn-topbar .sn-subtitle {
        color: var(--sn-text-dim);
        font-size: 0.8rem;
        margin: 0.15rem 0 0 0;
    }
    .sn-topbar .sn-badge {
        background-color: #eef2f8;
        color: var(--sn-text-dim);
        border: 1px solid var(--sn-field-border);
        border-radius: 4px;
        padding: 0.3rem 0.7rem;
        font-size: 0.78rem;
        white-space: nowrap;
    }

    .sn-section-header {
        color: var(--sn-text-dim);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
        border-bottom: 1px solid #d7deea;
        padding-bottom: 0.5rem;
        margin: 0.25rem 0 1rem 0;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: var(--sn-field) !important;
        border: 1px solid var(--sn-field-border) !important;
        color: var(--sn-text) !important;
        border-radius: 4px !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #8492a6 !important;
    }
    .stTextInput label, .stTextArea label, .stSelectbox label, .stCheckbox label {
        color: var(--sn-text) !important;
        font-size: 0.85rem !important;
    }
    .stSelectbox svg { color: var(--sn-text-dim) !important; }

    .sn-field-label {
        color: var(--sn-text);
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }
    .sn-field-label .sn-required-star {
        color: var(--sn-required);
        margin-right: 0.2rem;
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
    .kb-warning-banner {
        background-color: #fff3cd;
        border: 1px solid #ffe69c;
        color: #856404;
        padding: 0.5rem 0.9rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
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
        background-color: #f0f4fa;
        border-left: 3px solid var(--sn-accent);
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: var(--sn-text);
    }

    .applied-banner {
        background-color: #e7f1fb;
        border: 1px solid #bcd6f0;
        color: #1c5d99;
        padding: 0.45rem 0.8rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }

    /* ---- Top nav shell + breadcrumb bar (matches reference screenshot) ---- */
    .sn-shell-nav {
        background-color: #14171c;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 1rem;
        margin: -1rem -1rem 0.6rem -1rem;
        font-size: 0.85rem;
    }
    .sn-shell-nav .sn-brand {
        font-weight: 700;
        letter-spacing: 0.02em;
        font-size: 0.95rem;
        margin-right: 1.2rem;
        white-space: nowrap;
    }
    .sn-shell-nav .sn-brand small {
        display: block;
        font-size: 0.55rem;
        font-weight: 400;
        color: #b9bec7;
        letter-spacing: 0.04em;
    }
    .sn-shell-nav .sn-nav-links {
        display: flex;
        gap: 1.3rem;
        color: #d7dae0;
        flex: 1;
    }
    .sn-shell-nav .sn-record-pill {
        background-color: #1f232b;
        border-radius: 14px;
        padding: 0.25rem 0.9rem;
        font-size: 0.8rem;
        color: #e6e8eb;
        flex: 1;
        max-width: 480px;
        text-align: center;
        margin: 0 1rem;
    }
    .sn-shell-nav .sn-icons {
        display: flex;
        gap: 0.9rem;
        align-items: center;
        color: #c7cad1;
        font-size: 0.95rem;
    }
    .sn-shell-nav .sn-avatar {
        background-color: #2c79c7;
        color: #fff;
        border-radius: 50%;
        width: 1.5rem;
        height: 1.5rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        font-weight: 700;
    }

    .sn-breadcrumb-row {
        background-color: var(--sn-panel);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 1rem;
        margin: 0 -1rem 0.6rem -1rem;
        border-bottom: 1px solid #d7deea;
    }
    .sn-breadcrumb-row .sn-bc-left {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .sn-breadcrumb-row .sn-bc-back {
        border: 1px solid var(--sn-field-border);
        border-radius: 4px;
        width: 1.6rem;
        height: 1.6rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: var(--sn-accent);
        font-size: 0.85rem;
        background: #fff;
    }
    .sn-breadcrumb-row .sn-bc-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--sn-text);
        line-height: 1.1;
    }
    .sn-breadcrumb-row .sn-bc-subtitle {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--sn-text);
        line-height: 1.1;
    }
    .sn-breadcrumb-row .sn-bc-right {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        color: var(--sn-text-dim);
        font-size: 0.95rem;
    }
    .sn-breadcrumb-row .sn-bc-next {
        background-color: var(--sn-accent);
        color: #fff;
        border-radius: 4px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .sn-tab-row {
        display: flex;
        justify-content: space-between;
        padding: 0.45rem 1rem 0.35rem 1rem;
        margin: 0 -1rem 0.8rem -1rem;
        border-bottom: 2px solid #1a1a1a;
        font-size: 0.85rem;
        color: var(--sn-text);
    }

    .sn-readonly-field {
        background-color: #e4e6ea;
        border: 1px solid var(--sn-field-border);
        border-radius: 4px;
        padding: 0.4rem 0.6rem;
        font-size: 0.85rem;
        color: var(--sn-text-dim);
        min-height: 1.4rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize all Streamlit session_state keys used by this app."""
    defaults = {
        # Form values (ServiceNow-style fields)
        "state": "New",
        "requested_by": "",
        "requested_by_not_found": False,
        "call_back_number": "",
        "contact_type": "Phone",
        "requested_for": "",
        "location": "",
        "business_service": "",
        "situation": "",
        "configuration_item": "",
        "impact": "3 - Moderate / Limited",
        "urgency": "2 - Medium",
        "priority": "3 - Moderate",
        "assignment_group": "",
        "short_description": "",
        "description": "",
        "override_suggestion": False,

        # AI suggestion / modal state
        "ai_result": None,            # ClassificationResult | None
        "ai_result_snapshot": None,   # dict snapshot of AI suggestion at suggest-time, for feedback diffing
        "show_ai_modal": False,
        "show_ai_validation_modal": False,  # popup shown when short desc + description are empty
        "applied_suggestion": None,   # dict of last-applied values, for reference

        # Knowledge base availability (read-only check; training happens in training_app.py)
        "kb_ready": index_exists(),

        # Misc
        "submission_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_form() -> None:
    """Reset the ticket form fields and AI suggestion state after submission."""
    st.session_state["state"] = "New"
    st.session_state["requested_by"] = ""
    st.session_state["requested_by_not_found"] = False
    st.session_state["call_back_number"] = ""
    st.session_state["contact_type"] = "Phone"
    st.session_state["requested_for"] = ""
    st.session_state["location"] = ""
    st.session_state["business_service"] = ""
    st.session_state["situation"] = ""
    st.session_state["configuration_item"] = ""
    st.session_state["impact"] = "3 - Moderate / Limited"
    st.session_state["urgency"] = "2 - Medium"
    st.session_state["priority"] = "3 - Moderate"
    st.session_state["assignment_group"] = ""
    st.session_state["short_description"] = ""
    st.session_state["description"] = ""
    st.session_state["override_suggestion"] = False
    st.session_state["ai_result"] = None
    st.session_state["ai_result_snapshot"] = None
    st.session_state["applied_suggestion"] = None


storage_service.ensure_storage_files()
init_session_state()


# ---------------------------------------------------------------------------
# AI Suggest validation popup (st.dialog) — shown when Short Description and
# Description are both empty when the agent clicks AI Suggest.
# ---------------------------------------------------------------------------

@st.dialog("AI Suggest")
def ai_suggest_validation_dialog() -> None:
    st.warning("Please enter short description and description fields to proceed with AI suggestion.")
    if st.button("OK", use_container_width=True, type="primary", key="validation_ok"):
        st.session_state["show_ai_validation_modal"] = False
        st.rerun()


# ---------------------------------------------------------------------------
# AI Suggest modal (st.dialog)
# ---------------------------------------------------------------------------

@st.dialog("AI-Assisted Routing Suggestion", width="large")
def ai_suggest_dialog() -> None:
    result: ClassificationResult | None = st.session_state.get("ai_result")

    if result is None:
        st.info("No suggestion available. Close this dialog and click **AI Suggest** again.")
        if st.button("Cancel", use_container_width=True, key="dialog_cancel_no_result"):
            st.session_state["show_ai_modal"] = False
            st.rerun()
        return

    if result.warning and not result.business_service:
        st.warning(result.warning)
        if st.button("Cancel", use_container_width=True, key="dialog_cancel_warning"):
            st.session_state["show_ai_modal"] = False
            st.rerun()
        return

    # If confidence is below the configured threshold, don't show the
    # (unreliable) recommendation fields — show a generic message instead.
    if result.confidence < AI_SUGGEST_CONFIDENCE_THRESHOLD:
        st.info(
            "🤔 Couldn't find a confident suggestion. Please try providing a more "
            "accurate and detailed description."
        )
        if st.button("Cancel", use_container_width=True, key="dialog_cancel_low_confidence"):
            st.session_state["show_ai_modal"] = False
            st.rerun()
        return

    st.markdown("**AI Summary**")
    st.write(result.summary or "—")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommended Business Service**")
        st.write(result.business_service or "—")
        st.markdown("**Recommended Priority**")
        st.write(result.priority or "—")
    with col2:
        st.markdown("**Recommended Routing Rule**")
        st.write(result.assignment_group or "—")
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
        st.write(result.reasoning or "—")

    with st.expander(f"📋 Similar Historical Tickets ({len(result.similar_tickets)})", expanded=False):
        if not result.similar_tickets:
            st.caption("No similar tickets found. The knowledge base may need to be trained "
                       "(contact your administrator).")
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

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        apply_clicked = st.button("✅ Apply Suggestion", use_container_width=True, type="primary",
                                   key="dialog_apply")
    with btn_col2:
        cancel_clicked = st.button("Cancel", use_container_width=True, key="dialog_cancel")

    if apply_clicked:
        st.session_state["business_service"] = result.business_service
        st.session_state["assignment_group"] = result.assignment_group
        st.session_state["priority"] = (
            result.priority if result.priority in PRIORITY_OPTIONS else st.session_state["priority"]
        )
        st.session_state["applied_suggestion"] = {
            "business_service": result.business_service,
            "assignment_group": result.assignment_group,
            "priority": result.priority,
        }
        st.session_state["show_ai_modal"] = False
        st.rerun()

    if cancel_clicked:
        st.session_state["show_ai_modal"] = False
        st.rerun()


# ---------------------------------------------------------------------------
# Header — nav shell + breadcrumb row + tab row (matches reference screenshot)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="sn-shell-nav">
        <div class="sn-brand">IHG<small>HOTELS &amp; RESORTS</small></div>
        <div class="sn-nav-links">
            <span>All</span><span>Favorites</span><span>History</span><span>Workspaces</span>
        </div>
        <div class="sn-record-pill">IHG mySupport- Create Ticket | Intake Ticket ☆</div>
        <div class="sn-icons">
            <span>💬</span><span>❓</span><span>🔔</span><span class="sn-avatar">U</span>
        </div>
    </div>
    <div class="sn-breadcrumb-row">
        <div class="sn-bc-left">
            <div class="sn-bc-back">‹</div>
            <div>
                <div class="sn-bc-title">Intake Ticket</div>
                <div class="sn-bc-subtitle">New record</div>
            </div>
        </div>
        <div class="sn-bc-right">
            <span>📎</span><span>⚙️</span><span>⋯</span>
            <span class="sn-bc-next">Next</span>
        </div>
    </div>
    <div class="sn-tab-row">
        <span>Intake Details</span>
        <span>Record Preview</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("submission_message"):
    st.success(st.session_state["submission_message"])
    st.session_state["submission_message"] = None

if not st.session_state["kb_ready"]:
    st.markdown(
        '<div class="kb-warning-banner">⚠️ The knowledge base has not been trained yet. '
        "AI Suggest will still work but without historical grounding. "
        "Contact your administrator to run training.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Ticket form
# ---------------------------------------------------------------------------

left_col, right_col = st.columns(2)

# ----------------------------- LEFT SECTION -----------------------------
with left_col:
    st.session_state["state"] = st.selectbox(
        "State", STATE_OPTIONS,
        index=STATE_OPTIONS.index(st.session_state["state"]) if st.session_state["state"] in STATE_OPTIONS else 0,
    )

    st.session_state["requested_by"] = st.text_input(
        "＊ Requested by", value=st.session_state["requested_by"], placeholder="e.g. Maria Garcia",
    )
    st.session_state["requested_by_not_found"] = st.checkbox(
        "Requested by not found", value=st.session_state["requested_by_not_found"],
    )
    st.session_state["call_back_number"] = st.text_input(
        "Call back number", value=st.session_state["call_back_number"], placeholder="e.g. +1 555 123 4567",
    )
    st.session_state["contact_type"] = st.selectbox(
        "Contact type", CONTACT_TYPE_OPTIONS,
        index=CONTACT_TYPE_OPTIONS.index(st.session_state["contact_type"])
        if st.session_state["contact_type"] in CONTACT_TYPE_OPTIONS else 0,
    )
    st.session_state["requested_for"] = st.text_input(
        "＊ Requested for", value=st.session_state["requested_for"], placeholder="e.g. Front Desk Team",
    )
    st.session_state["location"] = st.text_input(
        "＊ Location", value=st.session_state["location"], placeholder="e.g. Hotel Property #4821",
    )

    # Business Service: Text input + Search + AI Suggest, side by side
    st.markdown(
        '<div class="sn-field-label"><span class="sn-required-star">＊</span>Business Service</div>',
        unsafe_allow_html=True,
    )
    biz_input_col, biz_search_col, biz_ai_col = st.columns([5, 1, 1.4])
    with biz_input_col:
        st.session_state["business_service"] = st.text_input(
            "Business Service", value=st.session_state["business_service"],
            placeholder="AI can suggest this →", label_visibility="collapsed",
        )
    with biz_search_col:
        st.button("🔍", key="business_service_search", use_container_width=True,
                   help="Search Business Service catalog")
    with biz_ai_col:
        ai_suggest_clicked = st.button(
            "🤖 AI Suggest", key="ai_suggest_btn", use_container_width=True,
            help="Get AI-recommended Business Service, Routing Rule, and Priority",
        )

    st.session_state["situation"] = st.text_input(
        "＊ Situation", value=st.session_state["situation"], placeholder="e.g. Network outage",
    )
    st.session_state["configuration_item"] = st.text_input(
        "Configuration Item", value=st.session_state["configuration_item"],
    )

# ----------------------------- RIGHT SECTION -----------------------------
with right_col:
    st.markdown('<div class="sn-field-label">Record type</div>', unsafe_allow_html=True)
    st.markdown('<div class="sn-readonly-field">&nbsp;</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sn-field-label" style="margin-top:0.5rem;">Routing rule</div>',
        unsafe_allow_html=True,
    )
    routing_rule_col, routing_info_col = st.columns([6, 1])
    with routing_rule_col:
        # NOTE: "Routing rule" replaces the previous "Assignment Group" field — same
        # session_state key (assignment_group) and same AI Suggest behavior, just
        # relabeled to match the ServiceNow screenshot. Still editable and still
        # populated by Apply Suggestion in the AI Suggest dialog.
        st.session_state["assignment_group"] = st.text_input(
            "Routing rule", value=st.session_state["assignment_group"],
             label_visibility="collapsed",
        )
    with routing_info_col:
        st.button("ℹ️", key="routing_rule_info", use_container_width=True,
                   help="Routing rule determines which team this ticket is assigned to.")

    st.session_state["override_suggestion"] = st.checkbox(
        "Override suggestion", value=st.session_state.get("override_suggestion", False),
    )

    st.markdown('<div class="sn-field-label" style="margin-top:0.5rem;">Template</div>', unsafe_allow_html=True)
    st.markdown('<div class="sn-readonly-field">&nbsp;</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    st.session_state["impact"] = st.selectbox(
        "Impact", IMPACT_OPTIONS,
        index=IMPACT_OPTIONS.index(st.session_state["impact"]) if st.session_state["impact"] in IMPACT_OPTIONS else 2,
    )
    st.session_state["urgency"] = st.selectbox(
        "Urgency", URGENCY_OPTIONS,
        index=URGENCY_OPTIONS.index(st.session_state["urgency"]) if st.session_state["urgency"] in URGENCY_OPTIONS else 1,
    )
    priority_index = (
        PRIORITY_OPTIONS.index(st.session_state["priority"])
        if st.session_state["priority"] in PRIORITY_OPTIONS else 2
    )
    st.session_state["priority"] = st.selectbox(
        "Priority", PRIORITY_OPTIONS, index=priority_index,
        help="AI Suggest can recommend a priority based on similar historical tickets.",
    )

    if st.session_state.get("applied_suggestion"):
        st.markdown(
            '<div class="applied-banner">✅ AI suggestion applied to Business Service, '
            "Routing rule, and Priority.</div>",
            unsafe_allow_html=True,
        )

# ----------------------------- LOWER SECTION -----------------------------
st.markdown('<div class="sn-section-header" style="margin-top:0.5rem;">Description</div>', unsafe_allow_html=True)
st.session_state["short_description"] = st.text_input(
    "Short description", value=st.session_state["short_description"],
    placeholder="e.g. Guest WiFi down on 3rd floor",
)
st.session_state["description"] = st.text_area(
    "Description", value=st.session_state["description"], height=140,
    placeholder="Describe the issue in as much detail as possible...",
)

st.markdown("<br>", unsafe_allow_html=True)
submit_col, _ = st.columns([1, 3])
with submit_col:
    submit_clicked = st.button("📤 Submit Ticket", use_container_width=True, type="primary")


# ----------------------------- AI SUGGEST ACTION -----------------------------
if ai_suggest_clicked:
    if not st.session_state["short_description"].strip() and not st.session_state["description"].strip():
        st.session_state["show_ai_validation_modal"] = True
    else:
        if not st.session_state["kb_ready"]:
            st.warning(
                "Knowledge base is not trained yet — proceeding without historical grounding. "
                "Contact your administrator to run training."
            )
        with st.spinner("Generating AI suggestion..."):
            result = classify_ticket(
                short_description=st.session_state["short_description"],
                description=st.session_state["description"],
                model_name=ACTIVE_MODEL,
            )
            st.session_state["ai_result"] = result
            st.session_state["ai_result_snapshot"] = {
                "business_service": result.business_service,
                "assignment_group": result.assignment_group,
                "priority": result.priority,
            }
            st.session_state["show_ai_modal"] = True


# ----------------------------- SUBMIT TICKET ACTION ---------------------------
if submit_clicked:
    if not st.session_state["short_description"].strip():
        st.error("Short Description is required to submit a ticket.")
    elif not st.session_state["business_service"].strip() or not st.session_state["assignment_group"].strip():
        st.error("Business Service and Routing Rule are required to submit a ticket.")
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
            "requested_by": st.session_state["requested_by"],
            "requested_for": st.session_state["requested_for"],
            "location": st.session_state["location"],
            "short_description": st.session_state["short_description"],
            "description": st.session_state["description"],
            "business_service": st.session_state["business_service"],
            "assignment_group": st.session_state["assignment_group"],
            "situation": st.session_state["situation"],
            "configuration_item": st.session_state["configuration_item"],
            "impact": st.session_state["impact"],
            "urgency": st.session_state["urgency"],
            "priority": st.session_state["priority"],
            "state": st.session_state["state"],
            "ai_summary": result.summary if result else "",
            "ai_confidence": result.confidence if result else None,
            "model_used": result.model_used if result else "",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })

        st.session_state["submission_message"] = f"Ticket {ticket['ticket_id']} submitted successfully!"
        reset_form()
        st.rerun()


# Open the modal(s) if flagged (called at the end so widgets above have
# already registered this run before a dialog takes over).
if st.session_state.get("show_ai_validation_modal"):
    ai_suggest_validation_dialog()
elif st.session_state.get("show_ai_modal"):
    ai_suggest_dialog()