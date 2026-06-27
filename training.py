"""
training.py

AI Ticket Router — Knowledge Base Training Console (Administrator App).

A standalone Streamlit application, independent from the agent-facing
Ticket Creation app. Run this on its own port (e.g. 8501) and restrict
access to administrators.

Responsibilities:
- Build / refresh the FAISS knowledge base from historical ticket data.
- Show training status and knowledge base health (tickets indexed,
  business service / assignment group counts, embedding dimensions).
- Display which model is currently configured for classification and
  embeddings (read-only, code-driven — never user-selectable).

This file contains NO backend logic of its own. It only calls into the
existing `services.training` and `services.vector_store` modules, which
are shared with `ticket.py`.

Run with:
    streamlit run training.py --server.port 8501
"""

from __future__ import annotations

import logging

import streamlit as st
from dotenv import load_dotenv

from services import storage_service
from services.llm_service import is_mock_mode
from services.model_registry import (
    fetch_available_models,
    get_default_embedding_model,
    get_default_model,
)
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
logger = logging.getLogger("training_app")

st.set_page_config(
    page_title="AI Ticket Router — Admin: Knowledge Base Training",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    :root {
        --sn-bg: #ffffff;
        --sn-panel: #f7f9fc;
        --sn-field: #ffffff;
        --sn-field-border: #cdd6e3;
        --sn-text: #1a2533;
        --sn-text-dim: #5b6a80;
    }
    /* Hide Streamlit's default chrome (header bar, toolbar, footer, decoration) */
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    footer { display: none !important; }
    #MainMenu { visibility: hidden !important; }

    .stApp { background-color: var(--sn-bg); }
    .main .block-container {
        padding-top: 0.8rem;
        max-width: 1200px;
    }
    .admin-header {
        background-color: var(--sn-panel);
        border: 1px solid #d7deea;
        border-left: 4px solid #2c79c7;
        border-radius: 6px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.3rem;
    }
    .admin-header h1 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--sn-text);
    }
    .admin-header p {
        margin: 0.25rem 0 0 0;
        font-size: 0.85rem;
        color: var(--sn-text-dim);
    }
    .panel-card {
        background-color: var(--sn-panel);
        border: 1px solid #d7deea;
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--sn-text);
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.7rem;
        border-bottom: 2px solid #2c79c7;
        padding-bottom: 0.4rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .status-ready { background-color: #d4edda; color: #155724; }
    .status-not-ready { background-color: #fff3cd; color: #856404; }
    .mock-banner {
        background-color: #fff3cd;
        border: 1px solid #ffe69c;
        color: #856404;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .model-info code {
        color: #1f7a3d;
        background-color: #f0f3f8;
        padding: 0.15rem 0.45rem;
        border-radius: 3px;
        font-size: 0.82rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; color: var(--sn-text); }
    div[data-testid="stMetricLabel"] { color: var(--sn-text-dim); }
    .stApp, .stMarkdown, p, label { color: var(--sn-text); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "training_stats": None,
        "kb_ready": index_exists(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


storage_service.ensure_storage_files()
init_session_state()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="admin-header">
        <h1>🛠️ Knowledge Base Training Console</h1>
        <p>Administrator tool · Builds the FAISS knowledge base used by the Ticket Creation app for
        AI-assisted routing suggestions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if is_mock_mode():
    st.markdown(
        '<div class="mock-banner">⚠️ <strong>Mock Mode Active</strong> — '
        "No GENAI_API_KEY found in your environment. Embedding generation will use "
        "the configured embedding model, but downstream classification in the Ticket "
        "Creation app will fall back to keyword heuristics. Add your GenAI Lab credentials "
        "to <code>.env</code> to enable real embeddings and LLM classification.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Current Knowledge Base Status
# ---------------------------------------------------------------------------

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">📊 Current Knowledge Base Status</div>', unsafe_allow_html=True)

status_col, action_col = st.columns([3, 1])

with status_col:
    if st.session_state["kb_ready"]:
        st.markdown(
            '<span class="status-pill status-ready">✅ Knowledge Base Ready</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "A trained FAISS index was found on disk. Re-train any time to refresh it "
            "from the latest ticket data."
        )
    else:
        st.markdown(
            '<span class="status-pill status-not-ready">⚠️ Not Trained Yet</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "No FAISS index found yet. Click **Train Knowledge Base** below to build it "
            "from historical ticket data."
        )

with action_col:
    train_clicked = st.button("🔁 Train Knowledge Base", use_container_width=True, type="primary")

st.markdown("</div>", unsafe_allow_html=True)

if train_clicked:
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


# ---------------------------------------------------------------------------
# Training Status / Metrics
# ---------------------------------------------------------------------------

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">📈 Training Status</div>', unsafe_allow_html=True)

stats = st.session_state.get("training_stats")
if stats:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tickets Indexed", stats["tickets_indexed"])
    m2.metric("Business Services", stats["business_service_count"])
    m3.metric("Assignment Groups", stats["assignment_group_count"])
    m4.metric("Embedding Dims", stats["embedding_dimensions"])
    st.caption(f"Status: **{stats['status']}**")
elif st.session_state["kb_ready"]:
    st.info(
        "A knowledge base exists on disk from a previous training run, but detailed "
        "stats for that run are not available in this session. Click **Train Knowledge "
        "Base** above to refresh and see live metrics."
    )
else:
    st.info("No training run has been performed in this session yet.")

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model Configuration (read-only — code-driven, never user-selectable)
# ---------------------------------------------------------------------------

st.markdown('<div class="panel-card model-info">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">⚙️ Configured Models (read-only)</div>', unsafe_allow_html=True)
st.caption(
    "Model selection is fully code-driven (DEFAULT_MODEL in .env → configured fallback → "
    "GenAI Lab model registry). End users never choose a model — this is shown here for "
    "administrator visibility only."
)

try:
    available_models = fetch_available_models()
    classification_model = get_default_model(available_models)
except Exception:  # noqa: BLE001
    classification_model = "(unavailable)"

try:
    embedding_model = get_default_embedding_model()
except Exception:  # noqa: BLE001
    embedding_model = "(unavailable)"

mc1, mc2 = st.columns(2)
with mc1:
    st.markdown("**Current Classification Model**")
    st.markdown(f"<code>{classification_model}</code>", unsafe_allow_html=True)
with mc2:
    st.markdown("**Current Embedding Model**")
    st.markdown(f"<code>{embedding_model}</code>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Recently submitted tickets (operational visibility for admins)
# ---------------------------------------------------------------------------

with st.expander("🗂️ Recently Submitted Tickets", expanded=False):
    submitted = storage_service.load_submitted_tickets()
    if not submitted:
        st.caption("No tickets submitted yet.")
    else:
        for t in reversed(submitted[-10:]):
            st.markdown(
                f"**{t.get('ticket_id', 'N/A')}** — {t.get('short_description', '')} "
                f"({t.get('priority', 'N/A')})"
            )
