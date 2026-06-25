# 🎫 AI-Powered ServiceNow Ticket Creation & Smart Routing Assistant

An AI assistant embedded into a ServiceNow-like incident creation screen that
recommends **Business Service**, **Assignment Group**, and **Priority** for
new tickets — grounded in historical resolved tickets (RAG) and continuously
improved by agent feedback.

Built for a hackathon: simple to run, but uses a real two-phase
embeddings + RAG + LLM architecture (no traditional ML classifiers).

---

## Architecture

```
PHASE 1 — Training / Knowledge Preparation
Historical Tickets → Load CSV → Clean Data → Mask PII →
Combine Title+Description → Generate Embeddings (GenAI Lab embedding model from /models) →
Store Metadata (JSON) → Build FAISS Index → Persist Knowledge Base

PHASE 2 — AI Classification / Inference
New Ticket → Preprocess Text → Generate Embedding →
Retrieve Top-3 Similar Tickets (FAISS) → Retrieve Feedback Examples →
Build LLM Prompt → LLM Classification → Confidence Validation →
Fallback Model (if confidence < threshold) → Return Suggestions →
Agent Review → Save Feedback → Reuse Feedback as Few-Shot Examples
```

No XGBoost / LightGBM / Random Forest / Logistic Regression — classification
is performed entirely by an LLM grounded with embeddings + similarity search
+ feedback-based few-shot examples.

---

## Project Structure

```
ticket-ai-router/
├── app.py                      # Streamlit UI (two-column ticket screen)
├── requirements.txt
├── .env.example
├── README.md
├── services/
│   ├── preprocess.py            # Text cleaning + PII masking
│   ├── embeddings.py            # GenAI Lab embedding API wrapper
│   ├── vector_store.py          # FAISS index build/save/load/search
│   ├── training.py              # Phase 1 pipeline
│   ├── classify.py              # Phase 2 pipeline (RAG + LLM + feedback)
│   ├── llm_service.py           # LangChain ChatOpenAI / GenAI Lab + mock mode
│   ├── model_registry.py        # Swagger-based model discovery + fallback list
│   └── storage_service.py       # All JSON persistence (no database)
├── data/
│   └── sample_tickets.csv       # 58 sample hospitality incidents, 10 combos
└── storage/
    ├── feedback.json
    ├── submitted_tickets.json
    ├── ticket_metadata.json
    └── faiss_index/             # FAISS index persisted here after training
```

---

## 1. VS Code Setup

1. Open the `ticket-ai-router` folder in VS Code (`File → Open Folder…`).
2. Install the **Python** extension if you don't already have it.
3. Open a terminal in VS Code: `` Ctrl+` `` (Windows/Linux) or `` Cmd+` `` (Mac).

## 2. Virtual Environment Setup

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

In VS Code, select this environment as your interpreter:
`Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv`.

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> Training calls the GenAI Lab embedding model configured in `.env`.
> (~80MB) from Hugging Face — this requires internet access once, then it's
> cached locally.

## 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your GenAI Lab credentials:

```
GENAI_BASE_URL=https://your-genai-lab-endpoint/v1
GENAI_API_KEY=your-api-key-here
DEFAULT_MODEL=genailab-maas-DeepSeek-V3-0324
FALLBACK_MODEL=azure/genailab-maas-gpt-4o-mini
DISABLE_SSL_VERIFY=false
GENAI_MODELS_URL=https://your-genai-lab-endpoint/openapi.json
CONFIDENCE_THRESHOLD=0.6
REQUEST_TIMEOUT_SECONDS=30
```

**Don't have GenAI Lab access yet?** Leave `GENAI_API_KEY` blank. The app
detects this and runs in **Mock Mode**, generating deterministic
keyword-based suggestions so you can demo the full UI and workflow without
any LLM access.

## 5. Run the App

```bash
streamlit run app.py
```

This opens the app in your browser, usually at `http://localhost:8501`.

## 6. Train the Knowledge Base

On first launch, expand the **"📚 Knowledge Base (Phase 1: Training)"**
section and click **🔁 Train Knowledge Base**. This will:

1. Load `data/sample_tickets.csv` (58 historical hospitality incidents)
2. Clean text and mask PII
3. Generate embeddings for each ticket
4. Build and persist a FAISS index to `storage/faiss_index/`
5. Save aligned metadata to `storage/ticket_metadata.json`

You'll see stats like:

```
Tickets Indexed: 58
Business Services: 10
Assignment Groups: 9
Embedding Dims: 384
Status: READY
```

The index is persisted to disk, so it auto-loads on every subsequent app
startup — you only need to retrain if you change `sample_tickets.csv`.

## 7. AI Suggest Workflow

1. Fill in **Caller**, **Short Description**, and **Description** on the
   left panel.
2. (Optional) Pick a specific **AI Model** from the dropdown.
3. Click **✨ AI Suggest**. The right panel populates with:
   - AI Summary
   - Suggested Business Service / Assignment Group / Priority
   - Confidence Score (color-coded badge)
   - Reasoning
   - Similar Historical Tickets (with similarity scores)
4. Click **✅ Apply Suggestions** to copy the AI's recommendations into the
   form fields — feel free to edit them afterward.
5. Click **📤 Submit Ticket** to save the ticket. If you edited any AI
   suggestion before submitting, that correction is automatically saved to
   `storage/feedback.json` and will be used as a few-shot example in future
   classifications (the feedback loop).

## 8. Model Registry Setup

The **AI Model** dropdown is populated by `services/model_registry.py`:

- If `GENAI_MODELS_URL` is set and reachable, the app attempts to parse a
  model list from the returned Swagger/OpenAPI JSON document.
- If that fails for any reason (unset, unreachable, unparseable), the app
  falls back to a static, known-good list:
  `genailab-maas-DeepSeek-V3-0324`, `azure/genailab-maas-gpt-4o-mini`, `gpt-4o`,
  `claude-sonnet`, `llama-3.1`.

The dropdown is **never** empty — there's always a usable model list.

---

## Troubleshooting

| Symptom | Likely Cause / Fix |
|---|---|
| `ModuleNotFoundError` on startup | Run `pip install -r requirements.txt` inside your activated virtual environment. |
| App says "No existing FAISS index found" | Click **Train Knowledge Base** — the index hasn't been built yet. |
| AI Suggest always shows "Mock Mode" banner | `GENAI_API_KEY` is empty in `.env`. Add your real key and restart the app (`Ctrl+C` then `streamlit run app.py` again). |
| Embedding call fails | Confirm `GENAI_API_KEY`, `GENAI_BASE_URL`, and `EMBEDDING_MODEL` are correct and that the embedding model appears in `GET /models`. |
| LLM calls always fail / time out | Verify `GENAI_BASE_URL` is correct and reachable, and that `DISABLE_SSL_VERIFY=true` if your lab uses self-signed certificates. |
| Model dropdown only shows the 5 fallback models | `GENAI_MODELS_URL` is unset or unreachable — this is expected fallback behavior, not an error. |
| Changes to `.env` don't seem to apply | Restart the Streamlit process; environment variables are only re-read on startup. |
| Want to reset all training/feedback/tickets data | Stop the app, delete the contents of `storage/feedback.json`, `storage/submitted_tickets.json`, `storage/ticket_metadata.json` (reset each to `[]`) and delete `storage/faiss_index/index.faiss`, then retrain. |

---

## Notes on Design Choices

- **No database**: per the project constraints, all persistence is plain
  JSON files under `storage/`, written atomically (write-to-temp-then-rename)
  to avoid partial writes if the app is interrupted mid-save.
- **PII masking**: applied to all free text (title, description, resolution
  notes) *before* embedding, storage, or LLM calls — emails, phone numbers,
  card-like numbers, IPs, SSNs, and labeled name fields are redacted.
- **Mock mode**: lets the entire app — training, retrieval, classification,
  feedback loop, UI — run and be demoed with zero external dependencies
  beyond Python packages, which is ideal for a hackathon judging environment
  with unreliable network access.
- **Confidence-based fallback**: if the primary/selected model's confidence
  falls below `CONFIDENCE_THRESHOLD` (default 0.6), the pipeline
  automatically retries once with `FALLBACK_MODEL` and keeps whichever
  result scored higher.


## GenAI Lab Model Configuration

Use `.env.example`. The app calls `${GENAI_BASE_URL}/models` and only uses models returned there. Chat dropdown excludes embedding/Whisper models. Embeddings use `EMBEDDING_MODEL=azure/genailab-maas-text-embedding-3-large`.
