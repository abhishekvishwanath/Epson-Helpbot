# Printer Support AI — Epson L3310 Helpbot

A customer-support chatbot for a direct-to-consumer printer brand. Customers ask questions in
plain English ("my printer prints blank pages", "what do the flashing lights mean?") and get
step-by-step answers **grounded only in the official Epson L3310 Series User's Guide**, with
clickable page citations. If the manual doesn't cover something, the assistant says so.

- **Ingestion:** structure-aware PDF parsing → hierarchical chunking (H3 400-word gate) →
  `all-MiniLM-L6-v2` embeddings → persistent ChromaDB, versioned and incremental.
- **Retrieval:** BM25 + dense → Reciprocal Rank Fusion → cross-encoder rerank → TOP_K = 5 →
  budgeted parent-context expansion.
- **Orchestration:** LangGraph (17 nodes, conditional routing for follow-ups, ambiguity,
  unsupported questions and LLM failure).
- **Generation:** Groq (`openai/gpt-oss-120b`, with failover) behind a provider-agnostic interface;
  citations are built from metadata and validated, never taken from model-written page numbers.
- **Memory:** persistent SQLite short-term session memory (last 20 messages, topic, state).
- **Frontend:** React + TypeScript + Vite + Tailwind customer chat page.

See [`docs/architecture.md`](docs/architecture.md), [`docs/evaluation_report.md`](docs/evaluation_report.md)
and the build log [`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md).

> **Model note:** the original brief specified the Anthropic Claude API. The product owner chose
> Groq-hosted open-weight models instead. Adding another provider means implementing `LLMClient`
> in `backend/app/generation/llm_client.py`.

---

## Quick start

### Prerequisites
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/) (or pip)
- Node.js 20+
- Tesseract OCR (`brew install tesseract`; optional — OCR is skipped with a warning if missing)
- A Groq API key (<https://console.groq.com>)
- The manual PDF (not committed — it is Epson-copyrighted). Download the *L3310 Series User's Guide*
  from <https://epson.sn> and save it as `data/raw/epson_l3310_users_guide.pdf`.

### Install
```bash
cp .env.example .env              # then set GROQ_API_KEY (and ADMIN_API_KEY if you want admin access by key)

cd backend
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt   # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd ../frontend && npm install && cd ..
```

### Ingest the manual (≈30 s; models download on first run)
```bash
backend/.venv/bin/python scripts/ingest_document.py
```
Re-running is a no-op when the PDF hasn't changed. `--force` re-parses (reusing unchanged
embeddings); `--rebuild` drops the collection (required after changing `EMBEDDING_MODEL`).
The server also auto-ingests the default manual on first start if the index is empty.

### Run
```bash
# terminal 1 — API on http://127.0.0.1:8000 (docs at /docs)
cd backend && .venv/bin/uvicorn app.main:app --port 8000

# terminal 2 — UI on http://localhost:5173 (proxies /api to the backend)
cd frontend && npm run dev
```

---

## API

| Method & path | Purpose |
|---|---|
| `POST /api/chat` | `{session_id?, message, product_model?}` → `{session_id, answer, response_type, citations[], retrieved_sources[], conversation_state, confidence, standalone_query, model, latency_ms}` |
| `POST /api/sessions` | Create a session |
| `GET /api/sessions/{id}` | Session messages + state (the unguessable ID is the access capability) |
| `DELETE /api/sessions/{id}` | Delete session data |
| `GET /api/documents` | Indexed manuals (product identification for the UI) |
| `GET /api/documents/{id}/pdf` | The manual PDF (citation links add `#page=N`) |
| `POST /api/admin/ingest` | Re-ingest the default/registered manual (background) |
| `POST /api/admin/documents` | Upload and ingest another manual (multipart: file, document_id, document_name, manufacturer, product_model) |
| `GET /api/admin/ingestion/status` | Documents, versions, recent jobs |
| `DELETE /api/admin/documents/{id}` | Remove a manual from the index |
| `GET /api/health` | Health, LLM/index readiness |

`response_type` is one of `answer`, `clarification`, `refusal`, `greeting`, `error` (LLM
unavailable: the response lists the most relevant manual pages instead).

**Admin access:** with `ADMIN_API_KEY` set, send it as `X-Admin-Key`. Without it, admin endpoints
work only from localhost when `ENVIRONMENT=development`; in production the key is mandatory.

Example:
```bash
curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"My printer is printing blank pages"}' | jq '.answer, .citations[].page_label'
```

---

## Configuration
All settings live in `.env` (see `.env.example`); they are validated at startup. Key ones:

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | Required for generation |
| `GROQ_MODEL` / `GROQ_FALLBACK_MODELS` / `GROQ_UNDERSTANDING_MODEL` | `openai/gpt-oss-120b` / `qwen/qwen3.8-27b,openai/gpt-oss-20b` / `openai/gpt-oss-20b` | Answer, failover, query-rewrite models |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it requires `--rebuild` |
| `RERANKER_ENABLED` / `RERANKER_MODEL` | `true` / `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| `H3_WORD_THRESHOLD`, `CHUNK_TARGET_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP` | 400, 550, 900, 60 | Chunking |
| `RETRIEVAL_TOP_K`, `DENSE_CANDIDATES`, `BM25_CANDIDATES`, `RRF_K` | 5, 20, 20, 60 | Retrieval |
| `LLM_CONTEXT_BUDGET_TOKENS` | 3500 | Retrieved-context budget |
| `SESSION_MESSAGE_LIMIT`, `SESSION_EXPIRY_DAYS` | 20, 7 | Memory |
| `DATABASE_URL` | `sqlite:///./data/sessions.db` | Sessions + document registry |

---

## Tests & evaluation
```bash
cd backend && .venv/bin/python -m pytest            # unit + integration (fake embedder/LLM)
.venv/bin/python -m pytest -m live_llm               # opt-in: real index + Groq
cd ../frontend && npm test                           # Vitest UI tests

backend/.venv/bin/python scripts/evaluate_rag.py --retrieval   # chunking comparison + ablations
backend/.venv/bin/python scripts/evaluate_rag.py --llm         # live answers, faithfulness judge, memory
```
Debug tools: `scripts/search_manual.py "query"`, `scripts/inspect_chunks.py [--grep|--show]`.

### Measured results (`docs/evaluation_report.md`)
34 verified questions from the manual + 7 unsupported + 2 ambiguous + 3 multi-turn follow-ups.

| Chunking (same hybrid retrieval) | Recall@5 | MRR | Context precision | Top-1 page hit |
|---|---|---|---|---|
| Fixed-size 400 tokens | 0.882 | 0.725 | 0.705 | 0.735 |
| Recursive 550 / 60 overlap | 0.941 | 0.770 | 0.745 | 0.706 |
| **Structure-based hierarchical** | **1.000** | **0.863** | **0.845** | **0.794** |

Ablations (structure chunks): reranker adds +0.047 MRR over hybrid without rerank; dense-only + rerank
matched hybrid + rerank on this set; BM25 alone without rerank is weakest (MRR 0.749).

Live answers (`gpt-oss-120b`, judged by `qwen3.8-27b`): faithfulness **0.931** (21/34 fully faithful),
≥1 correct page citation in **91%** of answers, unsupported questions declined **7/7**, ambiguous →
clarifying question **2/2**, follow-ups resolved **3/3**, memory persisted & sessions isolated **3/3**,
median latency **2.1 s**.

---

## Known limitations
- **Uncited additions:** the answer model occasionally adds plausible steps or causes not in the manual
  (≈7% of claims per the judge), despite explicit prompt rules. Citation markers are validated; claim
  content is not verified at runtime.
- **Pictograms:** indicator lights and some buttons are vector drawings with no text; they render as
  "symbol" chips. No vision model is available on the Groq account, so their meaning comes only from
  the surrounding text (the `VisionDescriber` hook is ready for one).
- **Groq free tier:** 8,000 tokens/min per model ⇒ only a few full answers per minute. The app fails
  over across three models and degrades to "these manual pages are relevant" when all are exhausted.
- **Reranker:** MS-MARCO cross-encoder under-ranks short code-lookup chunks (e.g. the maintenance box
  code is retrieved at rank 4); adds ~300 ms on CPU.
- Single-instance prototype: in-process rate limiter and session locks, SQLite, local ChromaDB.
- Layout assumptions: single-column body text; the parser is typography-driven and verified on one manual.
- Faithfulness is LLM-judged on 34 questions; no human-labelled evaluation yet.

## Production recommendations
1. Paid LLM tier (or self-hosted model) with streaming responses; keep the provider interface.
2. Runtime claim verification (cheap NLI/LLM check of each step against cited chunks) to drop
   unsupported steps before the answer is shown.
3. Vision descriptions for pictogram tables (lights/buttons), cached per figure at ingestion time.
4. Postgres for sessions/registry, Redis for rate limiting & session locks, managed vector DB or
   Chroma server; run ingestion as a queued worker.
5. Human-labelled evaluation set per manual, run in CI on every parser/chunker/prompt change; add
   per-manual ingestion QA (outline-match check already exists in tests).
6. Auth for admin (SSO), WAF, abuse monitoring, PII-safe analytics; consider a domain-tuned or
   multilingual reranker.


---

## Repository layout
```
backend/app/
  api/            routes (chat, sessions, documents, admin, health), dependencies, rate limiting
  core/           config, logging, exceptions, database
  ingestion/      pdf_parser, table/image extractors, structure_detector, chunker, metadata,
                  registry, ingestion_service, tokens, profiles (+ profile_data/)
  embeddings/     windowed sentence-transformer embeddings
  vectorstore/    ChromaDB service
  retrieval/      dense, bm25, fusion, reranker, context_expansion, corpus, retrieval_service
  graph/          LangGraph state, nodes, edges, workflow
  generation/     llm_client (Groq), prompts, response_validator
  memory/         SQLite session models/store, conversation memory
  evaluation/     baseline chunkers, metrics, in-memory store
  services/       wiring container, chat service
backend/tests/    unit/, integration/, evaluation/ (+ eval_set.json)
frontend/src/     components/, hooks/, pages/, services/, types/
scripts/          ingest_document.py, evaluate_rag.py, search_manual.py, inspect_chunks.py
docs/             architecture.md, evaluation_report.md, IMPLEMENTATION_LOG.md
data/             raw/ (PDFs, git-ignored), processed/, chroma_db/ (generated)
```
