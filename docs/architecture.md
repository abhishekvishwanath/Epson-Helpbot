# Architecture

Printer Support AI answers customer questions about a printer using only that printer's
official manual. It follows the reference two-phase design: **(1) data ingestion** and
**(2) search and retrieval**, orchestrated by LangGraph.

```
                 PHASE 1 — INGESTION                                   PHASE 2 — CHAT TURN
 ┌──────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
 │ PDF (+ optional glyph profile)           │        │ React UI  ──POST /api/chat──▶ FastAPI        │
 │   │ pdf_parser: spans → lines, furniture │        │                                │             │
 │   │   removal, printed page labels,      │        │ LangGraph workflow             ▼             │
 │   │   symbol glyphs, inline [icon]       │        │  receive_query → load_session_memory (SQLite)│
 │   │ table_extractor · image_extractor    │        │  → understand_query (LLM JSON / heuristics)  │
 │   │   (render + Tesseract OCR)           │        │     ├ greeting/out-of-scope/ambiguous → reply│
 │   ▼ structure_detector                   │        │     └ follow-up → resolve_followup_context   │
 │ Section tree (typography-driven levels)  │        │  → retrieve_documents (BM25 + dense → RRF)   │
 │   ▼ chunker (H3 400-word gate, fallback) │        │  → rerank_documents (cross-encoder, windows) │
 │ Chunks + metadata (stable IDs, pages)    │        │  → expand_parent_context (budgeted)          │
 │   ▼ embedding_service (MiniLM windows)   │        │  → assess_retrieval_confidence               │
 │ ChromaDB (persistent, versioned)  ───────┼──────▶ │     └ none → refusal                         │
 │ SQLite registry + ingestion jobs         │        │  → build_grounded_prompt → generate_response │
 └──────────────────────────────────────────┘        │     └ failure → "relevant pages" fallback    │
                                                      │  → validate_response → attach_citations      │
                                                      │  → update_session_memory → return_response   │
                                                      └──────────────────────────────────────────────┘
```

## Phase 1 — Ingestion (`backend/app/ingestion`)

| Step | Module | Notes |
|---|---|---|
| Parse | `pdf_parser.py` | Styled spans → visual lines (vertical-overlap banding keeps inline glyphs inline). Running headers/footers detected statistically from recurring y-positions; printed page labels read from them (never guessed). |
| Symbols | `profiles.py` | Optional per-document glyph map (data only). Without it: leading symbol → `•`, other → `[icon]`. Drawn pictograms inside text gaps become `[icon]`. |
| Tables | `table_extractor.py` | `find_tables` geometry; cell text rebuilt from spans; bold header detection; merged-cell carry-down. |
| Figures | `image_extractor.py` | Raster images + vector drawing clusters (frames/rules excluded) rendered to PNG, Tesseract OCR (confident, novel words only). `VisionDescriber` hook (null by default — never invents). |
| Structure | `structure_detector.py` | Body size = most common non-bold size; heading levels = recurring bold sizes clustered ±1pt. Blocks: paragraph, step, bullet, callout (note/important/caution/warning), label ("Solutions"), cross-link, table, figure. Heading stack → section tree. No manual-specific headings in code. |
| Chunk | `chunker.py` | See below. |
| Metadata | `metadata.py` | `Chunk` model, stable SHA-1 section IDs, generic type classifier. |
| Index | `ingestion_service.py`, `vectorstore/chroma_service.py` | Skip unchanged; reuse vectors for unchanged content hashes; write new version then delete stale; refuse mismatched embedding models. |

### Chunking rules
- Three levels: **parent section** (chapter) → **subsection** (`##`) → **child chunk**.
- **H3 gate:** a `##` subsection ≤ 400 words stays one chunk; larger ones split at `###`.
- `###` nodes split at their sub-headings above the 550-token target; deeper "problem"
  nodes (problem → causes → **Solutions**) stay whole up to 900 tokens.
- **Fallback** (only when still too big): procedure group → block → sentence. Tables split by
  row groups repeating the header. Callouts/figures/links stay with the preceding instruction;
  labels bind to what follows; mid-procedure continuations carry ~60 tokens overlap and a marker.
- Every chunk carries its breadcrumb (`[Doc > Chapter > Section > Topic]`), used for embedding,
  BM25 and the prompt.

### Storage layout
- One Chroma record per embedding **window** (`<chunk_id>#w<i>`). MiniLM truncates at 256
  word-pieces, so long chunks get several heading-prefixed windows; a chunk scores by its best window.
- Window 0 stores the full chunk text + all metadata and is the canonical chunk copy; BM25 and
  context expansion are built from it, so keyword and vector search always see the same corpus.
- `documents` registry (SQLite): version = source-hash prefix, status, embedding model, counts.
  Its signature (`doc@version` set) triggers in-process corpus/BM25 refresh.

## Phase 2 — Retrieval and generation

- **Hybrid search** (`retrieval/`): dense (20 candidates) + BM25 (20) → weighted RRF (k=60) →
  cross-encoder rerank of top 15 over ≤380-token paragraph windows → **TOP_K = 5**.
- **Context expansion**: adjacent split parts, same-subsection warnings for procedures, and
  actionable sections a troubleshooting hit explicitly cross-references — within a token budget.
- **Confidence**: rerank < −5 → *none* (refuse without calling the LLM); < 1 → *low* (prompt warns
  evidence may be partial); else *high*.
- **LLM**: provider-agnostic `LLMClient`; Groq implementation. Answers: `openai/gpt-oss-120b`;
  query understanding: `openai/gpt-oss-20b`; failover chain `qwen/qwen3.8-27b` → `gpt-oss-20b`
  (each Groq model has its own rate-limit bucket; a 429 switches model immediately).
- **Grounding**: sources are numbered `[S#]`; the model cites markers only. The validator
  normalises marker variants, drops unknown ones, removes page numbers absent from the evidence and
  builds structured citations from metadata (PDF deep links `…/pdf#page=N`).

## Session memory (`memory/`)
SQLite `chat_sessions` / `chat_messages`; last 20 messages, topic, unresolved issue, last
intent/query, recent questions; sliding 7-day expiry with an hourly janitor. History is used only to
*interpret* the question (standalone rewrite with an outline of the previous answer) and is passed to
the answer model as context, never as evidence — the manual is re-retrieved every turn.

## Multi-document readiness
Every chunk carries `document_id`, `manufacturer`, `product_model`, `document_version`; retrieval
filters by product model; `POST /api/admin/documents` ingests additional manuals; the UI reads the
active product from `GET /api/documents`.

## Security model (prototype)
Secrets only in `.env`; admin endpoints need `X-Admin-Key` (or localhost in development when no key
is set); uploads are PDF-only, size-limited, magic-byte checked and saved under a server-chosen name;
the PDF endpoint serves only registered files inside `RAW_DOCUMENTS_DIR`; session IDs are random
capabilities with no listing endpoint; per-IP rate limiting; CORS restricted to configured origins;
logs record metadata (intent, confidence, counts), not message text.
