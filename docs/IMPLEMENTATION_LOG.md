# Implementation log

## Decisions that deviate from the original brief
- **LLM provider: Groq, not Anthropic Claude.** The product owner chose Groq-hosted
  open-weight models (no Anthropic key available). The generation layer is
  provider-agnostic (`app/generation/llm_client.py`); Groq is the implemented provider.
- **Source PDF is not committed** (vendor copyright). Place it at
  `data/raw/epson_l3310_users_guide.pdf`.

## Phase 1 — Project initialization ✅
- Repo structure, Python 3.12 venv (uv), pinned `backend/requirements.txt`.
- `core/config.py` (pydantic-settings, validated, repo-root-relative paths),
  JSON structured logging, domain exceptions + handlers, `/api/health`.
- Vite + React 19 + TS + Tailwind v4 shell with `/api` dev proxy, Vitest.
- Tests: 5 backend tests passing; frontend production build OK.

## Phase 2 — PDF ingestion ✅
Manual inspected: 154 pages, PDF outline with 157 entries, body text MinionPro 10.5pt,
bold Myriad headings at 26/20/16/14–13.2/11.5pt, header breadcrumb + footer page number,
an icon font for bullets/buttons, vector-drawn illustrations and indicator lights.

- `pdf_parser.py` — styled spans → visual lines (vertical-overlap banding so taller
  inline glyphs stay inline; `[icon]` only where a drawing/image fills a text gap),
  statistical header/footer detection, printed page labels.
- `profiles.py` + `profile_data/epson_l3310_manual.json` — optional per-document glyph
  map (data only, sourced from the manual's own legends). Without it, symbols degrade
  to `•` / `[icon]`.
- `table_extractor.py` — `find_tables` geometry + span-based cell text (PyMuPDF's own
  table text drops spaces for this font), bold header detection, merged-cell carry-down.
- `image_extractor.py` — raster images + vector-drawing clusters (frames/rules removed)
  rendered to PNG, Tesseract OCR (confidence ≥75, words already on the page dropped),
  pluggable `VisionDescriber` (null by default — never invents descriptions).
- `structure_detector.py` — typography analysis (body size, recurring heading-size
  clusters → levels), block segmentation (paragraph/step/bullet/callout/label/link/
  table/figure), heading-stack section tree, pruning of empty/orphan sections.
- `document_builder.py` — orchestration, PDF validation, source SHA-256, Markdown render,
  `data/processed/<doc_id>/{structure.json,document.md,figures/}`.

Results on the manual: heading tree reproduces the PDF outline (≥95% exact level+title
match, asserted in tests), 139/154 printed page labels detected (cover/divider pages
have none), 177 figures (18 with OCR text). Full parse ≈24 s.

Known limitation: indicator-light pictograms in the status tables are vector art with
no text; they render as `[icon]`. The Status column text still describes the state.
A vision describer can be enabled later.

Tests: 18 passing (unit: line grouping, glyph fallback, typography, segmentation, tree;
integration: outline match, page labels, steps with icons, table rows, callouts).

## Phase 3 — Hierarchical chunking ✅
- `tokens.py` — counts with the `all-MiniLM-L6-v2` tokenizer (word-estimate fallback offline).
- `chunker.py` — three-level hierarchy (parent section → subsection → child chunk):
  `##` subsections kept whole under the **400-word H3 gate**, otherwise split at `###`;
  `###` split at sub-headings above the 550-token target; deeper "problem" nodes
  (problem → causes → **Solutions**) stay atomic up to 900 tokens. Fallback: procedure
  group → block → sentence; tables split by row groups with headers repeated;
  callouts/figures/cross-links attach to the preceding instruction; labels bind to what
  follows; mid-procedure continuations carry a ~60-token overlap and a "continued" marker.
- `metadata.py` — `Chunk` schema (hierarchy, stable SHA-1 section IDs, chunk IDs
  `doc::section::NN`, prev/next neighbours, PDF + printed page refs, token/word counts,
  fallback flags, content hash, figure IDs, cross-references), Chroma-safe flattening,
  generic chunk-type classifier (no manual-specific terms).
- `scripts/inspect_chunks.py` — stats / grep / show.

Manual result: 169 chunks, 16–818 tokens (median 199), 8 fallback splits, 164 with
printed page labels; types: 47 troubleshooting, 29 procedure, 8 warning, 5 table,
38 reference, 41 general, 1 faq.

Tests: 28 passing (6 new chunker unit tests incl. H3 gate, problem/solution cohesion,
procedure fallback with numbering + warning attachment, table row split, stable IDs;
4 integration tests on the manual).

## Phase 4 — Embeddings and vector indexing ✅
- `embedding_service.py` — `all-MiniLM-L6-v2` (normalised). The model truncates at 256
  word-pieces, so chunks are embedded as overlapping ~230-token windows, each prefixed
  with the heading path; dense search scores a chunk by its best window.
- `chroma_service.py` — persistent ChromaDB (cosine HNSW). One record per window; the
  primary window stores the full chunk text + all metadata and is the canonical chunk
  copy (BM25 and context expansion read it, so keyword and dense search can't drift).
  Collection metadata holds the embedding fingerprint; a mismatch raises
  `EmbeddingMismatchError` instead of mixing vector spaces.
- `registry.py` + `core/database.py` — SQLite (SQLAlchemy) document registry
  (id, model, version = source-hash prefix, status, counts) and ingestion jobs.
- `ingestion_service.py` — validate → skip if unchanged → parse → chunk → embed (reusing
  vectors for unchanged content hashes) → upsert new version → delete stale versions →
  mark ready → notify retrieval. Single-flight lock; failures recorded on the job.
- `scripts/ingest_document.py` (`--force`, `--rebuild`, `--pdf … --model …` for more manuals).

Manual: 168 chunks / 289 windows indexed in ~30 s; unchanged re-run skipped in 0.02 s;
`--force` re-parses and reuses all 168 embeddings.

Tests: 33 passing (5 new ingestion integration tests on a synthetic PDF with a fake
embedder: skip, versioned update with reuse + stale removal, delete, non-PDF rejection,
embedding-model mismatch).

## Phase 5 — Hybrid retrieval ✅
- `corpus.py` — chunk corpus loaded from Chroma primary records; auto-refreshes when the
  registry signature (document@version set) changes; title index for cross-references.
- `bm25_search.py` — BM25Okapi; tokenizer keeps technical tokens (`a4`, `2-sided`,
  `c9398`), light stemming (jammed→jam, flashing→flash, boxes→box), deepest headings boosted.
- `dense_search.py` — Chroma best-window cosine search with product/document filters.
- `fusion.py` — weighted RRF (`RRF_K`, `DENSE_WEIGHT`, `BM25_WEIGHT`).
- `reranker.py` — `cross-encoder/ms-marco-MiniLM-L-6-v2`, scored over paragraph-aligned
  ≤380-token windows (max) because the cross-encoder truncates at 512; `NullReranker`
  when `RERANKER_ENABLED=false`.
- `context_expansion.py` — only when needed: adjacent parts of split procedures,
  same-subsection warnings for procedures, actionable cross-referenced sections for
  troubleshooting hits; de-duplicated, primary hits first, within `LLM_CONTEXT_BUDGET_TOKENS`.
- `retrieval_service.py` — candidates (20+20) → RRF → rerank top 15 → TOP_K=5 → expand;
  confidence: rerank < −5 → none, < 1 → low, else high (dense-similarity fallback when
  reranking is off). Ablation switches for evaluation.
- `scripts/search_manual.py` — retrieval debugging CLI.

Calibration on 16 hand-written queries: clearly off-topic queries score ≈ −11 (→ none);
all manual questions score > 0. Warm latency ≈ 0.25–0.45 s on CPU.
Fixed during calibration: reranker truncation hid the end of long chunks; a table row
leaked into body text (now excluded by centre-point); reference expansion limited to
actionable targets. Observed weakness: the MS-MARCO cross-encoder under-ranks short
code-lookup chunks (maintenance box code at rank 4) — measured in Phase 9.

Tests: 41 passing (RRF math, tokenizer, BM25 filter, expansion rules/budget,
hybrid retrieval + product filter + ablations on a synthetic index).

## Phase 6 — LangGraph workflow + LLM (Groq) ✅
- `generation/llm_client.py` — provider-agnostic `LLMClient`; `GroqLLMClient` with timeouts,
  JSON mode, own retry/failover (a 429 switches model immediately rather than sleeping).
  Models available on the account (verified via the models API): `openai/gpt-oss-120b`
  (answers), `qwen/qwen3.8-27b` and `openai/gpt-oss-20b` (fallbacks), `gpt-oss-20b` for
  query understanding (separate per-model rate-limit bucket).
- `generation/prompts.py` — grounded system prompt (sources-only, no invented sections /
  pages / button names, `[icon]` rule, `[S#]` citations, history ≠ evidence, injection guard);
  JSON query-understanding prompt (intent, follow-up, standalone query, topic, clarification).
- `generation/response_validator.py` — normalises citation variants (`【S1】`, bare `S1`,
  `[ [S1] ]`), drops unknown markers, strips page numbers not present in the evidence,
  renumbers `[1]…`, builds structured `Citation`s from metadata (with PDF deep links).
- `graph/` — 14 specified nodes + 3 terminal responders; conditional edges for
  greeting/out-of-scope/ambiguous, follow-ups, no-evidence refusal, generation failure
  (degrades to "these manual sections look relevant" with citations). Per-node timings.
- `memory/conversation_manager.py` — memory interface + answer outline extraction so
  "the second method" resolves even when the previous answer is truncated.
- Retrieval refactored into `candidates` / `rerank` / `expand` stages (one per graph node).

Live verification (real index + Groq): blank pages → 4 correct citations (pp. 90, 60, 61, 61);
follow-up "How do I do the second method?" → resolved to "check and clean the print head";
off-topic → refusal; greeting → no LLM call. Issues found live and fixed: citation format
drift, invented button/light names in the lights table, a hallucinated "Wi-Fi section"
suggestion, wrong follow-up resolution, and 17–35 s latency caused by Groq's 8k tokens/min
per-model limit (context budget 6000→3500, output 1400→1000, model failover).

**Deployment note:** the Groq free tier (8,000 TPM per model) supports only a few full
answers per minute; production traffic needs a paid tier.

Tests: 59 passing + 1 opt-in live test (`pytest -m live_llm`, passing).

## Phase 7 — Session memory + API ✅
- `memory/memory_models.py`, `session_store.py` — SQLite tables `chat_sessions` /
  `chat_messages` (cascade delete), conversation state JSON, sliding expiry
  (`SESSION_EXPIRY_DAYS`), trimming to `SESSION_MESSAGE_LIMIT`, hourly expired-session janitor.
- `SQLiteConversationMemory` — persistent implementation of the memory interface; tracks
  current topic, unresolved issue, last intent/query, turn count, recent questions.
- Access model: session IDs are random UUID4 capabilities; no listing endpoint; unknown or
  expired IDs get a new session (never adopted).
- `services/chat_service.py` — per-session single-flight (409 on duplicate submit), request
  timeout (504). `api/rate_limit.py` — per-IP sliding window (429).
- Routes: `POST /api/chat`, `POST|GET|DELETE /api/sessions[/{id}]`, `GET /api/documents`,
  `GET /api/documents/{id}/pdf` (registered files inside RAW_DOCUMENTS_DIR only),
  `POST /api/admin/ingest`, `POST /api/admin/documents` (PDF upload: type, size, magic
  bytes, server-chosen path), `GET /api/admin/ingestion/status`,
  `DELETE /api/admin/documents/{id}`, `GET /api/health`. Admin: `X-Admin-Key` when
  `ADMIN_API_KEY` is set; otherwise only from localhost in development.
- Startup warm-up thread loads models/index and auto-ingests the default manual if none.

Live HTTP check: "Why are my prints coming out blurry?" → answer, 3 citations (pp. 89, 98, 61),
2.6 s; follow-up "How do I do the first one on a Mac?" → rewritten to "Align the print head
on a Mac", correct steps (p. 66); session state persisted in SQLite.

Tests: 68 passing (+9 API tests: persistent memory across requests and restart, isolation,
unknown IDs, trimming, delete/expiry, duplicate submit, validation, rate limit, admin access,
PDF path safety). Test DB is isolated in a temp dir.

## Phase 8 — Frontend ✅
React 19 + TypeScript + Vite + Tailwind v4 + lucide-react, customer-facing "PrintCare Support"
page (brand placeholder, active product from `/api/documents`, welcome + 5 suggested questions).
- `services/api.ts` — typed client, timeouts (AbortController), network/timeout/HTTP error mapping.
- `hooks/useChat.ts` — session ID in localStorage, conversation restored from
  `GET /api/sessions/{id}` on reload, single in-flight request (duplicate-submit guard),
  failed-message marking + retry, "New conversation" (deletes the server session).
- `AnswerMarkdown` — Markdown/GFM; `[n]` markers become citation chips that open the manual
  PDF at the cited page; `[Power button]`/`[Stop button]`/`[icon]` become symbol chips;
  "Important/Caution/Note" paragraphs and blockquotes render as amber callouts.
- `CitationList` — numbered sources with section, document and printed page, clickable.
- Composer (Enter to send, Shift+Enter newline, auto-grow, 2000-char cap), typing
  indicator, error banner with retry, responsive layout, ARIA labels/live regions.

Verified through the Vite dev proxy against the live backend.
Tests: 4 Vitest tests (marker/pictogram preprocessing; suggested question → cited answer →
follow-up carries session ID; duplicate-submit guard + server error; network error + retry).

Observed live: the model sometimes appends an uncited "finishing" step (e.g. "close the paper
support" for envelopes) despite an explicit prompt rule — quantified in Phase 9.
