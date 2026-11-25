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
