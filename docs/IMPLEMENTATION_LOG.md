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
