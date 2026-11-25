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
