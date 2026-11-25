"""Centralised, validated application configuration.

All values come from environment variables (or the repo-root ``.env``).
Relative paths are resolved against the repository root so the app behaves
the same no matter which directory it is launched from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -------------------------------------------------------
    app_name: str = "Printer Support AI"
    environment: str = Field(default="development", pattern="^(development|test|production)$")
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    admin_api_key: str = ""
    rate_limit_per_minute: int = Field(default=30, ge=1)
    request_timeout_seconds: float = Field(default=90.0, gt=0)

    # --- LLM (Groq) --------------------------------------------------------
    llm_provider: str = Field(default="groq", pattern="^(groq)$")
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    llm_max_output_tokens: int = Field(default=1400, ge=128)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_context_budget_tokens: int = Field(default=6000, ge=1000)

    # --- Embeddings / reranking -------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_window_tokens: int = Field(default=230, ge=32)
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Vector store ------------------------------------------------------
    chroma_persist_directory: Path = Path("./data/chroma_db")
    chroma_collection_name: str = "printer_manual"

    # --- Documents ---------------------------------------------------------
    raw_documents_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")
    default_document_path: Path = Path("./data/raw/epson_l3310_users_guide.pdf")
    default_document_id: str = "epson_l3310_manual"
    default_document_name: str = "Epson L3310 Series User's Guide"
    default_manufacturer: str = "Epson"
    default_product_model: str = "L3310"
    max_upload_mb: int = Field(default=50, ge=1)
    ocr_enabled: bool = True

    # --- Chunking ----------------------------------------------------------
    chunk_target_tokens: int = Field(default=550, ge=50)
    chunk_max_tokens: int = Field(default=900, ge=100)
    chunk_overlap: int = Field(default=60, ge=0)
    h3_word_threshold: int = Field(default=400, ge=50)

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1)
    dense_candidates: int = Field(default=20, ge=1)
    bm25_candidates: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, ge=0)
    bm25_weight: float = Field(default=1.0, ge=0)
    rerank_candidates: int = Field(default=15, ge=1)
    min_rerank_score: float = -5.0  # below → off-topic; >= min+6 → high confidence
    min_dense_similarity: float = 0.25

    # --- Sessions ----------------------------------------------------------
    session_message_limit: int = Field(default=20, ge=2)
    session_expiry_days: int = Field(default=7, ge=1)
    session_history_token_budget: int = Field(default=1500, ge=200)
    database_url: str = "sqlite:///./data/sessions.db"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip().startswith("["):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator(
        "chroma_persist_directory",
        "raw_documents_dir",
        "processed_dir",
        "default_document_path",
    )
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @field_validator("database_url")
    @classmethod
    def _resolve_sqlite(cls, value: str) -> str:
        prefix = "sqlite:///"
        if value.startswith(prefix) and not value.startswith(prefix + "/") and ":memory:" not in value:
            return prefix + str((REPO_ROOT / value[len(prefix):]).resolve())
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS must be <= CHUNK_MAX_TOKENS")
        if self.chunk_overlap >= self.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_TARGET_TOKENS")
        if self.environment == "production" and not self.admin_api_key:
            raise ValueError("ADMIN_API_KEY is required in production")
        return self

    @property
    def llm_configured(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
