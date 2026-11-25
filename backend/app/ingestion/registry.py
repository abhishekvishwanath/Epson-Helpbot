"""Document registry and ingestion-job tracking (SQLite via SQLAlchemy)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, Database


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_name: Mapped[str] = mapped_column(String(256))
    manufacturer: Mapped[str] = mapped_column(String(128), default="")
    product_model: Mapped[str] = mapped_column(String(128), index=True, default="")
    source_path: Mapped[str] = mapped_column(Text)
    source_hash: Mapped[str] = mapped_column(String(64))
    document_version: Mapped[str] = mapped_column(String(32))
    embedding_model: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|processing|ready|failed
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running")  # running|succeeded|skipped|failed
    message: Mapped[str] = mapped_column(Text, default="")
    stats_json: Mapped[str] = mapped_column(Text, default="{}")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def stats(self) -> dict:
        return json.loads(self.stats_json or "{}")


class DocumentRegistry:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, document_id: str) -> DocumentRecord | None:
        with self.db.session() as s:
            return s.get(DocumentRecord, document_id)

    def list(self) -> list[DocumentRecord]:
        with self.db.session() as s:
            return list(s.scalars(select(DocumentRecord).order_by(DocumentRecord.document_id)))

    def upsert(self, **fields) -> DocumentRecord:
        with self.db.session() as s:
            record = s.get(DocumentRecord, fields["document_id"])
            if record is None:
                record = DocumentRecord(**fields)
                s.add(record)
            else:
                for key, value in fields.items():
                    setattr(record, key, value)
            s.flush()
            return record

    def set_status(self, document_id: str, status: str, error: str = "") -> None:
        with self.db.session() as s:
            record = s.get(DocumentRecord, document_id)
            if record:
                record.status, record.error = status, error

    def delete(self, document_id: str) -> None:
        with self.db.session() as s:
            record = s.get(DocumentRecord, document_id)
            if record:
                s.delete(record)

    def signature(self) -> str:
        """Changes whenever the set of indexed (ready) document versions changes."""
        ready = [f"{r.document_id}@{r.document_version}" for r in self.list() if r.status == "ready"]
        return "|".join(sorted(ready))

    # -- jobs -------------------------------------------------------------
    def start_job(self, document_id: str) -> str:
        with self.db.session() as s:
            job = IngestionJob(document_id=document_id)
            s.add(job)
            s.flush()
            return job.job_id

    def finish_job(self, job_id: str, status: str, message: str = "", stats: dict | None = None) -> None:
        with self.db.session() as s:
            job = s.get(IngestionJob, job_id)
            if job:
                job.status, job.message = status, message
                job.stats_json = json.dumps(stats or {})
                job.finished_at = utcnow()

    def recent_jobs(self, limit: int = 10) -> list[IngestionJob]:
        with self.db.session() as s:
            return list(s.scalars(select(IngestionJob).order_by(IngestionJob.started_at.desc()).limit(limit)))
