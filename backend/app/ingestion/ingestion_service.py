"""End-to-end ingestion: PDF → structure → chunks → embeddings → ChromaDB.

Guarantees:
* unchanged sources (same hash + same embedding fingerprint) are skipped;
* chunks whose content hash is unchanged reuse their stored embeddings;
* the new version is written before stale records are deleted, so the index
  never goes empty during a re-ingest;
* every run is recorded as an ingestion job with status and statistics.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.chunker import ChunkingConfig, HierarchicalChunker
from app.ingestion.document_builder import file_sha256, open_pdf, parse_document, save_parsed
from app.ingestion.image_extractor import VisionDescriber
from app.ingestion.metadata import Chunk
from app.ingestion.registry import DocumentRegistry
from app.ingestion.tokens import get_token_counter
from app.vectorstore.chroma_service import ChromaService

logger = get_logger(__name__)


@dataclass
class DocumentSpec:
    path: Path
    document_id: str
    document_name: str
    manufacturer: str
    product_model: str


@dataclass
class IngestionResult:
    document_id: str
    status: str  # succeeded | skipped | failed
    document_version: str = ""
    chunks: int = 0
    embedded_chunks: int = 0
    reused_chunks: int = 0
    windows: int = 0
    stale_removed: int = 0
    seconds: float = 0.0
    message: str = ""
    warnings: list[str] = field(default_factory=list)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        store: ChromaService,
        embedder: EmbeddingService,
        describer: VisionDescriber | None = None,
        on_index_changed: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.store = store
        self.embedder = embedder
        self.describer = describer
        self.on_index_changed = on_index_changed
        self._lock = threading.Lock()

    def default_spec(self) -> DocumentSpec:
        s = self.settings
        return DocumentSpec(
            path=s.default_document_path,
            document_id=s.default_document_id,
            document_name=s.default_document_name,
            manufacturer=s.default_manufacturer,
            product_model=s.default_product_model,
        )

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def ingest(self, spec: DocumentSpec, *, force: bool = False) -> IngestionResult:
        if not self._lock.acquire(blocking=False):
            raise IngestionError("Another ingestion is already running")
        job_id = self.registry.start_job(spec.document_id)
        started = time.perf_counter()
        try:
            result = self._ingest(spec, force=force)
            result.seconds = round(time.perf_counter() - started, 2)
            self.registry.finish_job(job_id, result.status, result.message, asdict(result))
            if result.status == "succeeded" and self.on_index_changed:
                self.on_index_changed()
            return result
        except Exception as exc:
            self.registry.set_status(spec.document_id, "failed", str(exc))
            self.registry.finish_job(job_id, "failed", str(exc))
            logger.exception("ingestion_failed", extra={"document_id": spec.document_id})
            raise
        finally:
            self._lock.release()

    def _ingest(self, spec: DocumentSpec, *, force: bool) -> IngestionResult:
        open_pdf(spec.path).close()  # validates the file before any work
        source_hash = file_sha256(spec.path)
        version = source_hash[:12]
        model = self.embedder.model_name
        existing = self.registry.get(spec.document_id)
        if (
            not force
            and existing is not None
            and existing.status == "ready"
            and existing.source_hash == source_hash
            and existing.embedding_model == model
            and self.store.count_chunks(spec.document_id) == existing.chunk_count
        ):
            return IngestionResult(spec.document_id, "skipped", version, existing.chunk_count, message="unchanged source")

        self.registry.upsert(
            document_id=spec.document_id,
            document_name=spec.document_name,
            manufacturer=spec.manufacturer,
            product_model=spec.product_model,
            source_path=str(spec.path),
            source_hash=source_hash,
            document_version=version,
            embedding_model=model,
            status="processing",
            error="",
        )

        out_dir = self.settings.processed_dir / spec.document_id
        parsed = parse_document(
            spec.path,
            document_id=spec.document_id,
            document_name=spec.document_name,
            manufacturer=spec.manufacturer,
            product_model=spec.product_model,
            figures_dir=out_dir / "figures",
            ocr_enabled=self.settings.ocr_enabled,
            describer=self.describer,
        )
        save_parsed(parsed, out_dir)

        s = self.settings
        chunker = HierarchicalChunker(
            get_token_counter(s.embedding_model),
            ChunkingConfig(s.chunk_target_tokens, s.chunk_max_tokens, s.chunk_overlap, s.h3_word_threshold),
        )
        chunks = chunker.chunk(parsed)
        if not chunks:
            raise IngestionError("No content could be extracted from the document")
        (out_dir / "chunks.jsonl").write_text("\n".join(c.model_dump_json() for c in chunks))

        stats = self._index(spec.document_id, chunks)
        stale = self.store.delete_stale(spec.document_id, version)
        self.registry.upsert(
            document_id=spec.document_id,
            status="ready",
            chunk_count=len(chunks),
            page_count=parsed.page_count,
        )
        logger.info("ingestion_complete", extra={"document_id": spec.document_id, **stats, "stale_removed": stale})
        return IngestionResult(
            spec.document_id,
            "succeeded",
            version,
            chunks=len(chunks),
            stale_removed=stale,
            warnings=parsed.warnings,
            **stats,
        )

    def _index(self, document_id: str, chunks: list[Chunk]) -> dict[str, int]:
        # Vectors are reused for unchanged text; a full re-embed is done by
        # rebuilding the collection (required when the embedding model changes).
        previous = self.store.existing_hashes(document_id)
        embedded = reused = windows_total = 0
        pending: list[tuple[Chunk, list[str]]] = []
        for chunk in chunks:
            if previous.get(chunk.chunk_id) == chunk.content_hash:
                docs, embs = self.store.window_records(chunk.chunk_id)
                if len(embs):
                    # Same text: keep vectors, refresh metadata to the new version.
                    docs[0] = chunk.text_for_index
                    self.store.upsert_chunk(chunk, docs, embs)
                    reused += 1
                    windows_total += len(docs)
                    continue
            pending.append((chunk, self.embedder.windows(chunk.context_header, chunk.content)))

        flat = [w for _, ws in pending for w in ws]
        vectors = self.embedder.embed(flat)
        offset = 0
        for chunk, ws in pending:
            self.store.upsert_chunk(chunk, ws, vectors[offset : offset + len(ws)])
            offset += len(ws)
            embedded += 1
            windows_total += len(ws)
        return {"embedded_chunks": embedded, "reused_chunks": reused, "windows": windows_total}

    def delete(self, document_id: str) -> int:
        with self._lock:
            removed = self.store.delete_document(document_id)
            self.registry.delete(document_id)
        if self.on_index_changed:
            self.on_index_changed()
        return removed
