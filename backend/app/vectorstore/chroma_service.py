"""Persistent ChromaDB storage for chunk windows.

Record layout (one Chroma record per embedding window):

* id ``<chunk_id>#w<i>``; window 0 is the *primary* record and stores the full
  chunk text as its document plus all chunk metadata. It is the canonical copy
  from which the BM25 corpus and context expansion are built, so keyword and
  dense retrieval always see exactly the same chunks.
* windows ``i > 0`` store only the window text and the identifying metadata.

The collection metadata records the embedding model fingerprint; opening it
with a different model raises rather than silently mixing vector spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.ingestion.metadata import Chunk, ChunkType

logger = get_logger(__name__)


class EmbeddingMismatchError(AppError):
    status_code = 409
    code = "embedding_model_mismatch"


@dataclass
class DenseHit:
    chunk_id: str
    similarity: float  # cosine similarity of the best window
    window_index: int


def chunk_from_record(document: str, meta: dict) -> Chunk:
    header = meta.get("context_header", "")
    content = document[len(header):].lstrip("\n") if header and document.startswith(header) else document
    return Chunk(
        chunk_id=meta["chunk_id"],
        chunk_index=int(meta["chunk_index"]),
        document_id=meta["document_id"],
        document_name=meta["document_name"],
        document_version=meta["document_version"],
        manufacturer=meta.get("manufacturer", ""),
        product_model=meta.get("product_model", ""),
        parent_section=meta.get("parent_section", ""),
        subsection=meta.get("subsection", ""),
        topic=meta.get("topic", ""),
        heading_path=[p for p in meta.get("heading_path", "").split(" > ") if p],
        section_title=meta.get("section_title", ""),
        parent_id=meta.get("parent_id", ""),
        subsection_id=meta.get("subsection_id", ""),
        topic_id=meta.get("topic_id", ""),
        section_id=meta.get("section_id", ""),
        prev_chunk_id=meta.get("prev_chunk_id") or None,
        next_chunk_id=meta.get("next_chunk_id") or None,
        chunk_type=ChunkType(meta.get("chunk_type", ChunkType.GENERAL.value)),
        has_procedure=bool(meta.get("has_procedure")),
        has_warning=bool(meta.get("has_warning")),
        has_table=bool(meta.get("has_table")),
        pdf_page_start=int(meta["pdf_page_start"]),
        pdf_page_end=int(meta["pdf_page_end"]),
        page_start=meta.get("page_start") or None,
        page_end=meta.get("page_end") or None,
        token_count=int(meta.get("token_count", 0)),
        word_count=int(meta.get("word_count", 0)),
        fallback_split=bool(meta.get("fallback_split")),
        split_part=int(meta.get("split_part", 1)),
        split_total=int(meta.get("split_total", 1)),
        context_header=header,
        content=content,
        content_hash=meta.get("content_hash", ""),
        figure_ids=[f for f in meta.get("figure_ids", "").split(",") if f],
        cross_references=[x for x in meta.get("cross_references", "").split("\n") if x],
    )


class ChromaService:
    def __init__(self, persist_dir: Path, collection_name: str, fingerprint: dict[str, str | int]) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = collection_name
        self.fingerprint = fingerprint
        self.collection = self._open()

    def _open(self):
        existing = {c.name for c in self.client.list_collections()}
        if self.collection_name in existing:
            col = self.client.get_collection(self.collection_name, embedding_function=None)
            stored = {k: (col.metadata or {}).get(k) for k in self.fingerprint}
            if stored != self.fingerprint:
                raise EmbeddingMismatchError(
                    f"Collection '{self.collection_name}' was built with {stored}, but the app is configured "
                    f"for {self.fingerprint}. Re-run ingestion with --rebuild to re-embed all documents."
                )
            return col
        return self.client.create_collection(
            self.collection_name,
            metadata=dict(self.fingerprint),
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def reset(self) -> None:
        """Drop and recreate the collection (used for a full rebuild)."""
        if self.collection_name in {c.name for c in self.client.list_collections()}:
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            self.collection_name,
            metadata=dict(self.fingerprint),
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    # ------------------------------------------------------------------ write
    def upsert_chunk(self, chunk: Chunk, windows: list[str], embeddings: np.ndarray) -> None:
        base = chunk.chroma_metadata()
        base["context_header"] = chunk.context_header
        base["cross_references"] = "\n".join(chunk.cross_references)
        ids, docs, metas = [], [], []
        for i, window in enumerate(windows):
            ids.append(f"{chunk.chunk_id}#w{i}")
            if i == 0:
                docs.append(chunk.text_for_index)
                metas.append({**base, "window_index": 0, "is_primary": True})
            else:
                docs.append(window)
                metas.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_version": chunk.document_version,
                    "product_model": chunk.product_model,
                    "manufacturer": chunk.manufacturer,
                    "window_index": i,
                    "is_primary": False,
                })
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings.tolist())

    def delete_stale(self, document_id: str, keep_version: str) -> int:
        stale = self.collection.get(
            where={"$and": [{"document_id": document_id}, {"document_version": {"$ne": keep_version}}]}, include=[]
        )["ids"]
        if stale:
            self.collection.delete(ids=stale)
        return len(stale)

    def delete_document(self, document_id: str) -> int:
        ids = self.collection.get(where={"document_id": document_id}, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------- read
    def existing_hashes(self, document_id: str) -> dict[str, str]:
        res = self.collection.get(
            where={"$and": [{"document_id": document_id}, {"is_primary": True}]}, include=["metadatas"]
        )
        return {m["chunk_id"]: m["content_hash"] for m in res["metadatas"]}

    def window_records(self, chunk_id: str) -> tuple[list[str], np.ndarray]:
        res = self.collection.get(where={"chunk_id": chunk_id}, include=["documents", "embeddings", "metadatas"])
        order = sorted(range(len(res["ids"])), key=lambda i: res["metadatas"][i]["window_index"])
        docs = [res["documents"][i] for i in order]
        embs = np.asarray([res["embeddings"][i] for i in order], dtype=np.float32)
        return docs, embs

    def all_chunks(self, where: dict | None = None) -> list[Chunk]:
        clause = {"is_primary": True} if not where else {"$and": [{"is_primary": True}, where]}
        res = self.collection.get(where=clause, include=["documents", "metadatas"])
        chunks = [chunk_from_record(d, m) for d, m in zip(res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: (c.document_id, c.chunk_index))

    def count_chunks(self, document_id: str | None = None) -> int:
        clause = {"is_primary": True} if document_id is None else {"$and": [{"is_primary": True}, {"document_id": document_id}]}
        return len(self.collection.get(where=clause, include=[])["ids"])

    def query(self, embedding: np.ndarray, n_chunks: int, where: dict | None = None) -> list[DenseHit]:
        total = self.collection.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(n_chunks * 4, total),
            where=where or None,
            include=["distances", "metadatas"],
        )
        best: dict[str, DenseHit] = {}
        for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
            sim = 1.0 - float(dist)
            cid = meta["chunk_id"]
            if cid not in best or sim > best[cid].similarity:
                best[cid] = DenseHit(chunk_id=cid, similarity=sim, window_index=int(meta["window_index"]))
        return sorted(best.values(), key=lambda h: h.similarity, reverse=True)[:n_chunks]
