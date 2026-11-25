"""Inspect hierarchical chunks for the default (or a given) document.

Usage (from repo root):
    backend/.venv/bin/python scripts/inspect_chunks.py               # stats
    backend/.venv/bin/python scripts/inspect_chunks.py --grep "paper jam"
    backend/.venv/bin/python scripts/inspect_chunks.py --show <chunk_id>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.metadata import Chunk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--grep")
    parser.add_argument("--show")
    args = parser.parse_args()

    s = get_settings()
    doc_id = args.document_id or s.default_document_id
    path = s.processed_dir / doc_id / "chunks.jsonl"
    if not path.is_file():
        sys.exit(f"No chunks at {path}. Run scripts/ingest_document.py first.")
    chunks = [Chunk.model_validate(json.loads(line)) for line in path.read_text().splitlines()]

    if args.show:
        c = next((c for c in chunks if c.chunk_id == args.show), None)
        if not c:
            sys.exit("chunk not found")
        print(json.dumps(c.chroma_metadata(), indent=1))
        print("\n" + c.text_for_index)
        return
    if args.grep:
        needle = args.grep.lower()
        for c in chunks:
            if needle in c.text_for_index.lower():
                print(f"{c.chunk_id}  p{c.pdf_page_start}-{c.pdf_page_end}  {c.chunk_type.value:20} {' > '.join(c.heading_path)}")
        return

    tokens = [c.token_count for c in chunks]
    print(f"document: {doc_id}  chunks: {len(chunks)}")
    print(f"tokens  min {min(tokens)}  median {statistics.median(tokens):.0f}  mean {statistics.mean(tokens):.0f}  max {max(tokens)}")
    buckets = Counter(min(t // 100 * 100, 900) for t in tokens)
    print("token histogram:", ", ".join(f"{k}-{k + 99}: {buckets[k]}" for k in sorted(buckets)))
    print("types:", dict(Counter(c.chunk_type.value for c in chunks)))
    print("fallback-split chunks:", sum(c.fallback_split for c in chunks))
    print("with printed page labels:", sum(1 for c in chunks if c.page_start))


if __name__ == "__main__":
    main()
