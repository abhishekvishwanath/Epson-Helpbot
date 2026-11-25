"""Ingest a printer manual into the vector store.

Usage (from repo root):
    backend/.venv/bin/python scripts/ingest_document.py                 # default manual (from .env)
    backend/.venv/bin/python scripts/ingest_document.py --force         # re-parse even if unchanged
    backend/.venv/bin/python scripts/ingest_document.py --rebuild       # drop collection, re-embed everything
    backend/.venv/bin/python scripts/ingest_document.py --pdf path.pdf --document-id epson_l3250_manual \\
        --name "Epson L3250 Series User's Guide" --manufacturer Epson --model L3250
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.ingestion.ingestion_service import DocumentSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--document-id")
    parser.add_argument("--name")
    parser.add_argument("--manufacturer")
    parser.add_argument("--model")
    parser.add_argument("--force", action="store_true", help="re-ingest even if the source is unchanged")
    parser.add_argument("--rebuild", action="store_true", help="drop the collection (e.g. after changing embedding model)")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING")

    if args.rebuild:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))
        if settings.chroma_collection_name in {c.name for c in client.list_collections()}:
            client.delete_collection(settings.chroma_collection_name)
            print(f"Dropped collection '{settings.chroma_collection_name}'")

    from app.services.container import get_ingestion_service

    service = get_ingestion_service()
    spec = service.default_spec()
    if args.pdf:
        spec = DocumentSpec(
            path=args.pdf.resolve(),
            document_id=args.document_id or args.pdf.stem,
            document_name=args.name or args.pdf.stem,
            manufacturer=args.manufacturer or "",
            product_model=args.model or "",
        )
    result = service.ingest(spec, force=args.force or args.rebuild)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
