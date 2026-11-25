"""Debug hybrid retrieval from the command line.

    backend/.venv/bin/python scripts/search_manual.py "printer prints blank pages"
    backend/.venv/bin/python scripts/search_manual.py "paper jam" --no-rerank --model L3310
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.logging import configure_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--model", default=None, help="product model filter")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    configure_logging("WARNING")

    from app.services.container import get_retrieval_service

    result = get_retrieval_service().retrieve(
        args.query, product_model=args.model, top_k=args.top_k, use_reranker=not args.no_rerank
    )
    print(f"confidence={result.confidence} context_tokens={result.context_tokens} timings={result.timings_ms}")
    for item in result.context:
        c = item.chunk
        rr = f"{item.rerank_score:6.2f}" if item.rerank_score is not None else "   n/a"
        print(
            f"#{item.rank or '-':<2} rerank={rr} dense_rank={item.dense_rank} bm25_rank={item.bm25_rank} "
            f"[{item.source}] p{c.pdf_page_start}-{c.pdf_page_end} {c.chunk_type.value}: {' > '.join(c.heading_path)}"
        )


if __name__ == "__main__":
    main()
