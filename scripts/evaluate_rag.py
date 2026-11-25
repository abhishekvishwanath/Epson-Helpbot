"""RAG evaluation: chunking comparison, retrieval ablations and live answer quality.

    backend/.venv/bin/python scripts/evaluate_rag.py --retrieval        # no LLM calls
    backend/.venv/bin/python scripts/evaluate_rag.py --retrieval --llm  # + live answers & judge

Writes docs/evaluation_report.md and data/processed/eval_results.json.
Gold set: backend/tests/evaluation/eval_set.json (phrases are verified against the PDF first).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.evaluation import metrics as M  # noqa: E402

EVAL_SET = ROOT / "backend/tests/evaluation/eval_set.json"


# ---------------------------------------------------------------- dataset ---
def load_and_verify(pdf: Path) -> dict:
    import pymupdf

    data = json.loads(EVAL_SET.read_text())
    doc = pymupdf.open(pdf)
    problems = []
    for q in data["questions"]:
        page_text = " ".join(M.norm(doc[p - 1].get_text()) for p in q["pages"])
        missing = [ph for ph in q["phrases"] if M.norm(ph) not in page_text]
        if len(missing) == len(q["phrases"]):
            problems.append(f"{q['id']}: none of {q['phrases']} found on pages {q['pages']}")
    if problems:
        raise SystemExit("Gold set verification failed:\n" + "\n".join(problems))
    print(f"gold set verified: {len(data['questions'])} questions, all phrases present on their pages")
    return data


# --------------------------------------------------------------- retrieval ---
def build_strategies(settings: Settings, counter, embedder) -> dict[str, list]:
    from app.evaluation.baselines import fixed_size_chunks, page_texts, recursive_chunks
    from app.services.container import get_store

    pages = page_texts(settings.default_document_path)
    structure = get_store().all_chunks({"document_id": settings.default_document_id})
    if not structure:
        raise SystemExit("Index empty — run scripts/ingest_document.py first")
    return {
        "A. fixed-size (400 tok)": fixed_size_chunks(settings.default_document_id, pages, counter, 400),
        "B. recursive (550 tok, 60 overlap)": recursive_chunks(settings.default_document_id, pages, counter, 550, 60),
        "C. structure-based hierarchical": structure,
    }


def make_retriever(settings: Settings, chunks: list, embedder, reranker):
    from app.evaluation.inmemory_store import InMemoryStore
    from app.retrieval.corpus import ChunkCorpus
    from app.retrieval.dense_search import DenseSearcher
    from app.retrieval.retrieval_service import RetrievalService

    store = InMemoryStore(chunks, embedder)
    return RetrievalService(settings, ChunkCorpus(store, lambda: "static"), DenseSearcher(store, embedder), reranker), store


def run_retrieval_eval(data: dict, settings: Settings) -> dict:
    from app.ingestion.tokens import get_token_counter
    from app.retrieval.reranker import CrossEncoderReranker
    from app.services.container import get_embedder

    counter = get_token_counter(settings.embedding_model)
    embedder = get_embedder()
    reranker = CrossEncoderReranker(settings.reranker_model)
    strategies = build_strategies(settings, counter, embedder)

    def evaluate(retriever, **flags) -> dict:
        per_q, latencies = [], []
        for q in data["questions"]:
            t = time.perf_counter()
            res = retriever.retrieve(q["question"], expand=False, **flags)
            latencies.append((time.perf_counter() - t) * 1000)
            s = M.score([h.chunk for h in res.hits], set(q["pages"]), q["phrases"], k=settings.retrieval_top_k)
            per_q.append({"id": q["id"], "category": q["category"], **s.__dict__,
                          "top": [f"p{h.chunk.pdf_page_start}" + (f" {h.chunk.section_title}" if h.chunk.section_title else "") for h in res.hits[:3]]})
        keyword = [r for r in per_q if r["category"] == "keyword"]
        return {
            "recall@5": M.mean([r["recall_at_k"] for r in per_q]),
            "precision@5": M.mean([r["precision_at_k"] for r in per_q]),
            "mrr": M.mean([r["reciprocal_rank"] for r in per_q]),
            "context_precision": M.mean([r["context_precision"] for r in per_q]),
            "top1_page_hit": M.mean([r["top1_page_hit"] for r in per_q]),
            "keyword_recall@5": M.mean([r["recall_at_k"] for r in keyword]),
            "keyword_mrr": M.mean([r["reciprocal_rank"] for r in keyword]),
            "median_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1),
            "per_question": per_q,
        }

    results: dict = {"strategies": {}, "ablations": {}}
    for name, chunks in strategies.items():
        retriever, store = make_retriever(settings, chunks, embedder, reranker)
        tokens = [c.token_count for c in chunks]
        print(f"evaluating {name}: {len(chunks)} chunks / {store.windows} windows")
        out = evaluate(retriever)
        out["chunks"] = len(chunks)
        out["median_chunk_tokens"] = sorted(tokens)[len(tokens) // 2]
        results["strategies"][name] = out
        if name.startswith("C."):
            for label, flags in {
                "hybrid + rerank (production)": {},
                "hybrid, no rerank": {"use_reranker": False},
                "dense only + rerank": {"use_bm25": False},
                "BM25 only + rerank": {"use_dense": False},
                "dense only, no rerank": {"use_bm25": False, "use_reranker": False},
                "BM25 only, no rerank": {"use_dense": False, "use_reranker": False},
            }.items():
                print(f"  ablation: {label}")
                results["ablations"][label] = out if not flags else evaluate(retriever, **flags)
    return results


# --------------------------------------------------------------------- LLM ---
JUDGE_SYSTEM = """You are a strict fact-checker. You receive SOURCES from a printer manual and an ANSWER
written for a customer. Split the ANSWER into its individual factual claims and instructions (ignore
greetings, citation markers like [1], and statements that the manual does not cover something).
For each, decide if it is directly supported by the SOURCES (paraphrase is fine; added steps, causes,
names or numbers not in the SOURCES are unsupported). Return JSON only:
{"claims": [{"claim": "...", "supported": true|false}]}"""


def _with_backoff(fn, attempts: int = 6, wait: float = 25.0):
    from app.core.exceptions import LLMError

    for i in range(attempts):
        try:
            return fn()
        except LLMError:
            if i == attempts - 1:
                raise
            time.sleep(wait)


def run_llm_eval(data: dict, settings: Settings) -> dict:
    from app.core.database import Database
    from app.generation.llm_client import GroqLLMClient, parse_json_object
    from app.generation.response_validator import _NO_INFO_RE
    from app.graph.nodes import ChatNodes
    from app.graph.workflow import ChatWorkflow
    from app.ingestion.tokens import get_token_counter
    from app.memory.conversation_manager import SQLiteConversationMemory
    from app.memory.session_store import SessionStore
    from app.services.container import get_llm, get_retrieval_service

    if not settings.llm_configured:
        return {"skipped": "GROQ_API_KEY not configured"}
    retrieval = get_retrieval_service()
    counter = get_token_counter(settings.embedding_model)
    db = Database(f"sqlite:///{tempfile.mkdtemp()}/eval_sessions.db")
    db.create_all()
    store = SessionStore(db, settings.session_message_limit, settings.session_expiry_days)

    def workflow():
        # A fresh memory object per call: history must come from SQLite, not process state.
        return ChatWorkflow(ChatNodes(settings, retrieval, SQLiteConversationMemory(store), counter, get_llm()))

    def ask(message: str, session_id: str):
        for _ in range(6):
            res = workflow().run(message, session_id=session_id)
            if res.response_type != "error":
                return res
            time.sleep(25)  # all models rate-limited: wait for the per-minute quota to reset
        return res

    judge_model = settings.groq_fallback_models[0] if settings.groq_fallback_models else settings.groq_model
    judge = GroqLLMClient(settings.model_copy(update={"groq_model": judge_model, "groq_fallback_models": []}))

    def faithfulness(answer: str, chunk_ids: list[str]) -> dict:
        sources = "\n\n---\n\n".join(
            retrieval.corpus.get(cid).text_for_index for cid in chunk_ids if retrieval.corpus.get(cid)
        )
        resp = _with_backoff(lambda: judge.complete(
            JUDGE_SYSTEM, [{"role": "user", "content": f"SOURCES\n{sources}\n\nANSWER\n{answer}"}],
            max_tokens=1500, temperature=0.0, json_mode=True,
        ))
        claims = parse_json_object(resp.text).get("claims", [])
        supported = sum(1 for c in claims if c.get("supported"))
        return {"claims": len(claims), "supported": supported,
                "score": round(supported / len(claims), 3) if claims else 1.0,
                "unsupported": [c.get("claim") for c in claims if not c.get("supported")][:5]}

    def session():
        return store.create(settings.default_product_model).session_id

    answers = []
    prior = getattr(run_llm_eval, "prior", None)
    questions = [] if prior else data["questions"]
    for q in questions:
        res = ask(q["question"], session())
        gold = set(q["pages"])
        cited = [c for c in res.citations]
        correct = [c for c in cited if any(c.pdf_page <= p <= c.pdf_page_end for p in gold)]
        row = {
            "id": q["id"], "question": q["question"], "response_type": res.response_type, "model": res.model,
            "latency_ms": res.latency_ms, "citations": [f"p{c.pdf_page}" for c in cited],
            "any_citation_correct": bool(correct),
            "citation_precision": round(len(correct) / len(cited), 3) if cited else 0.0,
        }
        if res.response_type == "answer":
            row["faithfulness"] = faithfulness(res.answer, [s.chunk_id for s in res.retrieved_sources])
        answers.append(row)
        print(f"  {q['id']} {res.response_type} cites={row['citations']} ok={row['any_citation_correct']} "
              f"faith={row.get('faithfulness', {}).get('score')}")

    unsupported = []
    if prior:
        answers = prior["answers"]
        for u in prior["unsupported"]:
            u["declined"] = u["response_type"] == "refusal" or bool(_NO_INFO_RE.search(u["answer_excerpt"]))
        unsupported = prior["unsupported"]
    for u in ([] if prior else data["unsupported"]):
        res = ask(u["question"], session())
        declined = res.response_type == "refusal" or bool(_NO_INFO_RE.search(res.answer))
        row = {"id": u["id"], "question": u["question"], "response_type": res.response_type, "declined": declined,
               "answer_excerpt": res.answer[:240]}
        if res.response_type == "answer":
            row["faithfulness"] = faithfulness(res.answer, [s.chunk_id for s in res.retrieved_sources])
        unsupported.append(row)
        print(f"  {u['id']} {res.response_type} declined={declined}")

    ambiguous = prior["ambiguous"] if prior else []
    for a in ([] if prior else data["ambiguous"]):
        res = ask(a["question"], session())
        ok = res.response_type == "clarification" or res.answer.strip().endswith("?")
        ambiguous.append({"id": a["id"], "question": a["question"], "response_type": res.response_type,
                          "asked_clarification": ok, "answer_excerpt": res.answer[:200]})

    # Follow-ups: first turns of every conversation, then second turns (interleaved
    # sessions also exercise isolation); memory is reloaded from SQLite each turn.
    sessions = {f["id"]: session() for f in data["followups"]}
    for f in data["followups"]:
        ask(f["turns"][0], sessions[f["id"]])
    followups = []
    for f in data["followups"]:
        res = ask(f["turns"][1], sessions[f["id"]])
        pages_ok = any(any(c.pdf_page <= p <= c.pdf_page_end for p in f["expect_pages"]) for c in res.citations)
        terms_ok = any(t in (res.standalone_query + " " + res.answer).lower() for t in f["expect_terms"])
        stored = store.get(sessions[f["id"]])
        own_turns = [m.content for m in stored.messages if m.role == "user"]
        foreign = [t for other in data["followups"] if other["id"] != f["id"] for t in other["turns"]]
        followups.append({"id": f["id"], "turns": f["turns"], "standalone_query": res.standalone_query,
                          "citations": [f"p{c.pdf_page}" for c in res.citations], "resolved": pages_ok and terms_ok,
                          "memory_persisted": own_turns == f["turns"] and len(stored.messages) == 4,
                          "isolated": not any(t in own_turns for t in foreign)})
        print(f"  {f['id']} -> {res.standalone_query!r} resolved={pages_ok and terms_ok}")

    answered = [a for a in answers if a["response_type"] == "answer"]
    faith = [a["faithfulness"]["score"] for a in answered if "faithfulness" in a]
    return {
        "answers": answers,
        "unsupported": unsupported,
        "ambiguous": ambiguous,
        "followups": followups,
        "summary": {
            "questions": len(answers),
            "answered": len(answered),
            "faithfulness_mean": M.mean(faith),
            "fully_faithful_answers": f"{sum(1 for s in faith if s == 1.0)}/{len(faith)}",
            "page_accuracy_any_citation": M.mean([float(a["any_citation_correct"]) for a in answered]),
            "page_accuracy_citation_precision": M.mean([a["citation_precision"] for a in answered]),
            "unsupported_declined": f"{sum(u['declined'] for u in unsupported)}/{len(unsupported)}",
            "ambiguous_clarified": f"{sum(a['asked_clarification'] for a in ambiguous)}/{len(ambiguous)}",
            "followups_resolved": f"{sum(f['resolved'] for f in followups)}/{len(followups)}",
            "memory_persisted": f"{sum(f['memory_persisted'] for f in followups)}/{len(followups)}",
            "sessions_isolated": f"{sum(f['isolated'] for f in followups)}/{len(followups)}",
            "median_latency_ms": sorted(a["latency_ms"] for a in answers)[len(answers) // 2],
            "judge_model": judge_model,
            "answer_model": settings.groq_model,
        },
    }


# ------------------------------------------------------------------ report ---
def write_report(results: dict, path: Path) -> None:
    lines = ["# Evaluation report", "",
             "Generated by `scripts/evaluate_rag.py` on the Epson L3310 Series User's Guide. "
             "All numbers below are measured, not estimated.", ""]
    r = results.get("retrieval")
    if r:
        lines += ["## Chunking strategy comparison (same hybrid retrieval: BM25 + dense + RRF + cross-encoder, top-5)", "",
                  "Relevance = chunk overlaps the gold pages **and** contains a gold key phrase "
                  f"({len(results['dataset']['questions'])} questions).", "",
                  "| Strategy | Chunks | Median tokens | Recall@5 | Precision@5 | MRR | Context precision | Top-1 page hit | Keyword Recall@5 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, s in r["strategies"].items():
            lines.append(f"| {name} | {s['chunks']} | {s['median_chunk_tokens']} | {s['recall@5']} | {s['precision@5']} | "
                         f"{s['mrr']} | {s['context_precision']} | {s['top1_page_hit']} | {s['keyword_recall@5']} |")
        lines += ["", "## Retrieval ablations (structure-based chunks)", "",
                  "| Configuration | Recall@5 | Precision@5 | MRR | Context precision | Median latency (ms) |",
                  "|---|---|---|---|---|---|"]
        for name, s in r["ablations"].items():
            lines.append(f"| {name} | {s['recall@5']} | {s['precision@5']} | {s['mrr']} | {s['context_precision']} | {s['median_latency_ms']} |")
        prod = r["strategies"]["C. structure-based hierarchical"]["per_question"]
        misses = [q for q in prod if not q["recall_at_k"]]
        lines += ["", f"Misses (structure-based, production config): {len(misses)}"]
        for q in misses:
            lines.append(f"- `{q['id']}` top-3: {q['top']}")
    llm = results.get("llm")
    if llm and "summary" in llm:
        s = llm["summary"]
        lines += ["", "## Live answer evaluation", "",
                  f"Answer model `{s['answer_model']}`; faithfulness judged claim-by-claim by `{s['judge_model']}` "
                  "against the exact chunks given to the answer model (LLM-as-judge, not human-labelled).", "",
                  "| Metric | Result |", "|---|---|"]
        for key in ("questions", "answered", "faithfulness_mean", "fully_faithful_answers", "page_accuracy_any_citation",
                    "page_accuracy_citation_precision", "unsupported_declined", "ambiguous_clarified",
                    "followups_resolved", "memory_persisted", "sessions_isolated", "median_latency_ms"):
            lines.append(f"| {key.replace('_', ' ')} | {s[key]} |")
        lines += ["", "### Unsupported claims flagged by the judge"]
        for a in llm["answers"]:
            for claim in a.get("faithfulness", {}).get("unsupported", []):
                lines.append(f"- `{a['id']}` {claim}")
        lines += ["", "### Unsupported / off-topic questions"]
        for u in llm["unsupported"]:
            lines.append(f"- `{u['id']}` {u['question']} → **{u['response_type']}**, declined={u['declined']}: "
                         f"{u['answer_excerpt'][:160].replace(chr(10), ' ')}…")
        lines += ["", "### Follow-ups"]
        for f in llm["followups"]:
            lines.append(f"- `{f['id']}` {' → '.join(f['turns'])} ⇒ rewritten as “{f['standalone_query']}”, "
                         f"cited {f['citations']}, resolved={f['resolved']}")
    elif llm:
        lines += ["", f"## Live answer evaluation\n\nSkipped: {llm.get('skipped')}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true", help="chunking comparison + ablations (no LLM)")
    parser.add_argument("--llm", action="store_true", help="live answers, faithfulness judge, memory checks")
    parser.add_argument("--followups-only", action="store_true",
                        help="with --llm: reuse stored answers, rerun only follow-up/memory checks")
    args = parser.parse_args()
    configure_logging("ERROR")
    settings = get_settings()
    data = load_and_verify(settings.default_document_path)
    out = ROOT / "data/processed/eval_results.json"
    results = json.loads(out.read_text()) if out.is_file() else {}
    results["dataset"] = data
    if args.retrieval:
        results["retrieval"] = run_retrieval_eval(data, settings)
    if args.llm:
        if args.followups_only and "llm" in results:
            run_llm_eval.prior = results["llm"]
        results["llm"] = run_llm_eval(data, settings)
    out.write_text(json.dumps(results, indent=1, default=str))
    write_report(results, ROOT / "docs/evaluation_report.md")
    print(f"wrote {out} and docs/evaluation_report.md")


if __name__ == "__main__":
    main()
