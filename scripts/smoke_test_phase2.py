"""Phase 2 end-to-end smoke test.

Runs the full Phase 2 pipeline against the AAPL FY2025 10-K:
  download → hierarchical chunking → Chroma → LangGraph (retrieve + rerank +
  confidence gate) → Presidio PII check → answer with citations.

Also exercises:
  - Confidence-gate refusal on an unanswerable question
  - PII detection on a question containing a fake email address

Prerequisites
-------------
1. Ollama running with nomic-embed-text pulled.
2. A valid .env with GROQ_API_KEY.

Run with:
    uv run python scripts/smoke_test_phase2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from quaestor.config import settings

DIVIDER = "─" * 70


def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def check(label: str, condition: bool) -> None:
    icon = "✅" if condition else "✗ "
    print(f"  {icon}  {label}")
    if not condition:
        global _all_ok
        _all_ok = False


_all_ok = True


def main() -> None:
    global _all_ok

    print("\n" + "═" * 70)
    print("  QUAESTOR — Phase 2 Smoke Test")
    print("═" * 70)
    print(f"  LLM provider : {settings.llm_provider.value}")
    print(f"  Groq model   : {settings.groq_model}")
    print(f"  Embeddings   : {settings.ollama_embedding_model}")
    print(f"  Top-k        : {settings.retrieval_top_k}")

    # ------------------------------------------------------------------
    # 1. Download (idempotent — reuses Phase 1 download if present)
    # ------------------------------------------------------------------
    section("STEP 1 — Download AAPL 10-K (idempotent)")
    t0 = time.time()
    from quaestor.ingestion.loader import download_sec_filings, load_edgar_submission

    filing_dir = download_sec_filings(
        ticker="AAPL",
        form="10-K",
        limit=1,
        download_dir=settings.data_raw_dir / "sec_filings",
    )
    accession_dirs = [d for d in filing_dir.iterdir() if d.is_dir()]
    check("Filing directory found", len(accession_dirs) > 0)
    if not accession_dirs:
        sys.exit(1)
    submission_dir = accession_dirs[0]
    print(f"  Accession : {submission_dir.name}  ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Load
    # ------------------------------------------------------------------
    section("STEP 2 — Load & extract 10-K content")
    t0 = time.time()
    docs = load_edgar_submission(submission_dir)
    check("Docs loaded", len(docs) > 0)
    print(f"  {len(docs)} segment(s) in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Hierarchical chunking (Phase 2 strategy)
    # ------------------------------------------------------------------
    section("STEP 3 — Hierarchical chunking (parent 1024 / child 256)")
    t0 = time.time()
    from quaestor.ingestion.chunker import chunk_documents

    chunks = chunk_documents(
        docs,
        strategy="hierarchical",
        parent_chunk_size=1024,
        child_chunk_size=256,
    )
    check("Chunks produced", len(chunks) > 0)
    check("All chunks are child level",
          all(c.metadata.get("chunk_level") == "child" for c in chunks))
    check("parent_content present in metadata",
          all("parent_content" in c.metadata for c in chunks))
    print(f"  {len(docs)} segment(s) → {len(chunks)} child chunks in {time.time() - t0:.1f}s")
    print(f"  Sample metadata keys: {list(chunks[0].metadata.keys())}")

    # ------------------------------------------------------------------
    # 4. Index → ChromaDB
    # ------------------------------------------------------------------
    section("STEP 4 — Embed & index with nomic-embed-text → ChromaDB")
    t0 = time.time()
    print("  (May take 30–120 s depending on filing size…)")
    from langchain_ollama import OllamaEmbeddings

    from quaestor.ingestion.indexer import build_index

    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
    vector_store = build_index(chunks, embeddings=embeddings)
    indexed_count = vector_store._collection.count()
    check("Chunks indexed", indexed_count > 0)
    print(f"  Indexed {indexed_count} chunks in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 5. Build Phase 2 pipeline: reranker + LangGraph
    # ------------------------------------------------------------------
    section("STEP 5 — Build Phase 2 pipeline (reranker + LangGraph)")
    t0 = time.time()
    from quaestor.retrieval.reranker import _default_cross_encoder
    from quaestor.retrieval.graph import build_rag_graph, run_rag_graph

    cross_encoder = _default_cross_encoder()
    graph = build_rag_graph(
        vector_store=vector_store,
        cross_encoder=cross_encoder,
        confidence_threshold=-5.0,  # very permissive: test Q&A ability, not the gate
    )
    print(f"  Graph compiled in {time.time() - t0:.1f}s")
    check("Graph built", graph is not None)

    # ------------------------------------------------------------------
    # 6. Q&A — confident questions
    # ------------------------------------------------------------------
    questions = [
        "What was Apple's total net sales in the most recent fiscal year?",
        "What are the primary risk factors Apple discloses?",
        "What was Apple's net income in fiscal year 2025 compared to 2024?",
    ]

    section("STEP 6 — Q&A (confident questions)")
    for i, q in enumerate(questions, 1):
        print(f"\n  Q{i}: {q}")
        t0 = time.time()
        try:
            result = run_rag_graph(graph, q)
            elapsed = time.time() - t0
            check(f"Q{i} answered (not refused)", not result.refused)
            check(f"Q{i} has sources", len(result.sources) > 0)
            print(f"  A : {result.answer[:300]}")
            print(f"  Sources      : {result.sources}")
            print(f"  Top score    : {result.top_score:.3f}")
            print(f"  Time         : {elapsed:.1f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            _all_ok = False

    # ------------------------------------------------------------------
    # 7. Confidence-gate refusal test
    # ------------------------------------------------------------------
    section("STEP 7 — Confidence-gate refusal (unanswerable question)")

    # Build a very high-threshold graph that will always refuse
    strict_graph = build_rag_graph(
        vector_store=vector_store,
        cross_encoder=cross_encoder,
        confidence_threshold=999.0,  # impossibly high → always refuses
    )
    unanswerable = "What is Apple's projected revenue for fiscal year 2030?"
    print(f"  Q: {unanswerable}")
    try:
        result = run_rag_graph(strict_graph, unanswerable)
        check("Refusal triggered (refused=True)", result.refused is True)
        print(f"  Refusal text : {result.answer[:120]}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        _all_ok = False

    # ------------------------------------------------------------------
    # 8. PII guardrail test
    # ------------------------------------------------------------------
    section("STEP 8 — PII guardrail (input screening)")
    from quaestor.guardrails.input import detect_pii, redact_pii
    from quaestor.guardrails.input import _default_analyzer, _default_anonymizer

    pii_question = "My name is John Smith. What was Apple's revenue?"
    print(f"  Input  : {pii_question!r}")
    analyzer = _default_analyzer()
    anonymizer = _default_anonymizer()
    entities = detect_pii(pii_question, analyzer=analyzer)
    check("PII detected (PERSON)", any(e.entity_type == "PERSON" for e in entities))
    redacted = redact_pii(pii_question, analyzer=analyzer, anonymizer=anonymizer)
    check("PII redacted in output", "John Smith" not in redacted.redacted_text)
    print(f"  Redacted: {redacted.redacted_text!r}")
    print(f"  Entities: {[e.entity_type for e in entities]}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "═" * 70)
    status = "✅  ALL PHASE 2 CHECKS PASSED" if _all_ok else "⚠️   COMPLETED WITH ERRORS"
    print(f"  {status}")
    print("═" * 70 + "\n")

    if not _all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
