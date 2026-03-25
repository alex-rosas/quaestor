"""Phase 1 end-to-end smoke test.

Downloads AAPL 10-K from SEC EDGAR, indexes it with nomic-embed-text,
asks three questions via Groq, and prints the answers with citations.

Run with:
    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quaestor.config import settings
from quaestor.generation.chain import build_chain
from quaestor.ingestion.chunker import chunk_documents
from quaestor.ingestion.indexer import build_index
from quaestor.ingestion.loader import download_sec_filings, load_edgar_submission

DIVIDER = "─" * 70


def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def main() -> None:
    print("\n" + "═" * 70)
    print("  QUAESTOR — Phase 1 Smoke Test")
    print("═" * 70)
    print(f"  LLM provider : {settings.llm_provider.value}")
    print(f"  Groq model   : {settings.groq_model}")
    print(f"  Embeddings   : {settings.ollama_embedding_model} @ {settings.ollama_base_url}")
    print(f"  Chunk size   : {settings.chunk_size} / overlap {settings.chunk_overlap}")
    print(f"  Top-k        : {settings.retrieval_top_k}")

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    section("STEP 1 — Download AAPL 10-K from SEC EDGAR")
    t0 = time.time()
    filing_dir = download_sec_filings(
        ticker="AAPL",
        form="10-K",
        limit=1,
        download_dir=settings.data_raw_dir / "sec_filings",
    )
    print(f"  Saved to : {filing_dir}")

    # Find the accession sub-directory (e.g. 0000320193-25-000079/)
    accession_dirs = [d for d in filing_dir.iterdir() if d.is_dir()]
    if not accession_dirs:
        print("  ✗ No accession directories found.")
        sys.exit(1)
    submission_dir = accession_dirs[0]
    print(f"  Accession: {submission_dir.name}  ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Load
    # ------------------------------------------------------------------
    section("STEP 2 — Load & extract 10-K content")
    t0 = time.time()
    docs = load_edgar_submission(submission_dir)
    print(f"  Loaded {len(docs)} segment(s) in {time.time() - t0:.1f}s")
    sample = docs[0]
    print(f"  Sample [{sample.metadata.get('source','?').split('/')[-1]} seg 0]:")
    print(f"    {sample.page_content[:150].replace(chr(10),' ')}…")

    # ------------------------------------------------------------------
    # 3. Chunk
    # ------------------------------------------------------------------
    section("STEP 3 — Chunk (fixed-size)")
    t0 = time.time()
    chunks = chunk_documents(docs, strategy="fixed")
    print(f"  {len(docs)} segment(s) → {len(chunks)} chunks in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Index
    # ------------------------------------------------------------------
    section("STEP 4 — Embed & index with nomic-embed-text → ChromaDB")
    t0 = time.time()
    print("  (May take 30–120 s depending on filing size…)")
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
    vector_store = build_index(chunks, embeddings=embeddings)
    print(f"  Indexed {vector_store._collection.count()} chunks in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 5. Build chain
    # ------------------------------------------------------------------
    section("STEP 5 — Build RAG chain (Groq)")
    chain = build_chain(vector_store)
    print("  Chain ready.")

    # ------------------------------------------------------------------
    # 6. Questions
    # ------------------------------------------------------------------
    questions = [
        "What was Apple's total net sales in the most recent fiscal year?",
        "What are the primary risk factors Apple discloses in this filing?",
        "What was Apple's net income?",
    ]

    section("STEP 6 — Q&A")
    all_ok = True
    for i, q in enumerate(questions, 1):
        print(f"\n  Q{i}: {q}")
        t0 = time.time()
        try:
            result = chain.ask(q)
            elapsed = time.time() - t0
            print(f"  A : {result.answer[:500]}")
            print(f"  Sources : {result.sources}")
            print(f"  Time    : {elapsed:.1f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_ok = False

    print("\n" + "═" * 70)
    status = "✅ ALL STEPS PASSED" if all_ok else "⚠️  COMPLETED WITH ERRORS"
    print(f"  {status}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
