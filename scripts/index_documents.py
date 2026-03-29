"""Index all Quaestor source documents into ChromaDB.

Downloads SEC 10-K filings from EDGAR and indexes them alongside any
regulatory PDFs placed in data/raw/regulatory/ — all using hierarchical
chunking (256-char children for retrieval, 1024-char parents for generation).

Everything lands in the same ChromaDB collection so all documents are
queryable together.

Usage
-----
# Index all SEC tickers + any PDFs in data/raw/regulatory/
uv run python scripts/index_documents.py

# SEC filings only (skip regulatory)
uv run python scripts/index_documents.py --sec-only

# Regulatory PDFs only (skip SEC download)
uv run python scripts/index_documents.py --regulatory-only

# Single ticker
uv run python scripts/index_documents.py --ticker AAPL

# Dry run — show what would be indexed without touching ChromaDB
uv run python scripts/index_documents.py --dry-run

Prerequisites
-------------
1. Ollama running with nomic-embed-text pulled:
       ollama pull nomic-embed-text

2. .env configured with:
       GROQ_API_KEY=...
       SEC_REQUESTER_NAME=Your Name
       SEC_REQUESTER_EMAIL=your@email.com

3. For IFRS / PCAOB standards — place PDFs manually in data/raw/regulatory/
   before running:
       data/raw/regulatory/ifrs-9.pdf
       data/raw/regulatory/ifrs-15.pdf
       data/raw/regulatory/ifrs-16.pdf
       data/raw/regulatory/pcaob-as2101.pdf
       data/raw/regulatory/pcaob-as2201.pdf

   Free downloads:
       IFRS 9/15/16  → https://www.ifrs.org/issued-standards/list-of-standards/
                        (free account required; log in → "Download" button)
       PCAOB AS 2101 → https://pcaobus.org/Standards/Auditing/Pages/AS2101.aspx
       PCAOB AS 2201 → https://pcaobus.org/Standards/Auditing/Pages/AS2201.aspx
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quaestor.config import settings
from quaestor.ingestion.chunker import chunk_documents
from quaestor.ingestion.indexer import build_index, load_index
from quaestor.ingestion.loader import (
    download_sec_filings,
    load_edgar_submission,
    load_pdf,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEC_TICKERS = ["AAPL", "JPM", "JNJ", "XOM", "WMT"]

DIVIDER = "─" * 70

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def ok(msg: str) -> None:
    print(f"  ✅  {msg}")


def info(msg: str) -> None:
    print(f"  ℹ️   {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️   {msg}")


# ---------------------------------------------------------------------------
# SEC filing indexing
# ---------------------------------------------------------------------------

def index_sec_ticker(
    ticker: str,
    dry_run: bool = False,
) -> int:
    """Download (if needed) and index the latest 10-K for *ticker*.

    Returns the number of child chunks indexed, or 0 on failure.
    """
    section(f"SEC 10-K — {ticker}")
    t0 = time.time()

    # --- Download ---
    filing_base = settings.data_raw_dir / "sec_filings" / "sec-edgar-filings"
    ticker_dir = filing_base / ticker / "10-K"

    existing = sorted(ticker_dir.glob("*/")) if ticker_dir.exists() else []
    if existing:
        submission_dir = existing[-1]  # most recent filing folder
        info(f"Already downloaded → {submission_dir}")
    else:
        info(f"Downloading {ticker} 10-K from SEC EDGAR…")
        if not dry_run:
            try:
                result_dir = download_sec_filings(ticker, form="10-K", limit=1)
                dirs = sorted(result_dir.glob("*/"))
                if not dirs:
                    warn(f"No filing directories found after download for {ticker}")
                    return 0
                submission_dir = dirs[-1]
                ok(f"Downloaded → {submission_dir}")
            except Exception as exc:
                warn(f"Download failed for {ticker}: {exc}")
                return 0
        else:
            info(f"[dry-run] would download {ticker}")
            return 0

    # --- Load ---
    info(f"Loading documents from {submission_dir.name}…")
    if not dry_run:
        try:
            docs = load_edgar_submission(submission_dir)
            if not docs:
                warn(f"No documents loaded from {submission_dir}")
                return 0
            info(f"Loaded {len(docs)} page(s)")
        except Exception as exc:
            warn(f"Load failed for {ticker}: {exc}")
            return 0
    else:
        info("[dry-run] would load submission")
        return 0

    # --- Chunk ---
    info("Chunking (hierarchical: 1024-char parents / 256-char children)…")
    chunks = chunk_documents(
        docs,
        strategy="hierarchical",
        parent_chunk_size=1024,
        child_chunk_size=256,
        child_chunk_overlap=0,
    )
    info(f"Produced {len(chunks)} child chunk(s)")

    # --- Index ---
    info("Embedding and upserting into ChromaDB…")
    build_index(chunks)
    elapsed = time.time() - t0
    ok(f"{ticker} indexed — {len(chunks)} chunks in {elapsed:.1f}s")
    return len(chunks)


# ---------------------------------------------------------------------------
# Regulatory PDF indexing
# ---------------------------------------------------------------------------

def index_regulatory_pdfs(dry_run: bool = False) -> int:
    """Index all PDFs found in data/raw/regulatory/.

    Returns total child chunks indexed across all PDFs.
    """
    section("Regulatory PDFs (IFRS / PCAOB)")
    regulatory_dir = settings.data_raw_dir / "regulatory"

    if not regulatory_dir.exists():
        warn(f"{regulatory_dir} does not exist — creating it")
        regulatory_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(regulatory_dir.glob("*.pdf"))
    if not pdfs:
        warn(
            f"No PDFs found in {regulatory_dir}\n"
            "    Place IFRS and PCAOB PDFs there before running.\n"
            "    See script docstring for download links."
        )
        return 0

    total = 0
    for pdf_path in pdfs:
        info(f"Processing {pdf_path.name}…")
        if dry_run:
            info(f"[dry-run] would index {pdf_path.name}")
            continue
        try:
            docs = load_pdf(pdf_path)
            info(f"  Loaded {len(docs)} page(s)")
            chunks = chunk_documents(
                docs,
                strategy="hierarchical",
                parent_chunk_size=1024,
                child_chunk_size=256,
                child_chunk_overlap=0,
            )
            info(f"  Produced {len(chunks)} child chunk(s)")
            build_index(chunks)
            ok(f"{pdf_path.name} indexed — {len(chunks)} chunks")
            total += len(chunks)
        except Exception as exc:
            warn(f"Failed to index {pdf_path.name}: {exc}")

    return total


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(collection_size: int, new_chunks: int, elapsed: float) -> None:
    section("Summary")
    print(f"  New chunks indexed this run : {new_chunks:,}")
    print(f"  Total chunks in collection  : {collection_size:,}")
    print(f"  Total time                  : {elapsed:.1f}s")
    print(f"  Collection                  : {settings.chroma_collection_name}")
    print(f"  Persist dir                 : {settings.chroma_persist_dir}")
    print()
    print("  Query the index:")
    print('    uv run uvicorn quaestor.api.main:app --reload')
    print('    curl -X POST http://localhost:8000/ask \\')
    print('      -H "Content-Type: application/json" \\')
    print('      -d \'{"question": "What was Apple net income in FY2025?"}\'')
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index SEC 10-K filings and regulatory PDFs into ChromaDB."
    )
    parser.add_argument(
        "--ticker",
        metavar="SYMBOL",
        help="Index a single ticker instead of all (e.g. --ticker JPM)",
    )
    parser.add_argument(
        "--sec-only",
        action="store_true",
        help="Skip regulatory PDFs, index SEC filings only",
    )
    parser.add_argument(
        "--regulatory-only",
        action="store_true",
        help="Skip SEC downloads, index regulatory PDFs only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be indexed without downloading or modifying ChromaDB",
    )
    args = parser.parse_args()

    t_start = time.time()
    total_new = 0

    print(f"\n{'═' * 70}")
    print("  Quaestor — Document Indexing")
    print(f"{'═' * 70}")
    print(f"  Collection : {settings.chroma_collection_name}")
    print(f"  Backend    : {settings.vector_store_backend.value}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{'═' * 70}\n")

    # --- SEC filings ---
    if not args.regulatory_only:
        tickers = [args.ticker.upper()] if args.ticker else SEC_TICKERS
        for ticker in tickers:
            total_new += index_sec_ticker(ticker, dry_run=args.dry_run)

    # --- Regulatory PDFs ---
    if not args.sec_only and not args.ticker:
        total_new += index_regulatory_pdfs(dry_run=args.dry_run)

    # --- Final summary ---
    if not args.dry_run:
        try:
            vs = load_index()
            collection_size = vs._collection.count()
        except Exception:
            collection_size = -1
        print_summary(collection_size, total_new, time.time() - t_start)
    else:
        section("Dry Run Complete")
        info("No changes made to ChromaDB.")


if __name__ == "__main__":
    main()
