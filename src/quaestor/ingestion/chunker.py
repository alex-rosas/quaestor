"""Chunking strategies for Quaestor.

Phase 1 — fixed-size chunking (RecursiveCharacterTextSplitter).
Phase 2 — semantic chunking (SemanticChunker) and hierarchical chunking.

All strategies share the same public API so callers never need to change.

Public API
----------
chunk_documents(docs, strategy="fixed") -> list[Document]
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from quaestor.config import settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Lazy import — langchain-experimental is only required for semantic chunking.
try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:  # pragma: no cover
    SemanticChunker = None  # type: ignore[assignment,misc]


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


def _fixed_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Return a RecursiveCharacterTextSplitter using settings defaults."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )


def chunk_documents(
    docs: list[Document],
    strategy: ChunkStrategy | Literal["fixed", "semantic", "hierarchical"] = "fixed",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    breakpoint_threshold_type: Literal[
        "percentile", "standard_deviation", "interquartile", "gradient"
    ] = "percentile",
    embeddings: Embeddings | None = None,
) -> list[Document]:
    """Split *docs* into chunks using the chosen *strategy*.

    Args:
        docs:                      Source documents (e.g. from loader.load_pdf).
        strategy:                  Chunking strategy — ``"fixed"`` (Phase 1),
                                   ``"semantic"`` (Phase 2), or
                                   ``"hierarchical"`` (Phase 2, not yet implemented).
        chunk_size:                Override ``settings.chunk_size`` (fixed only).
        chunk_overlap:             Override ``settings.chunk_overlap`` (fixed only).
        breakpoint_threshold_type: How SemanticChunker detects topic boundaries.
                                   One of ``"percentile"`` (default),
                                   ``"standard_deviation"``, ``"interquartile"``,
                                   ``"gradient"``. Ignored for fixed strategy.
        embeddings:                Inject a custom embedding model (semantic only).
                                   Defaults to the provider in ``settings``.

    Returns:
        List of chunk Documents with preserved + extended metadata.
        All source metadata is kept; ``chunk_index`` is added to every chunk.

    Raises:
        ValueError:          If *docs* is empty or strategy is unknown.
        NotImplementedError: If *strategy* is ``"hierarchical"`` (Phase 2 TODO).
        ImportError:         If *strategy* is ``"semantic"`` and
                             ``langchain-experimental`` is not installed.
    """
    if not docs:
        raise ValueError("chunk_documents received an empty document list.")

    strategy = ChunkStrategy(strategy)

    if strategy == ChunkStrategy.FIXED:
        return _chunk_fixed(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if strategy == ChunkStrategy.SEMANTIC:
        return _chunk_semantic(
            docs,
            breakpoint_threshold_type=breakpoint_threshold_type,
            embeddings=embeddings,
        )

    if strategy == ChunkStrategy.HIERARCHICAL:
        raise NotImplementedError(
            "Hierarchical chunking is a Phase 2 feature not yet implemented. "
            "Use strategy='fixed' or strategy='semantic'."
        )

    raise ValueError(f"Unknown strategy: {strategy!r}")  # unreachable


# ---------------------------------------------------------------------------
# Fixed strategy
# ---------------------------------------------------------------------------

def _chunk_fixed(
    docs: list[Document],
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> list[Document]:
    """Split *docs* with fixed-size overlapping windows.

    Each output chunk inherits all metadata from its parent document and
    gains a ``chunk_index`` field (0-based position within that document).
    """
    splitter = _fixed_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    _tag_chunk_index(chunks)

    logger.info(
        "Fixed-size chunking: %d doc(s) → %d chunk(s) "
        "(chunk_size=%d, chunk_overlap=%d)",
        len(docs),
        len(chunks),
        chunk_size or settings.chunk_size,
        chunk_overlap or settings.chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Semantic strategy
# ---------------------------------------------------------------------------

def _chunk_semantic(
    docs: list[Document],
    breakpoint_threshold_type: str,
    embeddings: Embeddings | None,
) -> list[Document]:
    """Split *docs* at embedding-level semantic boundaries.

    Uses ``langchain_experimental.text_splitter.SemanticChunker`` which:
    1. Splits text into sentences.
    2. Embeds each sentence with the configured embedding model.
    3. Computes cosine similarity between adjacent sentence embeddings.
    4. Inserts a chunk boundary wherever similarity drops below the threshold.

    This keeps label + value pairs in the same chunk far more reliably than
    fixed-size splitting, addressing failure types F1 (mid-sentence breaks)
    and F2 (orphaned numbers) identified in the Phase 1 failure analysis.

    Args:
        docs:                      Source documents.
        breakpoint_threshold_type: Boundary detection method passed to
                                   ``SemanticChunker``.
        embeddings:                Embedding model. If ``None``, the model
                                   configured in ``settings`` is used.

    Raises:
        ImportError: If ``langchain-experimental`` is not installed.
    """
    if SemanticChunker is None:  # pragma: no cover
        raise ImportError(
            "langchain-experimental is required for semantic chunking. "
            "Run: uv add langchain-experimental"
        )

    if embeddings is None:
        from quaestor.ingestion.indexer import _get_embeddings
        embeddings = _get_embeddings()

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
    )
    chunks = splitter.split_documents(docs)

    _tag_chunk_index(chunks)

    logger.info(
        "Semantic chunking: %d doc(s) → %d chunk(s) (threshold_type=%s)",
        len(docs),
        len(chunks),
        breakpoint_threshold_type,
    )
    return chunks


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _tag_chunk_index(chunks: list[Document]) -> None:
    """Add a ``chunk_index`` field to each chunk (mutates in place).

    The index is 0-based and resets per source document so callers can
    sort or filter chunks by their position within the original document.
    """
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        idx = source_counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[src] = idx + 1
