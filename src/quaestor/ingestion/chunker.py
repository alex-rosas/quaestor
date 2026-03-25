"""Chunking strategies for Quaestor.

Phase 1 — fixed-size chunking (RecursiveCharacterTextSplitter).
Phase 2 — semantic chunking (SemanticChunker) and hierarchical chunking.

All strategies share the same public API so callers never need to change.

Public API
----------
chunk_documents(docs, strategy="fixed") -> list[Document]
"""

from __future__ import annotations

import hashlib
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
    parent_chunk_size: int = 1024,
    child_chunk_size: int = 256,
    child_chunk_overlap: int = 0,
) -> list[Document]:
    """Split *docs* into chunks using the chosen *strategy*.

    Args:
        docs:                      Source documents (e.g. from loader.load_pdf).
        strategy:                  Chunking strategy:
                                   ``"fixed"`` — Phase 1 baseline,
                                   ``"semantic"`` — embedding-level boundaries,
                                   ``"hierarchical"`` — parent/child split.
        chunk_size:                Override ``settings.chunk_size`` (fixed only).
        chunk_overlap:             Override ``settings.chunk_overlap`` (fixed only).
        breakpoint_threshold_type: Boundary detection for SemanticChunker.
                                   One of ``"percentile"`` (default),
                                   ``"standard_deviation"``, ``"interquartile"``,
                                   ``"gradient"``. Ignored for other strategies.
        embeddings:                Inject a custom embedding model (semantic only).
                                   Defaults to the provider in ``settings``.
        parent_chunk_size:         Parent window size in chars (hierarchical only).
                                   Default 1024.
        child_chunk_size:          Child window size in chars (hierarchical only).
                                   Default 256. Must be < parent_chunk_size.
        child_chunk_overlap:       Overlap between child chunks (hierarchical only).
                                   Default 0.

    Returns:
        List of chunk Documents with preserved + extended metadata.
        All source metadata is kept; ``chunk_index`` is added to every chunk.
        Hierarchical chunks additionally carry ``chunk_level``, ``parent_content``,
        ``parent_id``, and ``parent_index``.

    Raises:
        ValueError: If *docs* is empty, strategy is unknown, or
                    child_chunk_size >= parent_chunk_size.
        ImportError: If strategy is ``"semantic"`` and
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
        if child_chunk_size >= parent_chunk_size:
            raise ValueError(
                f"child_chunk_size ({child_chunk_size}) must be smaller than "
                f"parent_chunk_size ({parent_chunk_size})."
            )
        return _chunk_hierarchical(
            docs,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
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
# Hierarchical strategy
# ---------------------------------------------------------------------------

def _chunk_hierarchical(
    docs: list[Document],
    parent_chunk_size: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> list[Document]:
    """Split *docs* into parent windows, then split each parent into children.

    Only child chunks are returned — they are small enough for precise
    embedding retrieval. Each child carries its parent's full text in
    ``metadata["parent_content"]`` so the RAG chain can inject the larger
    context at generation time, recovering table structure that fixed-size
    splitting destroys (failure type F3 in the Phase 1 analysis).

    Metadata added to every child chunk:

    * ``chunk_level``    — always ``"child"``.
    * ``parent_content`` — the full text of the enclosing parent chunk.
    * ``parent_id``      — stable hash of ``source + parent start_index``.
    * ``parent_index``   — 0-based position of the parent within the source doc.
    * ``chunk_index``    — 0-based position of this child within its parent.

    Args:
        docs:               Source documents.
        parent_chunk_size:  Max characters per parent window.
        child_chunk_size:   Max characters per child window.
        child_chunk_overlap: Character overlap between consecutive children.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    all_children: list[Document] = []

    for doc in docs:
        parents = parent_splitter.split_documents([doc])
        for parent_idx, parent in enumerate(parents):
            pid = _parent_id(parent, parent_idx)
            children = child_splitter.split_documents([parent])
            for child_idx, child in enumerate(children):
                child.metadata.update({
                    "chunk_level":    "child",
                    "parent_content": parent.page_content,
                    "parent_id":      pid,
                    "parent_index":   parent_idx,
                    "chunk_index":    child_idx,
                })
                all_children.append(child)

    logger.info(
        "Hierarchical chunking: %d doc(s) → %d child chunk(s) "
        "(parent_size=%d, child_size=%d, child_overlap=%d)",
        len(docs),
        len(all_children),
        parent_chunk_size,
        child_chunk_size,
        child_chunk_overlap,
    )
    return all_children


def _parent_id(parent: Document, index: int) -> str:
    """Stable 12-hex-char ID for a parent chunk.

    Derived from source path and start_index so re-chunking the same
    document produces the same IDs (idempotent upserts).
    """
    source = parent.metadata.get("source", "")
    start  = parent.metadata.get("start_index", index)
    raw    = f"{source}::parent_start={start}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]  # noqa: S324


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
