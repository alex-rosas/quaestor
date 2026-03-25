"""Chunking strategies for Quaestor.

Phase 1 implements fixed-size chunking only (the intentional baseline).
Phase 2 slots in semantic and hierarchical strategies without changing callers.

Public API
----------
chunk_documents(docs, strategy="fixed") -> list[Document]
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from quaestor.config import settings

logger = logging.getLogger(__name__)


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
) -> list[Document]:
    """Split *docs* into chunks using the chosen *strategy*.

    Args:
        docs:          Source documents (e.g. from loader.load_pdf).
        strategy:      One of ``"fixed"`` (Phase 1), ``"semantic"`` or
                       ``"hierarchical"`` (Phase 2, raises NotImplementedError).
        chunk_size:    Override ``settings.chunk_size`` for this call.
        chunk_overlap: Override ``settings.chunk_overlap`` for this call.

    Returns:
        List of chunk Documents with preserved + extended metadata:
        all source metadata is kept; ``start_index`` is added.

    Raises:
        ValueError:          If *docs* is empty.
        NotImplementedError: If *strategy* is ``"semantic"`` or
                             ``"hierarchical"`` (Phase 2).
    """
    if not docs:
        raise ValueError("chunk_documents received an empty document list.")

    strategy = ChunkStrategy(strategy)

    if strategy == ChunkStrategy.FIXED:
        return _chunk_fixed(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if strategy == ChunkStrategy.SEMANTIC:
        raise NotImplementedError(
            "Semantic chunking is a Phase 2 feature. "
            "Use strategy='fixed' for Phase 1."
        )

    if strategy == ChunkStrategy.HIERARCHICAL:
        raise NotImplementedError(
            "Hierarchical chunking is a Phase 2 feature. "
            "Use strategy='fixed' for Phase 1."
        )

    raise ValueError(f"Unknown strategy: {strategy!r}")  # unreachable


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

    # Tag each chunk with its position so callers can sort/filter by order
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        idx = source_counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[src] = idx + 1

    logger.info(
        "Fixed-size chunking: %d doc(s) → %d chunk(s) "
        "(chunk_size=%d, chunk_overlap=%d)",
        len(docs),
        len(chunks),
        chunk_size or settings.chunk_size,
        chunk_overlap or settings.chunk_overlap,
    )
    return chunks
