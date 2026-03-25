"""Dense retrieval for Quaestor.

Phase 1: simple top-k similarity search against ChromaDB.
Phase 2 will slot in hybrid BM25 + dense retrieval without changing callers.

Public API
----------
retrieve(query, vector_store, top_k) -> list[Document]
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from quaestor.config import settings

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    vector_store: Chroma,
    top_k: int | None = None,
) -> list[Document]:
    """Return the top-k most relevant chunks for *query*.

    Performs cosine-similarity search against the ChromaDB collection and
    returns results ordered by descending relevance.

    Args:
        query:        Natural-language question from the user.
        vector_store: Indexed ChromaDB collection (from indexer.build_index
                      or indexer.load_index).
        top_k:        Number of chunks to return. Defaults to
                      ``settings.retrieval_top_k``.

    Returns:
        List of Documents ordered by relevance (most relevant first).
        Each Document retains the metadata from indexing (source, page, etc.).

    Raises:
        ValueError: If *query* is empty or blank.
        ValueError: If *top_k* is not a positive integer.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    top_k = top_k if top_k is not None else settings.retrieval_top_k

    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    logger.info("Retrieving top-%d chunks for query: %r", top_k, query[:80])

    results = vector_store.similarity_search(query, k=top_k)

    logger.info("Retrieved %d chunk(s).", len(results))
    return results
