"""Cross-encoder reranking for Quaestor.

Phase 2 — takes the top-k candidate chunks from dense retrieval and reranks
them with a cross-encoder so the LLM always receives the most relevant context
first, regardless of how noisy the initial vector-similarity ranking is.

Why rerank?
-----------
Dense retrieval (cosine similarity on embeddings) is fast but coarse: it
measures *query-document semantic proximity* globally.  A cross-encoder
scores each (query, document) pair jointly — it reads both texts together —
which yields much higher precision at the cost of latency.  The standard
two-stage pipeline is:
  1. Dense retrieval: retrieve top-k candidates cheaply (e.g. k=20).
  2. Cross-encoder rerank: score every candidate and keep top-n (e.g. n=5).
The LLM then sees only the n most relevant chunks, reducing hallucination
and improving faithfulness on financial tables (failure type F3).

Public API
----------
rerank(query, docs, cross_encoder=None, top_n=None) -> list[Document]
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document

from quaestor.config import settings

logger = logging.getLogger(__name__)

# Lazy import — sentence-transformers is only required for cross-encoder reranking.
try:
    from sentence_transformers import CrossEncoder as _CrossEncoderImpl
except ImportError:  # pragma: no cover
    _CrossEncoderImpl = None  # type: ignore[assignment, misc]


@runtime_checkable
class CrossEncoderProtocol(Protocol):
    """Structural interface expected by :func:`rerank`.

    Any object whose ``predict`` method accepts a list of ``[query, text]``
    string pairs and returns a sequence of floats is accepted.  This lets
    tests inject a deterministic stub without loading a real model.
    """

    def predict(self, sentences: list[list[str]]) -> list[float]: ...


def _default_cross_encoder() -> CrossEncoderProtocol:
    """Load the cross-encoder model named in ``settings.reranker_model``."""
    if _CrossEncoderImpl is None:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for cross-encoder reranking. "
            "Run: uv add sentence-transformers"
        )
    return _CrossEncoderImpl(settings.reranker_model)  # type: ignore[return-value]


def rerank(
    query: str,
    docs: list[Document],
    cross_encoder: CrossEncoderProtocol | None = None,
    top_n: int | None = None,
) -> list[Document]:
    """Rerank *docs* by relevance to *query* using a cross-encoder.

    The function scores every ``(query, doc.page_content)`` pair jointly,
    sorts by descending score, and returns the top *top_n* results.

    Args:
        query:         Natural-language question from the user.
        docs:          Candidate chunks from dense retrieval (e.g. from
                       :func:`~quaestor.retrieval.retriever.retrieve`).
        cross_encoder: Object with a ``predict(pairs)`` method.  Defaults to
                       loading ``settings.reranker_model`` via
                       ``sentence_transformers.CrossEncoder``.
        top_n:         How many top-ranked documents to return.  ``None``
                       returns all documents in reranked order.

    Returns:
        Documents ordered by descending cross-encoder score.  If *top_n* is
        set, only the first *top_n* are returned.  If *docs* is empty, an
        empty list is returned immediately (no model call).

    Raises:
        ValueError:  If *query* is empty or blank.
        ValueError:  If *top_n* is not a positive integer.
        ImportError: If ``sentence-transformers`` is not installed and no
                     *cross_encoder* is injected.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if top_n is not None and top_n <= 0:
        raise ValueError(f"top_n must be a positive integer, got {top_n}.")

    if not docs:
        logger.debug("rerank called with empty document list — returning [].")
        return []

    if cross_encoder is None:
        cross_encoder = _default_cross_encoder()

    pairs = [[query, doc.page_content] for doc in docs]
    scores: list[float] = list(cross_encoder.predict(pairs))

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    ranked_docs = [doc for _, doc in ranked]

    if top_n is not None:
        ranked_docs = ranked_docs[:top_n]

    logger.info(
        "Cross-encoder reranking: %d candidate(s) → %d returned (model=%s)",
        len(docs),
        len(ranked_docs),
        settings.reranker_model,
    )
    return ranked_docs
