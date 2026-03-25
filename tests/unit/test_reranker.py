"""Unit tests for src/quaestor/retrieval/reranker.py.

All tests are fully offline — a FakeCrossEncoder stub replaces the real
sentence-transformers model so no network access or GPU is required.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from quaestor.retrieval.reranker import CrossEncoderProtocol, rerank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(text: str, source: str = "test.pdf", page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


class FakeCrossEncoder:
    """Offline cross-encoder stub.

    Scores each (query, text) pair by counting how many words from the query
    appear in the text.  Deterministic, fast, and good enough for unit tests.
    """

    def predict(self, sentences: list[list[str]]) -> list[float]:
        scores = []
        for query, text in sentences:
            query_words = set(query.lower().split())
            overlap = sum(1 for w in text.lower().split() if w in query_words)
            scores.append(float(overlap))
        return scores


# ---------------------------------------------------------------------------
# rerank — happy path
# ---------------------------------------------------------------------------

class TestRerank:
    _ce = FakeCrossEncoder()

    def _docs(self) -> list[Document]:
        return [
            make_doc("Apple revenue was $383 billion in fiscal year 2024"),
            make_doc("JPMorgan net interest income rose 4 percent"),
            make_doc("Apple net income grew year over year"),
            make_doc("Unrelated text about weather and climate"),
        ]

    def test_returns_list_of_documents(self) -> None:
        result = rerank("Apple revenue", self._docs(), cross_encoder=self._ce)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_returns_all_docs_when_top_n_is_none(self) -> None:
        docs = self._docs()
        result = rerank("Apple revenue", docs, cross_encoder=self._ce)
        assert len(result) == len(docs)

    def test_top_n_limits_output(self) -> None:
        result = rerank("Apple revenue", self._docs(), cross_encoder=self._ce, top_n=2)
        assert len(result) == 2

    def test_top_n_one_returns_single_document(self) -> None:
        result = rerank("Apple revenue", self._docs(), cross_encoder=self._ce, top_n=1)
        assert len(result) == 1

    def test_most_relevant_doc_ranked_first(self) -> None:
        """The doc with the most query-word overlap should come first."""
        result = rerank("Apple revenue", self._docs(), cross_encoder=self._ce)
        # "Apple revenue was $383 billion..." matches "Apple" and "revenue"
        assert "Apple revenue" in result[0].page_content or "Apple" in result[0].page_content

    def test_reranking_changes_order(self) -> None:
        """A deliberately mis-ordered input should be corrected by reranking."""
        docs = [
            make_doc("Totally unrelated content about glaciers"),  # worst
            make_doc("Apple revenue Apple revenue Apple revenue"),  # best
        ]
        result = rerank("Apple revenue", docs, cross_encoder=self._ce)
        assert "Apple revenue" in result[0].page_content

    def test_metadata_preserved(self) -> None:
        docs = [make_doc("Apple revenue", source="aapl_10k.pdf", page=5)]
        result = rerank("Apple revenue", docs, cross_encoder=self._ce)
        assert result[0].metadata["source"] == "aapl_10k.pdf"
        assert result[0].metadata["page"] == 5

    def test_empty_docs_returns_empty_list(self) -> None:
        result = rerank("Apple revenue", [], cross_encoder=self._ce)
        assert result == []

    def test_single_doc_returned(self) -> None:
        docs = [make_doc("Single document")]
        result = rerank("Single document", docs, cross_encoder=self._ce)
        assert len(result) == 1

    def test_top_n_larger_than_docs_returns_all(self) -> None:
        docs = self._docs()
        result = rerank("Apple revenue", docs, cross_encoder=self._ce, top_n=100)
        assert len(result) == len(docs)

    def test_scores_are_descending(self) -> None:
        """Documents should be ordered best-first (scores non-increasing)."""
        docs = self._docs()
        pairs = [[d.page_content, "Apple revenue"] for d in docs]  # reversed intentionally
        # Use a real-score-aware fake: explicit score list
        class ExplicitScorer:
            def predict(self, sentences):
                return [3.0, 1.0, 2.0, 0.0]  # order: doc0 best, doc2 second

        result = rerank("Apple revenue", docs, cross_encoder=ExplicitScorer())
        # doc0 (score 3.0) first, doc2 (score 2.0) second, doc1 (score 1.0) third
        assert result[0] is docs[0]
        assert result[1] is docs[2]
        assert result[2] is docs[1]
        assert result[3] is docs[3]


# ---------------------------------------------------------------------------
# rerank — protocol conformance
# ---------------------------------------------------------------------------

class TestCrossEncoderProtocol:
    def test_fake_satisfies_protocol(self) -> None:
        ce = FakeCrossEncoder()
        assert isinstance(ce, CrossEncoderProtocol)


# ---------------------------------------------------------------------------
# rerank — validation errors
# ---------------------------------------------------------------------------

class TestRerankValidation:
    _ce = FakeCrossEncoder()

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            rerank("", [make_doc("text")], cross_encoder=self._ce)

    def test_blank_query_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            rerank("   ", [make_doc("text")], cross_encoder=self._ce)

    def test_zero_top_n_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            rerank("query", [make_doc("text")], cross_encoder=self._ce, top_n=0)

    def test_negative_top_n_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            rerank("query", [make_doc("text")], cross_encoder=self._ce, top_n=-1)
