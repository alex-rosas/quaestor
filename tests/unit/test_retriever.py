"""Unit tests for src/quaestor/retrieval/retriever.py.

Uses the same FakeEmbeddings + in-memory ChromaDB approach as test_indexer.py
so no Ollama or network access is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from quaestor.retrieval.retriever import retrieve


# ---------------------------------------------------------------------------
# Fake embeddings
# ---------------------------------------------------------------------------

class FakeEmbeddings(Embeddings):
    def __init__(self, size: int = 8) -> None:
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    def _vec(self, text: str) -> List[float]:
        seed = sum(ord(c) for c in text)
        return [(seed + i) % 100 / 100.0 for i in range(self.size)]


# ---------------------------------------------------------------------------
# Fixture — pre-built vector store
# ---------------------------------------------------------------------------

@pytest.fixture()
def vector_store(tmp_path: Path) -> Chroma:
    """Return a small in-process Chroma store with 5 documents."""
    docs = [
        Document(
            page_content=f"Apple revenue was $383 billion in fiscal year {2020 + i}",
            metadata={"source": "apple_10k.pdf", "page": i, "start_index": i * 200},
        )
        for i in range(5)
    ]
    store = Chroma(
        collection_name="test",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path),
    )
    store.add_documents(docs)
    return store


# ---------------------------------------------------------------------------
# retrieve — happy path
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_returns_list_of_documents(self, vector_store: Chroma) -> None:
        results = retrieve("Apple revenue", vector_store)
        assert isinstance(results, list)
        assert all(isinstance(r, Document) for r in results)

    def test_respects_top_k(self, vector_store: Chroma) -> None:
        results = retrieve("Apple revenue", vector_store, top_k=2)
        assert len(results) == 2

    def test_default_top_k_from_settings(self, vector_store: Chroma) -> None:
        from quaestor.config import settings
        results = retrieve("Apple revenue", vector_store)
        assert len(results) == settings.retrieval_top_k

    def test_top_k_capped_by_collection_size(self, vector_store: Chroma) -> None:
        """Requesting more results than documents should not raise."""
        results = retrieve("Apple revenue", vector_store, top_k=100)
        assert len(results) <= 5

    def test_results_have_source_metadata(self, vector_store: Chroma) -> None:
        results = retrieve("Apple revenue", vector_store, top_k=3)
        for r in results:
            assert "source" in r.metadata

    def test_results_have_page_metadata(self, vector_store: Chroma) -> None:
        results = retrieve("Apple revenue", vector_store, top_k=3)
        for r in results:
            assert "page" in r.metadata

    def test_top_k_one_returns_single_result(self, vector_store: Chroma) -> None:
        results = retrieve("Apple revenue", vector_store, top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# retrieve — validation errors
# ---------------------------------------------------------------------------

class TestRetrieveValidation:
    def test_empty_query_raises(self, vector_store: Chroma) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            retrieve("", vector_store)

    def test_blank_query_raises(self, vector_store: Chroma) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            retrieve("   ", vector_store)

    def test_zero_top_k_raises(self, vector_store: Chroma) -> None:
        with pytest.raises(ValueError, match="positive"):
            retrieve("Apple revenue", vector_store, top_k=0)

    def test_negative_top_k_raises(self, vector_store: Chroma) -> None:
        with pytest.raises(ValueError, match="positive"):
            retrieve("Apple revenue", vector_store, top_k=-1)
