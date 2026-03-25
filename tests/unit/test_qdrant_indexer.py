"""Unit tests for the Qdrant backend in src/quaestor/ingestion/indexer.py.

All tests are fully offline — they use an in-memory QdrantClient so no Docker
or network access is required.  FakeSparseEmbeddings replaces the real
fastembed BM25 model to keep tests fast and deterministic.
"""

from __future__ import annotations

from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient

from quaestor.ingestion.indexer import _doc_id, build_qdrant_index, load_qdrant_index


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDenseEmbeddings(Embeddings):
    """Deterministic 8-dim dense embeddings derived from text hash."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    def _vec(self, text: str) -> List[float]:
        seed = sum(ord(c) for c in text)
        return [(seed + i) % 100 / 100.0 for i in range(8)]


class FakeSparseEmbeddings(SparseEmbeddings):
    """Deterministic sparse vectors — one non-zero entry per unique word."""

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        return [self._sv(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._sv(text)

    def _sv(self, text: str) -> SparseVector:
        words = list(dict.fromkeys(text.lower().split()))  # stable unique order
        indices = [abs(hash(w)) % 10_000 for w in words]
        values = [1.0 / max(len(words), 1)] * len(words)
        return SparseVector(indices=indices, values=values)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunks(n: int = 3, source: str = "doc.pdf") -> list[Document]:
    return [
        Document(
            page_content=f"Apple revenue financial report chunk {i}",
            metadata={"source": source, "page": i, "start_index": i * 100},
        )
        for i in range(n)
    ]


def in_memory_client() -> QdrantClient:
    return QdrantClient(":memory:")


def _build(chunks, client, **kw) -> QdrantVectorStore:
    return build_qdrant_index(
        chunks,
        embeddings=FakeDenseEmbeddings(),
        sparse_embeddings=FakeSparseEmbeddings(),
        qdrant_client=client,
        **kw,
    )


# ---------------------------------------------------------------------------
# build_qdrant_index — happy path
# ---------------------------------------------------------------------------

class TestBuildQdrantIndex:
    def test_returns_qdrant_vector_store(self) -> None:
        client = in_memory_client()
        store = _build(make_chunks(3), client)
        assert isinstance(store, QdrantVectorStore)

    def test_empty_chunks_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_qdrant_index(
                [],
                embeddings=FakeDenseEmbeddings(),
                sparse_embeddings=FakeSparseEmbeddings(),
                qdrant_client=in_memory_client(),
            )

    def test_similarity_search_returns_results(self) -> None:
        client = in_memory_client()
        store = _build(make_chunks(5), client)
        results = store.similarity_search("Apple revenue", k=2)
        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)

    def test_source_metadata_preserved(self) -> None:
        client = in_memory_client()
        store = _build(make_chunks(3, source="jpm_10k.htm"), client)
        results = store.similarity_search("revenue", k=3)
        for r in results:
            assert r.metadata["source"] == "jpm_10k.htm"

    def test_page_metadata_preserved(self) -> None:
        client = in_memory_client()
        chunks = make_chunks(3)
        store = _build(chunks, client)
        results = store.similarity_search("revenue", k=3)
        pages = {r.metadata["page"] for r in results}
        assert pages == {0, 1, 2}

    def test_custom_collection_name(self) -> None:
        client = in_memory_client()
        store = _build(make_chunks(2), client, collection_name="custom_col")
        assert store.collection_name == "custom_col"

    def test_idempotent_upsert_hybrid(self) -> None:
        """Indexing the same chunks twice must not create duplicate points."""
        client = in_memory_client()
        chunks = make_chunks(3)
        _build(chunks, client, collection_name="idem_test")
        _build(chunks, client, collection_name="idem_test")
        info = client.get_collection("idem_test")
        assert info.points_count == 3

    def test_dense_only_mode(self) -> None:
        client = in_memory_client()
        store = build_qdrant_index(
            make_chunks(3),
            embeddings=FakeDenseEmbeddings(),
            retrieval_mode=RetrievalMode.DENSE,
            qdrant_client=client,
        )
        results = store.similarity_search("revenue", k=2)
        assert len(results) == 2

    def test_hybrid_mode_default(self) -> None:
        client = in_memory_client()
        store = _build(make_chunks(3), client)
        # Default retrieval mode should be HYBRID
        assert store.retrieval_mode == RetrievalMode.HYBRID

    def test_multiple_collections_isolated(self) -> None:
        """Two collections on the same client must not bleed into each other."""
        client = in_memory_client()
        _build(make_chunks(2, source="aapl.pdf"), client, collection_name="aapl")
        _build(make_chunks(3, source="jpm.pdf"), client, collection_name="jpm")
        assert client.get_collection("aapl").points_count == 2
        assert client.get_collection("jpm").points_count == 3


# ---------------------------------------------------------------------------
# load_qdrant_index
# ---------------------------------------------------------------------------

class TestLoadQdrantIndex:
    def test_loads_existing_collection(self) -> None:
        client = in_memory_client()
        _build(make_chunks(4), client, collection_name="existing")
        store = load_qdrant_index(
            collection_name="existing",
            embeddings=FakeDenseEmbeddings(),
            sparse_embeddings=FakeSparseEmbeddings(),
            qdrant_client=client,
        )
        assert isinstance(store, QdrantVectorStore)

    def test_search_after_load(self) -> None:
        client = in_memory_client()
        _build(make_chunks(4), client, collection_name="loaded")
        store = load_qdrant_index(
            collection_name="loaded",
            embeddings=FakeDenseEmbeddings(),
            sparse_embeddings=FakeSparseEmbeddings(),
            qdrant_client=client,
        )
        results = store.similarity_search("revenue", k=2)
        assert len(results) == 2

    def test_raises_if_collection_missing(self) -> None:
        client = in_memory_client()
        with pytest.raises(ValueError, match="does not exist"):
            load_qdrant_index(
                collection_name="nonexistent",
                embeddings=FakeDenseEmbeddings(),
                sparse_embeddings=FakeSparseEmbeddings(),
                qdrant_client=client,
            )

    def test_metadata_preserved_after_load(self) -> None:
        client = in_memory_client()
        _build(make_chunks(3, source="sec_filing.htm"), client, collection_name="meta_test")
        store = load_qdrant_index(
            collection_name="meta_test",
            embeddings=FakeDenseEmbeddings(),
            sparse_embeddings=FakeSparseEmbeddings(),
            qdrant_client=client,
        )
        results = store.similarity_search("revenue", k=3)
        for r in results:
            assert r.metadata["source"] == "sec_filing.htm"


# ---------------------------------------------------------------------------
# _doc_id — UUID format (shared by both backends)
# ---------------------------------------------------------------------------

class TestDocIdUuidFormat:
    def test_is_valid_uuid(self) -> None:
        import uuid
        doc = make_chunks(1)[0]
        id_str = _doc_id(doc, 0)
        # Should not raise
        parsed = uuid.UUID(id_str)
        assert str(parsed) == id_str

    def test_deterministic(self) -> None:
        doc = make_chunks(1)[0]
        assert _doc_id(doc, 0) == _doc_id(doc, 0)

    def test_different_docs_different_ids(self) -> None:
        chunks = make_chunks(3)
        ids = [_doc_id(c, i) for i, c in enumerate(chunks)]
        assert len(set(ids)) == 3
