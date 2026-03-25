"""Unit tests for src/quaestor/ingestion/indexer.py.

All tests use FakeEmbeddings (fixed-dimension random vectors) so Ollama is
never required.  ChromaDB is given a tmp_path directory per test so each
test starts with a clean, isolated collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from quaestor.ingestion.indexer import _doc_id, build_index, load_index


# ---------------------------------------------------------------------------
# Fake embeddings — no Ollama needed
# ---------------------------------------------------------------------------

class FakeEmbeddings(Embeddings):
    """Return deterministic fixed-dimension vectors from text hash."""

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
# Helpers
# ---------------------------------------------------------------------------

def make_chunks(n: int = 3, source: str = "doc.pdf") -> list[Document]:
    return [
        Document(
            page_content=f"chunk {i} content about finance",
            metadata={"source": source, "page": i, "start_index": i * 100},
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _doc_id
# ---------------------------------------------------------------------------

class TestDocId:
    def test_returns_string(self) -> None:
        doc = make_chunks(1)[0]
        assert isinstance(_doc_id(doc, 0), str)

    def test_deterministic(self) -> None:
        doc = make_chunks(1)[0]
        assert _doc_id(doc, 0) == _doc_id(doc, 0)

    def test_different_docs_different_ids(self) -> None:
        chunks = make_chunks(3)
        ids = [_doc_id(c, i) for i, c in enumerate(chunks)]
        assert len(set(ids)) == 3

    def test_same_content_different_source_different_id(self) -> None:
        doc_a = Document(page_content="text", metadata={"source": "a.pdf", "page": 0, "start_index": 0})
        doc_b = Document(page_content="text", metadata={"source": "b.pdf", "page": 0, "start_index": 0})
        assert _doc_id(doc_a, 0) != _doc_id(doc_b, 0)


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_returns_chroma_instance(self, tmp_path: Path) -> None:
        from langchain_chroma import Chroma
        chunks = make_chunks(3)
        store = build_index(chunks, persist_dir=tmp_path, embeddings=FakeEmbeddings())
        assert isinstance(store, Chroma)

    def test_empty_chunks_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_index([], persist_dir=tmp_path, embeddings=FakeEmbeddings())

    def test_collection_has_correct_count(self, tmp_path: Path) -> None:
        chunks = make_chunks(5)
        store = build_index(chunks, persist_dir=tmp_path, embeddings=FakeEmbeddings())
        assert store._collection.count() == 5

    def test_creates_persist_dir(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new_collection"
        chunks = make_chunks(2)
        build_index(chunks, persist_dir=new_dir, embeddings=FakeEmbeddings())
        assert new_dir.exists()

    def test_idempotent_upsert(self, tmp_path: Path) -> None:
        """Indexing the same chunks twice must not create duplicate entries."""
        chunks = make_chunks(3)
        build_index(chunks, persist_dir=tmp_path, embeddings=FakeEmbeddings())
        build_index(chunks, persist_dir=tmp_path, embeddings=FakeEmbeddings())
        store = build_index(chunks, persist_dir=tmp_path, embeddings=FakeEmbeddings())
        assert store._collection.count() == 3

    def test_custom_collection_name(self, tmp_path: Path) -> None:
        from langchain_chroma import Chroma
        chunks = make_chunks(2)
        store = build_index(
            chunks,
            collection_name="custom_col",
            persist_dir=tmp_path,
            embeddings=FakeEmbeddings(),
        )
        assert store._collection.name == "custom_col"

    def test_similarity_search_returns_results(self, tmp_path: Path) -> None:
        chunks = make_chunks(5)
        store = build_index(chunks, persist_dir=tmp_path, embeddings=FakeEmbeddings())
        results = store.similarity_search("finance", k=2)
        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)


# ---------------------------------------------------------------------------
# load_index
# ---------------------------------------------------------------------------

class TestLoadIndex:
    def test_load_existing_index(self, tmp_path: Path) -> None:
        from langchain_chroma import Chroma
        chunks = make_chunks(4)
        build_index(chunks, persist_dir=tmp_path, embeddings=FakeEmbeddings())
        store = load_index(persist_dir=tmp_path, embeddings=FakeEmbeddings())
        assert isinstance(store, Chroma)
        assert store._collection.count() == 4

    def test_raises_if_dir_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_index(persist_dir=tmp_path / "does_not_exist", embeddings=FakeEmbeddings())

    def test_raises_if_collection_empty(self, tmp_path: Path) -> None:
        """A directory that exists but contains an empty collection must raise."""
        # Create a valid but empty ChromaDB at tmp_path
        import chromadb
        client = chromadb.PersistentClient(path=str(tmp_path))
        client.get_or_create_collection("quaestor")

        with pytest.raises(ValueError, match="empty"):
            load_index(persist_dir=tmp_path, embeddings=FakeEmbeddings())

    def test_load_preserves_metadata(self, tmp_path: Path) -> None:
        chunks = make_chunks(3, source="apple_10k.pdf")
        build_index(chunks, persist_dir=tmp_path, embeddings=FakeEmbeddings())
        store = load_index(persist_dir=tmp_path, embeddings=FakeEmbeddings())
        results = store.similarity_search("finance", k=3)
        for r in results:
            assert r.metadata["source"] == "apple_10k.pdf"
