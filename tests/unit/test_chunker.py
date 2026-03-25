"""Unit tests for src/quaestor/ingestion/chunker.py."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from quaestor.ingestion.chunker import ChunkStrategy, chunk_documents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(text: str, source: str = "test.pdf", page: int = 0) -> Document:
    """Return a minimal Document for testing."""
    return Document(page_content=text, metadata={"source": source, "page": page})


def long_text(n_words: int = 300) -> str:
    """Generate deterministic lorem-style text of roughly *n_words* words."""
    word = "financial"
    return " ".join(f"{word}{i}" for i in range(n_words))


# ---------------------------------------------------------------------------
# Fixed strategy — basic behaviour
# ---------------------------------------------------------------------------

class TestFixedStrategy:
    def test_returns_list_of_documents(self) -> None:
        docs = [make_doc(long_text(200))]
        chunks = chunk_documents(docs)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_single_short_doc_stays_one_chunk(self) -> None:
        docs = [make_doc("Short text.")]
        chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1

    def test_long_doc_splits_into_multiple_chunks(self) -> None:
        docs = [make_doc(long_text(500))]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunk_size_respected(self) -> None:
        docs = [make_doc(long_text(500))]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk.page_content) <= 200 * 6  # generous char-to-token ratio

    def test_source_metadata_preserved(self) -> None:
        docs = [make_doc(long_text(200), source="apple_10k.pdf")]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata["source"] == "apple_10k.pdf"

    def test_page_metadata_preserved(self) -> None:
        docs = [make_doc(long_text(200), page=7)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata["page"] == 7

    def test_chunk_index_added(self) -> None:
        docs = [make_doc(long_text(300))]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_index_resets_per_document(self) -> None:
        docs = [
            make_doc(long_text(300), source="a.pdf"),
            make_doc(long_text(300), source="b.pdf"),
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        a_indices = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "a.pdf"]
        b_indices = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "b.pdf"]
        assert a_indices == list(range(len(a_indices)))
        assert b_indices == list(range(len(b_indices)))

    def test_start_index_present(self) -> None:
        docs = [make_doc(long_text(300))]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert "start_index" in chunk.metadata

    def test_multiple_documents_all_chunked(self) -> None:
        docs = [make_doc(long_text(300), source=f"doc{i}.pdf") for i in range(3)]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"doc0.pdf", "doc1.pdf", "doc2.pdf"}

    def test_explicit_fixed_strategy_string(self) -> None:
        docs = [make_doc("Some content.")]
        chunks = chunk_documents(docs, strategy="fixed")
        assert isinstance(chunks, list)

    def test_explicit_fixed_strategy_enum(self) -> None:
        docs = [make_doc("Some content.")]
        chunks = chunk_documents(docs, strategy=ChunkStrategy.FIXED)
        assert isinstance(chunks, list)

    def test_overlap_creates_shared_content(self) -> None:
        """With overlap > 0 adjacent chunks should share some characters."""
        text = " ".join(f"word{i}" for i in range(100))
        docs = [make_doc(text)]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=20)
        if len(chunks) >= 2:
            end_of_first = chunks[0].page_content[-10:]
            start_of_second = chunks[1].page_content[:40]
            assert any(w in start_of_second for w in end_of_first.split())


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_docs_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            chunk_documents([])


# ---------------------------------------------------------------------------
# Semantic strategy
# ---------------------------------------------------------------------------

class FakeEmbeddings:
    """Offline embeddings stub — returns a different vector per unique text.

    The SemanticChunker compares adjacent sentence embeddings. Using a
    hash-derived vector ensures varied similarity scores so the chunker
    actually produces splits without any network calls.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)  # noqa: S324
        # 8-dimensional vector derived from the hash
        return [(((h >> (i * 8)) & 0xFF) / 255.0) for i in range(8)]


class TestSemanticStrategy:
    """Tests for strategy='semantic'.

    SemanticChunker is mocked via the ``embeddings`` injection point so
    every test is fully offline — no Ollama required.
    """

    def _fake_emb(self) -> FakeEmbeddings:
        return FakeEmbeddings()

    def test_returns_list_of_documents(self) -> None:
        docs = [make_doc(long_text(200))]
        chunks = chunk_documents(docs, strategy="semantic", embeddings=self._fake_emb())
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_produces_at_least_one_chunk(self) -> None:
        docs = [make_doc(long_text(100))]
        chunks = chunk_documents(docs, strategy="semantic", embeddings=self._fake_emb())
        assert len(chunks) >= 1

    def test_source_metadata_preserved(self) -> None:
        docs = [make_doc(long_text(100), source="jpm_10k.htm")]
        chunks = chunk_documents(docs, strategy="semantic", embeddings=self._fake_emb())
        for chunk in chunks:
            assert chunk.metadata["source"] == "jpm_10k.htm"

    def test_page_metadata_preserved(self) -> None:
        docs = [make_doc(long_text(100), page=3)]
        chunks = chunk_documents(docs, strategy="semantic", embeddings=self._fake_emb())
        for chunk in chunks:
            assert chunk.metadata["page"] == 3

    def test_chunk_index_added(self) -> None:
        docs = [make_doc(long_text(200))]
        chunks = chunk_documents(docs, strategy="semantic", embeddings=self._fake_emb())
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_index_resets_per_document(self) -> None:
        docs = [
            make_doc(long_text(100), source="a.htm"),
            make_doc(long_text(100), source="b.htm"),
        ]
        chunks = chunk_documents(docs, strategy="semantic", embeddings=self._fake_emb())
        a_idx = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "a.htm"]
        b_idx = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "b.htm"]
        assert a_idx == list(range(len(a_idx)))
        assert b_idx == list(range(len(b_idx)))

    def test_enum_strategy_accepted(self) -> None:
        docs = [make_doc("Some content.")]
        chunks = chunk_documents(
            docs, strategy=ChunkStrategy.SEMANTIC, embeddings=self._fake_emb()
        )
        assert isinstance(chunks, list)

    def test_breakpoint_threshold_types_accepted(self) -> None:
        docs = [make_doc(long_text(100))]
        for threshold in ("percentile", "standard_deviation", "interquartile", "gradient"):
            chunks = chunk_documents(
                docs,
                strategy="semantic",
                breakpoint_threshold_type=threshold,  # type: ignore[arg-type]
                embeddings=self._fake_emb(),
            )
            assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Hierarchical strategy
# ---------------------------------------------------------------------------

class TestHierarchicalStrategy:
    """Tests for strategy='hierarchical'.

    All tests are fully offline — no embeddings required.
    Uses parent_chunk_size=300, child_chunk_size=100 for fast deterministic splits.
    """

    _P = 300  # parent size
    _C = 100  # child size

    def _chunks(self, docs: list, **kw):
        return chunk_documents(
            docs, strategy="hierarchical",
            parent_chunk_size=self._P, child_chunk_size=self._C,
            **kw,
        )

    def test_returns_list_of_documents(self) -> None:
        chunks = self._chunks([make_doc(long_text(200))])
        assert isinstance(chunks, list)
        assert all(isinstance(c, Document) for c in chunks)

    def test_all_chunks_are_children(self) -> None:
        chunks = self._chunks([make_doc(long_text(200))])
        assert all(c.metadata["chunk_level"] == "child" for c in chunks)

    def test_parent_content_present(self) -> None:
        chunks = self._chunks([make_doc(long_text(200))])
        assert all("parent_content" in c.metadata for c in chunks)
        assert all(len(c.metadata["parent_content"]) > 0 for c in chunks)

    def test_child_text_contained_in_parent(self) -> None:
        chunks = self._chunks([make_doc(long_text(200))])
        for chunk in chunks:
            assert chunk.page_content in chunk.metadata["parent_content"]

    def test_parent_id_present_and_stable(self) -> None:
        doc = make_doc(long_text(200))
        chunks_a = self._chunks([doc])
        chunks_b = self._chunks([doc])
        ids_a = [c.metadata["parent_id"] for c in chunks_a]
        ids_b = [c.metadata["parent_id"] for c in chunks_b]
        assert ids_a == ids_b  # deterministic

    def test_parent_index_present(self) -> None:
        chunks = self._chunks([make_doc(long_text(300))])
        assert all("parent_index" in c.metadata for c in chunks)

    def test_chunk_index_resets_per_parent(self) -> None:
        chunks = self._chunks([make_doc(long_text(300))])
        by_parent: dict[int, list[int]] = {}
        for c in chunks:
            pi = c.metadata["parent_index"]
            by_parent.setdefault(pi, []).append(c.metadata["chunk_index"])
        for indices in by_parent.values():
            assert indices == list(range(len(indices)))

    def test_more_children_than_parents(self) -> None:
        chunks = self._chunks([make_doc(long_text(400))])
        n_parents = len({c.metadata["parent_id"] for c in chunks})
        assert len(chunks) >= n_parents

    def test_source_metadata_preserved(self) -> None:
        chunks = self._chunks([make_doc(long_text(200), source="jpm_10k.htm")])
        assert all(c.metadata["source"] == "jpm_10k.htm" for c in chunks)

    def test_page_metadata_preserved(self) -> None:
        chunks = self._chunks([make_doc(long_text(200), page=5)])
        assert all(c.metadata["page"] == 5 for c in chunks)

    def test_multiple_docs_all_chunked(self) -> None:
        docs = [make_doc(long_text(200), source=f"doc{i}.htm") for i in range(3)]
        chunks = self._chunks(docs)
        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"doc0.htm", "doc1.htm", "doc2.htm"}

    def test_enum_strategy_accepted(self) -> None:
        chunks = chunk_documents(
            [make_doc(long_text(200))],
            strategy=ChunkStrategy.HIERARCHICAL,
            parent_chunk_size=self._P,
            child_chunk_size=self._C,
        )
        assert isinstance(chunks, list)

    def test_child_size_gte_parent_raises(self) -> None:
        with pytest.raises(ValueError, match="child_chunk_size"):
            chunk_documents(
                [make_doc("text")],
                strategy="hierarchical",
                parent_chunk_size=100,
                child_chunk_size=100,
            )


# ---------------------------------------------------------------------------
# Invalid strategy
# ---------------------------------------------------------------------------

class TestInvalidStrategy:
    def test_invalid_strategy_raises_value_error(self) -> None:
        docs = [make_doc("Some text.")]
        with pytest.raises(ValueError):
            chunk_documents(docs, strategy="magic")  # type: ignore[arg-type]
