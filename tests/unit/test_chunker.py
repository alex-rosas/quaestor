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
# Phase 2 stubs (hierarchical still pending)
# ---------------------------------------------------------------------------

class TestPhase2Stubs:
    def test_hierarchical_raises_not_implemented(self) -> None:
        docs = [make_doc("Some text.")]
        with pytest.raises(NotImplementedError):
            chunk_documents(docs, strategy="hierarchical")

    def test_invalid_strategy_raises_value_error(self) -> None:
        docs = [make_doc("Some text.")]
        with pytest.raises(ValueError):
            chunk_documents(docs, strategy="magic")  # type: ignore[arg-type]
