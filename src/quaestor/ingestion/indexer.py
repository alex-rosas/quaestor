"""Vector store indexing for Quaestor.

Embeds document chunks and upserts them into a ChromaDB (Phase 1) or Qdrant
(Phase 2) collection.  Qdrant supports hybrid BM25 + dense retrieval which
improves recall on financial documents with rare tokens (ticker symbols, GAAP
line-item names, regulatory article numbers).

Public API
----------
build_index(chunks)           — Chroma: embed + upsert; returns the collection
load_index()                  — Chroma: load existing collection
build_qdrant_index(chunks)    — Qdrant: hybrid BM25 + dense upsert
load_qdrant_index()           — Qdrant: open existing collection
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from quaestor.config import EmbeddingProvider, settings

if TYPE_CHECKING:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding factory
# ---------------------------------------------------------------------------

def _get_embeddings(provider: EmbeddingProvider | None = None) -> Embeddings:
    """Return the configured embedding model.

    Args:
        provider: Override ``settings.embedding_provider`` for this call.

    Returns:
        A LangChain ``Embeddings`` instance.

    Raises:
        ValueError: If the provider is unknown.
    """
    provider = provider or settings.embedding_provider

    if provider == EmbeddingProvider.OLLAMA:
        from langchain_ollama import OllamaEmbeddings  # type: ignore[import]

        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    if provider == EmbeddingProvider.HUGGINGFACE:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import]

        return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)

    raise ValueError(f"Unknown embedding provider: {provider!r}")


# ---------------------------------------------------------------------------
# Stable document ID
# ---------------------------------------------------------------------------

def _doc_id(doc: Document, index: int) -> str:
    """Return a stable, deterministic UUID string for *doc*.

    Built from the source path, page number, and start_index so that
    re-indexing the same document produces the same IDs (idempotent upserts).
    The result is formatted as a UUID so it satisfies both ChromaDB and
    Qdrant's point-ID requirements.
    """
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", 0)
    start = doc.metadata.get("start_index", index)
    raw = f"{source}::page={page}::start={start}"
    h = hashlib.md5(raw.encode()).hexdigest()  # noqa: S324 — not used for security
    return str(uuid.UUID(hex=h))


# ---------------------------------------------------------------------------
# ChromaDB client factory
# ---------------------------------------------------------------------------

def _chroma_client(persist_dir: Path | None = None) -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    persist_dir = Path(persist_dir or settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(
    chunks: list[Document],
    collection_name: str | None = None,
    persist_dir: Path | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Embed *chunks* and upsert them into a ChromaDB collection.

    Upserts are idempotent — re-indexing the same document will overwrite
    existing vectors rather than duplicate them, because IDs are derived
    from source metadata.

    Args:
        chunks:          List of chunk Documents (output of chunker).
        collection_name: Override ``settings.chroma_collection_name``.
        persist_dir:     Override ``settings.chroma_persist_dir``.
        embeddings:      Inject a custom embedding model (used in tests).

    Returns:
        A LangChain ``Chroma`` wrapper around the collection.

    Raises:
        ValueError: If *chunks* is empty.
    """
    if not chunks:
        raise ValueError("build_index received an empty chunk list.")

    collection_name = collection_name or settings.chroma_collection_name
    embeddings = embeddings or _get_embeddings()
    ids = [_doc_id(chunk, i) for i, chunk in enumerate(chunks)]

    logger.info(
        "Indexing %d chunks into collection '%s' at %s",
        len(chunks),
        collection_name,
        persist_dir or settings.chroma_persist_dir,
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir or settings.chroma_persist_dir),
    )
    vector_store.add_documents(documents=chunks, ids=ids)

    logger.info("Indexing complete — %d chunks stored.", len(chunks))
    return vector_store


def load_index(
    collection_name: str | None = None,
    persist_dir: Path | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Load an existing ChromaDB collection from disk.

    Args:
        collection_name: Override ``settings.chroma_collection_name``.
        persist_dir:     Override ``settings.chroma_persist_dir``.
        embeddings:      Inject a custom embedding model (used in tests).

    Returns:
        A LangChain ``Chroma`` wrapper around the collection.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
        ValueError:        If the collection has no documents.
    """
    collection_name = collection_name or settings.chroma_collection_name
    persist_dir = Path(persist_dir or settings.chroma_persist_dir)

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB persist directory not found: {persist_dir}. "
            "Run build_index first."
        )

    embeddings = embeddings or _get_embeddings()
    logger.info("Loading index from %s (collection: %s)", persist_dir, collection_name)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    count = vector_store._collection.count()
    if count == 0:
        raise ValueError(
            f"Collection '{collection_name}' exists but is empty. "
            "Run build_index first."
        )

    logger.info("Loaded collection with %d document(s).", count)
    return vector_store


# ---------------------------------------------------------------------------
# Qdrant backend — hybrid BM25 + dense retrieval
# ---------------------------------------------------------------------------

def _qdrant_client(url: str | None = None) -> QdrantClient:
    """Return a Qdrant client for the given *url*.

    Pass ``":memory:"`` (or set ``QDRANT_URL=:memory:``) to get an in-process
    client — useful for tests and local experimentation without Docker.
    """
    from qdrant_client import QdrantClient as _QC

    target = url or settings.qdrant_url
    if target == ":memory:":
        return _QC(":memory:")
    return _QC(url=target)


def _get_sparse_embeddings():
    """Return the default sparse BM25 encoder (fastembed Qdrant/bm25)."""
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name="Qdrant/bm25")


def _create_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    retrieval_mode,
) -> None:
    """Create a Qdrant collection with the correct vector configuration.

    * HYBRID → unnamed dense vector + ``"langchain-sparse"`` sparse vector.
    * DENSE  → unnamed dense vector only.
    * SPARSE → ``"langchain-sparse"`` sparse vector only.

    ``"langchain-sparse"`` is the default sparse vector name used by
    ``QdrantVectorStore`` (defined in ``langchain_qdrant``).
    """
    from langchain_qdrant import RetrievalMode
    from qdrant_client.models import (
        Distance,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )

    vectors_config = None
    sparse_vectors_config = None

    if retrieval_mode in (RetrievalMode.HYBRID, RetrievalMode.DENSE):
        vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)

    if retrieval_mode in (RetrievalMode.HYBRID, RetrievalMode.SPARSE):
        sparse_vectors_config = {
            "langchain-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )


def build_qdrant_index(
    chunks: list[Document],
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
    sparse_embeddings=None,
    retrieval_mode=None,
    qdrant_client: QdrantClient | None = None,
) -> QdrantVectorStore:
    """Embed *chunks* and upsert them into a Qdrant collection.

    Supports three retrieval modes (controlled by *retrieval_mode*):

    * ``RetrievalMode.HYBRID``  — BM25 sparse + dense vectors, RRF fusion.
      Best recall on financial docs with rare tokens (tickers, GAAP terms).
    * ``RetrievalMode.DENSE``   — dense-only (cosine similarity).
    * ``RetrievalMode.SPARSE``  — BM25-only.

    Upserts are idempotent — re-indexing the same chunks overwrites existing
    points because IDs are derived from source metadata (UUID-formatted MD5).

    Args:
        chunks:           Chunk Documents (output of chunker).
        collection_name:  Override ``settings.qdrant_collection_name``.
        embeddings:       Dense embedding model.  Defaults to
                          ``settings.embedding_provider``.
        sparse_embeddings: Sparse BM25 encoder.  Defaults to
                           ``FastEmbedSparse("Qdrant/bm25")``.
                           Pass a stub for offline tests.
        retrieval_mode:   ``RetrievalMode.HYBRID`` (default), ``.DENSE``,
                          or ``.SPARSE``.
        qdrant_client:    Inject a pre-built ``QdrantClient``
                          (e.g. ``QdrantClient(":memory:")`` in tests).

    Returns:
        A ``QdrantVectorStore`` ready for ``similarity_search``.

    Raises:
        ValueError: If *chunks* is empty.
    """
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    if not chunks:
        raise ValueError("build_qdrant_index received an empty chunk list.")

    collection_name = collection_name or settings.qdrant_collection_name
    embeddings = embeddings or _get_embeddings()
    sparse_embeddings = sparse_embeddings or _get_sparse_embeddings()
    retrieval_mode = retrieval_mode or RetrievalMode.HYBRID

    ids = [_doc_id(chunk, i) for i, chunk in enumerate(chunks)]
    client = qdrant_client or _qdrant_client()

    if not client.collection_exists(collection_name):
        # Probe vector size from a sample embedding
        sample_vec = embeddings.embed_query(chunks[0].page_content[:50])
        _create_qdrant_collection(client, collection_name, len(sample_vec), retrieval_mode)

    logger.info(
        "Qdrant indexing: %d chunk(s) → collection '%s' (mode=%s)",
        len(chunks),
        collection_name,
        retrieval_mode,
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=retrieval_mode,
    )
    store.add_documents(chunks, ids=ids)

    logger.info("Qdrant indexing complete — %d chunk(s) stored.", len(chunks))
    return store


def load_qdrant_index(
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
    sparse_embeddings=None,
    retrieval_mode=None,
    qdrant_client: QdrantClient | None = None,
) -> QdrantVectorStore:
    """Open an existing Qdrant collection for retrieval.

    Args:
        collection_name:  Override ``settings.qdrant_collection_name``.
        embeddings:       Dense embedding model.
        sparse_embeddings: Sparse BM25 encoder.
        retrieval_mode:   ``RetrievalMode.HYBRID`` (default).
        qdrant_client:    Inject a pre-built ``QdrantClient``.

    Returns:
        ``QdrantVectorStore`` connected to the existing collection.

    Raises:
        ValueError: If the collection does not exist or is empty.
    """
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    collection_name = collection_name or settings.qdrant_collection_name
    embeddings = embeddings or _get_embeddings()
    sparse_embeddings = sparse_embeddings or _get_sparse_embeddings()
    retrieval_mode = retrieval_mode or RetrievalMode.HYBRID

    client = qdrant_client or _qdrant_client()

    if not client.collection_exists(collection_name):
        raise ValueError(
            f"Qdrant collection '{collection_name}' does not exist. "
            "Run build_qdrant_index first."
        )

    info = client.get_collection(collection_name)
    point_count = info.points_count or 0
    if point_count == 0:
        raise ValueError(
            f"Qdrant collection '{collection_name}' exists but is empty. "
            "Run build_qdrant_index first."
        )

    logger.info(
        "Loaded Qdrant collection '%s' with %d point(s).",
        collection_name,
        point_count,
    )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=retrieval_mode,
    )
