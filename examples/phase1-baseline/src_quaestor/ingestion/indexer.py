"""Vector store indexing for Quaestor.

Embeds document chunks with nomic-embed-text (via Ollama) and upserts them
into a ChromaDB collection that persists to ``settings.chroma_persist_dir``.

Public API
----------
build_index(chunks)  — embed chunks and upsert into ChromaDB; returns the collection
load_index()         — load an existing persisted collection; raises if absent
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from quaestor.config import EmbeddingProvider, settings

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
    """Return a stable, deterministic ID for *doc*.

    Built from the source path, page number, and start_index so that
    re-indexing the same document produces the same IDs (idempotent upserts).
    """
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", 0)
    start = doc.metadata.get("start_index", index)
    raw = f"{source}::page={page}::start={start}"
    return hashlib.md5(raw.encode()).hexdigest()  # noqa: S324 — not used for security


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
