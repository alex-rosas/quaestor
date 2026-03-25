"""Quaestor — Phase 1 Streamlit demo.

Entry point: streamlit run app.py

All business logic lives in src/quaestor/.  This file only orchestrates
the UI and calls the library functions.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from quaestor.config import settings
from quaestor.generation.chain import Answer, build_chain
from quaestor.ingestion.chunker import chunk_documents
from quaestor.ingestion.indexer import build_index, load_index
from quaestor.ingestion.loader import load_pdf


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Quaestor — Financial Document Assistant",
    page_icon="📑",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _get_chain():
    """Return the cached RagChain from session state, or None."""
    return st.session_state.get("chain")


def _set_chain(chain) -> None:
    st.session_state["chain"] = chain


# ---------------------------------------------------------------------------
# Indexing logic
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading existing index…")
def _load_existing_index():
    """Try to load an existing persisted index. Returns None on failure."""
    try:
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        return load_index(embeddings=embeddings)
    except Exception:
        return None


def _index_uploaded_files(uploaded_files) -> None:
    """Chunk and index uploaded PDFs into an isolated collection."""
    tmp_dir = Path(".tmp_uploads")
    tmp_dir.mkdir(exist_ok=True)

    all_docs = []
    progress = st.progress(0, text="Reading PDFs…")

    for i, f in enumerate(uploaded_files):
        tmp_path = tmp_dir / f.name
        tmp_path.write_bytes(f.read())
        docs = load_pdf(tmp_path)
        non_empty = [d for d in docs if d.page_content.strip()]
        if not non_empty:
            st.warning(
                f"⚠️ **{f.name}** — no text could be extracted. "
                "This is usually a scanned (image-only) PDF. "
                "OCR support is not included in Phase 1."
            )
        else:
            all_docs.extend(non_empty)
            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Loaded {f.name} ({len(non_empty)} pages with text)",
            )

    if not all_docs:
        st.error("No readable text found in any uploaded file.")
        return

    # Show a content preview so the user can verify extraction quality
    with st.expander("📄 Extraction preview (first 300 chars)", expanded=False):
        st.text(all_docs[0].page_content[:300])

    progress.progress(1.0, text="Chunking…")
    chunks = chunk_documents(all_docs, strategy="fixed")
    st.caption(f"Produced {len(chunks)} chunks from {len(all_docs)} pages.")

    progress.progress(1.0, text="Embedding & indexing (this may take a minute)…")

    try:
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        # Use an isolated collection so uploaded docs never mix with the
        # persisted EDGAR index loaded via "Load existing index".
        vector_store = build_index(
            chunks,
            collection_name="quaestor_upload",
            embeddings=embeddings,
        )
    except Exception as e:
        st.error(
            f"Embedding failed: {e}\n\n"
            "Make sure Ollama is running and nomic-embed-text is pulled:\n"
            "`ollama pull nomic-embed-text`"
        )
        return

    chain = build_chain(vector_store)
    _set_chain(chain)
    progress.empty()
    st.success(
        f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s). "
        "Ready to answer questions."
    )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📑 Quaestor")
    st.caption("Financial document intelligence — ask questions, get cited answers.")

    # Sidebar — document ingestion
    with st.sidebar:
        st.header("Documents")

        # Option A: upload PDFs
        uploaded = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload SEC filings or regulatory standards.",
        )
        if uploaded:
            if st.button("Index uploaded files", type="primary"):
                _index_uploaded_files(uploaded)

        st.divider()

        # Option B: load existing index
        if st.button("Load existing index"):
            vector_store = _load_existing_index()
            if vector_store is not None:
                _set_chain(build_chain(vector_store))
                st.success("Index loaded.")
            else:
                st.warning(
                    "No existing index found at "
                    f"`{settings.chroma_persist_dir}`. "
                    "Upload and index documents first."
                )

        st.divider()
        st.caption(
            f"LLM: {settings.llm_provider.value} · "
            f"Embeddings: {settings.embedding_provider.value} · "
            f"top-k: {settings.retrieval_top_k}"
        )

    # Main area — Q&A
    chain = _get_chain()

    if chain is None:
        st.info(
            "👈 Upload PDFs and click **Index uploaded files**, "
            "or click **Load existing index** to get started."
        )
        return

    st.subheader("Ask a question")
    question = st.text_input(
        "Question",
        placeholder="What was Apple's total revenue in FY2023?",
        label_visibility="collapsed",
    )

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Thinking…"):
            try:
                result: Answer = chain.ask(question)
            except Exception as e:
                st.error(f"Error: {e}")
                return

        st.markdown("### Answer")
        st.markdown(result.answer)

        if result.sources:
            with st.expander("📎 Sources"):
                for src in result.sources:
                    st.markdown(f"- `{src}`")

        st.caption(f"Prompt: {result.prompt_version}")


if __name__ == "__main__":
    main()
