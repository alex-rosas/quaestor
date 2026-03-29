"""Quaestor — Phase 2 Streamlit demo.

Entry point: streamlit run app.py

All business logic lives in src/quaestor/.  This file only orchestrates
the UI and calls the library functions.

Phase 2 additions over Phase 1
-------------------------------
- Chunking strategy selector (fixed / semantic / hierarchical)
- LangGraph retrieval state machine (retrieve → rerank → confidence gate)
- Cross-encoder reranking (sentence-transformers, local, no API cost)
- Input guardrail: Presidio PII detection
- Confidence-gate refusal (low-score queries get a graceful refusal)
- Optional NLI hallucination check on the generated answer
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from quaestor.config import settings
from quaestor.ingestion.chunker import ChunkStrategy, chunk_documents
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
# Cached heavy objects
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading cross-encoder reranker…")
def _get_cross_encoder():
    from quaestor.retrieval.reranker import _default_cross_encoder
    return _default_cross_encoder()


@st.cache_resource(show_spinner="Loading NLI hallucination checker…")
def _get_nli_classifier():
    from quaestor.guardrails.output import _default_classifier
    return _default_classifier()


@st.cache_resource(show_spinner="Loading PII analyzer…")
def _get_pii_engines():
    from quaestor.guardrails.input import _default_analyzer, _default_anonymizer
    return _default_analyzer(), _default_anonymizer()


@st.cache_resource(show_spinner="Loading existing index…")
def _load_existing_index():
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        return load_index(embeddings=embeddings)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _get_vector_store():
    return st.session_state.get("vector_store")

def _set_vector_store(vs) -> None:
    st.session_state["vector_store"] = vs
    # Invalidate cached graph when vector store changes
    st.session_state.pop("rag_graph", None)
    st.session_state.pop("_graph_key", None)

def _get_rag_graph(vs, confidence_threshold: float):
    """Build (or retrieve cached) LangGraph for the current vector store."""
    cache_key = ("rag_graph", id(vs), confidence_threshold)
    if st.session_state.get("_graph_key") != cache_key or "rag_graph" not in st.session_state:
        from quaestor.retrieval.graph import build_rag_graph
        graph = build_rag_graph(
            vector_store=vs,
            cross_encoder=_get_cross_encoder(),
            confidence_threshold=confidence_threshold,
        )
        st.session_state["rag_graph"] = graph
        st.session_state["_graph_key"] = cache_key
    return st.session_state["rag_graph"]


# ---------------------------------------------------------------------------
# Indexing logic
# ---------------------------------------------------------------------------

def _index_uploaded_files(
    uploaded_files,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
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
                "This is usually a scanned (image-only) PDF."
            )
        else:
            all_docs.extend(non_empty)
            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Loaded {f.name} ({len(non_empty)} pages)",
            )

    if not all_docs:
        st.error("No readable text found in any uploaded file.")
        return

    with st.expander("📄 Extraction preview (first 300 chars)", expanded=False):
        st.text(all_docs[0].page_content[:300])

    progress.progress(1.0, text=f"Chunking ({strategy})…")

    try:
        if strategy == "semantic":
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(
                model=settings.ollama_embedding_model,
                base_url=settings.ollama_base_url,
            )
            chunks = chunk_documents(all_docs, strategy="semantic", embeddings=emb)
        else:
            chunks = chunk_documents(
                all_docs,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
    except Exception as e:
        st.error(f"Chunking failed: {e}")
        return

    st.caption(f"Produced **{len(chunks)}** chunks from {len(all_docs)} pages.")

    progress.progress(1.0, text="Embedding & indexing…")

    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        vs = build_index(
            chunks,
            collection_name="quaestor_upload",
            embeddings=embeddings,
        )
    except Exception as e:
        st.error(
            f"Embedding failed: {e}\n\n"
            "Make sure Ollama is running: `ollama pull nomic-embed-text`"
        )
        return

    _set_vector_store(vs)
    progress.empty()
    st.success(
        f"✅ Indexed **{len(chunks)}** chunks from {len(uploaded_files)} file(s). "
        "Ready to answer questions."
    )


# ---------------------------------------------------------------------------
# Sample questions (golden evaluation dataset — 20 questions)
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = {
    "📊 Factual": [
        "What were Apple's total net sales in fiscal year 2025?",
        "What was Apple's net income in fiscal year 2025?",
        "What was Apple's net income in fiscal year 2024?",
        "What was Apple's net income in fiscal year 2023?",
        "What are the primary risk factors Apple cites related to interest rates?",
        "What other primary risk factors does Apple disclose in its 10-K?",
        "On what date was Apple's fiscal year 2025 10-K filing submitted to the SEC?",
        "What does Apple identify as its main product and service categories?",
    ],
    "🔗 Multi-hop": [
        "How did Apple's net income change between fiscal year 2024 and fiscal year 2025, and what percentage growth does that represent?",
        "How did Apple's net income trend across fiscal years 2023, 2024, and 2025?",
        "How did Apple's total net sales compare between fiscal years 2024 and 2025, and what was the approximate growth rate?",
        "What combination of risk factors related to its global supply chain and geographic market concentration does Apple disclose?",
        "What do Apple's risk factor disclosures say about the relationship between macroeconomic conditions and consumer demand for its products?",
        "How does Apple describe its cybersecurity risk and what potential consequences does it disclose?",
        "How does Apple's exposure to foreign exchange risk interact with its international revenue base according to the 10-K?",
    ],
    "🚫 Unanswerable — confidence gate fires": [
        "What is Apple's projected total net sales for fiscal year 2026?",
        "What is Apple's internal target stock price or market capitalisation goal?",
        "How many new employees does Apple plan to hire in fiscal year 2026?",
        "What are Apple's specific plans for entering the autonomous vehicle market?",
        "What dividends per share did Apple pay in fiscal year 2020?",
    ],
}


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📑 Quaestor")
    st.caption(
        "Financial document intelligence · "
        "Phase 2: reranking · confidence gate · PII guardrail"
    )

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("📂 Documents")

        uploaded = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.subheader("Chunking")
        strategy = st.selectbox(
            "Strategy",
            options=["fixed", "hierarchical", "semantic"],
            index=0,
            help=(
                "**fixed** — baseline overlapping windows\n\n"
                "**hierarchical** — parent/child split; best for tables\n\n"
                "**semantic** — embedding-based boundaries (requires Ollama)"
            ),
        )
        col1, col2 = st.columns(2)
        chunk_size = col1.number_input("Chunk size", 128, 2048, 512, 64)
        chunk_overlap = col2.number_input("Overlap", 0, 512, 50, 10)

        if uploaded:
            if st.button("Index uploaded files", type="primary"):
                _index_uploaded_files(
                    uploaded, strategy, chunk_size, chunk_overlap
                )

        st.divider()

        if st.button("Load existing EDGAR index"):
            vs = _load_existing_index()
            if vs is not None:
                _set_vector_store(vs)
                st.success("Index loaded.")
            else:
                st.warning(
                    f"No index found at `{settings.chroma_persist_dir}`. "
                    "Run the smoke test first."
                )

        st.divider()

        st.subheader("⚙️ Phase 2 Settings")
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.5,
            help="Cross-encoder score below this → refusal instead of LLM answer.",
        )
        check_pii = st.checkbox("PII guardrail", value=True,
                                help="Detect PII in your question before sending to LLM.")
        check_hallucination = st.checkbox(
            "Hallucination check (slow)",
            value=False,
            help="NLI check on the answer. Downloads ~85 MB model on first use.",
        )

        st.divider()
        st.caption(
            f"LLM: {settings.llm_provider.value} · "
            f"Embed: {settings.embedding_provider.value} · "
            f"top-k: {settings.retrieval_top_k}"
        )

    # -----------------------------------------------------------------------
    # Main area — Q&A
    # -----------------------------------------------------------------------
    vs = _get_vector_store()

    if vs is None:
        st.info(
            "👈 Upload PDFs and click **Index uploaded files**, "
            "or click **Load existing EDGAR index** to get started."
        )
        return

    # Prefill text input when a sample question button is clicked
    if "_prefill" in st.session_state:
        st.session_state["question_input"] = st.session_state.pop("_prefill")
        st.session_state["_auto_run"] = True

    st.subheader("Ask a question")
    question = st.text_input(
        "Question",
        placeholder="What was Apple's total net sales in fiscal year 2025?",
        label_visibility="collapsed",
        key="question_input",
    )

    col_ask, col_clear = st.columns([1, 6])
    ask_clicked = col_ask.button("Ask", type="primary", disabled=not question)
    auto_run = st.session_state.pop("_auto_run", False)

    # Sample questions panel
    with st.expander("💡 Try these evaluation questions", expanded=not question):
        st.caption(
            "These are the 20 questions from Quaestor's golden evaluation dataset. "
            "The **unanswerable** group demonstrates the confidence gate — "
            "the system refuses rather than hallucinating."
        )
        for g_idx, (group, questions) in enumerate(SAMPLE_QUESTIONS.items()):
            st.markdown(f"**{group}**")
            cols = st.columns(2)
            for i, q in enumerate(questions):
                if cols[i % 2].button(q, key=f"sample_{g_idx}_{i}", use_container_width=True):
                    st.session_state["_prefill"] = q
                    st.rerun()
            st.markdown("")

    if ask_clicked or (auto_run and question):

        display_question = question
        graph_answer = None

        with st.status("Running pipeline…", expanded=True) as status:

            # --- Step 1: PII check ---
            if check_pii:
                st.write("🔍 Scanning for PII…")
                try:
                    analyzer, anonymizer = _get_pii_engines()
                    from quaestor.guardrails.input import detect_pii, redact_pii
                    entities = detect_pii(question, analyzer=analyzer)
                    if entities:
                        result = redact_pii(question, analyzer=analyzer, anonymizer=anonymizer)
                        display_question = result.redacted_text
                        st.warning(
                            f"⚠️ **PII detected** "
                            f"({', '.join(sorted({e.entity_type for e in entities}))}). "
                            "Sending redacted version to the LLM."
                        )
                    else:
                        st.write("✅ No PII found.")
                except Exception as e:
                    st.caption(f"PII check unavailable: {e}")
            else:
                st.write("⏭️ PII check disabled.")

            # --- Step 2: Retrieval + reranking ---
            st.write("📚 Retrieving and reranking passages…")
            try:
                from quaestor.retrieval.graph import run_rag_graph
                graph = _get_rag_graph(vs, confidence_threshold)
            except Exception as e:
                st.error(f"Failed to build pipeline: {e}")
                status.update(label="Pipeline error", state="error")
                st.stop()

            # --- Step 3: Confidence gate + generation ---
            st.write("🧠 Scoring confidence and generating answer…")
            try:
                graph_answer = run_rag_graph(graph, display_question)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                status.update(label="Pipeline error", state="error")
                st.stop()

            # --- Step 4: NLI hallucination check ---
            if check_hallucination and graph_answer and not graph_answer.refused:
                st.write("🛡️ Running hallucination check…")

            # Treat LLM self-refusals the same as gate refusals for display purposes
            _REFUSAL_PHRASE = "don't have enough information"
            is_any_refusal = graph_answer and (
                graph_answer.refused
                or _REFUSAL_PHRASE in (graph_answer.answer or "").lower()
            )

            _REFUSAL_PHRASE = "don't have enough information"
            is_llm_self_refusal = (
                graph_answer and not graph_answer.refused
                and _REFUSAL_PHRASE in (graph_answer.answer or "").lower()
            )
            if graph_answer and graph_answer.refused:
                status.update(label="🔒 Stopped before LLM — retrieval confidence too low", state="complete", expanded=False)
            elif is_llm_self_refusal:
                status.update(label="🤔 LLM reached but couldn't form a confident answer", state="complete", expanded=False)
            else:
                status.update(label="✅ Answer ready", state="complete", expanded=False)

        # --- Display answer ---
        if graph_answer is None:
            st.stop()

        _REFUSAL_PHRASE = "don't have enough information"
        is_llm_self_refusal = (
            not graph_answer.refused
            and _REFUSAL_PHRASE in (graph_answer.answer or "").lower()
        )

        if graph_answer.refused:
            st.info(
                "🔒 **Stopped at retrieval — the model was never called.**\n\n"
                "The system found passages in the indexed documents and scored how relevant "
                "they were to your question. That score fell below the confidence threshold, "
                "so the pipeline stopped there rather than pass weak evidence to the model.\n\n"
                "**What this means:** The answer is likely not in the indexed documents, "
                "or the question may be phrased differently from how it appears in the filing.\n\n"
                "_Try rephrasing, or check that the relevant filing is indexed._"
            )
        elif is_llm_self_refusal:
            st.info(
                "🤔 **Relevant passages were found, but the model couldn't form a confident answer.**\n\n"
                "The retrieval step succeeded — passages were scored, ranked, and sent to the model. "
                "After reading that context, the model determined it still lacked enough "
                "explicit information to give a reliable, cited answer.\n\n"
                "**What this means:** The documents were retrieved correctly, but the specific "
                "detail you're asking for may not be explicitly stated in those passages.\n\n"
                "_Try asking a more specific question, or check a different filing._"
            )
        else:
            st.markdown("### Answer")
            st.markdown(graph_answer.answer)

            # Reranker score badge
            score_color = "green" if graph_answer.top_score >= 1.0 else \
                          "orange" if graph_answer.top_score >= 0.0 else "red"
            st.caption(
                f"Retrieval confidence: :{score_color}[**{graph_answer.top_score:.2f}**]"
            )

            # --- Hallucination check ---
            if check_hallucination and not graph_answer.refused:
                try:
                    from quaestor.guardrails.output import check_hallucination
                    context = " ".join(graph_answer.sources) or graph_answer.answer
                    h_result = check_hallucination(
                        answer=graph_answer.answer,
                        context=context,
                        classifier=_get_nli_classifier(),
                    )
                    if h_result.is_hallucination:
                        st.warning(
                            f"⚠️ **NLI check**: answer may not be fully supported "
                            f"by the retrieved context "
                            f"(entailment score: {h_result.entailment_score:.2f})"
                        )
                    else:
                        st.success(
                            f"✅ **NLI check**: answer appears grounded in context "
                            f"(entailment score: {h_result.entailment_score:.2f})"
                        )
                except Exception as e:
                    st.caption(f"Hallucination check unavailable: {e}")

            # Sources
            if graph_answer.sources:
                with st.expander("📎 Sources"):
                    for src in graph_answer.sources:
                        st.markdown(f"- `{src}`")

        st.caption(f"Prompt: {graph_answer.prompt_version}")


if __name__ == "__main__":
    main()
