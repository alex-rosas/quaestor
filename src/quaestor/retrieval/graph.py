"""LangGraph retrieval state machine for Quaestor.

Phase 2 — wraps the retrieve → rerank → generate pipeline in a LangGraph
state machine with confidence thresholding and unanswerable refusal.

Why a state machine?
--------------------
The plain ``RagChain`` (Phase 1) always answers.  Financial QA needs a harder
guarantee: if retrieved evidence is too weak, the system must *refuse* rather
than hallucinate a plausible-sounding answer.  A cross-encoder score is a good
proxy for retrieval confidence — when the top-ranked chunk scores below
``settings.reranker_confidence_threshold`` we can be fairly sure the corpus
does not contain a relevant answer.

Graph topology
--------------
::

    START
      │
    [retrieve]        dense top-k similarity search
      │
    [rerank]          cross-encoder rescoring; top score stored in state
      │
    [check_confidence]─── score ≥ threshold ──► [generate] ─► END
                      │                                         ▲
                      └─── score < threshold  ──► [refuse]  ───┘

Nodes
-----
retrieve          Calls :func:`~quaestor.retrieval.retriever.retrieve`.
rerank            Scores every (query, chunk) pair with the cross-encoder,
                  sorts descending, stores top score + ranked docs in state.
generate          Formats context, calls the LLM, builds a ``GraphAnswer``.
refuse            Skips LLM; returns a canned refusal with ``refused=True``.

Public API
----------
build_rag_graph(vector_store, …) -> CompiledGraph
run_rag_graph(graph, question)   -> GraphAnswer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from quaestor.config import settings
from quaestor.generation.prompts import PROMPT_VERSION, RAG_PROMPT
from quaestor.retrieval.retriever import retrieve

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.language_models import BaseChatModel
    from langchain_core.output_parsers import StrOutputParser

    from quaestor.retrieval.reranker import CrossEncoderProtocol

logger = logging.getLogger(__name__)

# Canned refusal message — matches the wording in the Phase 1 prompt so
# downstream tests can use the same assertion string.
_REFUSAL_TEXT = (
    "I don't have enough information in the provided documents to answer this."
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    """Mutable state threaded through every graph node.

    Attributes:
        question:     Original user question.
        docs:         Chunks after reranking (best-first).
        top_score:    Cross-encoder score of the highest-ranked chunk.
        answer:       Generated or refusal text (populated at end).
        sources:      Deduplicated source filenames cited in the answer.
        refused:      True when the confidence check triggered a refusal.
        prompt_version: Prompt template version used during generation.
    """

    question: str
    docs: list[Document]
    top_score: float
    answer: str
    sources: list[str]
    refused: bool
    prompt_version: str


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------

@dataclass
class GraphAnswer:
    """Final structured output from :func:`run_rag_graph`.

    Attributes:
        question:       Original user question.
        answer:         Generated answer text (with inline citations) or
                        the canned refusal string.
        sources:        Deduplicated list of source filenames cited.
        refused:        ``True`` if the confidence threshold was not met.
        top_score:      Cross-encoder score of the best-ranked chunk.
        prompt_version: Prompt template version used.
    """

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    refused: bool = False
    top_score: float = 0.0
    prompt_version: str = PROMPT_VERSION


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _make_retrieve_node(vector_store: Chroma, top_k: int):
    """Return a LangGraph node that performs dense retrieval."""

    def _retrieve(state: RAGState) -> dict[str, Any]:
        docs = retrieve(state["question"], vector_store, top_k=top_k)
        logger.debug("retrieve node: %d chunk(s) returned", len(docs))
        return {"docs": docs}

    return _retrieve


def _make_rerank_node(cross_encoder: CrossEncoderProtocol, top_n: int | None):
    """Return a LangGraph node that reranks chunks and records the top score."""

    def _rerank(state: RAGState) -> dict[str, Any]:
        docs = state["docs"]
        if not docs:
            return {"docs": [], "top_score": -999.0}

        pairs = [[state["question"], doc.page_content] for doc in docs]
        raw_scores: list[float] = list(cross_encoder.predict(pairs))

        scored = sorted(zip(raw_scores, docs), key=lambda x: x[0], reverse=True)
        ranked_docs = [doc for _, doc in scored]
        top_score = float(scored[0][0])

        if top_n is not None:
            ranked_docs = ranked_docs[:top_n]

        logger.debug("rerank node: top_score=%.4f, kept %d chunk(s)", top_score, len(ranked_docs))
        return {"docs": ranked_docs, "top_score": top_score}

    return _rerank


def _make_generate_node(llm: BaseChatModel):
    """Return a LangGraph node that calls the LLM and produces an answer."""

    from langchain_core.output_parsers import StrOutputParser

    pipeline = RAG_PROMPT | llm | StrOutputParser()

    def _generate(state: RAGState) -> dict[str, Any]:
        docs = state["docs"]
        context = _format_context(docs)
        raw_answer: str = pipeline.invoke(
            {"context": context, "question": state["question"]}
        )
        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
        logger.debug("generate node: answer produced, sources=%s", sources)
        return {
            "answer": raw_answer,
            "sources": sources,
            "refused": False,
            "prompt_version": PROMPT_VERSION,
        }

    return _generate


def _refuse_node(state: RAGState) -> dict[str, Any]:
    """LangGraph node that returns a canned refusal without calling the LLM."""
    logger.debug(
        "refuse node: top_score=%.4f below threshold — refusing",
        state.get("top_score", -999.0),
    )
    return {
        "answer": _REFUSAL_TEXT,
        "sources": [],
        "refused": True,
        "prompt_version": PROMPT_VERSION,
    }


def _make_confidence_router(threshold: float):
    """Return a conditional-edge function that routes on cross-encoder score."""

    def _route(state: RAGState) -> Literal["generate", "refuse"]:
        if state["top_score"] >= threshold:
            return "generate"
        return "refuse"

    return _route


# ---------------------------------------------------------------------------
# Context formatter (same logic as chain.py — kept local to avoid coupling)
# ---------------------------------------------------------------------------

def _format_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_rag_graph(
    vector_store: Chroma,
    llm: BaseChatModel | None = None,
    cross_encoder: CrossEncoderProtocol | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
    confidence_threshold: float | None = None,
):
    """Compile and return the RAG LangGraph state machine.

    Args:
        vector_store:         Indexed ChromaDB collection.
        llm:                  Language model for answer generation.  Defaults
                              to the provider in ``settings``.
        cross_encoder:        Cross-encoder for reranking.  Defaults to
                              loading ``settings.reranker_model``.
        top_k:                Candidates to fetch from the vector store.
                              Defaults to ``settings.retrieval_top_k``.
        top_n:                Chunks to keep after reranking and pass to the
                              LLM.  ``None`` keeps all reranked chunks.
        confidence_threshold: Minimum cross-encoder score to attempt
                              generation.  Defaults to
                              ``settings.reranker_confidence_threshold``.

    Returns:
        A compiled LangGraph graph whose ``invoke({"question": …})`` method
        returns the final :class:`RAGState` dict.  Use :func:`run_rag_graph`
        to convert it to a :class:`GraphAnswer`.
    """
    # Resolve defaults
    if llm is None:
        from quaestor.generation.chain import _get_llm
        llm = _get_llm()

    if cross_encoder is None:
        from quaestor.retrieval.reranker import _default_cross_encoder
        cross_encoder = _default_cross_encoder()

    _top_k = top_k if top_k is not None else settings.retrieval_top_k
    _threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.reranker_confidence_threshold
    )

    # Build nodes
    retrieve_node = _make_retrieve_node(vector_store, _top_k)
    rerank_node = _make_rerank_node(cross_encoder, top_n)
    generate_node = _make_generate_node(llm)

    # Assemble graph
    builder: StateGraph = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)
    builder.add_node("refuse", _refuse_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_conditional_edges(
        "rerank",
        _make_confidence_router(_threshold),
        {"generate": "generate", "refuse": "refuse"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("refuse", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_rag_graph(graph: Any, question: str) -> GraphAnswer:
    """Run *graph* for *question* and return a structured :class:`GraphAnswer`.

    Args:
        graph:    Compiled graph from :func:`build_rag_graph`.
        question: Natural-language question from the user.

    Returns:
        :class:`GraphAnswer` with the answer text, sources, and metadata.

    Raises:
        ValueError: If *question* is empty or blank.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")

    initial_state: RAGState = {
        "question": question,
        "docs": [],
        "top_score": 0.0,
        "answer": "",
        "sources": [],
        "refused": False,
        "prompt_version": PROMPT_VERSION,
    }

    final_state: RAGState = graph.invoke(initial_state)

    return GraphAnswer(
        question=question,
        answer=final_state["answer"],
        sources=final_state.get("sources", []),
        refused=final_state.get("refused", False),
        top_score=final_state.get("top_score", 0.0),
        prompt_version=final_state.get("prompt_version", PROMPT_VERSION),
    )
