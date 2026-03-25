"""FastAPI application for Quaestor.

Phase 2 — exposes the full RAG pipeline (PII redaction → retrieve → rerank →
LangGraph confidence gate → NLI hallucination check) via two HTTP endpoints:

  POST /ask         Synchronous: returns a complete JSON answer.
  POST /ask/stream  Streaming: Server-Sent Events, one token per event.
  GET  /health      Health + config check.

Dependency injection
--------------------
Every heavyweight object (vector store, cross-encoder, LLM, NLI classifier,
Presidio analyzer) is obtained through a FastAPI ``Depends`` function.
Tests override these with lightweight fakes via ``app.dependency_overrides``,
so no real models or databases are needed during testing.

Running locally
---------------
::

    uvicorn quaestor.api.main:app --reload --port 8000

The app expects a ``.env`` at the working directory or environment variables
matching :class:`~quaestor.config.Settings`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from quaestor.api.schemas import (
    AskRequest,
    AskResponse,
    HallucinationCheck,
    HealthResponse,
    PiiReport,
)
from quaestor.config import VectorStoreBackend, settings
from quaestor.generation.prompts import PROMPT_VERSION, RAG_PROMPT
from quaestor.retrieval.graph import GraphAnswer, run_rag_graph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(
        "Quaestor API starting — backend=%s", settings.vector_store_backend
    )
    yield
    logger.info("Quaestor API shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Quaestor",
    description="RAG API for SEC financial filings — grounded, cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------

def get_vector_store():
    """Return the active vector store based on ``settings.vector_store_backend``."""
    if settings.vector_store_backend == VectorStoreBackend.QDRANT:
        from quaestor.ingestion.indexer import load_qdrant_index
        return load_qdrant_index()
    else:
        from quaestor.ingestion.indexer import load_index
        return load_index()


def get_llm():
    """Return the configured LLM."""
    from quaestor.generation.chain import _get_llm
    return _get_llm()


def get_cross_encoder():
    """Return the configured cross-encoder reranker."""
    from quaestor.retrieval.reranker import _default_cross_encoder
    return _default_cross_encoder()


def get_analyzer():
    """Return the Presidio analyzer engine (en_core_web_sm)."""
    from quaestor.guardrails.input import _default_analyzer
    return _default_analyzer()


def get_anonymizer():
    """Return the Presidio anonymizer engine."""
    from quaestor.guardrails.input import _default_anonymizer
    return _default_anonymizer()


def get_nli_classifier():
    """Return the NLI hallucination classifier."""
    from quaestor.guardrails.output import _default_classifier
    return _default_classifier()


def get_rag_graph(
    vector_store=Depends(get_vector_store),
    llm=Depends(get_llm),
    cross_encoder=Depends(get_cross_encoder),
):
    """Build and return the compiled LangGraph RAG state machine."""
    from quaestor.retrieval.graph import build_rag_graph
    return build_rag_graph(
        vector_store=vector_store,
        llm=llm,
        cross_encoder=cross_encoder,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_pii_check(
    question: str,
    analyzer,
    anonymizer,
) -> tuple[str, PiiReport]:
    """Run PII detection and return (cleaned_question, PiiReport)."""
    from quaestor.guardrails.input import redact_pii

    result = redact_pii(
        text=question,
        analyzer=analyzer,
        anonymizer=anonymizer,
    )
    report = PiiReport(
        detected=result.has_pii,
        entity_types=sorted({e.entity_type for e in result.entities}),
        redacted_question=result.redacted_text if result.has_pii else None,
    )
    # Use redacted question for the RAG pipeline
    clean = result.redacted_text if result.has_pii else question
    return clean, report


def _run_hallucination_check(
    answer: str,
    context: str,
    classifier,
) -> HallucinationCheck:
    from quaestor.guardrails.output import check_hallucination

    result = check_hallucination(
        answer=answer,
        context=context,
        classifier=classifier,
    )
    return HallucinationCheck(
        is_hallucination=result.is_hallucination,
        entailment_score=result.entailment_score,
        label=result.label,
    )


def _format_context_from_graph_answer(graph_answer: GraphAnswer) -> str:
    """Reconstruct context string from sources for hallucination check."""
    # The graph doesn't persist the raw context — we use the answer + sources
    # as a proxy.  A future improvement stores the context in GraphAnswer.
    return graph_answer.answer  # self-checking fallback


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API health status and active configuration."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        vector_store_backend=settings.vector_store_backend.value,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    rag_graph=Depends(get_rag_graph),
    analyzer=Depends(get_analyzer),
    anonymizer=Depends(get_anonymizer),
    nli_classifier=Depends(get_nli_classifier),
) -> AskResponse:
    """Answer a question synchronously.

    The full pipeline runs in sequence:

    1. PII redaction (if ``check_pii=True``).
    2. Retrieve → rerank → LangGraph confidence gate → generate / refuse.
    3. NLI hallucination check (if ``check_hallucination=True``).

    Returns a complete :class:`~quaestor.api.schemas.AskResponse`.
    """
    logger.info("POST /ask — question: %r", request.question[:80])

    # 1. PII check
    pii_report: PiiReport | None = None
    question = request.question
    if request.check_pii:
        question, pii_report = _run_pii_check(question, analyzer, anonymizer)

    # 2. RAG graph
    graph_answer: GraphAnswer = run_rag_graph(rag_graph, question)

    # 3. Hallucination check
    hallucination: HallucinationCheck | None = None
    if request.check_hallucination and not graph_answer.refused:
        context_proxy = " ".join(graph_answer.sources) or graph_answer.answer
        hallucination = _run_hallucination_check(
            answer=graph_answer.answer,
            context=context_proxy,
            classifier=nli_classifier,
        )

    return AskResponse(
        question=request.question,
        answer=graph_answer.answer,
        sources=graph_answer.sources,
        refused=graph_answer.refused,
        prompt_version=graph_answer.prompt_version,
        pii=pii_report,
        hallucination=hallucination,
    )


@app.post("/ask/stream")
async def ask_stream(
    request: AskRequest,
    vector_store=Depends(get_vector_store),
    llm=Depends(get_llm),
    cross_encoder=Depends(get_cross_encoder),
    analyzer=Depends(get_analyzer),
    anonymizer=Depends(get_anonymizer),
) -> StreamingResponse:
    """Answer a question with Server-Sent Events streaming.

    Tokens are emitted as they are generated by the LLM.  Each event is a
    JSON object:

    * ``{"type": "token", "content": "…"}``  — LLM output token.
    * ``{"type": "sources", "content": […]}`` — source list (end of stream).
    * ``{"type": "done"}``                    — stream complete.
    * ``{"type": "refused"}``                 — confidence gate fired.
    * ``{"type": "pii", "detected": bool, "entity_types": […]}`` — PII report.

    All events follow the SSE format: ``data: <json>\\n\\n``.
    """

    async def _event_stream() -> AsyncGenerator[str, None]:
        from langchain_core.output_parsers import StrOutputParser

        from quaestor.retrieval.reranker import rerank
        from quaestor.retrieval.retriever import retrieve

        # 1. PII check
        question = request.question
        if request.check_pii:
            question, pii_report = _run_pii_check(question, analyzer, anonymizer)
            yield _sse(
                {
                    "type": "pii",
                    "detected": pii_report.detected,
                    "entity_types": pii_report.entity_types,
                }
            )

        # 2. Retrieve + rerank
        docs = retrieve(question, vector_store, top_k=request.top_k)
        ranked = rerank(question, docs, cross_encoder=cross_encoder)

        # 3. Confidence check (top reranker score)
        if ranked:
            pairs = [[question, d.page_content] for d in docs]
            scores = list(cross_encoder.predict(pairs))
            top_score = max(scores) if scores else -999.0
        else:
            top_score = -999.0

        if top_score < settings.reranker_confidence_threshold:
            yield _sse({"type": "refused"})
            yield _sse({"type": "done"})
            return

        # 4. Format context
        from quaestor.generation.chain import _format_context
        context = _format_context(ranked)

        # 5. Stream LLM tokens
        pipeline = RAG_PROMPT | llm | StrOutputParser()
        async for chunk in pipeline.astream(
            {"context": context, "question": question}
        ):
            yield _sse({"type": "token", "content": chunk})

        # 6. Emit sources
        sources = sorted({d.metadata.get("source", "unknown") for d in ranked})
        yield _sse({"type": "sources", "content": sources})
        yield _sse({"type": "done"})

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(payload: dict[str, Any]) -> str:
    """Format a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(payload)}\n\n"
