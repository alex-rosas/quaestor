"""Pydantic request / response models for the Quaestor API.

All models use ``model_config = ConfigDict(frozen=True)`` so they are
hashable and safe to cache.  Field names follow snake_case throughout —
FastAPI serialises them to camelCase only if the client requests it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class HallucinationCheck(BaseModel):
    """NLI result embedded in :class:`AskResponse`."""

    model_config = ConfigDict(frozen=True)

    is_hallucination: bool = Field(
        description="True when the answer cannot be verified from the context."
    )
    entailment_score: float = Field(
        description="Model confidence that the answer is entailed by the context."
    )
    label: str = Field(
        description="Top NLI label: ENTAILMENT, NEUTRAL, or CONTRADICTION."
    )


class PiiReport(BaseModel):
    """PII summary embedded in :class:`AskResponse`."""

    model_config = ConfigDict(frozen=True)

    detected: bool = Field(description="True if any PII was found in the query.")
    entity_types: list[str] = Field(
        default_factory=list,
        description="Distinct PII entity types detected (e.g. PERSON, EMAIL_ADDRESS).",
    )
    redacted_question: str | None = Field(
        default=None,
        description="Question text after PII has been replaced with placeholders.",
    )


# ---------------------------------------------------------------------------
# /ask  (sync)
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Body for ``POST /ask`` and ``POST /ask/stream``."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(
        min_length=1,
        description="Natural-language question about the indexed documents.",
    )
    top_k: int | None = Field(
        default=None,
        gt=0,
        description="Override the number of chunks to retrieve.",
    )
    check_pii: bool = Field(
        default=True,
        description="Screen the question for PII before sending to the LLM.",
    )
    check_hallucination: bool = Field(
        default=False,
        description="Run NLI hallucination check on the generated answer.",
    )


class AskResponse(BaseModel):
    """Body returned by ``POST /ask``."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(description="Original user question.")
    answer: str = Field(description="Generated answer (may be a refusal).")
    sources: list[str] = Field(
        default_factory=list,
        description="Deduplicated source filenames cited in the answer.",
    )
    refused: bool = Field(
        description="True when the retrieval confidence was too low to answer."
    )
    prompt_version: str = Field(description="Prompt template version used.")
    pii: PiiReport | None = Field(
        default=None,
        description="PII screening result (populated when check_pii=True).",
    )
    hallucination: HallucinationCheck | None = Field(
        default=None,
        description="NLI result (populated when check_hallucination=True).",
    )


# ---------------------------------------------------------------------------
# /retrieve
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    """Body for ``POST /retrieve``."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        min_length=1,
        description="Natural-language search query.",
    )
    top_k: int = Field(
        default=5,
        gt=0,
        description="Number of chunks to return.",
    )


class RetrieveChunk(BaseModel):
    """Single retrieved chunk in :class:`RetrieveResponse`."""

    model_config = ConfigDict(frozen=True)

    chunk_text: str = Field(description="Raw text content of the chunk.")
    score: float = Field(description="Cosine similarity score (higher = more relevant).")
    metadata: dict = Field(
        default_factory=dict,
        description="Document metadata (source, page, section, etc.).",
    )


class RetrieveResponse(BaseModel):
    """Body returned by ``POST /retrieve``."""

    model_config = ConfigDict(frozen=True)

    results: list[RetrieveChunk] = Field(
        default_factory=list,
        description="Retrieved chunks ordered by relevance.",
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Body returned by ``GET /health``."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")
    vector_store_backend: str = Field(
        description="Active vector store backend (chroma | qdrant)."
    )
