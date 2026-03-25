"""Output guardrail — NLI-based hallucination check for Quaestor.

Phase 2 — screens every LLM-generated answer before it is returned to the
user.  The check uses a cross-encoder NLI (Natural Language Inference) model
to estimate whether the *answer* is logically entailed by the *context* that
was passed to the LLM.

Why NLI?
--------
Embedding similarity tells you if two texts are *topically related* but not
whether one *follows from* the other.  An NLI model is trained specifically
to distinguish three relationships:

* ENTAILMENT   — the hypothesis (answer) follows from the premise (context).
* NEUTRAL      — the hypothesis is unrelated or neither follows nor contradicts.
* CONTRADICTION— the hypothesis contradicts the premise.

An answer with low ENTAILMENT probability that doesn't clearly come from the
context is a hallucination candidate.

Model
-----
Default: ``cross-encoder/nli-deberta-v3-small`` (~85 MB, CPU-friendly).
Inject any callable with the same ``({"text": str, "text_pair": str})``
interface for offline testing.

Public API
----------
check_hallucination(answer, context, …) -> HallucinationResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from quaestor.config import settings

logger = logging.getLogger(__name__)

# Default NLI model — small, fast, CPU-friendly
_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# NLI label names as returned by the HuggingFace pipeline for this model
_LABEL_ENTAILMENT = "ENTAILMENT"
_LABEL_NEUTRAL = "NEUTRAL"
_LABEL_CONTRADICTION = "CONTRADICTION"


# ---------------------------------------------------------------------------
# Protocol for injectable classifier
# ---------------------------------------------------------------------------

@runtime_checkable
class NLIClassifier(Protocol):
    """Structural interface for an NLI text-classification pipeline.

    Any callable that accepts ``{"text": str, "text_pair": str}`` and
    returns a list of ``{"label": str, "score": float}`` dicts is accepted.
    This allows offline stubs in tests without downloading the real model.
    """

    def __call__(
        self, inputs: dict[str, str], **kwargs: Any
    ) -> list[dict[str, float]]: ...


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HallucinationResult:
    """Structured output of :func:`check_hallucination`.

    Attributes:
        is_hallucination:    ``True`` if entailment score is below the
                             configured threshold — i.e. the answer cannot
                             be verified from the retrieved context.
        entailment_score:    Model's confidence that the answer is entailed
                             by the context.  Range [0, 1].
        contradiction_score: Model's confidence that the answer contradicts
                             the context.  Range [0, 1].
        neutral_score:       Model's confidence for the NEUTRAL class.
        label:               Highest-scoring NLI label.
        model:               Name of the NLI model used.
    """

    is_hallucination: bool
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    label: str
    model: str = _DEFAULT_NLI_MODEL


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def _default_classifier() -> NLIClassifier:
    """Load the default NLI pipeline (downloaded on first call)."""
    from transformers import pipeline

    return pipeline(  # type: ignore[return-value]
        "text-classification",
        model=_DEFAULT_NLI_MODEL,
        top_k=None,  # return all labels
        device=-1,   # CPU
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_hallucination(
    answer: str,
    context: str,
    classifier: NLIClassifier | None = None,
    entailment_threshold: float = 0.5,
    model_name: str = _DEFAULT_NLI_MODEL,
) -> HallucinationResult:
    """Check whether *answer* is entailed by *context*.

    The NLI model treats *context* as the NLI premise and *answer* as the
    hypothesis.  A low ENTAILMENT score (below *entailment_threshold*)
    indicates the answer introduces claims not supported by the retrieved
    chunks — a hallucination.

    Args:
        answer:               Generated answer text from the LLM.
        context:              Retrieved context string passed to the LLM
                              (concatenation of chunk texts).
        classifier:           Injectable NLI classifier (for offline tests).
                              Defaults to loading *model_name* via
                              ``transformers.pipeline``.
        entailment_threshold: Minimum ENTAILMENT score to consider the answer
                              grounded.  Defaults to ``0.5``.
        model_name:           HuggingFace model identifier.  Ignored when
                              a custom *classifier* is injected.

    Returns:
        :class:`HallucinationResult` with scores and the ``is_hallucination``
        flag.

    Raises:
        ValueError: If *answer* or *context* is empty or blank.
    """
    if not answer or not answer.strip():
        raise ValueError("check_hallucination: answer must be a non-empty string.")
    if not context or not context.strip():
        raise ValueError("check_hallucination: context must be a non-empty string.")

    if classifier is None:
        classifier = _default_classifier()

    # Truncate to avoid model token limits (most NLI models max ~512 tokens)
    # A rough heuristic: 4 chars ≈ 1 token
    max_chars = 1800
    premise = context[:max_chars]
    hypothesis = answer[:max_chars]

    raw: list[dict[str, Any]] = classifier(
        {"text": premise, "text_pair": hypothesis}
    )

    # Normalise: ensure we have a flat list of {"label": str, "score": float}
    # Some pipeline versions return a nested list [[...]] for top_k=None
    if raw and isinstance(raw[0], list):
        raw = raw[0]

    scores: dict[str, float] = {item["label"].upper(): item["score"] for item in raw}

    entailment = scores.get(_LABEL_ENTAILMENT, 0.0)
    contradiction = scores.get(_LABEL_CONTRADICTION, 0.0)
    neutral = scores.get(_LABEL_NEUTRAL, 0.0)
    top_label = max(scores, key=lambda k: scores[k]) if scores else _LABEL_NEUTRAL

    is_hallucination = entailment < entailment_threshold

    logger.info(
        "NLI hallucination check: entailment=%.3f, contradiction=%.3f, "
        "neutral=%.3f → %s",
        entailment,
        contradiction,
        neutral,
        "HALLUCINATION" if is_hallucination else "GROUNDED",
    )

    return HallucinationResult(
        is_hallucination=is_hallucination,
        entailment_score=entailment,
        contradiction_score=contradiction,
        neutral_score=neutral,
        label=top_label,
        model=model_name,
    )
