"""Input guardrail — PII detection and redaction for Quaestor.

Phase 2 — screens every user query before it enters the RAG pipeline.
Financial analysts often paste raw document excerpts containing names, emails,
SSNs, or account numbers into queries.  Exposing that data to a remote LLM API
(Groq, Together) without sanitisation is a compliance risk.

Detection is done with Microsoft Presidio, which combines regex-based
``PatternRecognizer`` rules (email, phone, credit card, US SSN, IP address)
with a spaCy NER model (PERSON, ORGANIZATION, LOCATION).

Public API
----------
detect_pii(text, …)  -> list[PiiEntity]   — identify PII spans
redact_pii(text, …)  -> RedactionResult   — replace PII with type placeholders
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

# Default entity types to screen.  The list covers the PII most likely
# to appear in financial document queries.
DEFAULT_ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "US_BANK_NUMBER",
    "IBAN_CODE",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PiiEntity:
    """A single PII span detected in the input text.

    Attributes:
        entity_type: Presidio entity label (e.g. ``"PERSON"``).
        start:       Character offset of the span start.
        end:         Character offset of the span end (exclusive).
        score:       Confidence score in [0, 1].
        text:        The detected PII text.
    """

    entity_type: str
    start: int
    end: int
    score: float
    text: str


@dataclass
class RedactionResult:
    """Output of :func:`redact_pii`.

    Attributes:
        redacted_text: Input with PII replaced by ``<ENTITY_TYPE>`` tokens.
        entities:      List of all detected PII entities (pre-redaction).
        has_pii:       ``True`` if at least one entity was detected.
    """

    redacted_text: str
    entities: list[PiiEntity] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return len(self.entities) > 0


# ---------------------------------------------------------------------------
# Engine factories
# ---------------------------------------------------------------------------

def _default_analyzer() -> AnalyzerEngine:
    """Return an :class:`AnalyzerEngine` backed by ``en_core_web_sm``.

    Using the small spaCy model avoids a 500 MB download while still
    catching PERSON and ORGANIZATION entities.  Pattern-based recognizers
    (email, SSN, credit card) do not depend on the NLP model.
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    return AnalyzerEngine(nlp_engine=provider.create_engine())


def _default_anonymizer() -> AnonymizerEngine:
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_pii(
    text: str,
    entities: list[str] | None = None,
    min_score: float = 0.5,
    analyzer: AnalyzerEngine | None = None,
    language: str = "en",
) -> list[PiiEntity]:
    """Identify PII spans in *text*.

    Args:
        text:      Input string to screen (e.g. a user query).
        entities:  Entity types to detect.  Defaults to
                   :data:`DEFAULT_ENTITIES`.
        min_score: Minimum confidence to report an entity.  Defaults to
                   ``0.5``.
        analyzer:  Inject a custom :class:`~presidio_analyzer.AnalyzerEngine`
                   (used in tests to avoid loading spaCy).
        language:  Language code for the NLP engine.  Defaults to ``"en"``.

    Returns:
        List of :class:`PiiEntity` objects ordered by ``start`` position.
        Empty list if no PII is detected.

    Raises:
        ValueError: If *text* is empty or blank.
    """
    if not text or not text.strip():
        raise ValueError("detect_pii: text must be a non-empty string.")

    entities = entities or DEFAULT_ENTITIES
    analyzer = analyzer or _default_analyzer()

    raw_results = analyzer.analyze(
        text=text,
        entities=entities,
        language=language,
        score_threshold=min_score,
    )

    detected = [
        PiiEntity(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=r.score,
            text=text[r.start : r.end],
        )
        for r in sorted(raw_results, key=lambda x: x.start)
    ]

    if detected:
        logger.warning(
            "PII detected in input — %d entity/entities: %s",
            len(detected),
            [e.entity_type for e in detected],
        )
    else:
        logger.debug("No PII detected in input.")

    return detected


def redact_pii(
    text: str,
    entities: list[str] | None = None,
    min_score: float = 0.5,
    analyzer: AnalyzerEngine | None = None,
    anonymizer: AnonymizerEngine | None = None,
    language: str = "en",
) -> RedactionResult:
    """Detect and replace PII in *text* with ``<ENTITY_TYPE>`` placeholders.

    For example, ``"Email John at john@corp.com"`` becomes
    ``"Email <PERSON> at <EMAIL_ADDRESS>"``.

    Args:
        text:       Input string to sanitise.
        entities:   Entity types to redact.  Defaults to
                    :data:`DEFAULT_ENTITIES`.
        min_score:  Minimum confidence threshold.  Defaults to ``0.5``.
        analyzer:   Inject a custom :class:`~presidio_analyzer.AnalyzerEngine`.
        anonymizer: Inject a custom
                    :class:`~presidio_anonymizer.AnonymizerEngine`.
        language:   Language code.  Defaults to ``"en"``.

    Returns:
        :class:`RedactionResult` with the sanitised text and entity list.

    Raises:
        ValueError: If *text* is empty or blank.
    """
    if not text or not text.strip():
        raise ValueError("redact_pii: text must be a non-empty string.")

    entities = entities or DEFAULT_ENTITIES
    analyzer = analyzer or _default_analyzer()
    anonymizer = anonymizer or _default_anonymizer()

    from presidio_anonymizer.entities import OperatorConfig

    raw_results = analyzer.analyze(
        text=text,
        entities=entities,
        language=language,
        score_threshold=min_score,
    )

    detected = [
        PiiEntity(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=r.score,
            text=text[r.start : r.end],
        )
        for r in sorted(raw_results, key=lambda x: x.start)
    ]

    if raw_results:
        # Replace each entity with <ENTITY_TYPE>
        operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in entities
        }
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=raw_results,
            operators=operators,
        )
        redacted_text = anonymized.text
    else:
        redacted_text = text

    logger.debug(
        "redact_pii: %d entity/entities redacted from input.", len(detected)
    )
    return RedactionResult(redacted_text=redacted_text, entities=detected)
