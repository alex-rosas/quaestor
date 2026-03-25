"""Unit tests for src/quaestor/guardrails/input.py.

A FakeAnalyzer and FakeAnonymizer stub Presidio so spaCy is never loaded
and no network calls are made — all tests are fully offline.
"""

from __future__ import annotations

import pytest

from quaestor.guardrails.input import (
    DEFAULT_ENTITIES,
    PiiEntity,
    RedactionResult,
    detect_pii,
    redact_pii,
)


# ---------------------------------------------------------------------------
# Fakes — offline Presidio substitutes
# ---------------------------------------------------------------------------

class _FakeResult:
    """Mimics a Presidio RecognizerResult."""

    def __init__(self, entity_type: str, start: int, end: int, score: float) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


class FakeAnalyzer:
    """Returns a fixed list of RecognizerResult-like objects.

    Configure *hits* to inject any PII scenario you need.
    """

    def __init__(self, hits: list[_FakeResult] | None = None) -> None:
        self._hits = hits or []

    def analyze(self, text: str, entities, language: str, score_threshold: float):
        return [h for h in self._hits if h.score >= score_threshold]


class FakeAnonymizer:
    """Replaces detected spans with <ENTITY_TYPE> tokens."""

    def anonymize(self, text: str, analyzer_results, operators):
        result = text
        # Apply replacements right-to-left to preserve offsets
        for hit in sorted(analyzer_results, key=lambda x: x.start, reverse=True):
            placeholder = f"<{hit.entity_type}>"
            result = result[: hit.start] + placeholder + result[hit.end :]
        return _FakeAnonymizedResult(result)


class _FakeAnonymizedResult:
    def __init__(self, text: str) -> None:
        self.text = text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _person_hit(text: str, start: int) -> _FakeResult:
    return _FakeResult("PERSON", start, start + len(text), 0.85)

def _email_hit(text: str, start: int) -> _FakeResult:
    return _FakeResult("EMAIL_ADDRESS", start, start + len(text), 1.0)


# ---------------------------------------------------------------------------
# detect_pii — happy path
# ---------------------------------------------------------------------------

class TestDetectPii:
    def test_returns_list(self) -> None:
        result = detect_pii("Hello world", analyzer=FakeAnalyzer())
        assert isinstance(result, list)

    def test_no_pii_returns_empty(self) -> None:
        result = detect_pii("What is the net revenue?", analyzer=FakeAnalyzer())
        assert result == []

    def test_single_entity_detected(self) -> None:
        text = "My name is John"
        hits = [_person_hit("John", 11)]
        result = detect_pii(text, analyzer=FakeAnalyzer(hits))
        assert len(result) == 1
        assert result[0].entity_type == "PERSON"

    def test_multiple_entities_detected(self) -> None:
        text = "John sent john@test.com"
        hits = [_person_hit("John", 0), _email_hit("john@test.com", 10)]
        result = detect_pii(text, analyzer=FakeAnalyzer(hits))
        assert len(result) == 2

    def test_entity_fields_correct(self) -> None:
        text = "Email: john@test.com"
        hits = [_email_hit("john@test.com", 7)]
        result = detect_pii(text, analyzer=FakeAnalyzer(hits))
        entity = result[0]
        assert entity.entity_type == "EMAIL_ADDRESS"
        assert entity.start == 7
        assert entity.end == 7 + len("john@test.com")
        assert entity.text == "john@test.com"
        assert entity.score == 1.0

    def test_sorted_by_start(self) -> None:
        text = "John wrote john@test.com"
        hits = [_email_hit("john@test.com", 11), _person_hit("John", 0)]
        result = detect_pii(text, analyzer=FakeAnalyzer(hits))
        assert result[0].start < result[1].start

    def test_min_score_filters_low_confidence(self) -> None:
        text = "Possibly John"
        low_confidence_hit = _FakeResult("PERSON", 9, 13, 0.3)
        result = detect_pii(
            text,
            min_score=0.5,
            analyzer=FakeAnalyzer([low_confidence_hit]),
        )
        assert result == []

    def test_returns_pii_entity_instances(self) -> None:
        hits = [_person_hit("Alice", 0)]
        result = detect_pii("Alice", analyzer=FakeAnalyzer(hits))
        assert all(isinstance(e, PiiEntity) for e in result)


# ---------------------------------------------------------------------------
# detect_pii — validation
# ---------------------------------------------------------------------------

class TestDetectPiiValidation:
    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            detect_pii("", analyzer=FakeAnalyzer())

    def test_blank_text_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            detect_pii("   ", analyzer=FakeAnalyzer())


# ---------------------------------------------------------------------------
# redact_pii — happy path
# ---------------------------------------------------------------------------

class TestRedactPii:
    def test_returns_redaction_result(self) -> None:
        result = redact_pii(
            "Hello world",
            analyzer=FakeAnalyzer(),
            anonymizer=FakeAnonymizer(),
        )
        assert isinstance(result, RedactionResult)

    def test_no_pii_text_unchanged(self) -> None:
        text = "What is EBITDA?"
        result = redact_pii(text, analyzer=FakeAnalyzer(), anonymizer=FakeAnonymizer())
        assert result.redacted_text == text

    def test_no_pii_has_pii_false(self) -> None:
        result = redact_pii("revenue", analyzer=FakeAnalyzer(), anonymizer=FakeAnonymizer())
        assert result.has_pii is False

    def test_email_redacted(self) -> None:
        text = "Email john@test.com here"
        hits = [_email_hit("john@test.com", 6)]
        result = redact_pii(
            text, analyzer=FakeAnalyzer(hits), anonymizer=FakeAnonymizer()
        )
        assert "john@test.com" not in result.redacted_text
        assert "<EMAIL_ADDRESS>" in result.redacted_text

    def test_person_redacted(self) -> None:
        text = "My name is Alice"
        hits = [_person_hit("Alice", 11)]
        result = redact_pii(
            text, analyzer=FakeAnalyzer(hits), anonymizer=FakeAnonymizer()
        )
        assert "Alice" not in result.redacted_text
        assert "<PERSON>" in result.redacted_text

    def test_has_pii_true_when_entities_found(self) -> None:
        hits = [_person_hit("Bob", 0)]
        result = redact_pii("Bob", analyzer=FakeAnalyzer(hits), anonymizer=FakeAnonymizer())
        assert result.has_pii is True

    def test_entities_populated(self) -> None:
        text = "John john@test.com"
        hits = [_person_hit("John", 0), _email_hit("john@test.com", 5)]
        result = redact_pii(
            text, analyzer=FakeAnalyzer(hits), anonymizer=FakeAnonymizer()
        )
        assert len(result.entities) == 2

    def test_entities_are_pii_entity_instances(self) -> None:
        hits = [_email_hit("a@b.com", 0)]
        result = redact_pii("a@b.com", analyzer=FakeAnalyzer(hits), anonymizer=FakeAnonymizer())
        assert all(isinstance(e, PiiEntity) for e in result.entities)


# ---------------------------------------------------------------------------
# redact_pii — validation
# ---------------------------------------------------------------------------

class TestRedactPiiValidation:
    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            redact_pii("", analyzer=FakeAnalyzer(), anonymizer=FakeAnonymizer())

    def test_blank_text_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            redact_pii("   ", analyzer=FakeAnalyzer(), anonymizer=FakeAnonymizer())


# ---------------------------------------------------------------------------
# DEFAULT_ENTITIES constant
# ---------------------------------------------------------------------------

class TestDefaultEntities:
    def test_is_non_empty_list(self) -> None:
        assert isinstance(DEFAULT_ENTITIES, list)
        assert len(DEFAULT_ENTITIES) > 0

    def test_contains_person(self) -> None:
        assert "PERSON" in DEFAULT_ENTITIES

    def test_contains_email(self) -> None:
        assert "EMAIL_ADDRESS" in DEFAULT_ENTITIES
