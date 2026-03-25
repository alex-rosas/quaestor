"""Unit tests for src/quaestor/guardrails/output.py.

A FakeClassifier stub replaces the real DeBERTa NLI model so no model
download or GPU is required — all tests are fully offline.
"""

from __future__ import annotations

import pytest

from quaestor.guardrails.output import (
    HallucinationResult,
    NLIClassifier,
    check_hallucination,
)


# ---------------------------------------------------------------------------
# Fake NLI classifiers
# ---------------------------------------------------------------------------

class EntailmentClassifier:
    """Always returns ENTAILMENT with high confidence."""

    def __call__(self, inputs: dict, **kwargs) -> list[dict]:
        return [
            {"label": "ENTAILMENT", "score": 0.92},
            {"label": "NEUTRAL", "score": 0.05},
            {"label": "CONTRADICTION", "score": 0.03},
        ]


class ContradictionClassifier:
    """Always returns CONTRADICTION — answer contradicts context."""

    def __call__(self, inputs: dict, **kwargs) -> list[dict]:
        return [
            {"label": "CONTRADICTION", "score": 0.85},
            {"label": "NEUTRAL", "score": 0.10},
            {"label": "ENTAILMENT", "score": 0.05},
        ]


class NeutralClassifier:
    """Returns NEUTRAL — answer is off-topic relative to context."""

    def __call__(self, inputs: dict, **kwargs) -> list[dict]:
        return [
            {"label": "NEUTRAL", "score": 0.80},
            {"label": "ENTAILMENT", "score": 0.15},
            {"label": "CONTRADICTION", "score": 0.05},
        ]


class ExactThresholdClassifier:
    """Returns entailment_score == threshold exactly."""

    def __init__(self, entailment: float) -> None:
        remainder = (1.0 - entailment) / 2
        self._scores = [
            {"label": "ENTAILMENT", "score": entailment},
            {"label": "NEUTRAL", "score": remainder},
            {"label": "CONTRADICTION", "score": remainder},
        ]

    def __call__(self, inputs: dict, **kwargs) -> list[dict]:
        return self._scores


class NestedListClassifier:
    """Simulates a pipeline that returns [[{...}, ...]] (nested list)."""

    def __call__(self, inputs: dict, **kwargs) -> list[list[dict]]:
        return [[
            {"label": "ENTAILMENT", "score": 0.75},
            {"label": "NEUTRAL", "score": 0.20},
            {"label": "CONTRADICTION", "score": 0.05},
        ]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONTEXT = (
    "Apple's total net sales were $383.3 billion in fiscal year 2023, "
    "compared to $394.3 billion in fiscal year 2022."
)
GROUNDED_ANSWER = "Apple's net sales were $383.3 billion in FY2023."
HALLUCINATED_ANSWER = "Apple's net sales were $500 billion in FY2023."


# ---------------------------------------------------------------------------
# check_hallucination — grounded path
# ---------------------------------------------------------------------------

class TestGroundedAnswer:
    _clf = EntailmentClassifier()

    def test_returns_hallucination_result(self) -> None:
        result = check_hallucination(GROUNDED_ANSWER, CONTEXT, classifier=self._clf)
        assert isinstance(result, HallucinationResult)

    def test_not_hallucination(self) -> None:
        result = check_hallucination(GROUNDED_ANSWER, CONTEXT, classifier=self._clf)
        assert result.is_hallucination is False

    def test_entailment_score_high(self) -> None:
        result = check_hallucination(GROUNDED_ANSWER, CONTEXT, classifier=self._clf)
        assert result.entailment_score > 0.5

    def test_label_is_entailment(self) -> None:
        result = check_hallucination(GROUNDED_ANSWER, CONTEXT, classifier=self._clf)
        assert result.label == "ENTAILMENT"

    def test_scores_present(self) -> None:
        result = check_hallucination(GROUNDED_ANSWER, CONTEXT, classifier=self._clf)
        assert 0.0 <= result.entailment_score <= 1.0
        assert 0.0 <= result.contradiction_score <= 1.0
        assert 0.0 <= result.neutral_score <= 1.0


# ---------------------------------------------------------------------------
# check_hallucination — hallucination path (contradiction / neutral)
# ---------------------------------------------------------------------------

class TestHallucinatedAnswer:
    def test_contradiction_triggers_hallucination(self) -> None:
        result = check_hallucination(
            HALLUCINATED_ANSWER, CONTEXT, classifier=ContradictionClassifier()
        )
        assert result.is_hallucination is True

    def test_neutral_triggers_hallucination_below_threshold(self) -> None:
        result = check_hallucination(
            HALLUCINATED_ANSWER,
            CONTEXT,
            classifier=NeutralClassifier(),
            entailment_threshold=0.5,
        )
        # neutral classifier returns entailment=0.15 < 0.5
        assert result.is_hallucination is True

    def test_contradiction_label_set(self) -> None:
        result = check_hallucination(
            HALLUCINATED_ANSWER, CONTEXT, classifier=ContradictionClassifier()
        )
        assert result.label == "CONTRADICTION"


# ---------------------------------------------------------------------------
# check_hallucination — threshold boundary
# ---------------------------------------------------------------------------

class TestThresholdBoundary:
    def test_exactly_at_threshold_is_not_hallucination(self) -> None:
        """Entailment == threshold should NOT be flagged (>= condition)."""
        clf = ExactThresholdClassifier(entailment=0.5)
        result = check_hallucination(
            GROUNDED_ANSWER, CONTEXT, classifier=clf, entailment_threshold=0.5
        )
        assert result.is_hallucination is False

    def test_just_below_threshold_is_hallucination(self) -> None:
        clf = ExactThresholdClassifier(entailment=0.49)
        result = check_hallucination(
            GROUNDED_ANSWER, CONTEXT, classifier=clf, entailment_threshold=0.5
        )
        assert result.is_hallucination is True

    def test_custom_threshold_respected(self) -> None:
        """High threshold forces hallucination even for EntailmentClassifier."""
        result = check_hallucination(
            GROUNDED_ANSWER, CONTEXT,
            classifier=EntailmentClassifier(),
            entailment_threshold=0.99,
        )
        assert result.is_hallucination is True


# ---------------------------------------------------------------------------
# check_hallucination — nested list pipeline output
# ---------------------------------------------------------------------------

class TestNestedListOutput:
    def test_handles_nested_list(self) -> None:
        """Pipeline returning [[{...}]] should be flattened correctly."""
        result = check_hallucination(
            GROUNDED_ANSWER, CONTEXT, classifier=NestedListClassifier()
        )
        assert isinstance(result, HallucinationResult)
        assert result.entailment_score == pytest.approx(0.75)
        assert result.is_hallucination is False


# ---------------------------------------------------------------------------
# check_hallucination — validation
# ---------------------------------------------------------------------------

class TestValidation:
    _clf = EntailmentClassifier()

    def test_empty_answer_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            check_hallucination("", CONTEXT, classifier=self._clf)

    def test_blank_answer_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            check_hallucination("   ", CONTEXT, classifier=self._clf)

    def test_empty_context_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            check_hallucination(GROUNDED_ANSWER, "", classifier=self._clf)

    def test_blank_context_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            check_hallucination(GROUNDED_ANSWER, "   ", classifier=self._clf)


# ---------------------------------------------------------------------------
# NLIClassifier protocol
# ---------------------------------------------------------------------------

class TestNLIClassifierProtocol:
    def test_entailment_classifier_satisfies_protocol(self) -> None:
        assert isinstance(EntailmentClassifier(), NLIClassifier)

    def test_contradiction_classifier_satisfies_protocol(self) -> None:
        assert isinstance(ContradictionClassifier(), NLIClassifier)


# ---------------------------------------------------------------------------
# HallucinationResult dataclass
# ---------------------------------------------------------------------------

class TestHallucinationResult:
    def test_fields_accessible(self) -> None:
        r = HallucinationResult(
            is_hallucination=False,
            entailment_score=0.9,
            contradiction_score=0.05,
            neutral_score=0.05,
            label="ENTAILMENT",
        )
        assert r.is_hallucination is False
        assert r.entailment_score == pytest.approx(0.9)
        assert r.label == "ENTAILMENT"

    def test_default_model_name(self) -> None:
        r = HallucinationResult(
            is_hallucination=False,
            entailment_score=0.9,
            contradiction_score=0.05,
            neutral_score=0.05,
            label="ENTAILMENT",
        )
        assert "deberta" in r.model.lower() or "nli" in r.model.lower()
