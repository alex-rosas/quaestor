"""Unit tests for eval/evaluate.py.

Tests cover all offline-testable logic: dataset loading, validation,
sample construction, and RAGAS dataset conversion.  Functions that require
a live LLM or real vector store (run_pipeline_on_dataset,
run_ragas_evaluation) are tested with lightweight fakes via dependency
injection.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Adjust path so eval/ is importable from tests/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluate import (
    GOLDEN_DATASET_PATH,
    EvaluationSample,
    GoldenQuestion,
    build_evaluation_sample,
    load_golden_dataset,
    to_ragas_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_question(
    id: str = "Q01",
    q_type: str = "factual",
    question: str = "What was Apple's revenue?",
    ground_truth: str = "Apple's revenue was $416 billion.",
) -> GoldenQuestion:
    return GoldenQuestion(
        id=id,
        type=q_type,
        question=question,
        ground_truth=ground_truth,
        source_pages=[27],
        notes="test",
    )


def make_sample(
    question: str = "What was Apple's revenue?",
    answer: str = "Apple's revenue was $416 billion.",
    contexts: list[str] | None = None,
    ground_truth: str = "Apple's revenue was $416 billion.",
    refused: bool = False,
) -> EvaluationSample:
    return EvaluationSample(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        retrieved_contexts=contexts or ["Apple revenue was $416.1 billion."],
        refused=refused,
    )


def make_minimal_dataset_json(questions: list[dict]) -> dict:
    return {
        "version": "1.0",
        "created": "2026-03-25",
        "source_filing": "test",
        "description": "test",
        "question_types": {},
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# load_golden_dataset — the real file
# ---------------------------------------------------------------------------

class TestLoadGoldenDataset:
    def test_loads_real_file(self) -> None:
        questions = load_golden_dataset(GOLDEN_DATASET_PATH)
        assert len(questions) == 20

    def test_returns_golden_question_instances(self) -> None:
        questions = load_golden_dataset(GOLDEN_DATASET_PATH)
        assert all(isinstance(q, GoldenQuestion) for q in questions)

    def test_correct_type_counts(self) -> None:
        questions = load_golden_dataset(GOLDEN_DATASET_PATH)
        types = [q.type for q in questions]
        assert types.count("factual") == 8
        assert types.count("multi_hop") == 7
        assert types.count("unanswerable") == 5

    def test_all_questions_non_empty(self) -> None:
        questions = load_golden_dataset(GOLDEN_DATASET_PATH)
        assert all(q.question for q in questions)

    def test_all_ground_truths_non_empty(self) -> None:
        questions = load_golden_dataset(GOLDEN_DATASET_PATH)
        assert all(q.ground_truth for q in questions)

    def test_all_ids_unique(self) -> None:
        questions = load_golden_dataset(GOLDEN_DATASET_PATH)
        ids = [q.id for q in questions]
        assert len(set(ids)) == len(ids)

    def test_ids_sequential(self) -> None:
        questions = load_golden_dataset(GOLDEN_DATASET_PATH)
        ids = [q.id for q in questions]
        assert ids == [f"Q{i:02d}" for i in range(1, 21)]

    def test_unanswerable_have_empty_source_pages(self) -> None:
        questions = load_golden_dataset(GOLDEN_DATASET_PATH)
        for q in questions:
            if q.type == "unanswerable":
                assert q.source_pages == []


# ---------------------------------------------------------------------------
# load_golden_dataset — error cases
# ---------------------------------------------------------------------------

class TestLoadGoldenDatasetErrors:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_golden_dataset(Path("/nonexistent/golden_dataset.json"))

    def test_unknown_type_raises(self) -> None:
        bad = make_minimal_dataset_json([{
            "id": "Q01", "type": "UNKNOWN",
            "question": "q?", "ground_truth": "a.",
        }])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bad, f)
            tmp = Path(f.name)
        with pytest.raises(ValueError, match="unknown type"):
            load_golden_dataset(tmp)
        tmp.unlink()

    def test_empty_question_raises(self) -> None:
        bad = make_minimal_dataset_json([{
            "id": "Q01", "type": "factual",
            "question": "", "ground_truth": "a.",
        }])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bad, f)
            tmp = Path(f.name)
        with pytest.raises(ValueError, match="empty question"):
            load_golden_dataset(tmp)
        tmp.unlink()

    def test_empty_ground_truth_raises(self) -> None:
        bad = make_minimal_dataset_json([{
            "id": "Q01", "type": "factual",
            "question": "q?", "ground_truth": "",
        }])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bad, f)
            tmp = Path(f.name)
        with pytest.raises(ValueError, match="empty ground_truth"):
            load_golden_dataset(tmp)
        tmp.unlink()


# ---------------------------------------------------------------------------
# build_evaluation_sample
# ---------------------------------------------------------------------------

class TestBuildEvaluationSample:
    def test_returns_evaluation_sample(self) -> None:
        golden = make_question()
        sample = build_evaluation_sample(golden, "answer", ["context"])
        assert isinstance(sample, EvaluationSample)

    def test_question_passed_through(self) -> None:
        golden = make_question(question="Revenue?")
        sample = build_evaluation_sample(golden, "answer", ["ctx"])
        assert sample.question == "Revenue?"

    def test_ground_truth_passed_through(self) -> None:
        golden = make_question(ground_truth="$416 billion")
        sample = build_evaluation_sample(golden, "answer", ["ctx"])
        assert sample.ground_truth == "$416 billion"

    def test_answer_passed_through(self) -> None:
        golden = make_question()
        sample = build_evaluation_sample(golden, "Generated answer", ["ctx"])
        assert sample.answer == "Generated answer"

    def test_contexts_passed_through(self) -> None:
        golden = make_question()
        contexts = ["chunk A", "chunk B"]
        sample = build_evaluation_sample(golden, "answer", contexts)
        assert sample.retrieved_contexts == contexts

    def test_refused_defaults_false(self) -> None:
        golden = make_question()
        sample = build_evaluation_sample(golden, "answer", ["ctx"])
        assert sample.refused is False

    def test_refused_passed_through(self) -> None:
        golden = make_question(q_type="unanswerable")
        sample = build_evaluation_sample(golden, "I don't know", [], refused=True)
        assert sample.refused is True

    def test_question_type_propagated(self) -> None:
        golden = make_question(q_type="multi_hop")
        sample = build_evaluation_sample(golden, "answer", ["ctx"])
        assert sample.question_type == "multi_hop"


# ---------------------------------------------------------------------------
# to_ragas_dataset
# ---------------------------------------------------------------------------

class TestToRagasDataset:
    def test_returns_evaluation_dataset(self) -> None:
        from ragas.dataset_schema import EvaluationDataset
        samples = [make_sample()]
        dataset = to_ragas_dataset(samples)
        assert isinstance(dataset, EvaluationDataset)

    def test_sample_count_matches(self) -> None:
        samples = [make_sample() for _ in range(5)]
        dataset = to_ragas_dataset(samples)
        assert len(dataset.samples) == 5

    def test_user_input_mapped(self) -> None:
        samples = [make_sample(question="Revenue?")]
        dataset = to_ragas_dataset(samples)
        assert dataset.samples[0].user_input == "Revenue?"

    def test_response_mapped(self) -> None:
        samples = [make_sample(answer="$416 billion")]
        dataset = to_ragas_dataset(samples)
        assert dataset.samples[0].response == "$416 billion"

    def test_retrieved_contexts_mapped(self) -> None:
        contexts = ["chunk 1", "chunk 2"]
        samples = [make_sample(contexts=contexts)]
        dataset = to_ragas_dataset(samples)
        assert dataset.samples[0].retrieved_contexts == contexts

    def test_reference_mapped(self) -> None:
        samples = [make_sample(ground_truth="ref answer")]
        dataset = to_ragas_dataset(samples)
        assert dataset.samples[0].reference == "ref answer"

    def test_empty_samples_returns_empty_dataset(self) -> None:
        dataset = to_ragas_dataset([])
        assert len(dataset.samples) == 0


# ---------------------------------------------------------------------------
# run_pipeline_on_dataset — offline test via fake vector store + LLM
# ---------------------------------------------------------------------------

class FakeLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None, **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="$416 billion"))]
        )


class FakeVectorStore:
    """Minimal duck-typed vector store for pipeline tests."""

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        return [
            Document(
                page_content=f"Apple revenue context {i}",
                metadata={"source": "apple_10k.pdf", "page": i},
            )
            for i in range(min(k, 3))
        ]


class HighScoreCrossEncoder:
    def predict(self, sentences: list) -> list[float]:
        return [5.0] * len(sentences)


class TestRunPipelineOnDataset:
    def test_returns_one_sample_per_question(self) -> None:
        from eval.evaluate import run_pipeline_on_dataset

        questions = [make_question(id=f"Q{i:02d}") for i in range(1, 4)]
        samples = run_pipeline_on_dataset(
            questions=questions,
            vector_store=FakeVectorStore(),
            llm=FakeLLM(),
            cross_encoder=HighScoreCrossEncoder(),
            top_k=3,
        )
        assert len(samples) == 3

    def test_sample_questions_match_golden(self) -> None:
        from eval.evaluate import run_pipeline_on_dataset

        questions = [make_question(question="Revenue?")]
        samples = run_pipeline_on_dataset(
            questions=questions,
            vector_store=FakeVectorStore(),
            llm=FakeLLM(),
            cross_encoder=HighScoreCrossEncoder(),
        )
        assert samples[0].question == "Revenue?"

    def test_retrieved_contexts_non_empty(self) -> None:
        from eval.evaluate import run_pipeline_on_dataset

        questions = [make_question()]
        samples = run_pipeline_on_dataset(
            questions=questions,
            vector_store=FakeVectorStore(),
            llm=FakeLLM(),
            cross_encoder=HighScoreCrossEncoder(),
        )
        assert len(samples[0].retrieved_contexts) > 0
