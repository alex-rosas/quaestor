"""Unit tests for src/quaestor/retrieval/graph.py.

All tests are fully offline:
  - FakeEmbeddings   → in-memory Chroma (no Ollama)
  - FakeCrossEncoder → deterministic scores (no sentence-transformers download)
  - FakeLLM          → canned response (no Groq API key)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from quaestor.retrieval.graph import (
    GraphAnswer,
    RAGState,
    _REFUSAL_TEXT,
    build_rag_graph,
    run_rag_graph,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    def _vec(self, text: str) -> List[float]:
        seed = sum(ord(c) for c in text)
        return [(seed + i) % 100 / 100.0 for i in range(8)]


class HighScoreCrossEncoder:
    """Always returns a score well above any reasonable threshold."""

    def predict(self, sentences: list[list[str]]) -> list[float]:
        return [5.0] * len(sentences)


class LowScoreCrossEncoder:
    """Always returns a score well below any reasonable threshold."""

    def predict(self, sentences: list[list[str]]) -> list[float]:
        return [-5.0] * len(sentences)


class VariedScoreCrossEncoder:
    """Returns scores 3, 1, 2, 0 for the first four pairs (then 0)."""

    def predict(self, sentences: list[list[str]]) -> list[float]:
        preset = [3.0, 1.0, 2.0, 0.0]
        return [preset[i] if i < len(preset) else 0.0 for i in range(len(sentences))]


class FakeLLM(BaseChatModel):
    response: str = "Revenue was $383 billion. [Source: apple_10k.pdf, Page 3]"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vector_store(tmp_path: Path) -> Chroma:
    docs = [
        Document(
            page_content=f"Apple revenue was $383 billion in FY{2020 + i}",
            metadata={"source": "apple_10k.pdf", "page": i, "start_index": i * 200},
        )
        for i in range(4)
    ]
    store = Chroma(
        collection_name="test_graph",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path),
    )
    store.add_documents(docs)
    return store


@pytest.fixture()
def confident_graph(vector_store: Chroma):
    """Graph whose cross-encoder always returns high scores → generates answer."""
    return build_rag_graph(
        vector_store,
        llm=FakeLLM(),
        cross_encoder=HighScoreCrossEncoder(),
        confidence_threshold=0.0,
    )


@pytest.fixture()
def low_confidence_graph(vector_store: Chroma):
    """Graph whose cross-encoder always returns low scores → refuses."""
    return build_rag_graph(
        vector_store,
        llm=FakeLLM(),
        cross_encoder=LowScoreCrossEncoder(),
        confidence_threshold=0.0,
    )


# ---------------------------------------------------------------------------
# build_rag_graph — smoke tests
# ---------------------------------------------------------------------------

class TestBuildRagGraph:
    def test_returns_compiled_graph(self, vector_store: Chroma) -> None:
        graph = build_rag_graph(
            vector_store, llm=FakeLLM(), cross_encoder=HighScoreCrossEncoder()
        )
        assert graph is not None

    def test_graph_is_invokable(self, confident_graph) -> None:
        result = confident_graph.invoke(
            {
                "question": "What was Apple revenue?",
                "docs": [],
                "top_score": 0.0,
                "answer": "",
                "sources": [],
                "refused": False,
                "prompt_version": "v1",
            }
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# run_rag_graph — confident path (score ≥ threshold)
# ---------------------------------------------------------------------------

class TestConfidentPath:
    def test_returns_graph_answer(self, confident_graph) -> None:
        result = run_rag_graph(confident_graph, "Apple revenue")
        assert isinstance(result, GraphAnswer)

    def test_answer_is_non_empty(self, confident_graph) -> None:
        result = run_rag_graph(confident_graph, "Apple revenue")
        assert len(result.answer) > 0

    def test_refused_is_false(self, confident_graph) -> None:
        result = run_rag_graph(confident_graph, "Apple revenue")
        assert result.refused is False

    def test_sources_populated(self, confident_graph) -> None:
        result = run_rag_graph(confident_graph, "Apple revenue")
        assert len(result.sources) > 0

    def test_sources_are_strings(self, confident_graph) -> None:
        result = run_rag_graph(confident_graph, "Apple revenue")
        assert all(isinstance(s, str) for s in result.sources)

    def test_question_preserved(self, confident_graph) -> None:
        q = "What was Apple's total revenue?"
        result = run_rag_graph(confident_graph, q)
        assert result.question == q

    def test_top_score_positive(self, confident_graph) -> None:
        result = run_rag_graph(confident_graph, "Apple revenue")
        assert result.top_score > 0

    def test_prompt_version_present(self, confident_graph) -> None:
        result = run_rag_graph(confident_graph, "Apple revenue")
        assert result.prompt_version != ""


# ---------------------------------------------------------------------------
# run_rag_graph — low-confidence path (score < threshold → refuse)
# ---------------------------------------------------------------------------

class TestLowConfidencePath:
    def test_returns_graph_answer(self, low_confidence_graph) -> None:
        result = run_rag_graph(low_confidence_graph, "Apple revenue")
        assert isinstance(result, GraphAnswer)

    def test_refused_is_true(self, low_confidence_graph) -> None:
        result = run_rag_graph(low_confidence_graph, "Apple revenue")
        assert result.refused is True

    def test_answer_is_refusal_text(self, low_confidence_graph) -> None:
        result = run_rag_graph(low_confidence_graph, "Apple revenue")
        assert result.answer == _REFUSAL_TEXT

    def test_sources_empty_on_refusal(self, low_confidence_graph) -> None:
        result = run_rag_graph(low_confidence_graph, "Apple revenue")
        assert result.sources == []

    def test_top_score_negative(self, low_confidence_graph) -> None:
        result = run_rag_graph(low_confidence_graph, "Apple revenue")
        assert result.top_score < 0


# ---------------------------------------------------------------------------
# run_rag_graph — threshold boundary
# ---------------------------------------------------------------------------

class TestThresholdBoundary:
    def test_exactly_at_threshold_generates(self, vector_store: Chroma) -> None:
        """A score equal to the threshold should generate (≥ condition)."""
        class ExactThresholdEncoder:
            def predict(self, sentences):
                return [1.0] * len(sentences)

        graph = build_rag_graph(
            vector_store,
            llm=FakeLLM(),
            cross_encoder=ExactThresholdEncoder(),
            confidence_threshold=1.0,
        )
        result = run_rag_graph(graph, "Apple revenue")
        assert result.refused is False

    def test_just_below_threshold_refuses(self, vector_store: Chroma) -> None:
        """A score just below the threshold should refuse."""
        class BelowThresholdEncoder:
            def predict(self, sentences):
                return [0.99] * len(sentences)

        graph = build_rag_graph(
            vector_store,
            llm=FakeLLM(),
            cross_encoder=BelowThresholdEncoder(),
            confidence_threshold=1.0,
        )
        result = run_rag_graph(graph, "Apple revenue")
        assert result.refused is True

    def test_custom_threshold_respected(self, vector_store: Chroma) -> None:
        """Very high threshold should force refusal even from HighScoreEncoder."""
        graph = build_rag_graph(
            vector_store,
            llm=FakeLLM(),
            cross_encoder=HighScoreCrossEncoder(),
            confidence_threshold=100.0,  # impossible to meet
        )
        result = run_rag_graph(graph, "Apple revenue")
        assert result.refused is True


# ---------------------------------------------------------------------------
# run_rag_graph — reranking order
# ---------------------------------------------------------------------------

class TestRerankingOrder:
    def test_top_n_limits_context(self, vector_store: Chroma) -> None:
        """top_n=1 should pass only the best chunk to the LLM."""
        received_contexts: list[str] = []

        class CapturingLLM(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "capturing"

            def _generate(self, messages, stop=None, run_manager=None, **kw):
                received_contexts.append(str(messages))
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="ok"))]
                )

        graph = build_rag_graph(
            vector_store,
            llm=CapturingLLM(),
            cross_encoder=VariedScoreCrossEncoder(),
            confidence_threshold=0.0,
            top_n=1,
        )
        run_rag_graph(graph, "Apple revenue")
        # The context should contain exactly one chunk (one "FY20XX" entry)
        # "[Source:" appears twice when top_n=1: once in the prompt template's
        # format-instruction example and once for the actual chunk.
        # Counting actual chunk content (unique to the vector store) is cleaner.
        assert received_contexts[0].count("Apple revenue was $383 billion in FY") == 1


# ---------------------------------------------------------------------------
# run_rag_graph — validation
# ---------------------------------------------------------------------------

class TestRunRagGraphValidation:
    def test_empty_question_raises(self, confident_graph) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            run_rag_graph(confident_graph, "")

    def test_blank_question_raises(self, confident_graph) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            run_rag_graph(confident_graph, "   ")


# ---------------------------------------------------------------------------
# GraphAnswer dataclass
# ---------------------------------------------------------------------------

class TestGraphAnswer:
    def test_fields_accessible(self) -> None:
        ans = GraphAnswer(
            question="q", answer="a", sources=["s.pdf"], refused=False, top_score=1.5
        )
        assert ans.question == "q"
        assert ans.answer == "a"
        assert ans.sources == ["s.pdf"]
        assert ans.refused is False
        assert ans.top_score == 1.5

    def test_defaults(self) -> None:
        ans = GraphAnswer(question="q", answer="a")
        assert ans.sources == []
        assert ans.refused is False
        assert ans.top_score == 0.0
