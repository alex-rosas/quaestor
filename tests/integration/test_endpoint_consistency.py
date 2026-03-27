"""Integration tests: /ask and /ask/stream must behave identically.

Both endpoints share the same retrieval → rerank → confidence gate logic.
Any question that one endpoint refuses must also be refused by the other;
any question that one answers must produce the same text from the other.

This prevents regressions from the streaming gate divergence bug (Task 4):
previously the streaming endpoint read the confidence threshold from a
different code path than the compiled graph used by /ask.  Both now read
from ``settings.reranker_confidence_threshold`` at request time via the
``confidence_threshold`` key in ``RAGState``.

Fakes mirror the patterns in tests/unit/test_api.py — no real models or
network calls required.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from quaestor.api.main import (
    app,
    get_analyzer,
    get_anonymizer,
    get_cross_encoder,
    get_llm,
    get_nli_classifier,
    get_vector_store,
)


# ---------------------------------------------------------------------------
# Fakes (same patterns as unit/test_api.py)
# ---------------------------------------------------------------------------

class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1] * 8


class _FakeLLM(BaseChatModel):
    """Always returns the same canned answer so we can compare endpoints."""

    ANSWER: str = "Apple's total net sales were $416,161 million. [Source: aapl-10k.txt, Page 27]"

    @property
    def _llm_type(self) -> str:
        return "fake-consistency"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.ANSWER))]
        )


class _HighScoreCE:
    """Cross-encoder that always passes the confidence gate."""
    def predict(self, pairs: list) -> list[float]:
        return [5.0] * len(pairs)


class _LowScoreCE:
    """Cross-encoder that always fails the confidence gate."""
    def predict(self, pairs: list) -> list[float]:
        return [-5.0] * len(pairs)


class _FakeAnalyzer:
    def analyze(self, text, entities, language, score_threshold):
        return []


class _FakeAnonymizer:
    def anonymize(self, text, analyzer_results, operators):
        return type("R", (), {"text": text})()


class _FakeNli:
    def __call__(self, inputs: dict, **kwargs) -> list[dict]:
        return [
            {"label": "ENTAILMENT", "score": 0.92},
            {"label": "NEUTRAL", "score": 0.06},
            {"label": "CONTRADICTION", "score": 0.02},
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_stream_events(response) -> list[dict]:
    """Parse SSE lines from a streaming response into a list of event dicts."""
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _stream_answer(events: list[dict]) -> str:
    """Concatenate all token event content into the full answer string."""
    return "".join(e["content"] for e in events if e["type"] == "token")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _vector_store(tmp_path):
    docs = [
        Document(
            page_content="Apple's total net sales were $416,161 million in fiscal year 2025.",
            metadata={"source": "aapl-10k.txt", "page": 27, "start_index": 0},
        ),
        Document(
            page_content="Net income attributable to Apple Inc. was $93,736 million.",
            metadata={"source": "aapl-10k.txt", "page": 31, "start_index": 200},
        ),
    ]
    store = Chroma(
        collection_name="consistency_test",
        embedding_function=_FakeEmbeddings(),
        persist_directory=str(tmp_path),
    )
    store.add_documents(docs)
    return store


@pytest.fixture()
def confident_client(_vector_store):
    """Both endpoints answer — high CE scores pass the gate."""
    app.dependency_overrides = {
        get_vector_store: lambda: _vector_store,
        get_llm: lambda: _FakeLLM(),
        get_cross_encoder: lambda: _HighScoreCE(),
        get_analyzer: lambda: _FakeAnalyzer(),
        get_anonymizer: lambda: _FakeAnonymizer(),
        get_nli_classifier: lambda: _FakeNli(),
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


@pytest.fixture()
def refusing_client(_vector_store):
    """Both endpoints refuse — low CE scores fail the gate."""
    app.dependency_overrides = {
        get_vector_store: lambda: _vector_store,
        get_llm: lambda: _FakeLLM(),
        get_cross_encoder: lambda: _LowScoreCE(),
        get_analyzer: lambda: _FakeAnalyzer(),
        get_anonymizer: lambda: _FakeAnonymizer(),
        get_nli_classifier: lambda: _FakeNli(),
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# Tests — answer path
# ---------------------------------------------------------------------------

class TestAnswerPathConsistency:
    """When retrieval is confident, both endpoints must answer (not refuse)."""

    def test_ask_answers(self, confident_client) -> None:
        resp = confident_client.post("/ask", json={"question": "Apple net sales?"})
        assert resp.status_code == 200
        assert resp.json()["refused"] is False

    def test_stream_answers(self, confident_client) -> None:
        resp = confident_client.post("/ask/stream", json={"question": "Apple net sales?"})
        events = _collect_stream_events(resp)
        types = [e["type"] for e in events]
        assert "token" in types
        assert "refused" not in types

    def test_both_non_empty_answer(self, confident_client) -> None:
        ask_resp = confident_client.post("/ask", json={"question": "Apple net sales?"})
        stream_resp = confident_client.post("/ask/stream", json={"question": "Apple net sales?"})

        ask_answer = ask_resp.json()["answer"]
        stream_answer = _stream_answer(_collect_stream_events(stream_resp))

        assert len(ask_answer) > 0
        assert len(stream_answer) > 0

    def test_same_answer_text(self, confident_client) -> None:
        """Both endpoints must produce identical answer text for the same question.

        The FakeLLM always returns the same canned string, so any divergence
        in content would indicate a difference in the prompt or pipeline path.
        """
        q = "What were Apple's total net sales in FY2025?"
        ask_answer = confident_client.post("/ask", json={"question": q}).json()["answer"]
        stream_events = _collect_stream_events(
            confident_client.post("/ask/stream", json={"question": q})
        )
        stream_answer = _stream_answer(stream_events)

        assert ask_answer == stream_answer

    def test_sources_consistent(self, confident_client) -> None:
        """Both endpoints must cite the same source filenames."""
        q = "Apple revenue?"
        ask_sources = confident_client.post("/ask", json={"question": q}).json()["sources"]
        stream_events = _collect_stream_events(
            confident_client.post("/ask/stream", json={"question": q})
        )
        stream_sources = next(
            (e["content"] for e in stream_events if e["type"] == "sources"), []
        )
        assert sorted(ask_sources) == sorted(stream_sources)


# ---------------------------------------------------------------------------
# Tests — refusal path
# ---------------------------------------------------------------------------

class TestRefusalPathConsistency:
    """When retrieval is not confident, both endpoints must refuse."""

    def test_ask_refuses(self, refusing_client) -> None:
        resp = refusing_client.post("/ask", json={"question": "Apple net sales?"})
        assert resp.status_code == 200
        assert resp.json()["refused"] is True

    def test_stream_refuses(self, refusing_client) -> None:
        resp = refusing_client.post("/ask/stream", json={"question": "Apple net sales?"})
        events = _collect_stream_events(resp)
        types = [e["type"] for e in events]
        assert "refused" in types
        assert "token" not in types

    def test_both_refuse_same_question(self, refusing_client) -> None:
        """If /ask refuses, /ask/stream must also refuse for the same question."""
        q = "What was Apple's net income?"
        ask_refused = refusing_client.post("/ask", json={"question": q}).json()["refused"]
        stream_events = _collect_stream_events(
            refusing_client.post("/ask/stream", json={"question": q})
        )
        stream_refused = any(e["type"] == "refused" for e in stream_events)

        assert ask_refused is True
        assert stream_refused is True
