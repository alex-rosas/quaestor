"""Unit tests for src/quaestor/api/main.py.

Uses FastAPI's TestClient (sync) for /health and /ask, and httpx.AsyncClient
for the /ask/stream SSE endpoint.  All heavyweight dependencies (vector store,
LLM, cross-encoder, Presidio, NLI classifier) are replaced with lightweight
fakes via FastAPI's dependency_overrides mechanism — no network, no GPU.
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
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.5] * 8 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.5] * 8


class FakeLLM(BaseChatModel):
    response: str = "Apple revenue was $383 billion. [Source: apple_10k.pdf, Page 3]"

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
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response))]
        )


class HighScoreCrossEncoder:
    def predict(self, sentences: list) -> list[float]:
        return [5.0] * len(sentences)


class LowScoreCrossEncoder:
    def predict(self, sentences: list) -> list[float]:
        return [-5.0] * len(sentences)


class _FakeAnalyzerResult:
    def __init__(self, entity_type, start, end, score):
        self.entity_type, self.start, self.end, self.score = entity_type, start, end, score


class FakeAnalyzerNoPii:
    def analyze(self, text, entities, language, score_threshold):
        return []


class FakeAnalyzerWithPii:
    def analyze(self, text, entities, language, score_threshold):
        if "john@test.com" in text:
            return [_FakeAnalyzerResult("EMAIL_ADDRESS", text.index("john@test.com"),
                                        text.index("john@test.com") + 13, 1.0)]
        return []


class FakeAnonymizer:
    def anonymize(self, text, analyzer_results, operators):
        result = text
        for hit in sorted(analyzer_results, key=lambda x: x.start, reverse=True):
            result = result[: hit.start] + f"<{hit.entity_type}>" + result[hit.end :]
        return type("R", (), {"text": result})()


class FakeNliClassifier:
    """Returns ENTAILMENT — no hallucination."""

    def __call__(self, inputs: dict, **kwargs) -> list[dict]:
        return [
            {"label": "ENTAILMENT", "score": 0.90},
            {"label": "NEUTRAL", "score": 0.07},
            {"label": "CONTRADICTION", "score": 0.03},
        ]


# ---------------------------------------------------------------------------
# Fixtures — vector store + dependency overrides
# ---------------------------------------------------------------------------

@pytest.fixture()
def vector_store(tmp_path):
    docs = [
        Document(
            page_content=f"Apple revenue was $383 billion in FY{2020 + i}",
            metadata={"source": "apple_10k.pdf", "page": i, "start_index": i * 100},
        )
        for i in range(4)
    ]
    store = Chroma(
        collection_name="api_test",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path),
    )
    store.add_documents(docs)
    return store


@pytest.fixture()
def client(vector_store):
    """TestClient with all heavyweight dependencies overridden."""
    app.dependency_overrides = {
        get_vector_store: lambda: vector_store,
        get_llm: lambda: FakeLLM(),
        get_cross_encoder: lambda: HighScoreCrossEncoder(),
        get_analyzer: lambda: FakeAnalyzerNoPii(),
        get_anonymizer: lambda: FakeAnonymizer(),
        get_nli_classifier: lambda: FakeNliClassifier(),
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


@pytest.fixture()
def low_confidence_client(vector_store):
    """TestClient where cross-encoder always returns low scores → refusal."""
    app.dependency_overrides = {
        get_vector_store: lambda: vector_store,
        get_llm: lambda: FakeLLM(),
        get_cross_encoder: lambda: LowScoreCrossEncoder(),
        get_analyzer: lambda: FakeAnalyzerNoPii(),
        get_anonymizer: lambda: FakeAnonymizer(),
        get_nli_classifier: lambda: FakeNliClassifier(),
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_ok(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_version_present(self, client) -> None:
        resp = client.get("/health")
        assert "version" in resp.json()

    def test_backend_present(self, client) -> None:
        resp = client.get("/health")
        assert "vector_store_backend" in resp.json()


# ---------------------------------------------------------------------------
# POST /ask — happy path
# ---------------------------------------------------------------------------

class TestAsk:
    def test_200_ok(self, client) -> None:
        resp = client.post("/ask", json={"question": "What was Apple's revenue?"})
        assert resp.status_code == 200

    def test_answer_non_empty(self, client) -> None:
        resp = client.post("/ask", json={"question": "What was Apple's revenue?"})
        assert len(resp.json()["answer"]) > 0

    def test_question_echoed(self, client) -> None:
        q = "What was Apple's revenue in FY2023?"
        resp = client.post("/ask", json={"question": q})
        assert resp.json()["question"] == q

    def test_sources_list(self, client) -> None:
        resp = client.post("/ask", json={"question": "Apple revenue"})
        assert isinstance(resp.json()["sources"], list)

    def test_refused_false_on_confident_retrieval(self, client) -> None:
        resp = client.post("/ask", json={"question": "Apple revenue"})
        assert resp.json()["refused"] is False

    def test_pii_report_included_when_check_pii_true(self, client) -> None:
        resp = client.post("/ask", json={"question": "Apple revenue", "check_pii": True})
        assert resp.json()["pii"] is not None

    def test_pii_report_none_when_check_pii_false(self, client) -> None:
        resp = client.post("/ask", json={"question": "Apple revenue", "check_pii": False})
        assert resp.json()["pii"] is None

    def test_hallucination_included_when_requested(self, client) -> None:
        resp = client.post(
            "/ask", json={"question": "Apple revenue", "check_hallucination": True}
        )
        assert resp.json()["hallucination"] is not None

    def test_hallucination_none_when_not_requested(self, client) -> None:
        resp = client.post("/ask", json={"question": "Apple revenue"})
        assert resp.json()["hallucination"] is None

    def test_prompt_version_present(self, client) -> None:
        resp = client.post("/ask", json={"question": "Apple revenue"})
        assert resp.json()["prompt_version"] != ""

    def test_empty_question_rejected(self, client) -> None:
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422  # FastAPI validation error


# ---------------------------------------------------------------------------
# POST /ask — low-confidence → refusal
# ---------------------------------------------------------------------------

class TestAskRefusal:
    def test_refused_true(self, low_confidence_client) -> None:
        resp = low_confidence_client.post("/ask", json={"question": "Apple revenue"})
        assert resp.json()["refused"] is True

    def test_sources_empty_on_refusal(self, low_confidence_client) -> None:
        resp = low_confidence_client.post("/ask", json={"question": "Apple revenue"})
        assert resp.json()["sources"] == []

    def test_hallucination_not_run_on_refusal(self, low_confidence_client) -> None:
        resp = low_confidence_client.post(
            "/ask",
            json={"question": "Apple revenue", "check_hallucination": True},
        )
        assert resp.json()["hallucination"] is None


# ---------------------------------------------------------------------------
# POST /ask — PII detection
# ---------------------------------------------------------------------------

class TestAskPii:
    def test_pii_detected_flag(self, vector_store) -> None:
        app.dependency_overrides = {
            get_vector_store: lambda: vector_store,
            get_llm: lambda: FakeLLM(),
            get_cross_encoder: lambda: HighScoreCrossEncoder(),
            get_analyzer: lambda: FakeAnalyzerWithPii(),
            get_anonymizer: lambda: FakeAnonymizer(),
            get_nli_classifier: lambda: FakeNliClassifier(),
        }
        with TestClient(app) as c:
            resp = c.post(
                "/ask",
                json={"question": "revenue for john@test.com", "check_pii": True},
            )
        app.dependency_overrides = {}
        pii = resp.json()["pii"]
        assert pii["detected"] is True
        assert "EMAIL_ADDRESS" in pii["entity_types"]

    def test_no_pii_detected(self, client) -> None:
        resp = client.post(
            "/ask", json={"question": "What is EBITDA?", "check_pii": True}
        )
        assert resp.json()["pii"]["detected"] is False


# ---------------------------------------------------------------------------
# POST /ask/stream
# ---------------------------------------------------------------------------

class TestAskStream:
    def _collect_events(self, response) -> list[dict]:
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        return events

    def test_200_ok(self, client) -> None:
        resp = client.post(
            "/ask/stream",
            json={"question": "Apple revenue"},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200

    def test_content_type_is_event_stream(self, client) -> None:
        resp = client.post("/ask/stream", json={"question": "Apple revenue"})
        assert "text/event-stream" in resp.headers["content-type"]

    def test_done_event_present(self, client) -> None:
        resp = client.post("/ask/stream", json={"question": "Apple revenue"})
        events = self._collect_events(resp)
        types = [e["type"] for e in events]
        assert "done" in types

    def test_token_events_present(self, client) -> None:
        resp = client.post("/ask/stream", json={"question": "Apple revenue"})
        events = self._collect_events(resp)
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) >= 1

    def test_sources_event_present(self, client) -> None:
        resp = client.post("/ask/stream", json={"question": "Apple revenue"})
        events = self._collect_events(resp)
        sources_events = [e for e in events if e["type"] == "sources"]
        assert len(sources_events) == 1

    def test_refused_event_on_low_confidence(self, low_confidence_client) -> None:
        resp = low_confidence_client.post(
            "/ask/stream", json={"question": "Apple revenue"}
        )
        events = self._collect_events(resp)
        types = [e["type"] for e in events]
        assert "refused" in types
        assert "token" not in types

    def test_pii_event_when_check_pii(self, client) -> None:
        resp = client.post(
            "/ask/stream",
            json={"question": "Apple revenue", "check_pii": True},
        )
        events = self._collect_events(resp)
        pii_events = [e for e in events if e["type"] == "pii"]
        assert len(pii_events) == 1
