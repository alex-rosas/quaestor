"""Unit tests for src/quaestor/generation/chain.py.

The LLM is replaced with a FakeLLM that echoes a fixed string so no Groq
API key or network access is required.  Embeddings use FakeEmbeddings so
no Ollama is required either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, List, Optional

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from quaestor.generation.chain import Answer, RagChain, _format_context, build_chain
from quaestor.generation.prompts import PROMPT_VERSION


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


class FakeLLM(BaseChatModel):
    """Returns a canned response regardless of input."""

    response: str = "Apple's revenue was $383 billion. [Source: apple_10k.pdf, Page 3]"

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
        msg = AIMessage(content=self.response)
        return ChatResult(generations=[ChatGeneration(message=msg)])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vector_store(tmp_path: Path) -> Chroma:
    docs = [
        Document(
            page_content="Apple total net sales were $383.3 billion in FY2023.",
            metadata={"source": "apple_10k.pdf", "page": 3, "start_index": 0},
        ),
        Document(
            page_content="JPMorgan reported net revenue of $158.1 billion in 2023.",
            metadata={"source": "jpmorgan_10k.pdf", "page": 7, "start_index": 0},
        ),
    ]
    store = Chroma(
        collection_name="test",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path),
    )
    store.add_documents(docs)
    return store


@pytest.fixture()
def chain(vector_store: Chroma) -> RagChain:
    return build_chain(vector_store, llm=FakeLLM(), top_k=2)


# ---------------------------------------------------------------------------
# _format_context helper
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_includes_source(self) -> None:
        docs = [Document(page_content="text", metadata={"source": "a.pdf", "page": 1})]
        ctx = _format_context(docs)
        assert "a.pdf" in ctx

    def test_includes_page(self) -> None:
        docs = [Document(page_content="text", metadata={"source": "a.pdf", "page": 5})]
        ctx = _format_context(docs)
        assert "5" in ctx

    def test_includes_content(self) -> None:
        docs = [Document(page_content="revenue data", metadata={"source": "a.pdf", "page": 0})]
        ctx = _format_context(docs)
        assert "revenue data" in ctx

    def test_multiple_docs_separated(self) -> None:
        docs = [
            Document(page_content="chunk A", metadata={"source": "a.pdf", "page": 0}),
            Document(page_content="chunk B", metadata={"source": "b.pdf", "page": 1}),
        ]
        ctx = _format_context(docs)
        assert "chunk A" in ctx
        assert "chunk B" in ctx
        assert "---" in ctx

    def test_missing_metadata_graceful(self) -> None:
        docs = [Document(page_content="text", metadata={})]
        ctx = _format_context(docs)
        assert "unknown" in ctx


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------

class TestAnswer:
    def test_fields_accessible(self) -> None:
        ans = Answer(question="q", answer="a", sources=["s.pdf"])
        assert ans.question == "q"
        assert ans.answer == "a"
        assert ans.sources == ["s.pdf"]

    def test_prompt_version_default(self) -> None:
        ans = Answer(question="q", answer="a")
        assert ans.prompt_version == PROMPT_VERSION

    def test_sources_default_empty(self) -> None:
        ans = Answer(question="q", answer="a")
        assert ans.sources == []


# ---------------------------------------------------------------------------
# RagChain.ask
# ---------------------------------------------------------------------------

class TestRagChain:
    def test_returns_answer_instance(self, chain: RagChain) -> None:
        result = chain.ask("What was Apple's revenue?")
        assert isinstance(result, Answer)

    def test_answer_contains_text(self, chain: RagChain) -> None:
        result = chain.ask("What was Apple's revenue?")
        assert len(result.answer) > 0

    def test_question_preserved_in_answer(self, chain: RagChain) -> None:
        q = "What was Apple's revenue in FY2023?"
        result = chain.ask(q)
        assert result.question == q

    def test_sources_populated(self, chain: RagChain) -> None:
        result = chain.ask("What was Apple's revenue?")
        assert isinstance(result.sources, list)
        assert len(result.sources) > 0

    def test_sources_are_strings(self, chain: RagChain) -> None:
        result = chain.ask("What was Apple's revenue?")
        assert all(isinstance(s, str) for s in result.sources)

    def test_sources_are_deduplicated(self, tmp_path: Path) -> None:
        """Two chunks from the same source must produce one source entry."""
        docs = [
            Document(
                page_content=f"Revenue chunk {i}",
                metadata={"source": "apple_10k.pdf", "page": i, "start_index": i},
            )
            for i in range(3)
        ]
        store = Chroma(
            collection_name="dedup_test",
            embedding_function=FakeEmbeddings(),
            persist_directory=str(tmp_path),
        )
        store.add_documents(docs)
        c = build_chain(store, llm=FakeLLM(), top_k=3)
        result = c.ask("revenue")
        assert result.sources.count("apple_10k.pdf") == 1

    def test_prompt_version_in_answer(self, chain: RagChain) -> None:
        result = chain.ask("What is EBITDA?")
        assert result.prompt_version == PROMPT_VERSION

    def test_empty_question_raises(self, chain: RagChain) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            chain.ask("")

    def test_blank_question_raises(self, chain: RagChain) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            chain.ask("   ")


# ---------------------------------------------------------------------------
# build_chain factory
# ---------------------------------------------------------------------------

class TestBuildChain:
    def test_returns_rag_chain(self, vector_store: Chroma) -> None:
        c = build_chain(vector_store, llm=FakeLLM())
        assert isinstance(c, RagChain)

    def test_custom_top_k(self, vector_store: Chroma) -> None:
        c = build_chain(vector_store, llm=FakeLLM(), top_k=1)
        assert c._top_k == 1
