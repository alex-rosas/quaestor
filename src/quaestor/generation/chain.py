"""RAG chain for Quaestor.

Assembles retriever → prompt → LLM → structured answer into a single
callable.  The chain is stateless — build it once, call it many times.

Public API
----------
build_chain(vector_store)  -> RagChain
RagChain.ask(question)     -> Answer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from quaestor.config import LLMProvider, settings
from quaestor.generation.prompts import PROMPT_VERSION, RAG_PROMPT
from quaestor.retrieval.retriever import retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    """Structured response from the RAG chain.

    Attributes:
        question:      The original user question.
        answer:        The generated answer text (with inline citations).
        sources:       Deduplicated list of source filenames cited.
        prompt_version: Which prompt template version was used.
    """

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm(provider: LLMProvider | None = None) -> BaseChatModel:
    """Return the configured LLM backend.

    Args:
        provider: Override ``settings.llm_provider`` for this call.

    Returns:
        A LangChain ``BaseChatModel`` instance.

    Raises:
        ValueError: If the provider is unknown.
    """
    provider = provider or settings.llm_provider

    if provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key or None,  # None lets SDK read env
        )

    if provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama  # type: ignore[import]

        return ChatOllama(
            model=settings.ollama_llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
        )

    if provider == LLMProvider.TOGETHER:
        from langchain_together import ChatTogether  # type: ignore[import]

        return ChatTogether(
            model=settings.together_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.together_api_key or None,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def _format_context(docs: list) -> str:
    """Render retrieved chunks into a single context string for the prompt."""
    parts: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# RagChain
# ---------------------------------------------------------------------------

class RagChain:
    """Stateless RAG chain: retrieve → prompt → LLM → Answer.

    Args:
        vector_store: Indexed ChromaDB collection.
        llm:          Language model to use (injected for testing).
        top_k:        Override retrieval count.
    """

    def __init__(
        self,
        vector_store: Chroma,
        llm: BaseChatModel | None = None,
        top_k: int | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm or _get_llm()
        self._top_k = top_k or settings.retrieval_top_k

        # Build the LangChain LCEL pipeline
        self._pipeline = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> Answer:
        """Answer *question* using retrieved context.

        Args:
            question: Natural-language question from the user.

        Returns:
            An ``Answer`` with the generated text and cited sources.

        Raises:
            ValueError: If *question* is empty or blank.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        logger.info("RAG query: %r", question[:80])

        # 1. Retrieve
        docs = retrieve(question, self._vector_store, top_k=self._top_k)

        # 2. Format context
        context = _format_context(docs)

        # 3. Generate
        raw_answer: str = self._pipeline.invoke(
            {"context": context, "question": question}
        )

        # 4. Extract cited sources from metadata
        sources = sorted(
            {doc.metadata.get("source", "unknown") for doc in docs}
        )

        logger.info("Answer generated. Sources: %s", sources)
        return Answer(question=question, answer=raw_answer, sources=sources)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_chain(
    vector_store: Chroma,
    llm: BaseChatModel | None = None,
    top_k: int | None = None,
) -> RagChain:
    """Build and return a ``RagChain`` ready for querying.

    Args:
        vector_store: Indexed ChromaDB collection.
        llm:          Inject a custom LLM (used in tests / alternate providers).
        top_k:        Override retrieval count.

    Returns:
        Configured ``RagChain`` instance.
    """
    return RagChain(vector_store=vector_store, llm=llm, top_k=top_k)
