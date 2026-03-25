"""Unit tests for src/quaestor/generation/prompts.py."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from quaestor.generation.prompts import PROMPT_V1, PROMPT_VERSION, RAG_PROMPT


class TestPrompts:
    def test_rag_prompt_is_chat_prompt_template(self) -> None:
        assert isinstance(RAG_PROMPT, ChatPromptTemplate)

    def test_rag_prompt_aliases_v1(self) -> None:
        assert RAG_PROMPT is PROMPT_V1

    def test_prompt_version_string(self) -> None:
        assert PROMPT_VERSION == "v1"

    def test_prompt_has_context_variable(self) -> None:
        variables = RAG_PROMPT.input_variables
        assert "context" in variables

    def test_prompt_has_question_variable(self) -> None:
        variables = RAG_PROMPT.input_variables
        assert "question" in variables

    def test_prompt_renders_without_error(self) -> None:
        messages = RAG_PROMPT.format_messages(
            context="Apple revenue was $383 billion.",
            question="What was Apple's revenue?",
        )
        assert len(messages) == 2  # system + human

    def test_system_message_mentions_citations(self) -> None:
        messages = RAG_PROMPT.format_messages(context="ctx", question="q")
        system_text = messages[0].content
        assert "cite" in system_text.lower() or "citation" in system_text.lower()

    def test_system_message_mentions_unanswerable(self) -> None:
        messages = RAG_PROMPT.format_messages(context="ctx", question="q")
        system_text = messages[0].content
        assert "not" in system_text.lower() and "information" in system_text.lower()

    def test_human_message_contains_question(self) -> None:
        messages = RAG_PROMPT.format_messages(context="ctx", question="What is EBITDA?")
        human_text = messages[1].content
        assert "What is EBITDA?" in human_text

    def test_system_message_contains_context(self) -> None:
        ctx = "Revenue was $100B."
        messages = RAG_PROMPT.format_messages(context=ctx, question="q")
        system_text = messages[0].content
        assert ctx in system_text
