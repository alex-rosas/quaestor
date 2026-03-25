"""Versioned prompt templates for Quaestor.

All prompt changes are versioned so RAGAS eval runs can reference the exact
template used.  Import the latest version via the ``RAG_PROMPT`` constant.

Prompt design principles
-------------------------
- Citation is mandatory: the model must reference sources in every answer.
- Unanswerable by default: if context is insufficient the model must say so
  rather than hallucinate.
- Finance domain: language is scoped to SEC filings and regulatory standards.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# v1 — Phase 1 baseline
# ---------------------------------------------------------------------------

_SYSTEM_V1 = """\
You are Quaestor, a financial document assistant that answers questions
strictly based on the provided context from SEC filings and regulatory
standards.

Rules you must follow without exception:
1. Base every answer solely on the CONTEXT below.  Do not use prior knowledge.
2. Always cite your sources using the format [Source: <filename>, Page <n>].
   Include a citation for every factual claim.
3. If the context does not contain sufficient information to answer the
   question, respond with exactly:
   "I don't have enough information in the provided documents to answer this."
4. Do not speculate, infer, or extrapolate beyond what the context states.
5. Keep answers concise and factual.  Use bullet points for multi-part answers.

CONTEXT:
{context}
"""

_HUMAN_V1 = "Question: {question}"

PROMPT_V1 = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_V1),
        ("human", _HUMAN_V1),
    ]
)

# ---------------------------------------------------------------------------
# Latest alias — always points to the current production version
# ---------------------------------------------------------------------------

RAG_PROMPT = PROMPT_V1
PROMPT_VERSION = "v1"
