"""RAGAS evaluation harness for Quaestor.

Loads the golden dataset, runs the RAG pipeline on each question, and scores
the results with four RAGAS metrics:

* **Faithfulness**       — does the answer stick to retrieved context?
* **Answer Relevancy**   — is the answer on-topic for the question?
* **Context Precision**  — are the top-ranked chunks actually relevant?
* **Context Recall**     — did retrieval surface all the needed evidence?

Usage (live run, requires Groq key + indexed documents)::

    uv run python scripts/evaluate.py

    # optional flags
    uv run python scripts/evaluate.py \\
        --dataset eval/golden_dataset.json \\
        --output  benchmarks/ragas_results.json \\
        --chunking fixed          # or semantic / hierarchical \\
        --top-k   5

Public API (importable, all network-free parts)::

    from eval.evaluate import load_golden_dataset, build_evaluation_sample
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
DEFAULT_OUTPUT_PATH = Path("benchmarks") / "ragas_results.json"


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

@dataclass
class GoldenQuestion:
    """One row in the golden dataset.

    Attributes:
        id:           Short identifier (``"Q01"`` … ``"Q20"``).
        type:         ``"factual"``, ``"multi_hop"``, or ``"unanswerable"``.
        question:     Natural-language question text.
        ground_truth: Reference answer used by RAGAS metrics.
        source_pages: Filing page numbers where the answer can be found.
        notes:        Human-readable annotation.
    """

    id: str
    type: str
    question: str
    ground_truth: str
    source_pages: list[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvaluationSample:
    """One row ready for RAGAS.

    Attributes:
        question:           Original user question.
        ground_truth:       Reference answer.
        answer:             Pipeline-generated answer.
        retrieved_contexts: List of chunk texts returned by the retriever.
        refused:            ``True`` if the confidence gate fired.
        question_type:      ``"factual"``, ``"multi_hop"``, or ``"unanswerable"``.
    """

    question: str
    ground_truth: str
    answer: str
    retrieved_contexts: list[str]
    refused: bool = False
    question_type: str = "factual"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_golden_dataset(path: Path | str = GOLDEN_DATASET_PATH) -> list[GoldenQuestion]:
    """Load and validate the golden dataset JSON.

    Args:
        path: Path to ``golden_dataset.json``.  Defaults to
              :data:`GOLDEN_DATASET_PATH`.

    Returns:
        List of :class:`GoldenQuestion` in dataset order.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError:        If required fields are missing or the question
                           type is unknown.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    valid_types = {"factual", "multi_hop", "unanswerable"}
    questions: list[GoldenQuestion] = []

    for entry in raw["questions"]:
        q_type = entry.get("type", "")
        if q_type not in valid_types:
            raise ValueError(
                f"Question {entry.get('id')!r} has unknown type {q_type!r}. "
                f"Must be one of {valid_types}."
            )
        if not entry.get("question"):
            raise ValueError(f"Question {entry.get('id')!r} has empty question text.")
        if not entry.get("ground_truth"):
            raise ValueError(f"Question {entry.get('id')!r} has empty ground_truth.")

        questions.append(
            GoldenQuestion(
                id=entry["id"],
                type=q_type,
                question=entry["question"],
                ground_truth=entry["ground_truth"],
                source_pages=entry.get("source_pages", []),
                notes=entry.get("notes", ""),
            )
        )

    logger.info("Loaded %d golden questions from %s", len(questions), path)
    return questions


# ---------------------------------------------------------------------------
# Sample builder
# ---------------------------------------------------------------------------

def build_evaluation_sample(
    golden: GoldenQuestion,
    answer: str,
    retrieved_contexts: list[str],
    refused: bool = False,
) -> EvaluationSample:
    """Combine a golden question with pipeline output into an evaluation sample.

    Args:
        golden:             The golden question (question + ground_truth).
        answer:             The answer produced by the RAG pipeline.
        retrieved_contexts: Chunk texts retrieved for this question.
        refused:            Whether the confidence gate fired.

    Returns:
        :class:`EvaluationSample` ready to be converted to a RAGAS
        ``SingleTurnSample``.
    """
    return EvaluationSample(
        question=golden.question,
        ground_truth=golden.ground_truth,
        answer=answer,
        retrieved_contexts=retrieved_contexts,
        refused=refused,
        question_type=golden.type,
    )


# ---------------------------------------------------------------------------
# RAGAS dataset builder
# ---------------------------------------------------------------------------

def to_ragas_dataset(samples: list[EvaluationSample]):
    """Convert a list of :class:`EvaluationSample` to a RAGAS
    ``EvaluationDataset``.

    Args:
        samples: Evaluation samples from the pipeline run.

    Returns:
        ``ragas.dataset_schema.EvaluationDataset`` ready for
        ``ragas.evaluate()``.
    """
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    ragas_samples = [
        SingleTurnSample(
            user_input=s.question,
            response=s.answer,
            retrieved_contexts=s.retrieved_contexts,
            reference=s.ground_truth,
        )
        for s in samples
    ]
    return EvaluationDataset(samples=ragas_samples)


# ---------------------------------------------------------------------------
# Pipeline runner (live — needs real vector store + LLM)
# ---------------------------------------------------------------------------

def run_pipeline_on_dataset(
    questions: list[GoldenQuestion],
    vector_store,
    llm=None,
    cross_encoder=None,
    top_k: int = 5,
    confidence_threshold: float | None = None,
) -> list[EvaluationSample]:
    """Run the RAG pipeline on every golden question and collect samples.

    Args:
        questions:            Golden questions to evaluate.
        vector_store:         Indexed vector store (Chroma or Qdrant).
        llm:                  Language model.  Defaults to settings.
        cross_encoder:        Cross-encoder for reranking.
        top_k:                Retrieval candidate count.
        confidence_threshold: Override the default refusal threshold.

    Returns:
        List of :class:`EvaluationSample` — one per question.
    """
    from quaestor.retrieval.graph import build_rag_graph, run_rag_graph
    from quaestor.retrieval.retriever import retrieve

    graph = build_rag_graph(
        vector_store=vector_store,
        llm=llm,
        cross_encoder=cross_encoder,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )

    samples: list[EvaluationSample] = []

    for i, golden in enumerate(questions, 1):
        logger.info("[%d/%d] %s — %r", i, len(questions), golden.id, golden.question[:60])

        try:
            # Retrieve separately to capture context strings for RAGAS
            docs = retrieve(golden.question, vector_store, top_k=top_k)
            contexts = [d.page_content for d in docs]

            # Full pipeline (rerank + confidence gate + generate)
            graph_answer = run_rag_graph(graph, golden.question)

            sample = build_evaluation_sample(
                golden=golden,
                answer=graph_answer.answer,
                retrieved_contexts=contexts,
                refused=graph_answer.refused,
            )
        except Exception as exc:
            logger.error("Error on %s: %s", golden.id, exc)
            sample = build_evaluation_sample(
                golden=golden,
                answer=f"[ERROR: {exc}]",
                retrieved_contexts=[],
                refused=False,
            )

        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# RAGAS evaluation runner (live — needs LLM for faithfulness / relevancy)
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    samples: list[EvaluationSample],
    llm=None,
    embeddings=None,
    metrics: list | None = None,
) -> dict[str, Any]:
    """Score *samples* with RAGAS and return a results dict.

    Args:
        samples:    Evaluation samples from :func:`run_pipeline_on_dataset`.
        llm:        LangChain LLM wrapper for RAGAS judge metrics.
        embeddings: LangChain Embeddings for answer relevancy.
        metrics:    Override the default metric list.

    Returns:
        Dict with ``"scores"`` (per-metric averages) and ``"per_question"``
        (per-sample breakdowns).
    """
    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (  # noqa: PLC0415
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    if metrics is None:
        # strictness=1 avoids n>1 completions requests that Groq rejects (400)
        metrics = [Faithfulness(), AnswerRelevancy(strictness=1), ContextPrecision(), ContextRecall()]

    dataset = to_ragas_dataset(samples)

    ragas_llm = LangchainLLMWrapper(llm) if llm else None
    ragas_emb = LangchainEmbeddingsWrapper(embeddings) if embeddings else None

    from ragas.run_config import RunConfig

    # Limit concurrency to avoid overwhelming the LLM provider (Groq TPD
    # budget or Ollama queue depth).  max_workers=4 balances throughput
    # against rate-limit exposure; timeout=300s accommodates slow local models.
    run_cfg = RunConfig(max_workers=4, timeout=300)

    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        raise_exceptions=False,
        show_progress=True,
        run_config=run_cfg,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    scores = {col: float(df[col].mean()) for col in metric_cols if df[col].dtype.kind in ("f", "i")}
    logger.info("RAGAS scores: %s", scores)
    return {"scores": scores, "dataset": df.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results: dict[str, Any], output_path: Path | str) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results:     Output from :func:`run_ragas_evaluation`.
        output_path: Destination path (created if missing).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Results saved to %s", output_path)
