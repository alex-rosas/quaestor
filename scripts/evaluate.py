"""RAGAS evaluation runner for Quaestor.

Runs the full RAG pipeline over the 20-question golden dataset and scores
results with RAGAS faithfulness, answer relevancy, context precision, and
context recall.

Prerequisites
-------------
1. Indexed documents in the active vector store (run scripts/smoke_test.py
   or index via the Streamlit UI first).
2. A valid .env with GROQ_API_KEY (or set LLM_PROVIDER=ollama for local).

Usage
-----
::

    # Default: Chroma backend, fixed chunking, top-k=5
    uv run python scripts/evaluate.py

    # Qdrant backend
    VECTOR_STORE_BACKEND=qdrant uv run python scripts/evaluate.py

    # Custom output path
    uv run python scripts/evaluate.py --output benchmarks/my_run.json

    # Limit to first N questions (fast smoke test of the eval harness)
    uv run python scripts/evaluate.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make src/ and the project root importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("quaestor.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation over the Quaestor golden dataset."
    )
    parser.add_argument(
        "--dataset",
        default="eval/golden_dataset.json",
        help="Path to golden_dataset.json (default: eval/golden_dataset.json)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/ragas_results.json",
        help="Output path for results JSON (default: benchmarks/ragas_results.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking (faster, lower quality)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from quaestor.config import VectorStoreBackend, settings
    from quaestor.generation.chain import _get_llm
    from quaestor.ingestion.indexer import _get_embeddings

    from eval.evaluate import (
        GOLDEN_DATASET_PATH,
        load_golden_dataset,
        run_pipeline_on_dataset,
        run_ragas_evaluation,
        save_results,
    )

    # 1. Load golden dataset
    dataset_path = Path(args.dataset)
    logger.info("Loading golden dataset from %s", dataset_path)
    questions = load_golden_dataset(dataset_path)
    if args.limit:
        questions = questions[: args.limit]
        logger.info("Limited to first %d question(s)", len(questions))

    # 2. Load vector store
    logger.info("Loading vector store (backend=%s)", settings.vector_store_backend)
    if settings.vector_store_backend == VectorStoreBackend.QDRANT:
        from quaestor.ingestion.indexer import load_qdrant_index
        vector_store = load_qdrant_index()
    else:
        from quaestor.ingestion.indexer import load_index
        vector_store = load_index()

    # 3. Load models
    llm = _get_llm()
    embeddings = _get_embeddings()
    cross_encoder = None if args.no_rerank else None  # uses settings default

    # 4. Run pipeline on dataset
    logger.info("Running RAG pipeline on %d question(s) …", len(questions))
    samples = run_pipeline_on_dataset(
        questions=questions,
        vector_store=vector_store,
        llm=llm,
        cross_encoder=cross_encoder,
        top_k=args.top_k,
    )

    refused_count = sum(1 for s in samples if s.refused)
    logger.info(
        "Pipeline complete — %d answered, %d refused",
        len(samples) - refused_count,
        refused_count,
    )

    # 5. RAGAS evaluation
    logger.info("Running RAGAS evaluation …")
    results = run_ragas_evaluation(samples=samples, llm=llm, embeddings=embeddings)

    # 6. Annotate and save
    results["metadata"] = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(questions),
        "top_k": args.top_k,
        "vector_store_backend": settings.vector_store_backend.value,
        "llm_provider": settings.llm_provider.value,
        "rerank_enabled": not args.no_rerank,
        "refused_count": refused_count,
    }

    output_path = Path(args.output)
    save_results(results, output_path)

    # 7. Print summary
    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Results")
    print("=" * 60)
    for metric, score in results["scores"].items():
        print(f"  {metric:<30} {score:.4f}")
    print(f"\n  Questions evaluated : {len(questions)}")
    print(f"  Refused             : {refused_count}")
    print(f"  Results saved to    : {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
