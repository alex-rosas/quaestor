# Phase 1 Baseline — Fixed-Size Chunking Reference Implementation

This is the deliberately suboptimal baseline built to establish a controlled before/after comparison for Phase 2 improvements.

## Key Characteristics
- **Chunking**: Fixed-size 512 tokens, 50-token overlap
- **Vector Store**: ChromaDB (in-process)
- **Retrieval**: Dense-only semantic search
- **LLM**: Groq Llama 3.3 70B

## Results
- **RAGAS Faithfulness**: 0.61 (measured against 20-question golden dataset)
- **Query Latency**: 0.6–0.8s
- **Citation Accuracy**: 100% (manual verification)

## Why This Baseline?

Fixed-size chunking is the academic-standard reference configuration (RAGAS paper, LlamaIndex benchmarks). This implementation:
- Proves the pipeline end-to-end correctness
- Establishes a reproducible baseline for Phase 2 comparison
- Demonstrates where naive approaches fail on complex documents (JPMorgan 10-K table splits)

## Running This Version
```bash
# From this directory
uv venv
uv pip install -e .
uv run streamlit run app.py
```

Full technical walkthrough: [docs/architecture/phase1-baseline.md](../../docs/architecture/phase1-baseline.md)
