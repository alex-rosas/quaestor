# Quaestor

Production-grade RAG system for financial document intelligence.

**Status**: Phase 1 complete (fixed-size chunking baseline). Phase 2 development in progress.

_Full documentation in progress. See `docs/architecture/` for technical deep-dives._

## Quick Start
```bash
# Install dependencies
uv sync

# Run the Streamlit demo
uv run streamlit run app.py
```

## Project Structure

- `src/quaestor/` — Core RAG pipeline
- `examples/phase1-baseline/` — Phase 1 reference implementation
- `docs/` — Architecture documentation and case studies
- `benchmarks/` — Evaluation results across phases

