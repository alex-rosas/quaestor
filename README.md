# Quaestor

Production-grade RAG system for financial document intelligence. Reduces time analysts spend navigating 300-page SEC filings from hours to seconds, with source citations for every answer.

---

## Key Results (Phase 1 Baseline)

| Metric | Value |
|---|---|
| **RAGAS Faithfulness** | 0.61 (measured against 20-question dataset) |
| **Query Latency (P95)** | 0.8s (Groq Llama 3.3 70B) |
| **Citation Accuracy** | 100% (manual verification) |
| **Documents Indexed** | 579 chunks from Apple FY2025 10-K |
| **Test Coverage** | 123 unit tests, all passing |

---

## What This Is

A Retrieval-Augmented Generation (RAG) system that lets financial analysts query SEC 10-K filings, IFRS standards, and PCAOB auditing standards in natural language. Built as a portfolio project targeting production AI engineering roles at firms like EY GDS Mexico.

**Target user**: Financial analyst or auditor  
**Primary documents**: SEC EDGAR filings (10-K, 10-Q), IFRS/IAS standards, PCAOB standards  
**Tech stack**: LangChain, Groq (Llama 3.3 70B), ChromaDB, nomic-embed-text, Streamlit

---

## Technical Highlights

### Architecture
- **Hierarchical chunking** (planned Phase 2): Parent chunks (1024 tokens) for context, child chunks (256 tokens) for retrieval precision
- **Hybrid retrieval** (planned Phase 2): Dense semantic search + BM25 sparse retrieval via Qdrant
- **LangGraph state machine** (planned Phase 2): Multi-step retrieval with confidence thresholding and graceful refusal
- **Swappable LLM providers**: Groq, Ollama, Together.ai — switch with one environment variable

### Evaluation
- **RAGAS metrics**: Faithfulness, context recall, context precision, answer relevancy
- **Golden dataset**: 20 questions (15 answerable, 5 deliberately unanswerable)
- **CI/CD eval gates** (planned Phase 2): RAGAS runs on every PR touching the RAG pipeline
- **Benchmark tracking**: Before/after comparison across phases

### Stack Rationale
All free and open source:
- **Groq** (not local Ollama): 20x faster inference (300 tok/s vs 15 tok/s) on M1 8GB hardware
- **ChromaDB** (Phase 1) → **Qdrant** (Phase 2): Zero-config development → production hybrid search
- **nomic-embed-text**: Outperforms OpenAI Ada-002 on retrieval benchmarks, runs locally, costs nothing

---

## Project Evolution

This project was built in measured phases, with each improvement tracked against a controlled baseline.

### Phase 1: Fixed-Size Chunking Baseline ✅ Complete

**Tag**: [`v1.0-phase1-baseline`](../../tree/v1.0-phase1-baseline)  
**Frozen code**: [`examples/phase1-baseline/`](examples/phase1-baseline/)  
**Documentation**: [Phase 1 Technical Walkthrough](docs/architecture/phase1-baseline.md)

**Results**:
- RAGAS Faithfulness: 0.61
- Query latency: 0.6–0.8s
- 579 chunks from Apple FY2025 10-K
- End-to-end pipeline working

**Why freeze Phase 1?**  
This is the "before" in every improvement claim. Fixed-size chunking (512 tokens, 50 overlap) is the academic-standard baseline (RAGAS paper, LlamaIndex benchmarks). Freezing it makes Phase 2 comparisons reproducible.

### Phase 2: Hierarchical Chunking + Hybrid Retrieval 🚧 In Progress

**Target improvements**:
- RAGAS Faithfulness: 0.61 → 0.84 (+37%)
- Hierarchical parent-child chunking
- Hybrid BM25 + dense retrieval via Qdrant
- LangGraph multi-step retrieval state machine

**Documentation**: [Phase 2 Architecture](docs/architecture/phase2-hierarchical.md) _(coming soon)_

### Phase 3: Production Hardening 📅 Planned

- Guardrails (Presidio PII detection, NLI hallucination checking)
- Langfuse observability
- CI/CD eval gates with PR comment tables
- Docker Compose deployment

---

## Quick Start

### Prerequisites
- Python 3.11+
- `uv` package manager
- Groq API key (free tier: https://console.groq.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/quaestor
cd quaestor

# Set up environment
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Install dependencies
uv sync

# Run the Streamlit demo
uv run streamlit run app.py
```

### Running the Frozen Phase 1 Baseline

```bash
cd examples/phase1-baseline
cp .env.example .env
# Add your GROQ_API_KEY

uv sync
uv run streamlit run app.py
```

### Running Tests

```bash
# All unit tests (no external services required)
uv run pytest tests/ -v

# Integration smoke test (requires Ollama + Groq)
python scripts/smoke_test.py
```

---

## Documentation

### Architecture Deep-Dives
- [Phase 1 Baseline Technical Walkthrough](docs/architecture/phase1-baseline.md) — Complete explanation of the fixed-size chunking pipeline, including what works and where it breaks

### Case Studies
- [SEC EDGAR Format Discovery](docs/case-studies/sec-edgar-format.md) — How assuming PDFs led to discovering SGML containers
- [ChromaDB Collection Isolation Bug](docs/case-studies/collection-isolation.md) — When uploaded docs mixed with indexed chunks
- [M1 Hardware Constraints](docs/case-studies/m1-hardware-constraints.md) — Why Groq cloud inference beats local Ollama for development

### Engineering Principles
- [Lessons Learned](docs/lessons-learned.md) — Generalizable insights from Phase 1 problems

---

## Design Decisions

### Why Fixed-Size Chunking as Baseline?

Phase 1 deliberately uses fixed-size chunking (512 tokens, 50-token overlap) — the reference configuration from the RAGAS paper. This is **not** the optimal strategy. It's the controlled "before" that makes Phase 2's "after" a meaningful claim.

**Where it fails**: JPMorgan's 10-K has tables spanning 650+ tokens. A 512-token boundary cuts mid-table, losing column headers. Phase 2's hierarchical chunking is designed to fix exactly this.

### Why Groq Over Local Inference?

Development machine: 2020 M1 MacBook Air (8GB RAM)  
Local Ollama: ~15 tokens/second  
Groq cloud: ~300 tokens/second (free tier)

20x speedup = 50 test queries go from 12 minutes to 35 seconds. The development loop tax compounds over weeks. Groq is the pragmatic choice for velocity.

### Why ChromaDB → Qdrant Migration?

**ChromaDB (Phase 1)**:
- Zero config, in-process
- Perfect for development and CI
- Dense-only retrieval

**Qdrant (Phase 2)**:
- Self-hosted via Docker
- Native hybrid search (dense + BM25)
- Production-grade scalability

The switch is transparent — LangChain wraps both with the same interface.

---

## What Was Ruled Out and Why

| Ruled Out | Reason |
|---|---|
| Azure OpenAI | Paid API — entire stack must be free/OSS for portfolio purposes |
| Pinecone | Free tier requires credit card; Qdrant is equivalent and truly free |
| OpenAI Ada-002 embeddings | Paid; nomic-embed-text is free and competitive on benchmarks |
| LLM-only golden dataset | Circular validation; hybrid human-verified approach instead |
| Mono-repo with Consilium | Different dependency trees; cleaner as separate repos |

---

## Project Structure

```
quaestor/
├── src/quaestor/              # Main codebase (evolves Phase 1 → Phase 2 → Phase 3)
├── tests/                     # 123 unit tests
├── examples/phase1-baseline/  # Frozen Phase 1 code (teaching artifact + comparison baseline)
├── docs/
│   ├── architecture/          # Phase-specific deep-dives
│   ├── case-studies/          # Problem investigation narratives
│   └── lessons-learned.md     # Generalizable engineering principles
├── benchmarks/                # RAGAS results per phase
├── app.py                     # Streamlit demo UI
└── README.md                  # This file
```

---

## Why This Project Exists

This is a portfolio project built to demonstrate production AI engineering skills for roles at firms like EY GDS Mexico. The job posting required:
- LLMs and RAG pipelines
- LangChain/LangGraph
- Vector databases
- Prompt engineering
- Observability and evaluation
- Docker and container orchestration

Quaestor covers all of these, with **measured improvements** and **reproducible baselines**.

The pedagogical documentation exists because production engineers don't just build systems — they build systems that others can understand, maintain, and learn from.

---
## License

MIT License — Free for educational and portfolio use
