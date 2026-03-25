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

## Setup and Installation

### 1. Environment Setup
```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your API keys
# At minimum, you need ONE of:
#   - GROQ_API_KEY (recommended - free tier, fast)
#   - Local Ollama running (no API key needed)
#   - TOGETHER_API_KEY (fallback)
```

### 2. Install Dependencies
```bash
# Create virtual environment and install packages
uv sync
```

### 3. Download Sample Data (Optional)

The baseline was tested against Apple's FY2025 10-K. To reproduce:
```bash
# Download Apple 10-K
uv run python -c "
from sec_edgar_downloader import Downloader
dl = Downloader('YourName', 'your@email.com')
dl.get('10-K', 'AAPL', limit=1)
"
```

Or upload your own PDF via the Streamlit interface.

### 4. Run the Application
```bash
uv run streamlit run app.py
```

## Troubleshooting

**"KeyError: 'GROQ_API_KEY'"**
→ Copy `.env.example` to `.env` and add your Groq API key

**"Connection refused to localhost:11434"**
→ If using Ollama, start it first: `ollama serve`

**"No chunks retrieved"**
→ Index a document first via the Streamlit sidebar before querying
