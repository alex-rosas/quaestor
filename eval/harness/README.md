# Evaluation Harness

Infrastructure for running the RAGAS evaluation under free-tier API constraints (Groq 100k tokens/day per key).

All commands must be run from the **project root**.

## Usage

### Step 1 — Pipeline phase
```bash
bash eval/harness/run_pipeline.sh
```
Runs the RAG pipeline on all 20 questions. Rotates API keys automatically if the daily limit is hit. Saves progress to `benchmarks/ragas_results.checkpoint.json` after every question.

### Step 2 — RAGAS scoring phase
```bash
bash eval/harness/run_ragas.sh
```
Scores the pipeline answers with RAGAS, one question at a time. Rotates keys if needed. Saves progress to `benchmarks/ragas_results.ragas_checkpoint.json` after every question. Final results written to `benchmarks/ragas_results.json`.

### Check token budget
```bash
python eval/harness/check_limits.py <api_key>
```
Probes a Groq key and reports per-minute and daily token budgets. Daily limit resets at midnight UTC (6 PM Mexico City time).

## Files

| File | Purpose |
|---|---|
| `run_pipeline.sh` | Pipeline phase runner with key rotation and checkpointing |
| `run_ragas.sh` | RAGAS scoring phase runner with key rotation and checkpointing |
| `check_limits.py` | Token budget probe for any Groq API key |

## With unlimited tokens

If you have a key with no daily limit, the standard entry point works directly without the harness:

```bash
uv run python scripts/evaluate.py
```

The harness scripts are purely an infrastructure layer — they do not affect pipeline logic, retrieval, or scoring.
