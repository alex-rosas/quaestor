# Phase 2 Child-Chunk Baseline — RAGAS Results

**Run timestamp:** 2026-03-27 04:07:29 UTC
**Output file:** `eval/results/phase2_child_baseline.json`

## Configuration

| Parameter | Value |
|-----------|-------|
| Chunking | Child-only (256 chars, no parent injection) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Confidence threshold | 0.0 (default) |
| LLM provider | Groq `llama-3.3-70b-versatile` |
| Vector store | Chroma (1,110 documents) |
| Questions | 20 |

## Pipeline Phase (before RAGAS)

| Outcome | Count | Notes |
|---------|-------|-------|
| Gate-refused (refused=True) | 12 | Cross-encoder scored top chunk below threshold 0.0 — canned refusal text, no LLM call |
| TPD errors (429) | 6 | Groq rolling window exhausted (~99,924/100,000 tokens); empty contexts; `[ERROR: 429...]` response |
| Gate-passed, LLM refused on its own | 1 | Gate let the query through (score ≥ 0.0) but LLM said "I don't have enough information" |
| Gate-passed, correct answer | 1 | Q03: net income FY2024 = $93,736M — retrieved correct income statement chunk |
| **Total** | **20** | |

**About reproducibility:** The 6 TPD errors make this run non-reproducible. Someone running the same script on a fresh TPD budget will get different results (those 6 questions will either be answered or gate-refused, not error). The baseline must be rerun on a day with a full 100k token budget before it can serve as a valid denominator for Phase 3 comparison. See `genesis/BITACORA.md` for exact rerun instructions.

## RAGAS Scores

| Metric | Score |
|--------|-------|
| faithfulness | NaN |
| answer_relevancy | 0.0 |
| context_precision | 0.0 |
| context_recall | NaN |

### Why all scores are degenerate

These scores are not meaningful measurements — they are the expected artifact of a near-total refusal rate:

- **NaN faithfulness / context_recall**: RAGAS cannot compute these for refusal responses or empty contexts.
- **0.0 answer_relevancy / context_precision**: A handful of questions received non-NaN scores during the RAGAS evaluation pass, but all were 0.0 (the RAGAS LLM also hit TPD limits during its own 80-job evaluation phase, causing its judgements to fail).

### What this baseline actually establishes

This run is the **denominator** for Phase 3 comparison. Its value is not the numbers themselves but what they reveal about Phase 2 child-chunk-only retrieval:

1. The confidence gate is working correctly — it is refusing when retrieved chunks are too short to contain complete numerical answers.
2. The correct answer rate for factual numerical questions is ~5% (1/20) at threshold 0.0 with 256-char child chunks.
3. Parent context injection (Phase 3) is the intervention designed to close this gap: longer parent windows give the LLM enough context to answer questions the child chunks currently refuse.

## Phase 3 Target

After parent context injection is implemented, re-run with:

```
uv run python scripts/evaluate.py --output eval/results/phase3_parent_injection.json
```

Expected improvements: fewer refusals on answerable questions (Q1, Q2, Q5, Q8, Q9, Q12–Q15, Q17–Q19), stable refusals on genuinely unanswerable questions (Q16, Q18, Q19, Q20), and non-NaN RAGAS scores reflecting real answer quality.
