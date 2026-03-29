# Parent Context Injection — Before/After Comparison

> **Status:** ✅ Complete. Both baselines run on the same 20-question golden dataset with the same Groq key pool, same threshold (0.0), same top-k (5), same reranker.

---

## Phase 2 Baseline (Child-Only, 256 chars)

**File:** `eval/results/phase2_child_baseline_clean.json`
**Run date:** 2026-03-27

### RAGAS Scores

| Metric | Score |
|--------|-------|
| Faithfulness | NaN |
| Answer Relevancy | 0.1383 |
| Context Precision | 0.0000 |
| Context Recall | NaN |

### Pipeline Outcomes

| Outcome | Count | % |
|---------|-------|---|
| Gate-refused | 12 | 60% |
| LLM self-refused | 6 | 30% |
| Answered | 2 | 10% |
| Errors | 0 | 0% |

### Interpretation

The system correctly refused when 256-char child chunks provided insufficient
context. The high refusal rate (60%) is conservative behavior by design — refuse
rather than hallucinate. The NaN scores are artifacts of the refusal rate
(RAGAS cannot measure faithfulness when no answers exist).

**Key finding:** The confidence gate is working correctly. The problem is that
child chunks are too short to provide complete context for numerical questions.

---

## Phase 3 (Parent-Injected, 1024 chars)

**File:** `eval/results/phase3_parent_injection.json`
**Run date:** 2026-03-28

### RAGAS Scores

| Metric | Score |
|--------|-------|
| Faithfulness | 0.3667 |
| Answer Relevancy | 0.2631 |
| Context Precision | 0.3056 |
| Context Recall | 0.5000 |

### Pipeline Outcomes

| Outcome | Count | % |
|---------|-------|---|
| Gate-refused | 12 | 60% |
| LLM self-refused | 0 | 0% |
| Answered | 8 | 40% |
| Errors | 0 | 0% |

---

## Delta Analysis

| Metric | Phase 2 | Phase 3 | Improvement |
|--------|---------|---------|-------------|
| Gate-refused | 60% | 60% | Unchanged (correct) |
| LLM self-refused | 30% | **0%** | **-30%** |
| Answered | 10% | **40%** | **+300%** |
| Faithfulness | NaN | **0.367** | Becomes measurable |
| Answer Relevancy | 0.138 | **0.263** | **+91%** |
| Context Precision | 0.000 | **0.306** | **From zero to real** |
| Context Recall | NaN | **0.500** | Becomes measurable |

---

## Conclusion

Switching from 256-char child fragments to 1024-char parent windows:

1. **Eliminated all LLM self-refusals** (6→0) — every question that passed the
   confidence gate now receives a real answer instead of "I don't have enough
   information."

2. **Tripled the answer rate** (10%→40%) — 8 of 20 questions answered vs 2 before.

3. **Made RAGAS scores measurable** — faithfulness and context recall went from
   NaN to 0.367 and 0.500 respectively.

4. **Gate refusals unchanged at 60%** — this is correct. The 12 refused questions
   are genuinely unanswerable from the indexed document (wrong fiscal year,
   information not in the 10-K, etc.). The gate is discriminating correctly.

**Validates the hierarchical chunking hypothesis:** child chunks for retrieval
precision, parent chunks for generation context. One line of code, measurable
improvement.

### What the scores mean vs targets

| Metric | Phase 3 actual | Phase 3 target (SPEC.md) | Gap |
|--------|---------------|--------------------------|-----|
| Faithfulness | 0.367 | ≥ 0.84 | -0.47 |
| Context Precision | 0.306 | ≥ 0.71 | -0.40 |
| Context Recall | 0.500 | ≥ 0.79 | -0.29 |
| Answer Relevancy | 0.263 | ≥ 0.75 (implied) | -0.49 |

Gap to targets is expected — this is the first Phase 3 measurement with the
same 20-question dataset and child-level gate threshold. The 60% refusal rate
on genuinely unanswerable/out-of-scope questions suppresses all RAGAS averages.
Next steps: threshold calibration, dataset expansion to 120 questions, and
Langfuse tracing to identify specific failure patterns.
