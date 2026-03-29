# Phase 2 & 3 Baseline Summary

---

## Phase 2 Baseline — Child-Only Context (256 chars)

**Run date:** 2026-03-27
**Output file:** `eval/results/phase2_child_baseline_clean.json`
**Status:** ✅ Clean baseline (no TPD errors, all 20 questions evaluated)

### RAGAS Scores

| Metric | Score | Status |
|--------|-------|--------|
| Faithfulness | NaN | ✅ Expected (insufficient answers to measure) |
| Answer Relevancy | 0.138 | ✅ Expected (most responses were refusals) |
| Context Precision | 0.000 | ✅ Expected (chunks didn't enable answers) |
| Context Recall | NaN | ✅ Expected (insufficient answers to measure) |

### Pipeline Outcomes

| Outcome | Count | % |
|---------|-------|---|
| Gate-refused | 12 | 60% |
| LLM self-refused | 6 | 30% |
| Answered | 2 | 10% |
| Errors | 0 | 0% |

### Interpretation

The NaN and zero scores are not failures — they are the expected artifact of a
60% gate refusal rate. RAGAS cannot measure faithfulness when no answers exist.
The system refused rather than hallucinated; this is correct behavior for a
financial RAG system.

**Root cause:** 256-char child chunks are too short to contain complete numerical
answers. The gate correctly identifies them as insufficient context.

---

## Phase 3 Baseline — Parent-Injected Context (1024 chars)

**Run date:** 2026-03-28
**Output file:** `eval/results/phase3_parent_injection.json`
**Change:** One line in `graph.py` — `doc.metadata.get('parent_content', doc.page_content)`

### RAGAS Scores

| Metric | Score |
|--------|-------|
| Faithfulness | 0.367 |
| Answer Relevancy | 0.263 |
| Context Precision | 0.306 |
| Context Recall | 0.500 |

### Pipeline Outcomes

| Outcome | Count | % |
|---------|-------|---|
| Gate-refused | 12 | 60% |
| LLM self-refused | 0 | 0% |
| Answered | 8 | 40% |
| Errors | 0 | 0% |

---

## Before/After Delta

| Metric | Phase 2 | Phase 3 | Improvement |
|--------|---------|---------|-------------|
| Faithfulness | NaN | 0.367 | Measurable |
| Answer Relevancy | 0.138 | 0.263 | +91% |
| Context Precision | 0.000 | 0.306 | From zero to real |
| Context Recall | NaN | 0.500 | Measurable |
| LLM self-refusals | 30% | 0% | -30% |
| Answer rate | 10% | 40% | +300% |

**Full comparison:** `eval/results/COMPARISON.md`
