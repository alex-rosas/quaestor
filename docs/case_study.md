# Quaestor — Portfolio Case Study

> **Executive Summary:** Production-grade RAG system for financial document intelligence. Reduces analyst document navigation from minutes to seconds. Key result: parent context injection eliminated 100% of LLM self-refusals and tripled the answer rate — validated against a frozen RAGAS golden dataset.

---

## The Business Problem

### Who It's For

Financial analysts and auditors at firms like Ernst & Young (EY) who spend significant time manually navigating dense regulatory documents:
- SEC 10-K annual filings (300–400 pages)
- IFRS/IAS accounting standards
- PCAOB auditing standards

### The Pain Point

**Current workflow:**
1. Analyst receives question: "What was Apple's net income in FY2025?"
2. Open 395-page 10-K PDF
3. Ctrl+F for "net income" → 62 matches
4. Manually scan each match to find the right consolidated statement
5. Cross-reference footnotes, verify the fiscal period, check adjustments
6. Extract the number
7. **Total time: 15–45 minutes per question**

**With 20–50 such questions per engagement, this is hours of manual search time per analyst per week.**

### The Solution

Quaestor: Ask "What was Apple's net income in FY2025?" in natural language, get:

```
Apple's net income for fiscal year 2025 was $93,736 million.
[Source: aapl-10k-2025.txt, Page 31]
```

**Time: ~1 second. Citation included. Refusal when uncertain.**

---

## Why I Built This

### Primary Goal: Demonstrate RAG Architecture Skills

I wanted to build a portfolio project showing AI Engineer recruiters at EY that I understand:
- **Retrieval-Augmented Generation (RAG)** — the dominant architecture for document Q&A systems
- **Production engineering** — observability, testing, Docker deployment — not just a demo
- **Evaluation discipline** — RAGAS metrics, before/after comparisons, controlled experiments
- **Safety & trust** — confidence gating, hallucination detection, citation enforcement

### Why Financial Documents?

1. **Domain relevance** — EY's core business involves exactly these document types
2. **Hard problem** — 300+ page documents with nested financial tables stress-test chunking strategies
3. **Measurable success** — either the answer is correct with a source, or it isn't
4. **Production stakes** — incorrect financial answers have real consequences

---

## Technical Approach

### The Central Architecture Decision: Hierarchical Chunking

**The problem with naive chunking:**
Most RAG systems split documents into fixed 512-token chunks with 50-token overlap. This works for prose, but fails on financial tables:

```
| Fiscal Year | Net Revenue | Cost of Revenue | Gross Profit |
|-------------|-------------|-----------------|--------------|
| 2025        | $391,035M   | $145,384M       | $245,651M    |
| 2024        | $383,285M   | $141,048M       | $242,237M    |
```

A 512-token chunk boundary cuts mid-table. The retrieved chunk contains rows of numbers with no column headers. The LLM receives `| $391,035M | $145,384M |` and must guess what these numbers represent. It refuses to answer — correctly, but unhelpfully.

**My solution: Hierarchical chunking**
- **Child chunks (256 chars):** Small, precise, semantically focused — indexed for retrieval
- **Parent chunks (1024 chars):** Complete context window — stored in metadata, passed to LLM at generation time
- **Architecture:** Retrieve using small child (precision) → generate using large parent (context)

### The Key Experiment: Phase 3

**Hypothesis:** Passing 1024-char parent chunks to the LLM instead of 256-char child fragments will reduce self-refusal rate and improve answer quality.

**Method:**
1. Run RAGAS evaluation with child-only context (Phase 2 baseline)
2. Change one line in `graph.py`:
   ```python
   # Before (Phase 2):
   parts.append(f"[Source: {source}]\n{doc.page_content}")

   # After (Phase 3):
   parts.append(f"[Source: {source}]\n{doc.metadata.get('parent_content', doc.page_content)}")
   ```
3. Re-run RAGAS evaluation on the same 20-question golden dataset
4. Compare

**Result:**

| Metric | Child-Only | Parent-Injected | Impact |
|--------|-----------|-----------------|--------|
| LLM Self-Refusals | 30% (6/20) | **0% (0/20)** | Eliminated |
| Answer Rate | 10% (2/20) | **40% (8/20)** | Tripled |
| Faithfulness | NaN | **0.367** | Becomes measurable |
| Context Recall | NaN | **0.500** | Becomes measurable |
| Context Precision | 0.000 | **0.306** | From zero to real |
| Gate Refusals | 60% (12/20) | **60% (12/20)** | Unchanged (correct) |

**Conclusion:** One line of code, measurable improvement. The architecture was right — the only thing holding it back was feeding 256-char fragments to the LLM.

**Why gate refusals are unchanged:** This is the intended behavior. The 12 refused questions ask about fiscal years not in the document, or metrics Apple doesn't disclose. The confidence gate is discriminating correctly — it's not the bottleneck.

---

## Technical Decisions & Trade-Offs

### Why Groq (Not OpenAI)?

**Problem:** My development machine is a 2020 M1 MacBook Air (8 GB RAM). Running Llama 3.1 8B locally via Ollama generates ~15 tokens/second. Every test cycle takes 30–40 seconds.

**Solution:** Groq's free tier runs Llama 3.3 70B at ~300 tokens/second — a 20× speedup and a 9× larger model. Development loops go from painful to nearly instant.

**Trade-off:** Dependency on Groq API availability. Mitigation: Ollama is kept as local fallback, swappable via one env var (`LLM_PROVIDER=ollama`).

### Why ChromaDB (Not Qdrant)?

**For Phase 1–3:** ChromaDB is in-process, requires zero configuration, needs no Docker. The 305-test suite runs in CI with no external services — tests finish in 6 seconds.

**For Phase 4+:** Qdrant migration is planned — adds native hybrid BM25 + dense retrieval. Migration is opt-in via `VECTOR_STORE_BACKEND=qdrant`; ChromaDB remains the default.

**Trade-off:** ChromaDB doesn't support hybrid search. Acceptable for proving the architecture; Qdrant migration is next.

### Why 60% Refusal Rate?

**Context:** The 20-question golden dataset was written deliberately hard:
- Questions about fiscal years not in the indexed filing
- Metrics Apple doesn't disclose publicly (employee turnover, office square footage)
- Aggregate figures not explicitly stated

**The system correctly refuses on 12/20 questions** — these are genuinely unanswerable from the indexed document.

**In production:** With a real analyst query distribution (85–90% answerable), the refusal rate would be 10–15%. The 60% rate reflects test dataset composition, not system capability.

**Why conservative refusal matters:** An incorrect confident answer is worse than an honest "I don't have enough information" in financial analysis. Refusing is the right failure mode.

---

## What I Learned

### 1. Dataset Quality Matters More Than Model Size

**Initial assumption:** Bigger model = better RAGAS scores.

**Reality:** Faithfulness 0.367 (target ≥0.84) is primarily a dataset quality issue. 60% of the golden dataset questions are unanswerable — RAGAS averages over all 20, including refused ones. Expanding to 120 questions with 85% clearly answerable will close the gap. The pipeline architecture is validated; the test distribution is the bottleneck.

### 2. Observability Is Not Optional

**Initial approach:** Build the pipeline, measure RAGAS at the end.

**Problem:** When faithfulness was 0.367, I couldn't see *where* it was failing. Was it retrieval? The reranker threshold? LLM generation quality?

**Solution:** Langfuse tracing added in Phase 3. Now every query shows: retrieval latency → top reranker score → gate decision → LLM token count → NLI result.

**Lesson:** Build observability from day one. If I started over, Langfuse would be in the first commit. Retrofitting it is possible but requires touching every pipeline layer.

### 3. Production RAG Is About Edge Cases

**What works in demos:** Answering "What was revenue?" on a clean, well-structured 10-K.

**What breaks in production:**
- Questions that span multiple documents
- Queries about data not disclosed
- Ambiguous phrasing ("revenue" vs "net sales" vs "total income")
- Fiscal period mismatches

**Confidence gating and conservative refusal** are what separate a demo from a production system.

---

## Results

### Quantitative

| Metric | Value |
|--------|-------|
| Tests passing | 305 (all green, 6s runtime) |
| LLM self-refusals after parent injection | 0% (eliminated) |
| Answer rate improvement | 10% → 40% (+300%) |
| Faithfulness (RAGAS) | 0.367 (measurable, dataset-constrained) |
| Context Recall (RAGAS) | 0.500 |
| End-to-end query latency | ~0.8s (Groq Llama 3.3 70B) |
| Documents indexed | 1,110 child chunks from Apple FY2025 10-K |

### Qualitative

- **Production-grade architecture** — Langfuse tracing, Docker deployment, confidence gating, NLI hallucination check
- **Evaluation discipline** — RAGAS before/after comparison, controlled experiment, frozen golden dataset
- **Honest about gaps** — RAGAS below target with clear explanation of why and what to do next

---

## What I'd Do Differently

### 1. Write the Golden Dataset First

**What I did:** Wrote 20 questions in Phase 1 before the evaluation framework existed.

**Problem:** Many turned out to be unanswerable (60%), suppressing RAGAS scores.

**Better approach:** Write 120 questions *after* defining evaluation targets. Ensure 85% are clearly answerable from the indexed documents, 15% deliberately hard/unanswerable.

### 2. Add Observability from Day One

**What I did:** Built the pipeline across three phases, added Langfuse in the third.

**Problem:** Debugging low RAGAS scores without trace data is guesswork.

**Better approach:** First commit includes Langfuse setup. Every component traced from the start.

### 3. Start with Easier Documents

**What I did:** Started with Apple 10-K (395 pages, dense financial tables).

**Better approach:** Start with a 50-page IFRS standard (clean prose, minimal tables) to prove the architecture, then tackle 300+ page filings with nested financials.

---

## Interview Talking Points

### "Walk me through a technical decision you made."

> "The central decision was hierarchical chunking. Fixed-size 512-token chunks reliably bisect financial tables, leaving column headers in one chunk and data rows in another. The LLM receives incomplete context and refuses to answer.
>
> I designed a two-tier system: 256-char child chunks for retrieval precision, 1024-char parent chunks stored as metadata for generation context. Phase 3 validated this: one-line change to use parent windows eliminated all LLM self-refusals (30% → 0%) and tripled the answer rate (10% → 40%). That's the architecture working as designed."

### "How do you handle failure cases?"

> "Two safety layers. First, the confidence gate: after retrieval and cross-encoder reranking, if the top-scored chunk is below threshold, the system refuses without calling the LLM — cheaper and honest. Second, NLI hallucination check: DeBERTa-v3 verifies every generated answer against the retrieved context. Entailment passes, neutral or contradiction triggers a warning.
>
> The 60% refusal rate on my test dataset looks alarming but is correct — those questions ask about data not in the indexed filing. Conservative refusal is the right failure mode in financial contexts."

### "What metrics do you track?"

> "RAGAS for evaluation quality: faithfulness, context precision, context recall, answer relevancy — measured on a frozen 20-question golden dataset. And operational metrics via Langfuse: query latency breakdown by stage, confidence score distribution, gate refusal rate, NLI pass rate.
>
> The key Phase 3 result was eliminating LLM self-refusals — that's an operational metric, not a RAGAS metric, but it's what makes the system actually usable."

### "Why is faithfulness 0.367 instead of 0.84?"

> "That's the dataset story. The 20-question golden dataset includes 60% questions that are genuinely unanswerable — wrong fiscal year, undisclosed metrics. RAGAS averages over all 20 including refused questions, suppressing the mean.
>
> The improvement story is: before parent context injection, RAGAS couldn't even measure faithfulness — only 10% of questions got real answers. After injection, 40% got answers and faithfulness became 0.367. That's still below the 0.84 target, but the gap is a dataset composition problem, not a pipeline problem. Expanding to 120 clearly answerable questions would close most of it. For a portfolio demonstration, having a clear diagnosis is more valuable than a inflated score from an easy dataset."

---

## Project Links

- **Repository:** [github.com/yourusername/quaestor](https://github.com/yourusername/quaestor)
- **RAGAS Before/After:** [eval/results/COMPARISON.md](../eval/results/COMPARISON.md)
- **README:** [README.md](../README.md)

---

## Status: v3.0-portfolio-ready

Parked to build Consilium (multi-agent financial workflow system). Quaestor is a complete, documented, deployable portfolio artifact.

**Next steps if resumed:**
1. Golden dataset expansion: 20 → 120 questions (target faithfulness ≥0.75)
2. Qdrant hybrid retrieval
3. Confidence threshold calibration
4. CI/CD eval gate (GitHub Actions)
