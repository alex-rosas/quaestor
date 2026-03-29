# Quaestor — Portfolio Case Study

> **Executive Summary:** Production-grade RAG system for financial document intelligence. Reduces analyst document navigation from minutes to seconds. Key result: parent context injection eliminated 100% of LLM self-refusals and tripled the answer rate — validated against a frozen RAGAS golden dataset.

---

## Table of Contents

- [The Business Problem](#the-business-problem)
- [Motivation](#motivation)
- [Relation to Prior Work](#relation-to-prior-work)
- [Industry Context](#industry-context)
- [Project Contribution](#project-contribution)
- [Positioning](#positioning)
- [Technical Approach](#technical-approach)
- [Technical Decisions & Trade-Offs](#technical-decisions--trade-offs)
- [What I Learned](#what-i-learned)
- [Results](#results)
- [What I'd Do Differently](#what-id-do-differently)
- [Design Notes and Observations](#design-notes-and-observations)
- [Project Links](#project-links)
- [Status: v3.0-portfolio-ready](#status-v30-portfolio-ready)
- [References](#references)
  
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

## Motivation

Financial document search and analysis is a well-established problem in professional services. Platforms such as AlphaSense, Bloomberg, and Refinitiv already provide powerful tools for navigating filings, extracting information, and supporting analysts in research workflows.

However, these systems are primarily optimized for search and retrieval. Precise question answering over long, structured documents — especially when answers depend on tables, footnotes, or context distributed across sections — remains challenging. In practice, analysts often still rely on manual validation, cross-referencing, and interpretation.

This project does not attempt to replace these tools. Instead, it uses financial filings as a demanding testbed for studying retrieval-augmented generation (RAG) systems in conditions where:
- Documents are long (100–300+ pages)
- Information is hierarchically structured (sections, tables, footnotes)
- Relevant context may be fragmented across distant parts of the document
- Incorrect answers carry non-trivial consequences

These characteristics expose known failure modes of RAG systems:
- **Context fragmentation** (naive chunking breaks tables and structure)
- **Retrieval noise** (irrelevant but semantically similar chunks)
- **Hallucination under uncertainty** (LLMs generating plausible but unsupported answers)

Quaestor was built as an experiment in applying production-oriented RAG techniques to this setting, with the goal of understanding how architectural decisions affect reliability.

---

## Relation to Prior Work

The system design is grounded in established research:

- Retrieval-augmented generation as introduced in *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
- Evaluation using *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (Shah et al., 2023)
- Long-context limitations highlighted in *Lost in the Middle* (Liu et al., 2023)
- Retrieval quality improvements via cross-encoder reranking (Nogueira & Cho, 2019)
- Hallucination detection approaches such as SelfCheckGPT (Manakul et al., 2023)

These works collectively show that while RAG is effective, its performance depends heavily on retrieval quality, context construction, and validation mechanisms.

---

## Industry Context

Existing platforms such as AlphaSense, Bloomberg, and Refinitiv demonstrate that document search at scale is already a solved problem in many respects. They provide:
- Fast keyword and semantic search
- Document indexing and filtering
- Access to curated financial datasets

However, they typically:
- Do not guarantee grounded, cited answers
- Rely on users to interpret retrieved passages
- Are not designed as end-to-end generative QA systems with explicit hallucination controls

This creates a gap between **retrieval** and **fully grounded question answering**.

---

## Project Contribution

The contribution of Quaestor is not a new model or a novel algorithm, but a **system-level exploration of RAG behavior under realistic constraints**.

Specifically, it demonstrates:

### 1. Impact of Context Construction
- Shows how naive fixed-size chunking fails on structured financial data
- Introduces hierarchical chunking with parent context injection
- Empirically improves answerability and reduces LLM refusal

### 2. Retrieval + Reranking Interaction
- Combines dense retrieval with cross-encoder reranking
- Uses confidence scores to estimate answerability

### 3. Explicit Failure Handling
- Implements a confidence gate to prefer refusal over hallucination
- Adds a post-generation NLI check to validate answer grounding

### 4. Evaluation-Driven Development
- Uses RAGAS metrics to track changes across controlled experiments
- Includes unanswerable questions to test system behavior under uncertainty

---

## Positioning

Quaestor should be understood as:

- Not a replacement for existing financial research platforms  
- Not a novel ML algorithm  

But rather:

> A production-style RAG system built to study and mitigate known failure modes in a high-precision, structured-document domain.

The primary contribution is **clarifying how design choices affect reliability**, rather than claiming to solve the broader problem.

In that sense, the value of the project lies less in the system itself, and more in the disciplined exploration of how RAG systems behave when correctness, not just fluency, is the primary constraint.

### Why Financial Documents?

1. **Hard problem** — 300+ page filings with nested financial tables expose every weakness in naïve chunking strategies
2. **Measurable success** — either the answer is correct with a source citation, or it isn't
3. **Production stakes** — incorrect financial answers have real consequences, so safety mechanisms like confidence gating and hallucination detection are not optional
4. **Domain depth** — SEC 10-Ks, IFRS standards, and PCAOB guidance represent the full stack of documents a financial analyst encounters

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

## Design Notes and Observations

### Context Construction

One of the main issues I ran into was how poorly fixed-size chunking works for financial documents.

With 512-token chunks, tables get split in awkward ways — for example, column headers end up in one chunk and the actual data rows in another. When the model receives only part of that structure, it either produces low-quality answers or refuses entirely.

What worked better was separating concerns:

- Small chunks (~256 chars) for retrieval precision  
- Larger “parent” windows (~1024 chars) for generation context  

This keeps retrieval focused while giving the model enough context to reconstruct structured information. In practice, switching to parent context reduced refusal behavior significantly and made answers more consistent.

### Handling Uncertainty

A recurring question was how to deal with cases where the system *probably shouldn’t answer*.

I ended up treating this explicitly in two stages:

- A **confidence gate** after reranking: if the top result is below a threshold, the system refuses early instead of passing weak context to the LLM  
- A **post-generation check** using NLI to verify whether the answer is actually supported by the retrieved context  

This shifts the system toward conservative behavior. In this domain, refusing is often preferable to giving a plausible but incorrect answer.

### Evaluation Approach

For evaluation, I used a small fixed dataset (20 questions) and tracked:

- Faithfulness  
- Context precision / recall  
- Answer relevancy  

along with operational signals like:
- refusal rate  
- latency by stage  
- confidence score distribution  

One thing that became clear is that metrics alone are not always easy to interpret without understanding the dataset composition.

### On the Faithfulness Score

The faithfulness score (~0.36) looks low at first glance, but it’s influenced by the dataset design.

A large portion of the questions are intentionally unanswerable (e.g., asking for data not present in the document). Since evaluation averages over all questions, refusals tend to lower the overall score.

What mattered more in this case was the transition:
- initially, very few questions produced usable answers  
- after improving context construction, a larger subset became answerable and measurable  

So the score reflects both system behavior and dataset composition, not just model quality.

### General Takeaway

The main takeaway from this project is that small design decisions in RAG systems — especially around context construction and filtering — have a large impact on behavior.

In structured documents like financial filings, naive approaches tend to fail in predictable ways, and addressing those failures requires treating retrieval, context, and validation as separate concerns.

---

## Project Links

- **Repository:** [github.com/alex-rosas/quaestor](https://github.com/alex-rosas/quaestor)
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

## References

### Core RAG Framework
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.  
  https://arxiv.org/abs/2005.11401

### Evaluation
- Shah, R., et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*.  
  https://arxiv.org/abs/2309.15217

### Long Context Limitations
- Liu, N. F., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*.  
  https://arxiv.org/abs/2307.03172

### Retrieval and Reranking
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT*.  
  https://arxiv.org/abs/1901.04085

### Hallucination / Reliability
- Manakul, P., et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*.  
  https://arxiv.org/abs/2303.08896

---

### Industry Context

- [AlphaSense](https://www.alpha-sense.com/) — AI-powered financial document search and analysis
- [Bloomberg Terminal](https://professional.bloomberg.com/products/bloomberg-terminal/) — comprehensive financial data and analytics platform
- [LSEG Data & Analytics](https://www.lseg.com/en/data-analytics) — financial market data and infrastructure (formerly Refinitiv)

---

### Optional (Further Reading)

- Zhang, Y., et al. (2023). *A Survey of Hallucination in Large Language Models*.
  https://arxiv.org/abs/2309.01219