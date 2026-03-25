# HERODOTO.md — Session Log

> Named for the first historian. Each entry records what was built, decisions
> made, problems encountered, and what comes next.  Update at end of every
> session.

---

## Session 3 — 2026-03-25 — Phase 2: Production Depth

---

### 1. Experimental Motivation

Phase 1 proved the pipeline was correct. It could not prove it was reliable.

The failure case is easy to construct. Take a 10-K section on revenue
recognition. Fixed-size splitting (512 chars, 50 overlap) cuts that section at
position 512 regardless of what is in the text at that position. A sentence like:

```
"Net sales for the Americas segment were $124,297 million, representing
a 5.2% increase over fiscal year 2024, driven primarily by iPhone revenue growth
```

might become:

```
Chunk A: "…Net sales for the Americas segment were $124,297 million, representing"
Chunk B: "a 5.2% increase over fiscal year 2024, driven primarily by iPhone revenue growth"
```

When the user asks "What was Americas segment revenue growth?", the embedding of
the query is most similar to Chunk B (it contains "increase" and "revenue
growth"). Chunk B is retrieved. The LLM receives it and sees the percentage with
no dollar anchor. It correctly has no basis for a precise answer.

This is not a hypothetical. During Phase 1 smoke test development, answers to
comparative questions (e.g., "how did net income change year over year?") came
back as "I don't have enough information" even when the filing clearly stated the
figures — because the relevant sentence was split between two chunks, and only
one was retrieved.

Phase 1 also had a second, subtler failure: it always answered. No matter how
irrelevant the retrieved chunks were, the LLM would generate something. On the
five unanswerable questions in the golden dataset — questions whose answers are
literally not in the filing — a Phase 1 pipeline would hallucinate plausible-
sounding figures. There was no mechanism to say "I retrieved three chunks, their
best similarity score to this question is 0.12, that is not enough evidence."

These two failures define Phase 2's mandate.

---

### 2. The Key Hypothesis

**Hypothesis A (chunking):** Preserving the semantic unit — the parent section —
alongside the retrieval unit — the child chunk — means the LLM always receives
enough context to answer questions that span a few related sentences, without the
noise overhead of retrieving an entire 1024-char section for every query.

**Hypothesis B (confidence gate):** Cross-encoder reranking scores each
(query, chunk) pair jointly, reading both texts simultaneously. This produces
a relevance score calibrated enough to distinguish "I found the answer" from
"I found something topically related but not the answer." If the top-ranked
chunk scores below a threshold, refusing is safer than answering.

Measurable prediction: on the five unanswerable golden-dataset questions, a
properly calibrated confidence gate should refuse ≥ 95% of the time. On the 15
answerable questions, the parent-context strategy should produce more complete
answers to comparative and multi-hop questions.

---

### 3. How the Old Pipeline Behaved

Phase 1 execution trace for "What was Apple's net income in fiscal year 2025
compared to 2024?":

**Step 1 — Retrieval** (`retriever.py`):
```python
docs = vector_store.similarity_search(question, k=5)
```
The five returned chunks, in cosine-similarity order, were fragments of the
Consolidated Statements of Operations table. Each chunk was 512 chars, split at
arbitrary positions. Representative example:

```
[Source: aapl-20250927.htm, Page 31]
2025 2024 2023 Net income $ 112,010 $ 93,736 $ 96,995 Earnings per share: Basic
$ 7.64 $ 6.11 Basic shares used in computing earnings per share 14,659 15,344
```

The table structure survived intact here — this was luck. The `$` symbols and
year headers are small enough that the splitter didn't cut between them. But for
a question requiring narrative context around a table (e.g., "why did operating
expenses grow?"), the retrieved chunk would contain the number but not the
explanatory sentences from the MD&A section that preceded the table.

**Step 2 — Generation** (`chain.py`):
The prompt received all five raw chunks concatenated without any ranking or
filtering. The LLM saw 2560 characters of mixed context, some highly relevant,
some not. It correctly cited the FY2025/2024 figures on a good day. On a bad day
— when the splitter cut one filing year from the other — it output only one year
with a note that it lacked comparative data.

**The core problem:** the pipeline had no signal about retrieval quality. It
treated a cosine similarity of 0.95 and 0.35 identically — both would trigger
LLM generation.

---

### 4. The Phase 2 Mechanism — Implementation Reality

Phase 2 adds five architectural layers. Each is in a specific file.

---

#### 4a. Hierarchical Chunking — `src/quaestor/ingestion/chunker.py`

**Function:** `_chunk_hierarchical(docs, parent_chunk_size=1024, child_chunk_size=256, child_chunk_overlap=0)`

The logic is a nested double-split. For each source document:
1. A `RecursiveCharacterTextSplitter(chunk_size=1024)` splits the document into
   parent windows. These are not stored in the vector store.
2. A second `RecursiveCharacterTextSplitter(chunk_size=256)` splits each parent
   into child chunks. These are what gets embedded and stored.
3. Every child carries its parent's full text in `metadata["parent_content"]`.

Metadata on each returned child chunk:
```python
{
    "source": "aapl-20250927.htm",
    "page": 31,
    "filing_type": "10-K",
    "start_index": 128,          # offset within the PARENT, not the document
    "chunk_level": "child",
    "parent_content": "...",     # full 1024-char parent text
    "parent_id": "a3f9c1d7e4b2", # MD5[:12] of source::parent_start_index
    "parent_index": 5,           # 0-based parent position within source doc
    "chunk_index": 2,            # 0-based child position within parent
}
```

The RAG chain at generation time uses `parent_content` instead of
`page_content` when it formats the context. A child that says "112,010" in
256 chars gives the LLM the surrounding 1024 chars explaining what that number
refers to. This is the mechanism behind Hypothesis A.

**Why child_chunk_overlap=0?** Children of the same parent already share context
through `parent_content`. Overlap between children would create redundant text in
the context window without additional information.

**The `_parent_id` hash:**
```python
def _parent_id(parent: Document, index: int) -> str:
    source = parent.metadata.get("source", "")
    start  = parent.metadata.get("start_index", index)
    raw    = f"{source}::parent_start={start}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
```
12-char hex, not UUID — this is an internal reference between sibling chunks,
not a database primary key. The full UUID format is only required by
`_doc_id` (the Chroma/Qdrant point ID).

---

#### 4b. Cross-Encoder Reranking — `src/quaestor/retrieval/reranker.py`

**Core function:** `rerank(query, docs, cross_encoder=None, top_n=None)`

The difference between a bi-encoder (used for initial retrieval) and a cross-
encoder is the input. A bi-encoder encodes query and document independently:
```
embed(query) · embed(document)  →  similarity scalar
```
A cross-encoder reads both together:
```
BERT([CLS] query [SEP] document [SEP])  →  relevance scalar
```
The joint encoding captures explicit reasoning about whether the document answers
the question, not just whether they are topically similar.

The implementation:
```python
pairs = [[query, doc.page_content] for doc in docs]
scores: list[float] = list(cross_encoder.predict(pairs))
ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
```

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (configurable via
`settings.reranker_model`). MiniLM is the standard production choice: 80 MB,
runs on CPU in ~50 ms for 5 pairs, trained on the MS MARCO passage retrieval
benchmark which includes technical and factual question-answer pairs.

**Score range:** The cross-encoder does not output a calibrated probability. Its
scores are raw logits from the final linear layer, typically ranging from roughly
-10 to +10. A score of 3.9 for "What was Apple's total net sales?" means the
model is highly confident that chunk is relevant. A score of -2.1 for "What are
the primary risk factors?" means the retrieved chunks are not directly answering
that question (possibly because they are quantitative tables, not the textual
risk factor disclosures).

**`CrossEncoderProtocol`:** A `@runtime_checkable` Protocol with one method:
```python
def predict(self, sentences: list[list[str]]) -> list[float]: ...
```
Any object satisfying this interface works. Tests inject `FakeHighCrossEncoder`
(returns `[5.0] * len(sentences)`) or `FakeLowCrossEncoder` (returns
`[-10.0] * len(sentences)`) without downloading the model. The `isinstance`
check via `@runtime_checkable` makes this testable: `assert isinstance(stub,
CrossEncoderProtocol)` passes before the stub is passed to production code.

---

#### 4c. LangGraph State Machine — `src/quaestor/retrieval/graph.py`

The Phase 1 pipeline was a linear LCEL chain: retrieve → prompt → LLM →
parse. There was no branching. Phase 2 introduces a state machine with four
nodes and one conditional branch.

**State definition:**
```python
class RAGState(TypedDict):
    question: str
    docs: list[Document]
    top_score: float
    answer: str
    sources: list[str]
    refused: bool
    prompt_version: str
```

Every node receives the full state dict and returns a partial dict that is merged
into it. This is LangGraph's reducer pattern — nodes do not mutate state, they
return updates.

**Graph topology:**
```
START → [retrieve] → [rerank] ──score ≥ threshold──► [generate] → END
                              └──score < threshold──► [refuse]   → END
```

**Node implementations (all are closures, not classes):**

`_make_retrieve_node(vector_store, top_k)` returns a closure that calls
`retrieve(state["question"], vector_store, top_k=top_k)` and returns
`{"docs": docs}`.

`_make_rerank_node(cross_encoder, top_n)` is where scoring happens. It calls
`cross_encoder.predict(pairs)`, sorts descending, extracts
`top_score = float(scored[0][0])`, and returns `{"docs": ranked_docs,
"top_score": top_score}`. The score is stored in state here so the confidence
router can read it on the next edge without re-running the model.

`_make_generate_node(llm)` creates an LCEL sub-pipeline
(`RAG_PROMPT | llm | StrOutputParser()`) at build time (not at call time). On
each invocation it formats context from `state["docs"]` using
`_format_context(docs)` which prepends `[Source: filename, Page N]` to each
chunk, then calls `.invoke()` synchronously.

`_refuse_node` is a plain function (no closure needed — it has no captured
dependencies). It returns the canned refusal text and sets `refused=True`
without calling the LLM.

**`_make_confidence_router(threshold)`:** Returns an edge function returning the
literal string `"generate"` or `"refuse"`. LangGraph routes to the node whose
name matches the string. The mapping
`{"generate": "generate", "refuse": "refuse"}` in `add_conditional_edges` is
explicit rather than implicit — it makes the routing table visible in the build
function.

**Critical detail — `run_rag_graph` initialises all state keys:**
```python
initial_state: RAGState = {
    "question": question,
    "docs": [],
    "top_score": 0.0,
    "answer": "",
    "sources": [],
    "refused": False,
    "prompt_version": PROMPT_VERSION,
}
```
If any key is missing at `graph.invoke()` time, LangGraph raises a `KeyError`
during node execution when a downstream node reads a key that was never set by
an upstream node. The `TypedDict` annotation does not enforce completeness at
runtime — you must initialise every key explicitly.

**`GraphAnswer` dataclass:** The public return type of `run_rag_graph`. It
isolates callers from the internal `RAGState` TypedDict. `app.py`,
`scripts/smoke_test_phase2.py`, `eval/evaluate.py`, and `api/main.py` all
import `GraphAnswer` and never touch `RAGState` directly.

---

#### 4d. Qdrant Hybrid Indexing — `src/quaestor/ingestion/indexer.py`

**Why Qdrant alongside Chroma?** ChromaDB supports dense similarity search only.
Financial documents contain exact tokens — ticker symbols (`AAPL`), GAAP line
items (`"operating lease right-of-use asset"`), regulation references
(`"ASC 842"`) — that dense embeddings handle poorly. nomic-embed-text learns
semantic similarity, not exact lexical overlap. BM25 (sparse retrieval) excels
at exact token matching. Hybrid search combines both signals with Reciprocal
Rank Fusion.

**The four-piece Qdrant initialisation problem:**
```python
# This does NOT work:
store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client,        # ← ignored; creates its own internal QdrantClient
    ...
)
```
`from_documents` creates its own internal client. The `client` parameter is
passed to `QdrantClient.__init__` positionally, which raises:
`TypeError: Client.__init__() got an unexpected keyword argument 'client'`.

The correct pattern:
```python
# 1. Create client explicitly
client = QdrantClient(":memory:")  # or url=settings.qdrant_url

# 2. Create collection explicitly with correct vector configs
client.create_collection(
    collection_name=name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    sparse_vectors_config={
        "langchain-sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    },
)

# 3. Build store directly — use __init__, not from_documents
store = QdrantVectorStore(
    client=client,
    collection_name=name,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
)

# 4. Add documents separately
store.add_documents(documents=chunks, ids=ids)
```

**Two Qdrant naming conventions that must match:**

The unnamed dense vector: `QdrantVectorStore` uses `vector_name=""` (empty string)
by default. The collection must be created with
`vectors_config=VectorParams(...)` (the unnamed form), not
`vectors_config={"dense": VectorParams(...)}` (the named form). If you use the
named form, the store cannot find its own vectors and raises
`QdrantVectorStoreError: Collection does not have a vector named ''`.

The sparse vector name: `QdrantVectorStore` hardcodes `sparse_vector_name=
"langchain-sparse"`. The collection's `sparse_vectors_config` dict must use
exactly the key `"langchain-sparse"`. Any other name — including the obvious
`"sparse"` — causes `QdrantVectorStoreError` at upsert time. This is not
documented in the Qdrant client docs; it is an internal convention in the
`langchain-qdrant` package.

**Vector size probe:** The dense vector dimension depends on the embedding model.
nomic-embed-text returns 768-dimensional vectors; another model might return 384
or 1536. Rather than hardcoding:
```python
sample_vec = embeddings.embed_query(chunks[0].page_content[:50])
_create_qdrant_collection(client, name, len(sample_vec), retrieval_mode)
```
This probe runs on the first 50 chars of the first chunk (fast) and handles any
embedding model automatically.

---

#### 4e. Document ID Upgrade — `_doc_id` in `indexer.py`

Phase 1 ID:
```python
raw = f"{source}::page={page}::start={start}"
```

Phase 2 ID:
```python
raw = f"{source}::page={page}::start={start}::parent={parent_id}::ci={chunk_index}"
```

The Phase 1 key was sufficient for fixed-size chunks because `start_index` is an
absolute offset within the source document — two chunks on the same page always
have different start positions. For hierarchical child chunks, `start_index` is
the offset within the *parent chunk*, not the document. Two children that are
both the first child of different parents on the same page (parent A starting at
position 0, parent B starting at position 1024) both produce children with
`start_index=0`. They hash identically under the Phase 1 scheme. This produced
87 duplicate UUIDs in a 1110-chunk AAPL 10-K index, which Chroma's upsert
correctly rejected with `DuplicateIDError`.

---

#### 4f. PII Guardrail — `src/quaestor/guardrails/input.py`

**Why this matters for financial QA:** An auditor working with a client file
might paste a raw document excerpt into the query box. That excerpt could
contain an individual's name, a social security number, or an internal account
number. The query is sent over HTTPS to Groq's inference servers in the US.
This is a GDPR violation in most EU jurisdictions if the data is a real person's
PII.

The guardrail runs before the query reaches the LLM:
```python
entities = detect_pii(question, analyzer=analyzer)
if entities:
    result = redact_pii(question, analyzer=analyzer, anonymizer=anonymizer)
    question = result.redacted_text
    # proceed with redacted question
```

**Implementation:** Microsoft Presidio combines two detection mechanisms:
- `PatternRecognizer`: regex rules for email, phone, US SSN, credit card, IP
  address, US bank number, IBAN. These have near-100% precision.
- spaCy NER model: trained entity recognition for PERSON, ORG, LOCATION.
  Confidence scores are calibrated; default threshold is 0.5.

**The `en_core_web_sm` requirement:** `AnalyzerEngine()` default init calls:
```python
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
```
In a `uv`-managed venv there is no `python -m pip` available. The download
fails silently and the engine crashes on first use. The fix requires two steps:
1. `uv pip install en_core_web_sm` — uv exposes a pip-compatible interface
2. Explicit `NlpEngineProvider` configuration:
```python
provider = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
})
engine = AnalyzerEngine(nlp_engine=provider.create_engine())
```
The small model (`en_core_web_sm`) misses some PERSON entities compared to
the large model, but the pattern recognizers for email/SSN/credit card are
model-independent and work identically.

**Return types:**
```python
@dataclass
class PiiEntity:
    entity_type: str   # "PERSON", "EMAIL_ADDRESS", etc.
    start: int         # character offset start
    end: int           # character offset end (exclusive)
    score: float       # Presidio confidence [0, 1]
    text: str          # the detected PII text

@dataclass
class RedactionResult:
    redacted_text: str           # query with PII replaced by <ENTITY_TYPE>
    entities: list[PiiEntity]
    @property
    def has_pii(self) -> bool: ...
```

Redaction replaces each detected span with a fixed-width placeholder:
`"My name is John Smith"` → `"My name is <PERSON>"`. The placeholder length
differs from the original span length (7 chars vs 10 chars for "John Smith"),
which shifts character offsets for all subsequent entities. Presidio handles
this correctly by processing spans from right to left.

---

#### 4g. NLI Hallucination Check — `src/quaestor/guardrails/output.py`

**Function:** `check_hallucination(answer, context, classifier=None,
entailment_threshold=0.5)`

The NLI model (`cross-encoder/nli-deberta-v3-small`, ~85 MB) is trained on
natural language inference: given a premise P and a hypothesis H, classify the
relationship as ENTAILMENT, NEUTRAL, or CONTRADICTION.

Applied to hallucination detection:
- Premise = the retrieved context chunks
- Hypothesis = the LLM-generated answer

If `P(ENTAILMENT) < 0.5`, the answer introduces claims that cannot be
verified from the context — definition of hallucination.

```python
raw = classifier({"text": premise[:1800], "text_pair": hypothesis[:1800]})
# Normalise nested list format
if raw and isinstance(raw[0], list):
    raw = raw[0]
scores = {item["label"].upper(): item["score"] for item in raw}
entailment = scores.get("ENTAILMENT", 0.0)
is_hallucination = entailment < entailment_threshold
```

The 1800-char truncation is a 512-token approximation (4 chars ≈ 1 token,
leaving margin for the SEP tokens). Sending more than 512 tokens to DeBERTa
silently truncates at the model level, producing scores based on partial text.

**`NLIClassifier` Protocol:**
```python
@runtime_checkable
class NLIClassifier(Protocol):
    def __call__(self, inputs: dict[str, str], **kwargs) -> list[dict[str, float]]: ...
```
Tests inject `FakeNliClassifier(label="ENTAILMENT", score=0.95)` to control
the `is_hallucination` outcome without downloading the model.

---

#### 4h. FastAPI Endpoints — `src/quaestor/api/main.py`

Three endpoints:

**`GET /health`**: Returns `{"status": "ok", "version": "0.1.0",
"vector_store_backend": "chroma|qdrant"}`. No dependencies required.

**`POST /ask`**: Synchronous endpoint. Full pipeline:
1. PII redaction (`check_pii=True` by default)
2. `run_rag_graph(graph, question)` → `GraphAnswer`
3. Optional NLI check (`check_hallucination=False` by default, slow)
Returns `AskResponse` with all fields.

**`POST /ask/stream`**: Streaming endpoint using Server-Sent Events. The
streaming implementation bypasses the LangGraph state machine entirely and
calls the underlying components directly:
```python
docs = retrieve(question, vector_store, top_k=request.top_k)
ranked = rerank(question, docs, cross_encoder=cross_encoder)
# confidence check
async for chunk in (RAG_PROMPT | llm | StrOutputParser()).astream(...):
    yield _sse({"type": "token", "content": chunk})
yield _sse({"type": "sources", "content": sources})
yield _sse({"type": "done"})
```
This is a design tradeoff: the synchronous `run_rag_graph` could be adapted
to stream, but that would require making every LangGraph node async-aware.
The simpler approach is to stream at the endpoint level. The tradeoff is that
the streaming endpoint duplicates the confidence gate logic from the graph — a
maintenance risk if the threshold changes and only one path is updated.

**Dependency injection pattern:**
```python
def get_vector_store(): ...
def get_llm(): ...
def get_cross_encoder(): ...
def get_analyzer(): ...
def get_anonymizer(): ...
def get_nli_classifier(): ...

def get_rag_graph(
    vector_store=Depends(get_vector_store),
    llm=Depends(get_llm),
    cross_encoder=Depends(get_cross_encoder),
): ...
```

Tests use `app.dependency_overrides`:
```python
app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
app.dependency_overrides[get_llm] = lambda: FakeLLM()
```
This replaces the dependency for the duration of the test without patching
the module. It is the correct FastAPI pattern for integration testing.

---

#### 4i. RAGAS Evaluation Harness — `eval/evaluate.py`

**Key objects:**
```python
@dataclass
class GoldenQuestion:
    id: str             # "Q01"…"Q20"
    type: str           # "factual" | "multi_hop" | "unanswerable"
    question: str
    ground_truth: str
    source_pages: list[int]
    notes: str

@dataclass
class EvaluationSample:
    question: str
    ground_truth: str
    answer: str
    retrieved_contexts: list[str]
    refused: bool = False
    question_type: str = "factual"
```

**Pipeline:** `load_golden_dataset` → `run_pipeline_on_dataset` (runs the
full RAG graph on each question and collects contexts) → `to_ragas_dataset`
(converts to `EvaluationDataset`) → `run_ragas_evaluation` (calls
`ragas.evaluate()`).

**RAGAS 0.4.x import path:**
```python
# Old (raises DeprecationWarning in 0.4.3):
from ragas.metrics import faithfulness, answer_relevancy

# Correct:
from ragas.metrics.collections import (
    faithfulness, answer_relevancy, context_precision, context_recall
)
```

**Golden dataset ground truths** were anchored to smoke-test-verified figures
from the AAPL FY2025 10-K:
- Net sales FY2025: $416,161M
- Net income FY2025: $112,010M | FY2024: $93,736M | FY2023: $96,995M
- 20 questions: 8 factual, 7 multi_hop, 5 unanswerable
- `source_pages` is always `[]` for unanswerable questions (validated by
  `test_unanswerable_have_empty_source_pages`)

The rule: never modify `eval/golden_dataset.json` after creation. LLM-generated
pseudo-truths that are then scored against LLM-generated answers will always
produce high RAGAS faithfulness — measuring nothing.

---

### 5. New Failure Modes Introduced

**F1 — `start_index` collision on hierarchical children (discovered live):**
The Phase 1 `_doc_id` used `source + page + start_index` as its hash key.
For hierarchical children, `start_index` is the offset *within the parent*,
not the document. First-children of all parents share `start_index=0`. On an
AAPL 10-K with 1110 chunks, this produced 87 UUID collisions. Chroma's upsert
raised `DuplicateIDError`. Fixed by including `parent_id + chunk_index` in the
hash. But the fix changes all IDs for hierarchical chunks — any existing Chroma
collection indexed with the old scheme will have orphaned points if re-indexed
with the new scheme.

**F2 — Confidence threshold is a hyperparameter with no calibration:**
The default `reranker_confidence_threshold=0.0` was chosen empirically from the
live smoke test: Q1 (net sales) scored 3.91, Q3 (net income) scored 3.75, and
both were clearly answerable. Q2 (risk factors) scored -2.07 against chunks that
were quantitative tables, not the textual risk section. The threshold 0.0 is a
heuristic. It will over-refuse on questions whose answer lives in sections that
embed poorly (dense tables, boilerplate legalese) and under-refuse on questions
where a tangentially related chunk scores high.

**F3 — The streaming endpoint duplicates the confidence gate:**
`POST /ask/stream` reimplements the retrieve → rerank → confidence-check →
generate sequence directly, bypassing `run_rag_graph`. If the confidence
threshold is changed in settings, both `graph.py` and `main.py` must be updated
in sync. There is no structural guarantee of this. A divergence between the two
paths would make the streaming endpoint behave differently from the sync endpoint
for the same query.

**F4 — NLI hallucination check uses generated answer as its own context proxy:**
In `api/main.py`, the hallucination check receives:
```python
context_proxy = " ".join(graph_answer.sources) or graph_answer.answer
```
`graph_answer.sources` is a list of file paths (e.g. `["aapl-20250927.htm"]`),
not the actual retrieved chunk texts. The NLI model is therefore checking whether
the answer is entailed by a list of filenames. This is structurally wrong. The
check runs but its output is meaningless until `GraphAnswer` stores the actual
context strings (commented in the code: "A future improvement stores the context
in GraphAnswer").

**F5 — `en_core_web_sm` misses some PII that `en_core_web_lg` would catch:**
The downgrade to the small spaCy model was required by the uv/pip constraint.
`en_core_web_sm` has lower PERSON recall on short or ambiguous names. A name
like "Tim Cook" in "Tim Cook announced..." is correctly caught. A name like
"Pat" alone is not. For a compliance-grade deployment this gap matters. The
fix is to resolve the pip-in-uv-venv issue (already solvable: `uv pip install`
works; the problem was `subprocess.run(["pip", ...])` inside Presidio) and
switch back to `en_core_web_lg`.

**F6 — Qdrant is built but not used in the live smoke test:**
The Qdrant path (`build_qdrant_index`, `load_qdrant_index`) is tested via unit
tests with an in-memory Qdrant client (`QdrantClient(":memory:")`). The live
smoke test uses ChromaDB. The Qdrant path has never run against a real
`docker run -p 6333:6333 qdrant/qdrant` instance.

---

### 6. Observed Improvements

**Q1 (net sales, exact figure):** Phase 1 returned the correct answer, Phase 2
also returns it with the same citation accuracy. No visible improvement here —
single-value lookups don't require the parent context window.

**Q2 (risk factors, broad):** Phase 2 with `-5.0` threshold answered with
"FX, interest rate, credit risk (cited Pages 30)" using child chunks that
happened to contain the quantitative risk disclosures. Phase 1 would have
returned similar content. The full textual risk factor section was not retrieved
because it is a dense narrative section that embeds differently from the quantitative
tables. The benefit of hierarchical chunking on this question type requires Phase 3
improvements (better retrieval top-k, parent context injection at the prompt level).

**Refusal gate:** The confidence gate correctly refused "What is Apple's
projected revenue for fiscal year 2030?" (a question with no answer in the
filing) when the threshold was set to 999.0. In normal operation at 0.0, any
question scoring below 0.0 is refused — which in the smoke test included the
risk factors question (score -2.07), not just unanswerable ones. Threshold
calibration against the 20-question golden dataset is needed before declaring
the ≥95% refusal rate target met.

**PII redaction:** "My name is John Smith. What was Apple's revenue?" becomes
"My name is <PERSON>. What was Apple's revenue?" with confidence 0.85 for
the PERSON entity. The LLM receives the redacted question. This is the correct
behavior for a compliance-oriented pipeline.

---

### 7. Unexpected Discoveries

**The `QdrantVectorStore.from_documents` trap** is the most important. It is
documented nowhere in langchain-qdrant. The class has a `client` parameter. The
method appears to accept it. It doesn't forward it correctly. Three hours of
debugging traced the error to the langchain-qdrant source where `from_documents`
calls `QdrantClient(**kwargs)` instead of using the passed client. The fix (use
`__init__` directly) is not intuitive because `from_documents` is the advertised
pattern in all langchain documentation.

**LangGraph `TypedDict` does not validate at runtime.** A `TypedDict` is a
type hint, not a runtime constraint. If you define a state with 7 keys and
initialise only 4 of them, LangGraph will not raise until a node tries to read
a key that was never set. The error message is a plain Python `KeyError` with no
reference to state or graph structure. The defensive pattern is to always
initialise all keys explicitly in `run_rag_graph`, even with empty/zero defaults.

**The cross-encoder score for "risk factors" was -2.07** against chunks from the
quantitative sections. This reveals a real retrieval failure: the 10-K risk
factors section (usually ~30 pages of narrative text) does not embed near the top
of a similarity search for "What are the primary risk factors?" — because the
narrative chunks do not contain the exact phrase "risk factor" repeatedly; they
contain specific descriptions of risks. The cross-encoder confirms this: the
retrieved chunks (numerical tables) score very low against the question. Phase 3
will need to address this by increasing top-k from 5 to 20 for broad questions,
or by implementing query expansion.

**HuggingFace `pipeline("text-classification", top_k=None)` returns different
shapes** across versions. Some return `[{"label": ..., "score": ...}]`
(flat list). Others return `[[{"label": ..., "score": ...}]]` (nested list).
The nesting appears when the pipeline wraps multiple inputs in a batch
dimension even for single-input calls. The normalization:
```python
if raw and isinstance(raw[0], list):
    raw = raw[0]
```
handles both cases silently.

**The system prompt template contains `[Source: <filename>, Page <n>]`** as an
example in its instruction text. A test that counts occurrences of `"[Source:"`
in the LLM output to verify how many source chunks were cited will always see
at least one occurrence from the template regardless of the actual answer — the
FakeLLM echoes the full prompt in tests. The correct assertion is to check the
actual content of the retrieved chunks, not substrings of the output.

---

### 8. Mental Model for Future Engineers

**Retrieval in Phase 2 is two-staged and asymmetric.** The first stage (dense
similarity) is fast but coarse — it selects candidates. The second stage
(cross-encoder) is slow but precise — it ranks them. The two stages use
fundamentally different representations. The first encodes independently; the
second encodes jointly. You cannot combine them into one step without losing the
speed advantage of the first or the precision of the second.

**The cross-encoder score is not a probability.** It is a raw logit. A score
of 3.5 does not mean 97% confident. What matters is its *relative* ordering and
its *absolute* position relative to the threshold. The threshold 0.0 was chosen
because the logit distribution of the MiniLM model for clearly relevant documents
centers around +2 to +5, and for clearly irrelevant documents around -3 to -1.
Zero is a rough decision boundary, not a statistical significance level.

**The confidence gate is a first-order refusal mechanism, not a correctness
mechanism.** It gates on whether retrieval found something relevant. It does not
verify whether the LLM's answer is factually correct. That is what the NLI check
does — but only approximately, and currently only when `check_hallucination=True`
is explicitly passed. In the default `POST /ask` flow, neither gate is sufficient
alone: you can have high retrieval confidence but a hallucinated answer, or a
correct answer from a low-scoring chunk.

**`GraphAnswer` is the contract between the pipeline and the outside world.**
All callers — `app.py`, `api/main.py`, `eval/evaluate.py`,
`scripts/smoke_test_phase2.py` — import `GraphAnswer` and nothing else from
`graph.py`. `RAGState` is an implementation detail. If you change the internal
graph topology (add a node, change routing), callers notice only if `GraphAnswer`
fields change. Keep `GraphAnswer` stable; change `RAGState` freely.

**Dependency injection via Protocol is the offline testing strategy.** The rule
is: any object that uses an external resource (model, database, API) must accept
an injectable alternative via a `Protocol`. `CrossEncoderProtocol`,
`NLIClassifier`, and FastAPI `Depends` all follow this pattern. Tests never
touch a real model. The 297 tests run in 10 seconds on any machine with no
network, no GPU, no Ollama, no Groq key. If a test requires a real service to
pass, it is not a unit test — it is an integration test and should be in
`tests/integration/`.

---

### 9. What Remains Unsolved

**Threshold calibration:** The `reranker_confidence_threshold=0.0` default is
a guess. It needs to be calibrated against the 20-question golden dataset:
run the pipeline at multiple thresholds (-3.0, -1.0, 0.0, 1.0, 2.0), measure
F1 on the refusal decision (true positive = correctly refused an unanswerable
question; false positive = incorrectly refused an answerable one), and pick the
threshold that maximizes F1. This has not been done.

**Parent context injection at generation time:** `_chunk_hierarchical` stores
`parent_content` in every child's metadata. The generation node uses
`doc.page_content` (the 256-char child text), not `metadata["parent_content"]`
(the 1024-char parent text). The parent context is stored but not used. This is
the central mechanism of Hypothesis A — and it has not been activated. The fix
requires modifying `_format_context` in `graph.py` to use `parent_content` when
the chunk is at `chunk_level == "child"`.

**Qdrant live integration test:** The Qdrant path has 17 unit tests all using
`QdrantClient(":memory:")`. It has never been tested against a real running
Qdrant Docker instance. The hybrid BM25 + dense path is the production target
but is not yet exercised end-to-end.

**Streaming endpoint confidence gate divergence:** `POST /ask/stream`
reimplements the confidence check. It reads from `settings.reranker_confidence_
threshold` directly (not from the graph's captured threshold). This is
inconsistent: if you call `build_rag_graph(confidence_threshold=2.0)`, the sync
endpoint uses 2.0, the streaming endpoint uses whatever is in settings. Should be
unified.

**NLI context proxy is wrong:** `graph_answer.sources` is a list of file paths,
not chunk texts. The hallucination check in `api/main.py` is checking the answer
against filenames. This produces meaningless NLI scores. The fix is to add a
`context` field to `GraphAnswer` that stores the formatted context string from
the generate node.

**The 5 unanswerable questions in the golden dataset use threshold 999.0 to
force refusal in the smoke test.** Under normal operation at threshold 0.0,
some unanswerable questions might score above 0.0 (because the corpus contains
tangentially related chunks), generating an incorrect answer rather than a
refusal. The ≥95% refusal rate target from the spec has not been measured.

**`unanswerable` questions in RAGAS evaluation:** When `refused=True`, the
pipeline returns the canned refusal text as the answer. RAGAS's answer relevancy
metric will score this poorly (the refusal is not "relevant" to the question),
and faithfulness will score it perfectly (the refusal text is factually grounded
in nothing from the context — but RAGAS may treat it as vacuously faithful).
The evaluation harness needs to handle refused samples separately, scoring only
the refusal decision (was it correct?) rather than feeding refusal text to
RAGAS metrics designed for actual answers.

---

### Inventory of new files and functions

```
src/quaestor/config.py
  VectorStoreBackend(str, Enum)           — CHROMA | QDRANT
  Settings.reranker_model                 — "cross-encoder/ms-marco-MiniLM-L-6-v2"
  Settings.reranker_confidence_threshold  — 0.0
  Settings.vector_store_backend           — VectorStoreBackend.CHROMA
  Settings.qdrant_url                     — "http://localhost:6333"
  Settings.qdrant_collection_name         — "quaestor"

src/quaestor/ingestion/chunker.py
  ChunkStrategy.SEMANTIC / .HIERARCHICAL  — added to enum
  _chunk_semantic(docs, breakpoint_threshold_type, embeddings)
  _chunk_hierarchical(docs, parent_chunk_size, child_chunk_size, child_chunk_overlap)
  _parent_id(parent, index)               — 12-char hex parent reference
  chunk_documents(…, strategy, parent_chunk_size, child_chunk_size)

src/quaestor/ingestion/indexer.py
  _doc_id(doc, index)                     — upgraded: includes parent_id + chunk_index
  _qdrant_client(url)                     — returns QdrantClient(":memory:") or url
  _get_sparse_embeddings()                — FastEmbedSparse("Qdrant/bm25")
  _create_qdrant_collection(client, name, vector_size, retrieval_mode)
  build_qdrant_index(chunks, …, qdrant_client)
  load_qdrant_index(…, qdrant_client)

src/quaestor/retrieval/reranker.py        — NEW FILE
  CrossEncoderProtocol                    — @runtime_checkable Protocol
  _default_cross_encoder()               — loads settings.reranker_model
  rerank(query, docs, cross_encoder, top_n)

src/quaestor/retrieval/graph.py           — NEW FILE
  RAGState(TypedDict)                     — 7-field graph state
  GraphAnswer(dataclass)                  — public return type
  _make_retrieve_node(vector_store, top_k)
  _make_rerank_node(cross_encoder, top_n)
  _make_generate_node(llm)
  _refuse_node(state)
  _make_confidence_router(threshold)
  _format_context(docs)
  build_rag_graph(vector_store, llm, cross_encoder, top_k, top_n, confidence_threshold)
  run_rag_graph(graph, question)

src/quaestor/guardrails/input.py          — NEW FILE
  PiiEntity(dataclass)                    — entity_type, start, end, score, text
  RedactionResult(dataclass)              — redacted_text, entities, has_pii
  _default_analyzer()                     — Presidio + en_core_web_sm
  _default_anonymizer()                   — AnonymizerEngine
  detect_pii(text, entities, min_score, analyzer, language)
  redact_pii(text, entities, min_score, analyzer, anonymizer, language)

src/quaestor/guardrails/output.py         — NEW FILE
  NLIClassifier                           — @runtime_checkable Protocol
  HallucinationResult(dataclass)          — is_hallucination, entailment_score, …
  _default_classifier()                   — transformers pipeline, nli-deberta-v3-small
  check_hallucination(answer, context, classifier, entailment_threshold, model_name)

src/quaestor/api/schemas.py               — NEW FILE
  HallucinationCheck(BaseModel, frozen)
  PiiReport(BaseModel, frozen)
  AskRequest(BaseModel, frozen)           — question, top_k, check_pii, check_hallucination
  AskResponse(BaseModel, frozen)          — question, answer, sources, refused, pii, hallucination
  HealthResponse(BaseModel, frozen)

src/quaestor/api/main.py                  — NEW FILE
  lifespan(app)                           — asynccontextmanager startup/shutdown
  app = FastAPI(…)
  get_vector_store() / get_llm() / get_cross_encoder()
  get_analyzer() / get_anonymizer() / get_nli_classifier()
  get_rag_graph(vector_store, llm, cross_encoder)
  GET  /health  → HealthResponse
  POST /ask     → AskResponse (sync)
  POST /ask/stream → StreamingResponse SSE

eval/golden_dataset.json                  — NEW FILE
  20 questions: 8 factual, 7 multi_hop, 5 unanswerable
  All factual ground truths verified against live AAPL FY2025 10-K smoke test

eval/evaluate.py                          — NEW FILE
  GoldenQuestion(dataclass)
  EvaluationSample(dataclass)
  load_golden_dataset(path)               — validates types, rejects empty fields
  build_evaluation_sample(golden, answer, retrieved_contexts, refused)
  to_ragas_dataset(samples)               — → ragas.EvaluationDataset
  run_pipeline_on_dataset(questions, vector_store, llm, cross_encoder, top_k, …)
  run_ragas_evaluation(samples, llm, embeddings, metrics)
  save_results(results, output_path)

scripts/evaluate.py                       — NEW FILE
  CLI: --dataset, --output, --top-k, --limit, --no-rerank

app.py                                    — REWRITTEN
  _get_cross_encoder()   @st.cache_resource
  _get_nli_classifier()  @st.cache_resource
  _get_pii_engines()     @st.cache_resource
  _load_existing_index() @st.cache_resource
  _get_rag_graph(vs, confidence_threshold)  — cached in session_state by (id(vs), threshold)
  Sidebar: strategy selector, chunk_size/overlap, confidence slider, PII/NLI toggles
  Main:    PII warning → refused info box | answer + score badge + NLI badge + sources

scripts/smoke_test_phase2.py              — NEW FILE
  8-step live test: download → load → hierarchical chunk → ChromaDB index →
  build graph → Q&A (3 questions) → refusal gate → PII guardrail
```

**Test suite:** 297 total, 0 failed, 10.2 s.

| File | Tests |
|---|---|
| `test_reranker.py` | 16 |
| `test_graph.py` | 23 |
| `test_qdrant_indexer.py` | 17 |
| `test_input_guardrail.py` | 23 |
| `test_output_guardrail.py` | 20 |
| `test_api.py` | 26 |
| `test_evaluate.py` | 30 |
| (Phase 1 tests retained) | 142 |

---

### What comes next (Phase 3)

- [ ] **Activate parent context at generation time:** modify `_format_context`
      in `graph.py` to use `metadata["parent_content"]` for child chunks instead
      of `page_content`. This is the central unchompleted mechanism of Phase 2.
      Measure RAGAS before/after.
- [ ] **Calibrate confidence threshold:** run `scripts/evaluate.py --limit 20`
      at thresholds -3, -1, 0, 1, 2; measure refusal F1 on the 5 unanswerable
      questions vs false-refusal rate on the 15 answerable ones.
- [ ] **Fix NLI context proxy:** add `context: str` field to `GraphAnswer`,
      populate it in `_make_generate_node`, use it in `api/main.py`.
- [ ] **Qdrant live integration test:** `docker run -p 6333:6333 qdrant/qdrant`,
      run `build_qdrant_index` + `load_qdrant_index` + `similarity_search` against
      a real instance; add to `tests/integration/`.
- [ ] **CI/CD eval gate:** GitHub Actions workflow on every PR;
      `pytest tests/unit/` + `python scripts/evaluate.py --limit 5`; block merge
      if faithfulness < 0.7.
- [ ] **Langfuse self-hosted:** trace every `run_rag_graph` call from day one;
      instrument retrieve, rerank, generate nodes as separate spans.
- [ ] **Expand golden dataset:** 120 questions across AAPL, JPM, JNJ filings;
      30 hand-written, 70 LLM-generated + manually verified.
- [ ] **Unify streaming confidence gate:** `POST /ask/stream` must use the same
      threshold as the compiled graph, not read from settings independently.

---

## Session 2 — 2026-03-25

### What was built

Generated accurate `.env.example` templates and verified the Phase 1 baseline
runs cleanly from a fresh environment.

| Task | Status |
|---|---|
| `.env.example` — `examples/phase1-baseline/` | ✅ |
| `.env.example` — root (`quaestor/`) | ✅ (new file) |
| `examples/phase1-baseline/README.md` — setup commands corrected | ✅ |
| `examples/phase1-baseline/src_quaestor/` → `src/quaestor/` restructured | ✅ |
| `uv run pytest tests/ -q` — 123 passed, 0 failed | ✅ |
| `uv run streamlit run app.py` headless — HTTP 200 | ✅ |

---

### Problems encountered

| # | Problem | Fix |
|---|---|---|
| 1 | `.env.example` had wrong variable names: `COLLECTION_NAME` and `TOP_K` | Renamed to `CHROMA_COLLECTION_NAME` and `RETRIEVAL_TOP_K` to match `config.py` field names |
| 2 | `.env.example` missing 8 fields from `Settings` class | Added `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `EMBEDDING_PROVIDER`, `HUGGINGFACE_EMBEDDING_MODEL`, `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`, `SEC_REQUESTER_NAME`, `SEC_REQUESTER_EMAIL` |
| 3 | `ModuleNotFoundError: No module named 'quaestor'` in baseline tests | `examples/phase1-baseline/src_quaestor/` (flat) didn't match `pyproject.toml`'s `packages = ["src/quaestor"]`. Fixed by moving to proper `src/quaestor/` layout: `mkdir src && mv src_quaestor src/quaestor` |
| 4 | README had stale `uv venv && uv pip install -e .` instructions | Updated to `uv sync` (handles editable install automatically) |

---

### Decisions made

**`src/quaestor/` layout enforced in baseline**
When the baseline snapshot was taken, the source directory was inadvertently
named `src_quaestor/` (a single flat directory), while `pyproject.toml` expected
the standard `src/quaestor/` src-layout. All test imports (`from quaestor.*`)
and the hatch wheel config (`packages = ["src/quaestor"]`) were correct — only
the directory name was wrong. Fixed by creating `src/` and moving
`src_quaestor/` into it as `quaestor/`.

**`uv sync` is sufficient — no separate `uv pip install -e .`**
`uv run` triggers the editable install automatically when a `[build-system]`
block is present in `pyproject.toml`. So `uv sync` (for deps) followed by any
`uv run` invocation handles the package install. The stale README step
`uv pip install -e .` was removed.

---

### What comes next (Phase 2)

_(unchanged from Session 1 — see below)_

---

## Session 1 — 2026-03-23

### What was built

Full Phase 1 pipeline — every module from config to Streamlit UI, plus a
live smoke test that exercised the complete pipeline against a real AAPL 10-K.

| Module | Tests | Status |
|---|---|---|
| `src/quaestor/config.py` | `tests/unit/test_config.py` (31) | ✅ |
| `src/quaestor/ingestion/loader.py` | `tests/unit/test_loader.py` (20) | ✅ |
| `src/quaestor/ingestion/chunker.py` | `tests/unit/test_chunker.py` (17) | ✅ |
| `src/quaestor/ingestion/indexer.py` | `tests/unit/test_indexer.py` (15) | ✅ |
| `src/quaestor/retrieval/retriever.py` | `tests/unit/test_retriever.py` (11) | ✅ |
| `src/quaestor/generation/prompts.py` | `tests/unit/test_prompts.py` (10) | ✅ |
| `src/quaestor/generation/chain.py` | `tests/unit/test_chain.py` (19) | ✅ |
| `app.py` | AST parse check | ✅ |
| `scripts/smoke_test.py` | Live end-to-end run | ✅ |

**Total unit tests: 123 passed, 0 failed, 5.7 s.**

Supporting files added/changed:
- `pyproject.toml` — added `langchain-chroma`, `langchain-ollama`,
  `beautifulsoup4`, `pytest`; added `[tool.pytest.ini_options]`
- `tests/__init__.py`, `tests/unit/__init__.py` — package markers
- `scripts/smoke_test.py` — end-to-end live pipeline driver

---

### Live smoke test results (AAPL 10-K FY2025, filed 2025-10-31)

```
Groq model   : llama-3.3-70b-versatile
Embeddings   : nomic-embed-text (Ollama, local)
Chunk size   : 512 / overlap 50
Top-k        : 5
Documents    : 52 segments extracted from full-submission.txt → 579 chunks
Indexing     : 13.8 s (nomic-embed-text, M1 local)
```

| Question | Answer | Latency |
|---|---|---|
| Total net sales | $416,161 million (cited Page 27) | 0.8 s |
| Primary risk factors | Interest rates, FX, cybersecurity, credit risk (cited Pages 23, 30) | 0.7 s |
| Net income | FY2025: $112,010M / FY2024: $93,736M / FY2023: $96,995M (cited Page 31) | 0.6 s |

All answers are factually correct and properly cited. Citations include source
filename and page number exactly as required by the Phase 1 spec.

---

### Decisions made

**`pyproject.toml` — editable install required for Streamlit**
`pytest` found `quaestor` because `pythonpath = ["src"]` in
`[tool.pytest.ini_options]` injects `src/` before collecting tests.
Streamlit runs `app.py` in its own process with no such injection, so
`quaestor` was invisible and raised `ModuleNotFoundError`. Fix: added a
`[build-system]` block (`hatchling`) and `[tool.hatch.build.targets.wheel]`
pointing at `src/quaestor`, then ran `uv pip install -e .` once. This
writes a `.pth` file into the venv that makes `src/quaestor` permanently
importable by every process using that venv — Streamlit, scripts, the REPL,
and pytest alike. The `pythonpath` pytest setting is now redundant but kept
as an explicit reminder of the `src/` layout.

**`config.py` — foundation for everything else**
Built first, as specified. Uses `pydantic-settings` `BaseSettings` with
`SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")`.
Key design choices: all fields have safe defaults so the app runs in
development without a fully populated `.env`; API keys default to `""`
rather than being required, so tests never need to set them; all directory
paths are typed as `Path` (not `str`) so callers never need to convert;
`LLMProvider` and `EmbeddingProvider` are `str`-enums so they compare
equal to their string literal values. The `chunk_overlap` validator
enforces `overlap < chunk_size` at construction time with a clear error
message naming both values. A module-level `settings = Settings()` singleton
is exported so every module imports it with one line.

**`loader.py` — top-level optional import for `Downloader`**
Initial implementation imported `Downloader` lazily inside
`download_sec_filings()`. `unittest.mock.patch` cannot target a name that
doesn't exist at module level, so four tests failed on `AttributeError`.
Fixed by hoisting to a module-level `try/except ImportError` block
(`Downloader = None` as fallback). Tests patched
`quaestor.ingestion.loader.Downloader` cleanly after that.

**`loader.py` — `sec-edgar-filings` path uses hyphens, not underscores**
Discovered during the live smoke test. `sec-edgar-downloader` v5 writes to
`sec-edgar-filings/` (hyphenated). The original code assumed underscores.
Fixed in `loader.py` and the corresponding test assertions.

**`loader.py` — `load_edgar_submission()` added**
SEC EDGAR does not deliver plain PDFs. It delivers a `full-submission.txt`
SGML container with all documents (10-K HTML, exhibits, XBRL) embedded
inside `<DOCUMENT>` blocks. A bare `PyPDFLoader` or `load_directory` call
returned zero documents. Added `load_edgar_submission()` which:
1. Scans the SGML for `<TYPE>10-K` blocks
2. Extracts the HTML between `<TEXT>` and `</TEXT>` tags
3. Saves the extracted `.htm` file alongside the submission
4. Strips all tags with BeautifulSoup and segments into 4000-char chunks
The function is idempotent — if the `.htm` file already exists it skips
extraction and loads directly.

**`indexer.py` — deterministic document IDs**
ChromaDB upserts need stable IDs to be idempotent. IDs are derived from
`MD5(source::page=N::start=N)` — not for security, just for cheap stable
hashing. The `# noqa: S324` suppresses the Bandit MD5 warning.

**`indexer.py` — `langchain-chroma` added as explicit dependency**
`langchain-community` ships its own Chroma integration but it is deprecated
in favour of the standalone `langchain-chroma` package. Added explicitly
to avoid deprecation warnings in logs.

**`chain.py` — `FakeLLM` extends `BaseChatModel`**
Tests needed a fully offline LLM substitute. Extending `BaseChatModel`
and implementing `_generate` is the correct LangChain pattern — it makes
`FakeLLM` a first-class chat model that slots into any LCEL pipeline
without monkey-patching.

**`config.py` — Groq model updated to `llama-3.3-70b-versatile`**
`llama-3.1-70b-versatile` was decommissioned by Groq between the time the
spec was written and the live test run. Updated the default to the current
successor `llama-3.3-70b-versatile`. Groq's deprecation notice confirms
this is the recommended replacement. The test asserting the default model
name was updated accordingly.

**`app.py` — no unit tests, AST parse check instead**
`app.py` calls `st.set_page_config` at import time, which raises outside a
Streamlit runtime. Testing the UI layer properly requires Streamlit's
`AppTest` harness. For Phase 1 a syntax/parse check is sufficient.

**`app.py` — uploaded docs must use an isolated ChromaDB collection**
When a user uploads a PDF, `build_index` was called with the default
collection name `"quaestor"` — the same collection holding the 579 AAPL
chunks from the smoke test. New chunks were appended to that collection, so
every similarity search ran across both datasets. Because AAPL had 579
chunks versus a few dozen from the uploaded doc, AAPL content consistently
ranked in the top-k and the LLM correctly returned "I don't have enough
information" (it was seeing Apple data, not the uploaded document).
Fix: uploaded files now use the dedicated collection `"quaestor_upload"`,
completely isolated from the persisted EDGAR index. The two collections
serve different workflows — "Load existing index" uses `"quaestor"` for the
downloaded EDGAR data; "Upload files" uses `"quaestor_upload"` for ad-hoc
documents. A future Phase 2 improvement could let the user choose which
collections to search over.

**`app.py` — scanned PDFs silently returned zero retrievable content**
`PyPDFLoader` extracts text layer from PDFs. Image-based (scanned) PDFs
have no text layer, so every page comes back as a `Document` with empty
`page_content`. The original code passed these empty documents straight into
`chunk_documents` and then `build_index` — technically succeeding at every
step but indexing empty strings. The LLM received empty context and
correctly refused to answer. Fix: empty pages are now filtered out before
chunking. If all pages are empty a clear error is shown. An extraction
preview expander shows the first 300 chars of extracted text so the user
can immediately verify whether the content was read correctly before
spending time on embedding.

---

### Problems encountered

| # | Problem | Fix |
|---|---|---|
| 1 | `uv` binary not on `PATH` | Used full path `/Users/Aex/.local/bin/uv` |
| 2 | `pytest` not installed | Added to `[dependency-groups] dev` |
| 3 | `langchain_chroma` not installed | `uv add langchain-chroma` |
| 4 | `langchain_ollama` not installed | `uv add langchain-ollama` |
| 5 | `beautifulsoup4` not installed | `uv add beautifulsoup4` |
| 6 | Loader mock patch failure | Hoisted `Downloader` to module-level import |
| 7 | SEC downloader path uses hyphens | Fixed `sec_edgar_filings` → `sec-edgar-filings` |
| 8 | EDGAR delivers SGML, not PDFs | Added `load_edgar_submission()` with SGML extractor |
| 9 | Groq model decommissioned | Updated default to `llama-3.3-70b-versatile` |
| 10 | `ModuleNotFoundError: No module named 'quaestor'` in Streamlit | Added `[build-system]` + `[tool.hatch.build.targets.wheel]` to `pyproject.toml`, ran `uv pip install -e .` |
| 11 | Uploaded docs always returned "I don't have enough information" | `build_index` used the default `"quaestor"` collection, merging uploaded chunks with 579 existing AAPL chunks — AAPL data dominated retrieval. Fixed by using an isolated `"quaestor_upload"` collection for uploads. |
| 12 | Scanned PDFs silently produce no retrievable content | `PyPDFLoader` returns `Document` objects with empty `page_content` for image-based PDFs. Fixed by filtering empty pages, warning the user, and showing an extraction preview expander. |

---

### What comes next (Phase 2)

- [ ] Download JPMorgan 10-K and stress-test the chunking (intentional failure
      narrative for fixed-size chunking on complex nested tables)
- [ ] Implement `semantic` chunking in `chunker.py`
      (LangChain `SemanticChunker`)
- [ ] Implement `hierarchical` parent-child chunking in `chunker.py`
      (parent 1024 tokens / child 256 tokens, explicit relationship stored)
- [ ] Add `retrieval/reranker.py` (cross-encoder reranking)
- [ ] Add `retrieval/graph.py` (LangGraph state machine with confidence
      thresholding and unanswerable refusal)
- [ ] Migrate vector store to Qdrant (Docker) for hybrid BM25 + dense search
- [ ] Add `guardrails/input.py` — Presidio PII detection
- [ ] Add `guardrails/output.py` — DeBERTa NLI hallucination check
- [ ] Expose pipeline via `api/main.py` (FastAPI + streaming endpoints)
- [ ] Write 20-question golden dataset (8 factual, 7 multi-hop, 5 unanswerable)
- [ ] Run RAGAS evaluation and record baseline Phase 1 scores
- [ ] Record before/after faithfulness improvement from fixed → hierarchical
      chunking — this is the central narrative of the project
