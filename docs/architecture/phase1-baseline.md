# QUAESTOR
## Phase 1 — Technical Walkthrough

*A production-grade RAG system for financial document intelligence*
*March 2026*

---

## 1. The Problem

Financial analysts and auditors at firms like EY spend a disproportionate amount of time doing something mechanical: navigating dense documents to find specific numbers or disclosures. A single Apple 10-K filing is over 400 pages. A JPMorgan 10-K exceeds 300 pages with deeply nested financial tables.

The question "What was Apple's operating income in FY2025?" has a precise, citable answer buried somewhere in those pages — but finding it manually means scrolling, searching, cross-referencing footnotes, and second-guessing whether you found the right line item.

Quaestor is a Retrieval-Augmented Generation (RAG) system that answers these questions in natural language, cites the exact page and section, and refuses to guess when the answer is not in the documents.

### 1.1 What is a 10-K Filing?

A 10-K is an annual financial report that every publicly traded company in the United States is required to file with the SEC (Securities and Exchange Commission). It is the most comprehensive financial document a company produces.

A typical 10-K includes:

- Audited financial statements (income statement, balance sheet, cash flows)
- Management's Discussion and Analysis (MD&A) — narrative explanation of results
- Risk factors — a catalogue of everything that could go wrong
- Business description — what the company actually does and how it makes money
- Notes to financial statements — the fine print that accountants care about most

#### Example: Why JPMorgan?

JPMorgan's 10-K is intentionally chosen as a stress-test, but "stress-test" needs to be precise: it is not merely a long or complex document. It is specifically chosen because its internal structure produces three concrete, predictable failure modes when processed with fixed-size chunking.

> **Failure mode 1 — Table splits.** A single consolidated income statement in JPMorgan's filing spans approximately 650–800 tokens. A 512-token chunk boundary cuts through the middle of the table — typically at row 8 or 9 of 12. The retrieved chunk contains rows of numbers with no column headers. The LLM receives `| 47.2 | 39.8 | 12.1 |` and must guess whether those are billions or millions, which years they represent, and which line item they belong to. The 50-token overlap does not help: the last 50 tokens of the preceding chunk are from the table body rows 6–7, not from the header row that labels the columns — which appeared hundreds of tokens earlier.
>
> **Failure mode 2 — Cross-reference blindness.** JPMorgan uses phrases like "See Note 12 for further detail" and "as discussed in Item 7A" more than 40 times throughout the filing. Note 12 may be 80 pages later. Fixed-size chunking processes the document linearly: the chunk containing the reference and the chunk containing the referenced content are stored as independent vectors with no relationship. The retriever has no mechanism to follow the reference. A question whose answer is in Note 12 will only succeed if the retriever happens to surface Note 12 directly — which depends on semantic similarity, not on the document's own navigation structure.
>
> **Failure mode 3 — Numerical density.** A single page of JPMorgan's income statement contains 30–50 distinct monetary figures from different line items. A 512-token chunk from this section may contain values for net interest income, non-interest revenue, provision for credit losses, compensation expense, and net income simultaneously — all as bare numbers. When the query is "What was net income in FY2024?", the retriever finds the chunk (it is semantically relevant), but the LLM must distinguish the correct figure from 30 other figures in the same context. On a structurally clean document like Apple's 10-K, this ambiguity rarely occurs. On JPMorgan's, it is the norm.

> *Apple's 10-K is the **working** document: dense but well-structured enough that fixed-size chunking retrieves the right chunks most of the time. JPMorgan is the **failing** document. The experimental design requires both: Apple proves the pipeline is correct; JPMorgan exposes exactly where it breaks. This is not a bug in Phase 1 — it is the controlled 'before' that makes Phase 2's 'after' a meaningful claim.*

### 1.2 What is SEC EDGAR?

EDGAR (Electronic Data Gathering, Analysis, and Retrieval) is the SEC's public database of corporate filings. Every 10-K, 10-Q, 8-K, and proxy statement filed since 1996 is freely accessible at `edgar.sec.gov`. No API key required, no subscription — it is a public good.

#### The Unexpected Format Discovery

The critical discovery during Phase 1 development: SEC EDGAR does not deliver PDFs. It delivers SGML containers — a format called `full-submission.txt` that wraps all filing documents inside structured markup blocks.

> *When the loader was first tested with PyPDFLoader (LangChain's standard PDF reader), it returned zero documents. Investigation revealed that EDGAR's format looks like this:*

```
-----BEGIN PRIVACY-ENHANCED MESSAGE-----
<SUBMISSION>
  <DOCUMENT>
    <TYPE>10-K
    <TEXT>
      ...the actual 10-K HTML content...
    </TEXT>
  </DOCUMENT>
</SUBMISSION>
```

> *This required writing a custom `load_edgar_submission()` function that extracts only the `<TYPE>10-K` block, strips HTML tags with BeautifulSoup, and saves clean text. Counterintuitively, this turned out better than PDFs — HTML from EDGAR is clean structured prose with no layout artifacts.*

---

## 2. What is Retrieval-Augmented Generation?

A Large Language Model (LLM) like Llama 3.3 or GPT-4 is trained on a massive corpus of text and learns to generate coherent responses. But it has two fundamental limitations for financial document work:

1. **Knowledge cutoff**: The model was trained on data up to a certain date. Apple's FY2025 10-K was filed in October 2025. Any model with a 2024 cutoff has never seen it.
2. **Hallucination**: When the model does not know something, it sometimes generates plausible-sounding but incorrect information. In financial analysis this is catastrophic — a confident wrong number is worse than an admission of uncertainty.

RAG solves both problems by splitting the problem into two stages: retrieval and generation.

### 2.1 The RAG Pipeline in Plain English

- **Step 1 — Ingest**: Load the documents, split them into manageable chunks, convert each chunk into a numerical vector (an embedding), and store those vectors in a database.
- **Step 2 — Retrieve**: When a user asks a question, convert the question into the same kind of vector. Find the chunks whose vectors are most similar to the question vector.
- **Step 3 — Generate**: Give the LLM the question and the retrieved chunks as context. Instruct it to answer using only that context and to cite its sources.

#### The Key Analogy

> *RAG turns a closed-book exam into an open-book exam. The LLM no longer needs to memorize facts — it needs to read the relevant passage and extract the answer. Citation accuracy becomes testable and verifiable.*

### 2.2 What is an Embedding?

An embedding is a way of representing text as a list of numbers (a vector) such that texts with similar meaning end up with similar numbers. The vector lives in a high-dimensional space — nomic-embed-text produces 768-dimensional vectors.

> *For example: The sentence "Apple reported revenue of $416 billion" might become a vector like [0.23, -0.15, 0.89, ... 768 numbers total]. The sentence "Apple's total net sales were $416.2 billion" would produce a different but very similar vector — maybe [0.24, -0.14, 0.88, ...]. A completely unrelated sentence like "The weather is sunny today" would produce a vector pointing in a completely different direction.*

Similarity is measured using cosine similarity — the angle between two vectors. A value of 1.0 means identical meaning, near 0 means unrelated.

#### Concrete Example from Apple 10-K

> *During indexing of Apple's FY2025 10-K, 579 chunks were created. When a user asks "What was Apple's total revenue?", that question gets converted to a query vector. ChromaDB compares this vector against all 579 chunk vectors and finds that chunk #47 (which contains "Total net sales: $416,161 million") has a cosine similarity of 0.91 to the query — the highest score. That chunk gets retrieved and passed to the LLM.*

### 2.3 What is a Vector Store?

A vector store is a database optimized for one specific operation: given a query vector, find the *k* stored vectors that are most similar to it. This is called Approximate Nearest Neighbor (ANN) search.

A regular database like PostgreSQL could do this with a brute-force scan, but at the cost of reading every single row. Vector stores use specialized index structures (HNSW, IVF) that make this search fast even across millions of vectors.

#### Why ChromaDB for Phase 1?

> *In Phase 1, the vector store is ChromaDB — an open-source, in-process database that requires no server, no Docker, and no configuration. It stores vectors in a local `.chroma/` directory. This design decision was deliberate: the entire test suite (123 tests) runs in CI with no external services. For Phase 2, ChromaDB will be replaced with Qdrant, which adds hybrid search (dense + sparse retrieval) and runs as a proper server.*

---

## 3. Architecture Overview

Quaestor is structured as a pipeline of four independent layers. Each layer has its own module, its own tests, and clearly defined inputs and outputs. No layer knows about the implementation details of any other layer.

### 3.1 Repository Map

Every file discussed in this document maps to exactly one entry below. Read this before opening any source file.

```
quaestor/
  src/quaestor/
    config.py              # All settings: .env vars -> pydantic model (singleton)
    ingestion/
      loader.py            # load_pdf()  load_edgar_submission()
      chunker.py           # chunk_documents() -- RecursiveCharacterTextSplitter
      indexer.py           # build_index()  load_index()
    retrieval/
      retriever.py         # retrieve(query, index) -> List[Document]
    generation/
      prompts.py           # system prompt template (versioned)
      chain.py             # build_chain()  ask(query, chain) -> str
  tests/
    test_config.py         # 31 tests
    test_loader.py         # 20 tests
    test_chunker.py        # 17 tests
    test_indexer.py        # 15 tests
    test_retriever.py      # 11 tests
    test_chain.py          # 29 tests
  app.py                   # Streamlit UI -- zero business logic
  smoke_test.py            # Integration test (requires live Ollama + Groq)
  .env                     # API keys and runtime flags (git-ignored)
  pyproject.toml           # Dependencies managed by uv
```

Two directories are absent from the repository and created at runtime: `.chroma/` (ChromaDB vector store, written on first `build_index()` call) and `data/raw/` (downloaded EDGAR filings, git-ignored). If either is missing when you run the app, the relevant step will fail with a clear error.

The entry point for both indexing and querying is `app.py`. There is no CLI in Phase 1 — everything flows through Streamlit. Orchestration logic lives in `app.py`; business logic lives in `src/`.

### 3.2 Layer Responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| Configuration | `config.py` | Central settings loaded from environment variables |
| Ingestion | `loader.py`, `chunker.py`, `indexer.py` | Load documents, split into chunks, embed and store |
| Retrieval | `retriever.py` | Embed query, run similarity search, return top-k |
| Generation | `prompts.py`, `chain.py` | Format prompt, send to Groq, parse citations |
| UI | `app.py` | Streamlit demo interface |

### 3.3 Data Flow Through the Pipeline

1. User uploads a PDF or selects an EDGAR ticker → `loader.py` extracts text
2. Text → `chunker.py` splits into 512-token chunks with 50-token overlap
3. Chunks → `indexer.py` embeds with nomic-embed-text and upserts to ChromaDB
4. User types a question → `retriever.py` embeds query, returns top-5 chunks
5. Top-5 chunks + question → `chain.py` formats prompt, sends to Groq, returns answer
6. Answer displayed in Streamlit with source expander showing provenance

#### Example: End-to-End Query

> *User query: "What was Apple's net income in FY2025?"*
> *→ `retriever.py` embeds this question into a 768-dimensional vector*
> *→ ChromaDB searches all 579 Apple 10-K chunk vectors*
> *→ Top result: chunk #127 from Page 31 (cosine similarity 0.88)*
> *→ Chunk text: "Net income ... $112,010 million ... fiscal year 2025"*
> *→ `chain.py` inserts this into the prompt template*
> *→ Groq's Llama 3.3 70B reads the context and generates:*
> *"Apple's net income for FY2025 was $112,010 million. [Source: apple-10k-2025.txt, Page 31]"*
> *→ Streamlit displays the answer with citation*

---

## 4. The Technology Stack

### 4.1 LLM Inference: Groq + Ollama

An LLM is what reads the retrieved context and generates the answer. The choice of where to run it is a hardware constraint problem.

#### The M1 Bottleneck

> *The development machine is a 2020 M1 MacBook Air with 8GB of unified memory. Running Llama 3.1 8B locally via Ollama generates about 15 tokens per second. A typical RAG response is 150–300 tokens, so each query takes 10–20 seconds. Over a day of development that accumulates to hours of waiting.*
>
> *Groq is a cloud service that runs open-source LLMs on custom silicon (Language Processing Units) at 300 tokens per second for Llama 3.3 70B on the free tier. That's a 20x speedup versus local inference, and the model is 70 billion parameters — roughly 9x larger than what the M1 can run locally.*

The key design decision: all three providers (Groq, Ollama, Together.ai) are wrapped identically by LangChain. Switching between them requires changing one environment variable — `LLM_PROVIDER` — with zero code changes.

### 4.2 Embeddings: nomic-embed-text via Ollama

Embeddings are separate from LLM inference and run locally on the M1. nomic-embed-text is a 270MB model that runs entirely on Apple Silicon unified memory. It does not conflict with Groq because Groq handles LLM calls in the cloud — the M1 only runs the embedding model.

#### Why This Model?

> *nomic-embed-text outperforms OpenAI's Ada-002 on several standard retrieval benchmarks (MTEB) while costing nothing. The alternative — paying per embedding call — adds operational cost and a network dependency for every indexing run. A local model eliminates both. It runs entirely offline, which also means the embedding step works without any external service during development or in CI.*

### 4.3 Vector Store: ChromaDB

ChromaDB is an open-source vector database that runs in-process — meaning it runs inside the same Python process as the rest of the application, with no separate server, no Docker container, and no network calls. It persists data to a local directory (`.chroma/`) and loads it on startup.

For Phase 1, this is the right choice. The entire test suite can run in CI with no external services. The Streamlit demo can run offline.

#### The Phase 2 Migration Plan

> *For Phase 2, ChromaDB will be replaced with Qdrant (self-hosted via Docker), which adds hybrid dense + sparse retrieval. The switch is transparent to the retriever — LangChain wraps both with the same interface. Hybrid search means combining semantic similarity (dense vectors) with keyword matching (BM25), which improves retrieval of exact financial terms like ticker symbols and line item names.*

### 4.4 LangChain and LCEL

LangChain is a framework for building LLM-powered applications. It provides document loaders, text splitters, vector store integrations, and LLM provider wrappers.

LCEL (LangChain Expression Language) is LangChain's way to compose pipeline steps declaratively:

```python
chain = retriever | prompt_template | llm | output_parser
```

Each component has a well-defined input and output type. The chain is lazy — it does not execute until you call `.invoke()`. This makes each step independently testable.

#### How LCEL Execution Actually Works

The `|` operator is not magic. It is Python's `__or__` dunder method, overloaded by LangChain's `Runnable` base class. Every component in the chain — the retriever, the prompt template, the LLM, the output parser — inherits from `Runnable`, which requires exactly one method: `invoke(input) -> output`.

When Python evaluates `retriever | prompt_template`, it calls the `__or__` dunder method on the retriever, which returns a `RunnableSequence` object. Piping with `llm` extends that sequence. The final `chain` object is a `RunnableSequence` of four steps. *Nothing has run yet.*

When `chain.invoke({"query": "What was net income?"})` is called, execution proceeds in order:

1. `retriever.invoke({"query": "..."})` → `List[Document]`
   Embeds the query with nomic-embed-text and returns the top-k chunks from ChromaDB.
2. `prompt_template.invoke({context, question})` → `ChatPromptValue`
   Formats the system prompt, inserting the retrieved chunk text and the original question.
3. `llm.invoke(ChatPromptValue)` → `AIMessage`
   Sends the formatted prompt to Groq (or Ollama) and returns the model's response object.
4. `output_parser.invoke(AIMessage)` → `str`
   Extracts `.content` from the response object and returns a plain Python string.

> *A common point of confusion: the retriever outputs `List[Document]`, but the prompt template needs both `context` (the chunks) **and** `question` (the original query). Where does `question` come from if the retriever never produced it?*
>
> *Answer: LCEL passes the original input dict **alongside** each step's output through the sequence. The prompt template receives the retrieved documents as `context` and pulls `question` directly from the original `{"query": "..."}` dict via its input variable mapping. The key name difference (`query` vs `question`) is resolved by how the `ChatPromptTemplate` variables are declared in `prompts.py`.*

### 4.5 Configuration: pydantic-settings

Pydantic is a Python library for data validation using type annotations. Every configurable parameter in Quaestor lives in a single `Settings` class that reads from environment variables.

#### Why Enums for Provider Names

> *The `LLMProvider` field is an enum, not a string. This means a typo in the `.env` file (e.g., `LLM_PROVIDER=grq`) raises a validation error at application startup with a clear message, rather than silently falling through to unexpected behavior mid-pipeline. This is the right failure mode: loud and early.*

---

## 5. Module-by-Module Build Walkthrough

### 5.1 config.py — The Foundation

`config.py` is the first module built. Its only job is to provide a validated settings singleton that every other module imports.

#### Key Design Decisions

- Enum types for provider names — validation at startup, not at call time
- Path type for directory paths — avoids string concatenation bugs
- `chunk_overlap` validator — raises `ValueError` if overlap ≥ chunk_size
- Single module-level singleton: `settings = Settings()`

#### Which Settings Control What — and When

Not all settings are equal. Some take effect only at index-build time; changing them after an index exists has no effect until you rebuild. Others affect every query. Getting them wrong produces silent failures rather than loud errors.

| Setting | When active | Consequence if wrong |
|---|---|---|
| `LLM_PROVIDER` | Every query | Wrong provider invoked; Groq key rejected by Ollama endpoint |
| `GROQ_API_KEY` | Every query (Groq only) | 401 error on first query |
| `OLLAMA_BASE_URL` | Every query (Ollama only) | Connection refused if Ollama not running |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Index build only | No effect at query time; stale index retains old parameters |
| `COLLECTION_NAME` | Build **and** query | Mismatch creates an empty collection at query time; retriever returns nothing |
| `CHROMA_PERSIST_DIR` | Build **and** query | Different path at query time creates a new empty store |
| `TOP_K` | Every query | Too low: LLM misses relevant context. Too high: prompt exceeds context window |

> *Three settings are **re-index triggers**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `COLLECTION_NAME`. Changing any of these in `.env` after an index has been built does **not** update the existing vectors — the old index persists unchanged. You must either delete `.chroma/` and rebuild from scratch, or use a new `COLLECTION_NAME` to build a parallel index.*
>
> *`COLLECTION_NAME` also controls collection isolation: the EDGAR workflow writes to `"quaestor"` and the PDF upload workflow writes to `"quaestor_upload"`. Querying the wrong collection returns results from a different document set with no error.*

### 5.2 ingestion/loader.py

`loader.py` handles two distinct source types: arbitrary PDFs uploaded by users, and SEC EDGAR filings downloaded programmatically.

#### The SGML Discovery

> *The biggest surprise: SEC EDGAR does not deliver PDFs. It delivers SGML containers that look like:*

```xml
<DOCUMENT>
  <TYPE>10-K
  <TEXT>...HTML content...</TEXT>
</DOCUMENT>
```

> *When PyPDFLoader was first tried, it returned zero documents. The fix required writing `load_edgar_submission()` that extracts the `<TYPE>10-K` block, strips HTML with BeautifulSoup, and saves clean text.*
>
> *Counterintuitively, this turned out better than PDFs. PDF text extraction often produces layout artifacts — merged table columns, broken hyphenation, header noise. HTML from EDGAR is clean structured prose.*

### 5.3 ingestion/chunker.py — Where Quality is Won or Lost

Chunking is where most RAG quality is won or lost. The core problem: an LLM has a context window limit, and a 400-page document far exceeds it. The pipeline must split the document into pieces small enough to fit in the prompt while preserving enough context that each piece is still meaningful.

#### Phase 1: Fixed-Size Chunking with Overlap (The Intentional Baseline)

In Phase 1: `chunk_size=512` tokens, `chunk_overlap=50` tokens.

These are not arbitrary numbers. The 512-token ceiling derives directly from the input context limit of BERT-based sentence-transformer embedding models — the dominant open-source embedding family as of 2023 (all-MiniLM-L6-v2, all-mpnet-base-v2, and their derivatives).[^1] A chunk larger than the embedding model's context window is silently truncated during indexing — the tail of the chunk is embedded as if it does not exist. This makes 512 the natural ceiling for any chunk that must be fully and faithfully represented in the vector store, and it became the de facto reference size for fixed-size chunking baselines in the RAG evaluation literature.[^2] The 50-token overlap — approximately 10% of chunk size — is the most common accompanying choice in the same literature,[^3] representing the minimum redundancy that smooths hard boundary cuts without doubling storage costs. Phase 2 will report RAGAS faithfulness scores against this exact configuration, making the before/after comparison directly reproducible.

> *The design logic is adversarial by construction. Three choices compound to maximise failure rate:*
>
> *1. **512 tokens is the worst chunk size for financial tables.** It is too large to stay inside a paragraph, and too small to contain a full table. It reliably bisects structured content. A chunk size of 128 would produce many small, semantically incomplete fragments; 2048 would mostly capture whole sections. 512 sits precisely in the gap where it cuts financial tables mid-body.*
>
> *2. **50-token overlap is too small to bridge a table header.** Table headers in financial filings typically appear once, 200–600 tokens before the data rows. A 50-token overlap brings back the last 50 tokens of the preceding chunk — body rows, not headers. The column labels that make the numbers interpretable are permanently out of reach for any chunk after the first.*
>
> *3. **JPMorgan is the document that maximises exposure to both problems.** Its tables are longer than 512 tokens, its cross-references span tens of pages, and its numerical density is extreme. Apple's 10-K has shorter tables and cleaner section structure, which is why the Phase 1 demo works on Apple and struggles on JPMorgan.*
>
> *If Phase 2 had used easier documents or a non-standard baseline, the improvement claim would be ambiguous. Using the academic-standard parameters against the hardest test document makes the before/after comparison unambiguous and reproducible.*

#### What Does Overlap Mean?

> *The last 50 tokens of Chunk N become the first 50 tokens of Chunk N+1. This creates redundancy so that context doesn't get completely severed at chunk boundaries.*
>
> *For example, if a document has 1000 tokens total:*
> - *Chunk 1: tokens 0–512*
> - *Chunk 2: tokens 462–974 (starts 50 tokens before Chunk 1 ended)*
> - *Chunk 3: tokens 924–1000 (starts 50 tokens before Chunk 2 ended)*
>
> *Without overlap, Chunk 1 would be tokens 0–512, Chunk 2 would be 512–1024, with a hard cut at exactly token 512. With 50-token overlap, there's a 50-token 'buffer zone' that appears in both chunks.*

#### Why Overlap Helps (But Doesn't Solve Everything)

Overlap reduces the severity of bad splits but doesn't prevent them. The chunks still cut at fixed token counts regardless of sentence or paragraph structure.

> *Consider this sentence: "Revenue is recognized when the performance obligation is satisfied, which occurs when control of the promised goods or services is transferred to the customer."*
>
> *If the chunk boundary falls at token 500 and this sentence spans tokens 485–530:*
> - *Chunk 1 (tokens 0–512): includes the full sentence*
> - *Chunk 2 (tokens 462–974): includes the full sentence (starts at token 462, before the sentence)*
>
> *This works well. But what if a critical definition spans tokens 500–650?*
> - *Chunk 1 (tokens 0–512): gets "...revenue is recognized when the performance obligation"*
> - *Chunk 2 (tokens 462–974): gets "when the performance obligation is satisfied, which occurs when control of..."*
>
> *The overlap captured "when the performance obligation" in both chunks, but the complete definition is still fragmented. The retriever might find Chunk 2 (which has "is satisfied" and "control of") but not realize it needs Chunk 1 to get the full context of what triggers recognition.*
>
> *This is why Phase 2 switches to hierarchical chunking: parent chunks of 1024 tokens provide full context, while child chunks of 256 tokens enable precise retrieval. The parent-child relationship means the system can retrieve a small precise chunk but pass the full parent context to the LLM.*

### 5.4 ingestion/indexer.py

`indexer.py` takes text chunks and stores them in ChromaDB as searchable vectors.

#### Idempotent Upserts

> *Each chunk receives a stable ID derived from `MD5(source_path + page_number + start_index)`. ChromaDB's upsert operation either creates a new record or updates an existing one with the same ID.*
>
> *Why this matters: Re-running the indexer on the same document is safe — no duplicates are created. When Apple files a 10-K amendment, you can re-index without manual cleanup.*

### 5.5 retrieval/retriever.py

`retriever.py` is intentionally thin — a clean interface over ChromaDB's similarity search, designed so Phase 2 can swap in hybrid retrieval without callers knowing.

When a query arrives, it is embedded with nomic-embed-text. ChromaDB computes cosine similarity between the query vector and all stored chunk vectors, returning the top-k most similar chunks.

### 5.6 generation/prompts.py + chain.py

`prompts.py` defines a versioned system prompt. Versioning matters because prompt changes affect output quality in ways that are hard to detect without eval runs.

#### The Three Hard Rules

The system prompt enforces three non-negotiable rules:

1. **Context-only answers**: The LLM must answer using only the provided chunks. No drawing on training knowledge.
2. **Mandatory citations**: Every factual claim must be followed by `[Source: filename, Page N]`.
3. **Graceful refusal**: If the context does not contain enough information, respond with "I don't have enough information to answer this question based on the provided documents."

> *This refusal is not a failure — it is a feature. The refusal rate on unanswerable questions is a metric in Phase 2. A system that says 'I don't know' when appropriate is more trustworthy than one that always guesses.*

---

## 6. Runtime Walkthrough: One Query, Every Function

The previous section described each module statically. This section traces a single real query through every function call in order — from the user pressing Enter to the cited answer appearing in Streamlit.

**Query**: "What was Apple's total net sales in FY2025?"

### 6.1 Phase A — Indexing (runs once, before any query)

Indexing must have completed before a query can succeed. The sequence runs when the user clicks "Index Documents" in the Streamlit sidebar.

1. **`app.py`** receives the ticker `"AAPL"` from the sidebar input.
2. **`loader.load_edgar_submission("AAPL")`** is called.
   - Downloads `full-submission.txt` from `edgar.sec.gov`
   - Finds the `<TYPE>10-K` block inside the SGML container
   - Strips HTML tags with BeautifulSoup
   - Returns `List[Document]` — 52 documents, one per SGML segment
3. **`chunker.chunk_documents(docs, settings)`** is called.
   - Instantiates `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)`
   - Splits each document's text, preserving source and page metadata
   - Returns `List[Document]` — 579 chunks
4. **`indexer.build_index(chunks, settings)`** is called.
   - Instantiates `OllamaEmbeddings(model="nomic-embed-text")`
   - Calls `Chroma.from_documents(chunks, embeddings, persist_directory=".chroma/")`
   - For each chunk: Ollama converts the text to a 768-dimensional vector
   - ChromaDB stores all 579 vectors plus their metadata to disk
   - **Total time: 13.8 seconds on M1**

After this step, `.chroma/` contains 579 vectors. The app is ready to answer queries.

### 6.2 Phase B — Query Execution (runs on every question)

1. **`app.py`** receives the query string from the Streamlit chat input.
2. **`indexer.load_index(settings)`** is called.
   - Opens the existing ChromaDB at `.chroma/` — does not re-embed anything
   - Returns the `Chroma` object (the vector store handle)
3. **`chain.build_chain(index, settings)`** is called.
   - Creates `retriever = index.as_retriever(search_kwargs={"k": 5})`
   - Creates `ChatGroq(model="llama-3.3-70b-versatile")`
   - Creates `ChatPromptTemplate` from the versioned system prompt in `prompts.py`
   - Assembles the LCEL sequence: `retriever | prompt | llm | StrOutputParser()`
   - *Nothing has executed yet — this is a lazy pipeline object*
4. **`chain.ask(query, chain)`** is called, which calls `chain.invoke({"query": query})`.
   - **Step 1 — Retrieval**: `retriever.invoke({"query": "What was Apple's total net sales..."})`
     - Ollama embeds the query: 768-dim vector produced in ≈0.1 s
     - ChromaDB runs cosine similarity against all 579 chunk vectors
     - Returns the 5 chunks with highest similarity scores
     - *Top result: chunk #47, "Total net sales $416,161 million", cosine sim 0.91*
   - **Step 2 — Prompt assembly**: `prompt.invoke({context, question})`
     - Inserts the 5 chunk texts and the query into the prompt template (see box below)
     - Returns a `ChatPromptValue` ready for the LLM
   - **Step 3 — Generation**: `ChatGroq.invoke(prompt_value)`
     - HTTP POST to `api.groq.com/openai/v1/chat/completions`
     - Groq returns response in ≈0.7 s at 300 tok/s
     - Returns `AIMessage` object containing the raw response string
   - **Step 4 — Parsing**: `StrOutputParser.invoke(ai_message)`
     - Extracts `ai_message.content` — the plain answer string
5. **`app.py`** renders the answer string and the source chunks in a Streamlit expander.

### 6.3 The Prompt That Actually Gets Sent to the LLM

This is what Groq receives. Nothing is hidden or inferred.

```
SYSTEM:
You are a financial document analyst. Answer questions using ONLY the
context provided. Cite every factual claim with [Source: filename, Page N].
If the context does not contain the answer, respond with:
"I don't have enough information to answer this question based on the
provided documents." Do not use any knowledge outside the provided context.

CONTEXT:
[Chunk 1 - apple-10k-2025.txt, Page 27]
Net sales by reportable segment and product for 2025, 2024 and 2023 were
as follows: Total net sales $416,161 million [2025] $391,035 million [2024]

[Chunk 2 - apple-10k-2025.txt, Page 26]
The following table shows net sales and the percentage change from the
prior year for each reportable segment...

[Chunks 3-5 omitted for brevity]

QUESTION:
What was Apple's total net sales in FY2025?

--- LLM RESPONSE ---
Apple's total net sales in FY2025 were $416,161 million.
[Source: apple-10k-2025.txt, Page 27]
```

Three things to notice: (1) the system prompt rules are injected verbatim — the LLM is not trusted to infer citation behaviour; (2) the chunk text is raw extracted prose including table fragments; (3) the LLM is never given the query without context — it cannot draw on training memory.

---

## 7. Problems Encountered — and What They Teach

Every problem below was real. They are listed because the way a problem is solved reveals more about the system's design than the happy path ever does.

### Problem 1: SEC EDGAR Does Not Deliver PDFs

> *The SGML container format was completely unexpected. PyPDFLoader returned zero documents. The root cause was an unchecked assumption: "EDGAR is a document repository, therefore it delivers standard document formats."*
>
> ***Lesson**: Inspect the actual data before writing code to parse it. Ten minutes with a text editor would have revealed the format. Now a standing rule: download one example and read it before designing any loader.*
>
> ***Silver lining**: The HTML extracted from EDGAR is higher quality source material than a PDF text layer would have been.*

### Problem 2: ChromaDB Collection Isolation

> *After indexing the Apple 10-K (579 chunks), uploading a different document and asking questions about it returned Apple financial data. Both workflows were writing to the same ChromaDB collection.*
>
> ***Fix**: Isolate the two workflows into separate collections: `"quaestor"` for persisted EDGAR index and `"quaestor_upload"` for ad-hoc uploaded documents.*
>
> ***Lesson**: This is a classic integration bug — each function worked correctly in isolation but their interaction produced wrong behavior. A test that indexes two separate document sets and verifies that queries against each return only the right results would have caught this.*

### Problem 3: Scanned PDFs Silently Index Empty Strings

> *PyPDFLoader extracts the text layer. Scanned PDFs are images with no text layer. Every page returned an empty Document. The pipeline succeeded at every step and indexed empty strings.*
>
> ***Fix**: Filter empty pages before chunking, warn the user, and add an extraction preview showing the first 300 characters. Users can now verify text extraction before committing time to embedding.*

### Problem 4: The Groq Model Was Decommissioned Mid-Build

> *Between writing the spec and the first live test, Groq replaced `llama-3.1-70b-versatile` with `llama-3.3-70b-versatile`. The pipeline failed with "model not found".*
>
> ***Lesson**: All model names live in `config.py` with environment variable overrides. The fix was one line. If model names had been scattered across five files it would have been a risky search-and-replace.*

### Problem 5: ModuleNotFoundError in Streamlit

> *pytest found the `quaestor` package because `pyproject.toml` injects `src/` into the Python path. Streamlit runs `app.py` in its own process with no such injection.*
>
> ***Fix**: Add a `[build-system]` block to `pyproject.toml` and run `uv pip install -e .` — an editable install that makes `src/quaestor` permanently importable by every process.*

---

## 8. Testing Strategy

There are 123 unit tests. All of them run in under 6 seconds. None require Ollama, a Groq API key, or any network access.

### 8.1 Fake LLM and Fake Embeddings

LangChain provides `BaseChatModel` as an abstract base class. The test suite implements a `FakeLLM` that extends it and returns a canned response string regardless of input.

> *Using fakes instead of mocks is a deliberate choice. A mock asserts that a specific method was called with specific arguments — it tests the interaction. A fake implements the interface and returns predictable outputs — it tests the behavior. For pipeline components, behavior is what matters.*

### 8.2 Test Coverage by Module

| Module | Tests | Key Scenarios |
|---|---|---|
| `config.py` | 31 | Defaults, enum validation, Path type, overlap validator |
| `loader.py` | 20 | PDF loading, SGML extraction, empty page filtering |
| `chunker.py` | 17 | Chunk count, overlap, metadata preservation |
| `indexer.py` | 15 | Build, load, upsert idempotency, collection naming |
| `retriever.py` | 11 | Top-k results, metadata passthrough, empty collection |
| `chain + prompts` | 29 | Answer type, citation format, refusal string, context injection |

### 8.3 Integration Test: smoke_test.py

The smoke test is the only test that requires live services (Ollama + Groq). It runs the full pipeline against the real Apple FY2025 10-K filing and verifies that each step produces expected output.

### 8.4 How to Run the Tests Locally

```bash
# Unit tests only -- no Ollama, no Groq, no network
uv run pytest tests/ -v

# Single module
uv run pytest tests/test_chain.py -v

# Integration smoke test (requires Ollama running + GROQ_API_KEY in .env)
python smoke_test.py
```

> ***What is not yet runnable**: RAGAS evaluation against a golden dataset. The golden dataset and the evaluation harness are Phase 2 deliverables. In Phase 1 there is no automated quality score — only the smoke test verifying that the pipeline returns a non-empty cited answer. Manual spot-checking of retrieval quality is the only evaluation available at this stage.*

---

## 9. Live Results

The following results are from a live run against the Apple FY2025 10-K filing (filed 2025-10-31). All numbers are real.

### 9.1 Pipeline Performance Metrics

| Metric | Value |
|---|---|
| SGML segments extracted | 52 |
| Chunks after fixed-size splitting (512 tokens, 50 overlap) | 579 |
| Embedding time (nomic-embed-text on M1) | 13.8 seconds |
| Query latency (Groq Llama 3.3 70B) | 0.6 – 0.8 seconds |
| Unit test suite runtime (123 tests) | 5.8 seconds |

### 9.2 Sample Query Results Against Apple FY2025 10-K

| Question | Answer | Source Cited |
|---|---|---|
| Total net sales FY2025? | $416,161 million | Page 27 |
| Net income FY2025? | $112,010 million | Page 31 |
| Primary risk factors? | Interest rates, FX, cybersecurity | Pages 23, 30 |

---

## 10. What Phase 1 Proves — and What Comes Next

### 10.1 What Phase 1 Proves

- The end-to-end pipeline is correct. Documents load, chunk, embed, index, retrieve, and generate cited answers.
- The infrastructure is solid. Config, testing, packaging, provider switching, and idempotent indexing all work as designed.
- The abstraction boundaries hold. Swapping LLM providers requires one env var change.
- The citations are accurate. Every answer includes a verifiable filename and page number.

### 10.2 What Phase 1 Does Not Prove

- That fixed-size chunking is the right strategy — it is deliberately the wrong strategy.
- That retrieval quality is high — there is no measurement yet, only manual verification.
- That the system handles unanswerable questions reliably — refusal rate is not measured.
- That the pipeline is observable — there is no tracing, no latency breakdown.

### 10.3 Phase 2 Targets

| Change | Expected Impact |
|---|---|
| Hierarchical parent-child chunking | Higher RAGAS faithfulness — section boundaries preserved |
| Hybrid BM25 + dense retrieval via Qdrant | Better recall on exact financial terms |
| LangGraph retrieval state machine | Confidence thresholding and unanswerable refusal |
| RAGAS evaluation golden dataset | Quantitative before/after comparison |
| Langfuse tracing from first call | Full visibility into retrieval and latency *(not yet wired — zero observability exists in Phase 1)* |

The Phase 1 baseline is not a draft — it is a deliberate, complete, and working system built to be beaten. The fixed-size chunking is the 'before'. Phase 2 is the 'after'. The distance between them, measured in RAGAS faithfulness points, is the story.

---

## Footnotes

[^1]: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," *Proceedings of EMNLP 2019*, arXiv:1908.10084. The 512-token input limit applies to all BERT-based sentence-transformers; inputs exceeding this window are silently truncated by the tokenizer.

[^2]: Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," arXiv:2312.10997 (2023), surveys chunking strategies and uses 512-token fixed chunks as the reference baseline condition for comparison; Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation," arXiv:2309.15217 (2023), evaluates retrieval quality against this configuration.

[^3]: The 10% overlap convention appears across LangChain's text-splitter documentation (<https://python.langchain.com/docs/concepts/text_splitters/>) and in the LlamaIndex chunking guides (<https://docs.llamaindex.ai/en/stable/optimizing/production_rag/>), reflecting empirical consensus that smaller overlaps produce too many hard cuts while larger overlaps inflate storage without proportional retrieval gain.
