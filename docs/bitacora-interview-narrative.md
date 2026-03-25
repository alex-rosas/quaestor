# BITACORA.md — Quaestor: My Interview Narrative

> Written in first person. This is how I explain this project out loud.
> Read this before every interview. Update it at the end of every phase.
> This is not a technical spec — it is my story.

---

## Phase 1 — The working demo

### What I built and why

I built Quaestor — a RAG system that lets financial analysts query SEC filings
and regulatory standards in natural language and get cited answers back. The
idea came from a real problem: analysts at firms like EY spend hours manually
navigating 300-page 10-K filings looking for specific numbers or disclosures.
I wanted to reduce that to a natural language question with a sourced answer.

The first question I had to answer was: what does "working" mean for Phase 1?
I decided it meant one thing — you open a browser, type a question about
Apple's financials, and get a factually correct answer with the exact page and
section it came from. Everything else was deferred.

---

### The stack and why I chose it

**LLM: Groq (free tier) with Llama 3.3 70B**

My development machine is a 2020 M1 MacBook Air with 8GB of RAM. Running a
large language model locally at that constraint gives me about 15 tokens per
second — which means every test cycle takes 30–40 seconds. Over a full day of
development that adds up to hours of waiting.

Groq solves this completely. It runs Llama 3.3 70B — a much better model than
I could run locally — at 300 tokens per second on its free tier. That's a 20x
speedup. The development loop went from painful to nearly instant. I keep
Ollama with Llama 3.1 8B as a local fallback for offline work.

The key architectural decision is that all three providers — Groq, Ollama,
Together.ai — are swappable with a single environment variable. LangChain
wraps them identically. This means I can switch from development to demo to
fallback without touching a single line of code.

**Embeddings: nomic-embed-text via Ollama (local)**

This is where the M1 actually works in my favor. Embeddings are lightweight —
nomic-embed-text is about 270MB and runs fast on Apple Silicon unified memory.
Critically, it does not conflict with LLM inference because Groq handles LLM
calls in the cloud. My M1 only runs embeddings. That's the right division of
labor for my hardware.

I chose nomic-embed-text because it outperforms OpenAI's Ada-002 on several
retrieval benchmarks and costs nothing. There was no reason to pay for
embeddings when a free local model is competitive.

**Vector store: ChromaDB**

For Phase 1 I used ChromaDB. The reason is simple — it runs in-process,
requires zero configuration, and needs no Docker. I can run the full pipeline
including tests in CI without spinning up any external service. It persists to
a local `.chroma/` directory.

For Phase 2 I will migrate to Qdrant, which supports hybrid dense and sparse
retrieval natively. But starting with ChromaDB let me focus on getting the
pipeline right before worrying about the retrieval architecture.

**Framework: LangChain**

LangChain gives me document loaders, text splitters, retrieval chains, and
prompt templates out of the box. More importantly it has native wrappers for
all three of my LLM providers — swapping providers is one line. The RAG chain
is built with LCEL (LangChain Expression Language), which composes retrieval,
prompting, and generation as a declarative pipeline. Each step is explicit and
independently testable. For Phase 2 I will add LangGraph for the multi-step
retrieval state machine, which layers on top of LangChain cleanly.

---

### How I structured the code

I used a `src/` layout with `pyproject.toml` and `uv` for dependency
management. The project is divided into four layers that mirror the RAG
pipeline:

- `ingestion/` — loading documents and downloading from SEC EDGAR
- `retrieval/` — querying the vector store
- `generation/` — the RAG chain that produces cited answers
- `api/` — FastAPI endpoints (Phase 2)

All configuration flows through a single `config.py` using `pydantic-settings`.
Nothing in the project reads an environment variable directly — everything goes
through the central settings object. This means if I need to change a
parameter, I change it in one place and it propagates everywhere.

I made a specific choice to use Pydantic enums for `LLMProvider` and
`EmbeddingProvider` rather than raw strings. This means a typo in the `.env`
file fails immediately at startup with a clear validation error rather than
silently using the wrong provider mid-pipeline.

**Indexing is idempotent by design.**
Each chunk gets a stable ID derived from `MD5(source::page::start_index)`.
Calling `build_index` twice on the same document overwrites existing vectors
rather than creating duplicates. This matters in a real system because
documents get updated — Apple files a 10-K amendment, a standard gets revised
— and you need to be able to re-index without corrupting the collection. Most
demos ignore this. Production can't.

**The prompt is the first anti-hallucination layer.**
The system prompt enforces three hard rules: answer only from the provided
context, cite every factual claim with `[Source: filename, Page N]`, and if
the context is insufficient respond with a specific refusal string rather than
guessing. That refusal string matters — it makes "I don't know" a deliberate
feature rather than a failure mode. It's also what Phase 2 measures directly:
the golden dataset includes five deliberately unanswerable questions, and the
target is a ≥95% correct refusal rate.

**Testing is fully offline.**
There are 123 unit tests and all of them run without Ollama, without a Groq
API key, and without any network access. Every test that touches the LLM uses
a `FakeLLM` — a class that extends LangChain's `BaseChatModel` and returns a
canned response. Every test that touches embeddings uses `FakeEmbeddings` that
returns deterministic vectors derived from a hash of the input text. The
entire suite runs in 6 seconds. That's a production testing discipline: the
test suite is an asset that can be run anywhere, not a liability that requires
a live environment.

---

### The problems I actually ran into

This is the part most people skip in interviews. I think it's the most
important part.

**Problem 1 — SEC EDGAR doesn't deliver PDFs**

This was the biggest discovery. I assumed SEC EDGAR would give me clean PDFs.
It doesn't. It delivers SGML containers — a format called `full-submission.txt`
that wraps all filing documents (the 10-K HTML, exhibits, XBRL data) inside
`<DOCUMENT>` blocks with metadata headers.

When I tried to load these with LangChain's `PyPDFLoader` I got zero documents
back. The loader was looking for a PDF text layer that wasn't there.

The fix was writing a custom `load_edgar_submission()` function that:
1. Scans the SGML for `<TYPE>10-K` blocks
2. Extracts the HTML between `<TEXT>` and `</TEXT>` tags
3. Saves the extracted `.htm` file to disk
4. Strips all HTML tags with BeautifulSoup
5. Segments the clean text into 4000-character chunks

This function is idempotent — if the `.htm` file already exists it skips
extraction and loads directly. That matters because re-running the pipeline
shouldn't re-parse a 400-page document every time.

This turned out to be a better outcome than getting PDFs. The SGML container
preserves document structure in a way that PDF text layers don't — PDF
extraction often produces layout artifacts, broken table rows, and merged
columns. The HTML extracted from EDGAR is cleaner source material for
chunking.

**Problem 2 — Collection isolation for uploaded documents**

The Streamlit app has two modes: load an existing indexed EDGAR filing, or
upload your own PDF. When I first built the upload path, I made it write to
the same ChromaDB collection as the EDGAR data.

The result was subtle and confusing. After indexing the Apple 10-K (579
chunks), every query against an uploaded document returned Apple financial
data. The LLM would correctly say "I don't have enough information" about the
uploaded content — because it was retrieving Apple chunks, not the user's
document.

The fix was isolating the two workflows into separate ChromaDB collections:
`"quaestor"` for the persisted EDGAR index and `"quaestor_upload"` for ad-hoc
uploaded documents. They never mix. In Phase 2 I will extend this to let users
choose which collections to search across.

This is a good example of a bug that only appears at integration — each
individual function worked correctly, but their combination produced wrong
behavior. The unit tests for `build_index` and `retrieve` both passed. A test
that indexed two separate document sets and verified that queries against each
returned only the right results would have caught this immediately. That test
is now on the Phase 2 list.

**Problem 3 — Scanned PDFs silently return nothing**

`PyPDFLoader` extracts the text layer from PDFs. Image-based scanned PDFs
have no text layer — every page comes back as a `Document` with empty
`page_content`. The pipeline succeeded at every step but indexed empty
strings. The LLM received empty context and correctly refused to answer, which
looked like a pipeline bug.

The fix was filtering out empty pages before chunking, warning the user
clearly, and adding an extraction preview that shows the first 300 characters
of extracted text. Now the user can verify immediately whether the document
was read correctly before spending 15 seconds on embedding.

**Problem 4 — The Groq model was decommissioned**

Between the time I wrote the spec and the first live test, Groq decommissioned
`llama-3.1-70b-versatile` and replaced it with `llama-3.3-70b-versatile`. The
pipeline failed on the first real query with a model not found error.

This is a good example of why all model names live in `config.py` with
environment variable overrides rather than being hardcoded anywhere. The fix
was one line — changing the default value in `config.py` and updating the
corresponding test assertion. If model names had been scattered across five
files it would have been a search-and-replace with risk of missing one.

**Problem 5 — Module not found in Streamlit**

`pytest` found the `quaestor` package because `pyproject.toml` injects `src/`
into the Python path for tests. Streamlit runs `app.py` in its own process
with no such injection, so `import quaestor` raised `ModuleNotFoundError`.

The fix was adding a `[build-system]` block to `pyproject.toml` and running
`uv pip install -e .` — an editable install that writes a `.pth` file into
the virtual environment making `src/quaestor` permanently importable by every
process using that environment. Streamlit, scripts, the REPL, and pytest all
find the package from then on.

**Problem 6 — Mock patching requires module-level imports**

One subtler Python detail worth knowing. `download_sec_filings()` originally
imported `Downloader` lazily inside the function body. `unittest.mock.patch`
works by replacing a name on a module — it can't patch a name that doesn't
exist at module level. Four tests failed with `AttributeError: module has no
attribute 'Downloader'`.

The fix was hoisting the import to the top of the module inside a
`try/except ImportError` block, with `Downloader = None` as the fallback.
This is the correct pattern for optional dependencies that need to be
testable: the name exists at module level, the test patches it, and the
real code checks for `None` before using it.

---

### The live results

After building all seven modules I ran a smoke test against a real Apple
FY2025 10-K filing (filed 2025-10-31). These are the actual numbers:

- 52 document segments extracted from the SGML container
- 579 chunks after fixed-size splitting (512 tokens, 50 overlap)
- 13.8 seconds to embed all 579 chunks with nomic-embed-text on M1
- Query latency: 0.6–0.8 seconds end to end

| Question | Answer | Source cited |
|---|---|---|
| Total net sales | $416,161 million | Page 27 |
| Net income FY2025 | $112,010 million | Page 31 |
| Primary risk factors | Interest rates, FX, cybersecurity | Pages 23, 30 |

All answers are factually correct. All citations include the source filename
and page number.

---

### What I would do differently

**Investigate the data format before writing any code.**
I assumed SEC EDGAR delivered clean PDFs. I should have downloaded one filing
and opened it in a text editor before writing a single line of loader code.
Ten minutes of inspection would have revealed the SGML format immediately.
The lesson is general: understand your data source completely before designing
the code that consumes it.

**Write the collection isolation test first.**
The bug where uploaded documents mixed with EDGAR chunks was a classic
integration test failure — it only appeared when two features interacted. A
test that indexes two separate document sets and verifies that queries against
each return only the right results would have caught this before it ever
reached the UI. Now it's on the list.

**Instrument from day one.**
In Phase 1 I have no visibility into what happens inside a query beyond the
final answer. I don't know which chunks were retrieved, what their similarity
scores were, or where latency is spent. Phase 2 will add Langfuse tracing from
the first call — not retrofitted later. The cost of retrofitting observability
is always higher than the cost of building it in from the start.

---

### What this phase proves

Fixed-size chunking works. The pipeline is end-to-end correct. Citations are
accurate. The infrastructure is solid — configuration, testing, packaging,
provider switching, and idempotent indexing all work as designed.

What it doesn't prove yet is that this approach is better than alternatives.
That's the Phase 2 story — replacing fixed-size with hierarchical chunking,
measuring the RAGAS improvement, and showing before and after numbers. The
fixed-size baseline I built in Phase 1 is intentionally the "before": 512-token
windows with 50-token overlap, hard boundaries that cut mid-sentence regardless
of content. A sentence discussing revenue recognition can be split mid-clause
between two chunks, destroying the context the retriever needs. Hierarchical
chunking — parent chunks of 1024 tokens for context, child chunks of 256 tokens
for retrieval precision — is designed to fix exactly that. Phase 1 gives Phase 2
something concrete to beat.

---

> Update this file at the end of every phase. Add a new section for Phase 2
> covering the chunking improvement narrative, the RAGAS numbers, and whatever
> new problems you ran into. The problems section is the most important part —
> it's what separates someone who built something from someone who understands it.
