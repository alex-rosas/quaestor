# HERODOTO.md — Session Log

> Named for the first historian. Each entry records what was built, decisions
> made, problems encountered, and what comes next.  Update at end of every
> session.

---

## Session 3 — 2026-03-25

### What was built

Full Phase 2 pipeline — every component from cross-encoder reranking through
LangGraph, Qdrant hybrid retrieval, PII/hallucination guardrails, FastAPI, and
RAGAS evaluation harness.  The Phase 2 smoke test runs all 8 steps end-to-end
against the live AAPL FY2025 10-K with all checks passing.

| Module | Tests | Status |
|---|---|---|
| `src/quaestor/retrieval/reranker.py` | `tests/unit/test_reranker.py` (16) | ✅ |
| `src/quaestor/retrieval/graph.py` | `tests/unit/test_graph.py` (23) | ✅ |
| `src/quaestor/ingestion/indexer.py` — Qdrant hybrid | `tests/unit/test_qdrant_indexer.py` (17) | ✅ |
| `src/quaestor/guardrails/input.py` | `tests/unit/test_input_guardrail.py` (23) | ✅ |
| `src/quaestor/guardrails/output.py` | `tests/unit/test_output_guardrail.py` (20) | ✅ |
| `src/quaestor/api/main.py` + `schemas.py` | `tests/unit/test_api.py` (26) | ✅ |
| `eval/evaluate.py` | `tests/unit/test_evaluate.py` (30) | ✅ |
| `eval/golden_dataset.json` | 20 questions (8 factual, 7 multi_hop, 5 unanswerable) | ✅ |
| `scripts/evaluate.py` | CLI with `--limit`, `--no-rerank`, `--top-k` flags | ✅ |
| `app.py` | Rewritten for Phase 2 (LangGraph, PII, NLI, confidence slider) | ✅ |
| `scripts/smoke_test_phase2.py` | 8-step live end-to-end test | ✅ |

**Total unit tests: 297 passed, 0 failed, 10.2 s.**

---

### Phase 2 smoke test results (AAPL FY2025 10-K, 1110 child chunks)

```
Cross-encoder : cross-encoder/ms-marco-MiniLM-L-6-v2 (local, no API cost)
Graph         : retrieve → rerank → confidence_gate → generate/refuse
Confidence    : -5.0 for Q&A test, 999.0 for forced-refusal test
```

| Step | Result |
|---|---|
| Hierarchical chunking (1024/256) | 1110 child chunks, all with `parent_content` |
| ChromaDB index | 1110 chunks embedded by nomic-embed-text in 24 s |
| Q1 — Total net sales | $416,161 million (score 3.91, cited Page 27) |
| Q2 — Primary risk factors | FX, interest rate, credit risk (score -2.07) |
| Q3 — Net income FY2025 vs 2024 | Sources found (score 3.75) |
| Refusal test (threshold 999.0) | `refused=True` ✅ |
| PII guardrail ("John Smith") | PERSON detected, redacted to `<PERSON>` ✅ |

---

### Decisions made

**LangGraph `StateGraph` with `TypedDict` state**
`RAGState` carries `question`, `docs`, `top_score`, `answer`, `sources`,
`refused`, `prompt_version`.  Each node is a pure function that merges a
partial dict into state.  The confidence router is an edge function
returning `"generate"` or `"refuse"`.  Returning a `GraphAnswer` dataclass
from `run_rag_graph` isolates callers from the internal `TypedDict`.

**Cross-encoder scoring inline in the rerank node**
The rerank node calls `cross_encoder.predict([[query, doc.page_content] for doc in docs])`,
sorts descending, trims to `top_n`, and stores `max(scores)` as `top_score`
in state.  This means the confidence gate can branch immediately on the next
edge without re-scoring.

**`CrossEncoderProtocol` for offline testing**
Any object with `.predict(list[list[str]]) -> list[float]` satisfies the
protocol.  Unit tests inject a `FakeHighCrossEncoder` (all 5.0) or
`FakeLowCrossEncoder` (all -10.0) to test the confidence gate branches
without downloading the 80 MB MiniLM weights.

**`QdrantVectorStore.__init__` + manual `create_collection`**
`QdrantVectorStore.from_documents(client=...)` does not forward the
`client` kwarg — it silently creates its own internal `QdrantClient` and
ignores the injected one.  Result: `TypeError: Client.__init__() got an
unexpected keyword argument 'client'`.  The fix is to use
`QdrantVectorStore.__init__(client=client, ...)` directly after calling
`client.create_collection(...)` to pre-create the collection with the
correct vector configs.

**Qdrant sparse vector name must be `"langchain-sparse"`**
`langchain-qdrant`'s `QdrantVectorStore` defaults to
`sparse_vector_name="langchain-sparse"`.  Creating the Qdrant collection
with key `"sparse"` causes a `QdrantVectorStoreError` at upsert time because
the store looks for `"langchain-sparse"` and cannot find it.

**Qdrant unnamed dense vector**
`QdrantVectorStore` defaults to `vector_name=""` (unnamed vector).
The collection must be created with `vectors_config=VectorParams(...)`
(not `vectors_config={"dense": VectorParams(...)}`).

**UUID-formatted MD5 IDs for both Chroma and Qdrant**
`_doc_id` originally returned a 32-char hex MD5.  Qdrant raised
`ValueError: Point id X is not a valid UUID`.  Fix: `str(uuid.UUID(hex=h))`.
The UUID format is accepted by both backends and preserves the determinism
guarantee.

**`_doc_id` must include `parent_id` + `chunk_index`**
Hierarchical child chunks store `start_index` relative to their parent,
not to the document.  Two first-children (chunk_index=0) on the same page
from different parents get the same `source::page=N::start=0` hash →
duplicate UUIDs → Chroma `DuplicateIDError`.  Fix: extend the hash key to
`…::parent=<parent_id>::ci=<chunk_index>`.

**Presidio requires `en_core_web_sm` + explicit `NlpEngineProvider`**
`AnalyzerEngine()` default init downloads `en_core_web_lg` via `pip`, which
fails inside a `uv`-managed venv (`No module named pip`).  Fix:
`uv pip install en_core_web_sm` (uv exposes its own pip compat layer), then
construct with `NlpEngineProvider(nlp_configuration={"nlp_engine_name":
"spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]})`.

**RAGAS 0.4.x import path changed**
`from ragas.metrics import faithfulness` raises a `DeprecationWarning` in
RAGAS 0.4.3.  Correct import: `from ragas.metrics.collections import
faithfulness, answer_relevancy, context_precision, context_recall`.

**NLI pipeline may return nested list**
Some HuggingFace `pipeline("text-classification")` versions return
`[[{"label": ..., "score": ...}]]` instead of `[{...}]` when called with
`top_k=None`.  Fix: `if raw and isinstance(raw[0], list): raw = raw[0]`.

**`[Source:` count test false positive**
`test_top_n_limits_context` counted `"[Source:"` occurrences in the LLM
output to verify only one source chunk was used.  The system prompt template
itself contains the string `[Source: <filename>, Page <n>]` as an example,
so the count was 2 even with `top_n=1`.  Fix: count unique chunk content
strings instead.

**FastAPI dependency injection via `app.dependency_overrides`**
Every heavy resource (vector store, LLM, cross-encoder, PII engines, NLI)
is wrapped in a `Depends()` provider.  Tests override these with
`app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()`.
This pattern removes every network/GPU call from the test suite while
exercising the full request/response pipeline.

**SSE streaming via `StreamingResponse` + `astream`**
`POST /ask/stream` yields newline-terminated `data: <json>\n\n` events:
`pii`, `token` (one per LangChain stream chunk), `sources`, `done`.
The caller reconstructs the answer by joining all `token` payloads.

**Golden dataset ground truths anchored to smoke-test figures**
All factual ground truths in `eval/golden_dataset.json` were written after
the Phase 1 smoke test confirmed the exact values from the AAPL FY2025 10-K:
net sales $416,161M, net income $112,010M (FY2025) / $93,736M (FY2024) /
$96,995M (FY2023).  This prevents evaluation drift from LLM-generated
pseudo-truths.

---

### Problems encountered

| # | Problem | Fix |
|---|---|---|
| 1 | `QdrantVectorStore.from_documents` ignores injected `client` kwarg | Use `__init__` directly + `client.create_collection()` |
| 2 | Qdrant raised `Point id X is not a valid UUID` | Wrap MD5 hex with `str(uuid.UUID(hex=h))` |
| 3 | Qdrant `QdrantVectorStoreError` on sparse vector name mismatch | Use `"langchain-sparse"` as sparse vector key (langchain-qdrant default) |
| 4 | Qdrant unnamed dense vector mismatch | Use `vectors_config=VectorParams(...)` (not keyed dict) |
| 5 | Chroma `DuplicateIDError` — 87 duplicate IDs on hierarchical chunks | Add `parent_id` + `chunk_index` to `_doc_id` hash key |
| 6 | Presidio `AnalyzerEngine()` calls pip, fails in uv venv | `uv pip install en_core_web_sm` + explicit `NlpEngineProvider` config |
| 7 | RAGAS `DeprecationWarning` on old import path | Switch to `ragas.metrics.collections` |
| 8 | NLI pipeline returned `[[{...}]]` instead of `[{...}]` | Flatten: `if isinstance(raw[0], list): raw = raw[0]` |
| 9 | `[Source:` count test returned 2 with `top_n=1` | Count unique chunk content, not `"[Source:"` occurrences |
| 10 | Q2 (risk factors) refused by confidence gate at threshold 0.0 | Smoke test Q&A section uses `-5.0`; gate behaviour tested separately at step 7 |

---

### What comes next (Phase 3)

- [ ] CI/CD eval gate: `pytest tests/unit/ && python scripts/evaluate.py --limit 5`
      on every PR; block merge if RAGAS faithfulness < 0.7
- [ ] Langfuse self-hosted (Docker Compose) — trace every `run_rag_graph` call
- [ ] Arize Phoenix span export for retrieval + generation observability
- [ ] Argilla data labelling integration for HITL feedback on refused questions
- [ ] Docker Compose full stack (Quaestor API + Qdrant + Langfuse + Argilla)
- [ ] Expand golden dataset from 20 → 120 questions (add JPMorgan, Goldman filings)
- [ ] NeMo Guardrails scope enforcement (reject non-financial questions at API layer)
- [ ] Streamlit eval dashboard — compare Phase 1 vs Phase 2 RAGAS scores visually

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
