[![CI](https://github.com/AbhinavM28/Werby/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhinavM28/Werby/actions/workflows/ci.yml)

# 🏗️ Werby — AI Engineering Copilot

**Werby** is a Retrieval-Augmented Generation (RAG) system that lets warehouse and industrial engineers ask natural-language questions against their own engineering documentation — equipment manuals, SOPs, spec sheets, safety procedures — and get grounded, source-cited answers.

Ask *"What is the rated load of the AS/RS crane?"* and Werby retrieves the relevant passages from your ingested manuals, feeds them to an LLM under strict grounding rules, and returns the answer **with citations to the exact chunks it used** — because in an industrial setting, a hallucinated torque spec is a safety incident, not a bug.

> **Status:** actively developed, pre-deployment. The RAG pipeline, provider abstraction, reranking stage, CI quality gates, and evaluation harness are complete and tested — see [Evaluation & Results](#evaluation--results) for real measured numbers and [Roadmap](#roadmap) for what's next. A hosted demo and walkthrough recording will follow deployment.

## How RAG works here

```
WRITE PATH (ingestion)
  PDF/TXT/MD ──▶ text extraction ──▶ semantic chunking ──▶ OpenAI embeddings ──▶ ChromaDB

READ PATH (query)
  question ──▶ embed ──▶ dense search (wide candidate pool, retrieve_n)
           ──▶ cross-encoder rerank (optional) ──▶ narrow to top-k
           ──▶ relevance threshold ──▶ below cutoff? refuse, skip the LLM call
           ──▶ context assembly ──▶ LLM (grounded system prompt)
           ──▶ answer + cited sources
```

## Architecture

```
werby/
├── app/
│   ├── main.py                     # App factory + domain-error → HTTP mapping
│   ├── api/
│   │   ├── deps.py                 # Composition root: all dependency wiring
│   │   └── v1/routes.py            # Thin HTTP endpoints
│   ├── core/
│   │   ├── config.py               # Typed settings (pydantic-settings, .env)
│   │   ├── logging.py              # Structured logging setup
│   │   └── exceptions.py           # Domain exceptions (HTTP-agnostic)
│   ├── schemas/rag.py              # Pydantic API contracts
│   └── services/
│       ├── document_processor.py   # Text extraction + semantic chunking (pure)
│       ├── vector_store.py         # VectorStore ABC + ChromaDB implementation
│       ├── llm_service.py          # Prompt engineering + generation
│       ├── rag_service.py          # Retrieve → rerank → filter → generate orchestration
│       ├── evaluation.py           # Retrieval hit-rate & faithfulness evaluation harness
│       └── providers/
│           ├── base.py             # EmbeddingProvider / LLMProvider / Reranker ABCs
│           ├── openai_provider.py  # OpenAI embeddings + chat (batched, retried)
│           ├── ollama_provider.py  # Fully local inference via Ollama's REST API
│           ├── local_embeddings.py # In-process sentence-transformers embeddings
│           └── local_reranker.py   # In-process cross-encoder reranking
├── frontend/streamlit_app.py       # Pure HTTP client UI (swappable for React)
├── scripts/
│   ├── ingest.py                   # Bulk CLI ingestion (reuses IngestionService)
│   └── evaluate.py                 # CLI for the evaluation harness
├── data/eval/dataset.yaml          # Versioned evaluation question set
├── tests/                          # Unit tests (mocked externals, no network)
├── .github/workflows/ci.yml        # Lint (ruff) + type-check (mypy) + test on every PR
├── Dockerfile                      # Multi-stage, non-root, healthchecked
├── docker-compose.yml
└── pyproject.toml
```

### Key design decisions

**Layered architecture.** Routes → services → infrastructure. Routes only translate HTTP; services hold all business logic and know nothing about HTTP; the composition root (`app/api/deps.py`) is the only place concrete implementations are chosen.

**Dependency inversion at the vector store.** Everything depends on the `VectorStore` interface, not ChromaDB. Migrating to pgvector or Qdrant later means writing one new subclass and changing one line of wiring.

**Domain exceptions, mapped once.** Services raise `WerbyError` subclasses; a single exception handler in `main.py` maps each to the right HTTP status. Business logic stays reusable from the API, the CLI, or a future worker queue.

**Idempotent ingestion.** Chunk IDs are deterministic (`filename::chunk_N`), so re-uploading a revised manual updates vectors instead of duplicating them.

**Grounded prompting.** The system prompt forbids answering outside the retrieved context, requires inline `[Source N]` citations, and demands exact quotation of safety-critical values.

**Pluggable AI providers.** `LLMProvider` and `EmbeddingProvider` interfaces (`app/services/providers/`) make the AI backend a `.env` choice. `LLM_PROVIDER=ollama` + `EMBEDDING_PROVIDER=ollama` runs Werby **fully locally with zero external network calls** — no API keys, no per-token cost, documentation never leaves the machine. Built for proprietary and export-controlled engineering documentation.

**Retrieve-then-rerank.** Dense search ranks by vector similarity, which is really vocabulary overlap in disguise — a long, jargon-dense document can outscore a short, specific one that's the actually correct answer. A `Reranker` interface (mirroring the provider ABCs — same argument, different backend) scores `(query, chunk)` pairs jointly with a cross-encoder instead of comparing independent embeddings, catching exactly this failure. It's off by default and opt-in (`RERANK_ENABLED=true`, local cross-encoder via the `local` extra), and deliberately widens retrieval to a candidate pool (`retrieve_n`) before narrowing back to `top_k` — a reranker can only promote a chunk that's *in* the pool it's given, never recover one dense search missed outright.

**Relevance threshold.** Chunks scoring below a configurable cutoff are discarded before context is built; if none survive, Werby refuses to answer instead of generating from irrelevant context — and skips the LLM call entirely, saving its cost and latency. In an industrial safety setting, an honest "I don't know" beats a fluent wrong answer about a torque spec or lockout procedure.

**Evaluation harness as a first-class citizen, not an afterthought.** `EvaluationService` runs a versioned question set (`data/eval/dataset.yaml`) against the real vector store and a real LLM, measuring retrieval hit-rate (both pre- and post-rerank, so reranking's actual contribution is a number, not a guess), keyword pass-rate, and LLM-judged faithfulness. It's what caught the retrieval failure that motivated the reranker in the first place — see [Evaluation & Results](#evaluation--results).

**Embedding-compatibility guard.** Vectors from different embedding models are not comparable; mixing them silently ruins retrieval instead of erroring. The Chroma collection is stamped with the model that built it, and the app refuses to start on a mismatch with instructions to re-ingest — converting a silent data-corruption bug into an actionable startup error.

**Resilience.** All OpenAI calls are batched where possible and retried with exponential backoff (tenacity); only persistent failures surface as errors.

## Quickstart

```bash
git clone <your-repo-url> && cd werby
python -m venv .venv && source .venv/bin/activate
make install                      # or: pip install -e ".[dev,frontend]"
cp .env.example .env              # add your OPENAI_API_KEY

make run                          # API → http://localhost:8000/docs
make frontend                     # UI  → http://localhost:8501
```

Bulk-ingest a folder of documents:

```bash
python -m scripts.ingest ./my_engineering_docs
```

Evaluate retrieval quality against the versioned dataset:

```bash
python -m scripts.evaluate data/eval/dataset.yaml                # hit-rate, keyword & LLM-judge faithfulness
python -m scripts.evaluate data/eval/dataset.yaml --skip-judge   # skip the LLM-judge call and its cost
```

Enable cross-encoder reranking (off by default — see [Retrieve-then-rerank](#key-design-decisions)):

```bash
pip install -e ".[dev,local]"        # local extra: sentence-transformers, for the reranker
echo "RERANK_ENABLED=true" >> .env
```

### Docker

```bash
docker compose up --build         # API on :8000, Chroma persisted in a volume
```

## API

Interactive docs at `http://localhost:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness probe |
| POST | `/api/v1/documents` | Upload & ingest one document |
| GET | `/api/v1/documents` | Corpus statistics |
| DELETE | `/api/v1/documents/{name}` | Remove a document |
| POST | `/api/v1/query` | Ask a question, get a cited answer |

## Testing & quality

```bash
make test       # pytest — unit tests, no network required
make lint       # ruff + mypy
make format
```

## Evaluation & Results

Unit tests prove the code behaves correctly against mocks. `EvaluationService` (`app/services/evaluation.py`) measures whether the *system* — real vector store, real LLM — actually retrieves the right thing and answers faithfully, run against a versioned question set (`data/eval/dataset.yaml`): 37 cases (31 in-scope, 6 deliberately out-of-scope) across the six OSHA standards and equipment manuals currently ingested, including cross-document near-misses designed to expose ranking *confusion*, not just outright retrieval failure.

- **Retrieval hit-rate** — does the expected source document appear in the results, measured at three checkpoints: *dense-only* (pure embedding search), *pre-rerank* (the wide candidate pool after hybrid fusion, `retrieve_n`), and *post-rerank* (the final `top_k` that actually reaches the LLM). The gap between dense-only and pre-rerank is hybrid retrieval's measured contribution; the gap between pre-rerank and post-rerank is reranking's — two independent features, two independently measured effects, not estimates.
- **Score distribution** — the top similarity score per question, split between in-scope and deliberately out-of-scope questions. This is what `relevance_threshold` (see [Configuration](#configuration)) is actually tuned against.
- **Faithfulness** — a free, deterministic keyword check (with OR-group support for questions that have more than one textually-correct answer drawn from different true clauses of the same source), plus an optional LLM-judge that flags answer claims unsupported by the retrieved context. The judge produces two signals (a list of unsupported claims, and a summary `yes`/`no`); when they disagree with each other, that's recorded as its own **inconsistent** outcome instead of being silently resolved one way or the other.

### Reranking: measured, not assumed

The reranker exists because this harness caught a concrete, reproducible failure during a stress test: a question about one document's lockout/tagout procedure retrieved five chunks from a *different*, more vocabulary-dense OSHA standard, and zero from the document that actually answers it — a ranking failure between two relevant documents, which a relevance threshold cannot fix (both candidates score as "relevant enough"). Re-running the same harness against the same corpus with only `RERANK_ENABLED` changed:

| Metric | Reranking OFF | Reranking ON |
|---|---|---|
| Hit-rate | 96.8% (30/31) | **100.0%** (31/31) |
| Keyword pass-rate | 90.3% | 87.1% |

The one miss, every time: `nearmiss-crane-loto-vs-osha-standard` — *"What is the lockout/tagout procedure for the AS/RS stacker crane?"* With reranking off, the correct document is findable but buried at rank 18 of the 20-candidate pool (dense search's top-5 never sees it). The cross-encoder promotes it straight to rank 1. Keyword pass-rate is nearly identical on both sides (the small delta is run-to-run LLM phrasing variance, not a reranking effect — the model isn't perfectly deterministic even at low temperature) — reranking's effect here is purely on *which document wins*, not on answer quality once it does.

```bash
python -m scripts.evaluate data/eval/dataset.yaml --skip-judge                       # reranking off (default)
RERANK_ENABLED=true python -m scripts.evaluate data/eval/dataset.yaml --skip-judge    # reranking on
```

Reported honestly, not oversold: 37 questions across six documents is enough to validate the harness and the reranker's real effect, and it's already caught two live issues before they'd have been embarrassing anywhere else — an LLM-judge that contradicted its own verdict, and a keyword check too brittle to tell a correct, reworded answer from a wrong one. It still isn't a large-scale statistical benchmark. Growing the dataset further as more documentation gets ingested is on the [roadmap](#roadmap).

### Hybrid Retrieval (Dense + BM25): measured, not assumed

Dense (embedding) search is weak on exact identifiers — a part number or standard number is one token among many in a chunk's vector, easily outweighed by the surrounding prose. BM25 lexical search is the opposite: no notion of paraphrase, but a rare token that matches gets a large weight precisely because it's rare. Hybrid retrieval fuses both result lists by rank (Reciprocal Rank Fusion, not raw score — see `reciprocal_rank_fusion()` in `app/services/rag_service.py` for why the two scores aren't comparable) before reranking. Re-running the same harness with only `HYBRID_ENABLED` changed, in both provider modes:

| Metric | Cloud OFF | Cloud ON | Local OFF | Local ON |
|---|---|---|---|---|
| Hit-rate | 100.0% (31/31) | 100.0% (31/31) | 100.0% (31/31) | 100.0% (31/31) |
| Dense-only hit-rate | 100.0% (31/31) | 100.0% (31/31) | 100.0% (31/31) | 100.0% (31/31) |
| Keyword pass-rate | 87.1% | 83.9% | 90.3% | 90.3% |
| Faithfulness pass-rate | 91.9% | 91.9% | 27.0% | 24.3% |
| Avg. answer latency | 1.77s (median 1.67s) | 1.71s (median 1.59s) | 28.2s¹ (median 27.8s)¹ | 28.6s (median 27.6s) |

¹ Local OFF latency is carried over from the [Cloud vs. Local](#cloud-vs-local-air-gapped-operation) benchmark below (identical corpus, dataset, and settings) rather than re-captured in this run; the Local ON figure confirms it's consistent with fresh measurement.

Reported honestly: on this corpus, hybrid retrieval shows **no hit-rate improvement** — dense search with reranking already finds every in-scope document, so there's nothing left for BM25 to rescue. This is an expected ceiling effect, not a failed feature; the corpus doesn't yet contain the kind of exact-identifier-vs-paraphrase near-miss that would demonstrate hybrid's value the way the stress-test corpus demonstrated reranking's. Latency overhead is negligible (BM25 search over 185 chunks is a low-millisecond operation, dwarfed by LLM generation either way). Keyword and faithfulness deltas are within the same run-to-run LLM-phrasing variance already noted for reranking above.

**Known limitation, not yet fixed:** BM25 scores are unbounded and not on the `[0, 1]` cosine scale, so chunks found *only* by lexical search are exempted from `relevance_threshold` entirely (`RetrievedChunk.bypass_relevance_filter`) rather than compared against a cosine-calibrated bar — see that field's docstring for the reasoning. This run caught the real cost of that choice: two deliberately out-of-scope questions (*"a good recipe for sourdough bread"*, *"symptoms of seasonal allergies"*) picked up a coincidental lexical match and got answered instead of refused, in **both** provider modes, with reported top scores of 4.11 and 4.67 — nonsense on a similarity scale, and a sign the exemption is too permissive as written. Cloud mode's answers stayed faithful anyway (`gpt-4o-mini` recognized the context didn't actually address the question), but the safety net a threshold is supposed to provide was bypassed by construction, not honored. `mean_out_of_scope_score` and `score_gap` are omitted from the table above for exactly this reason — mixing cosine and raw-BM25 units in one average produces a number (a *negative* gap) that looks alarming but isn't a real regression, just an artifact of comparing incompatible scales. Recalibrating this (a minimum BM25-score floor, or normalizing lexical scores before fusion) is tracked on the [roadmap](#roadmap) before hybrid retrieval should be trusted in production alongside ambiguous or adversarial queries.

```bash
python -m scripts.evaluate data/eval/dataset.yaml                       # hybrid off (default)
HYBRID_ENABLED=true python -m scripts.evaluate data/eval/dataset.yaml   # hybrid on
```

### Cloud vs. Local (Air-Gapped) Operation

"Cloud or fully local" is a `.env` choice, not a code change (see [Pluggable AI providers](#key-design-decisions)). Both runs below used the identical dataset, corpus, and `RERANK_ENABLED=true` — only `LLM_PROVIDER`/`EMBEDDING_PROVIDER` and the target Chroma collection differ.

| Metric | Cloud (OpenAI: gpt-4o-mini + text-embedding-3-small) | Local (Ollama: llama3.2:3b + nomic-embed-text) |
|---|---|---|
| Hit-rate | 100.0% (31/31) | 100.0% (31/31) |
| Keyword pass-rate | 87.1% | 87.1% |
| Faithfulness pass-rate | **91.9%** | 27.0% |
| Avg. answer latency | **1.85s** (median 1.77s) | 28.2s (median 27.8s) |

Retrieval quality is identical between the two — `nomic-embed-text` finds the right document just as reliably as OpenAI's embeddings on this corpus. What doesn't transfer is faithfulness and speed. `llama3.2:3b` is used as *both* the answer generator and the LLM-judge in local mode, and a 3B model is a weak, inconsistent judge of its own output — 6 inconsistent verdicts against 2 for cloud, the same self-contradiction pattern the `INCONSISTENT` verdict exists to catch, just happening far more often at this model size. CPU-bound local generation also runs roughly 15× slower per query. Score distributions shift too: local mode's gap between in-scope and out-of-scope scores is smaller (0.12 vs. 0.23 for cloud, from the same [reranking table](#reranking-measured-not-assumed) metric above) — `relevance_threshold` is tuned against cloud-scale scores today and would need separate calibration before trusting it in local mode.

None of this is a defect in local mode — it's the honestly-measured cost of the tradeoff it exists for: **complete data privacy and zero external dependencies** for proprietary or export-controlled documentation, in exchange for a smaller model's answer quality and CPU-bound latency. This benchmark ran a 3B chat model specifically; `app/core/config.py`'s own default (`OLLAMA_CHAT_MODEL=llama3.1:8b`) is larger and would likely narrow the faithfulness and latency gap at the cost of memory and generation speed — not yet benchmarked.

**Operational note:** local mode needs a one-time *online* setup — pulling the Ollama models (`ollama pull llama3.2:3b`, `ollama pull nomic-embed-text`) and downloading the sentence-transformers cross-encoder for reranking (cached after first load) — after which it runs with zero network access. This benchmark ran fully offline that way, with Ollama's server started as `CUDA_VISIBLE_DEVICES="" ollama serve` (CPU-only inference — GPU acceleration would change the latency numbers above materially).

```bash
# cloud
LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai python -m scripts.evaluate data/eval/dataset.yaml
# local, fully offline after the one-time model download
LLM_PROVIDER=ollama EMBEDDING_PROVIDER=ollama python -m scripts.evaluate data/eval/dataset.yaml
```

## Configuration

All settings load from environment variables / `.env` and are validated at startup — see `app/core/config.py` for every knob (models, chunk size, top-k, paths). Secrets never live in code.

## Roadmap

- [x] Pluggable LLM/embedding providers — OpenAI, Ollama (fully local / air-gapped), sentence-transformers
- [x] CI-enforced quality gates — ruff, mypy, and a hermetic pytest suite required on every PR
- [x] Evaluation harness — retrieval hit-rate (dense/pre-rerank/post-rerank), score distribution, keyword + LLM-judge faithfulness (see [Evaluation & Results](#evaluation--results))
- [x] Configurable relevance threshold — refuses to answer, and skips the LLM call, when nothing retrieved clears the bar
- [x] Cross-encoder reranking stage — retrieve-then-rerank, off by default, its contribution measured by the eval harness rather than assumed
- [x] Grow the evaluation dataset alongside the ingested corpus — 37 cases across six documents, including cross-document near-misses
- [x] Hybrid retrieval (BM25 + dense, fused via Reciprocal Rank Fusion) — off by default; see [Hybrid Retrieval](#hybrid-retrieval-dense--bm25-measured-not-assumed) for its measured effect and a known scoring limitation
- [ ] Recalibrate lexical-only relevance scoring — normalize BM25 scores (or otherwise replace the current threshold-bypass exemption) before trusting hybrid retrieval against ambiguous or out-of-scope queries
- [ ] pgvector backend behind the existing `VectorStore` interface
- [ ] Conversation memory / multi-turn queries
- [ ] Auth + multi-tenant corpora
- [ ] React frontend against the same API
- [ ] Deploy and record a live demo

## License

MIT
