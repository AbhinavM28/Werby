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

- **Retrieval hit-rate** — does the expected source document appear in the results, measured at two checkpoints: *pre-rerank* (the wide dense-search candidate pool, `retrieve_n`) and *post-rerank* (the final `top_k` that actually reaches the LLM). The gap between the two is reranking's measured contribution, not an estimate.
- **Score distribution** — the top similarity score per question, split between in-scope and deliberately out-of-scope questions. This is what `relevance_threshold` (see [Configuration](#configuration)) is actually tuned against.
- **Faithfulness** — a free, deterministic keyword check (with OR-group support for questions that have more than one textually-correct answer drawn from different true clauses of the same source), plus an optional LLM-judge that flags answer claims unsupported by the retrieved context. The judge produces two signals (a list of unsupported claims, and a summary `yes`/`no`); when they disagree with each other, that's recorded as its own **inconsistent** outcome instead of being silently resolved one way or the other.

### Reranking: measured, not assumed

The reranker exists because this harness caught a concrete, reproducible failure during a stress test: a question about one document's lockout/tagout procedure retrieved five chunks from a *different*, more vocabulary-dense OSHA standard, and zero from the document that actually answers it — a ranking failure between two relevant documents, which a relevance threshold cannot fix (both candidates score as "relevant enough"). Re-running the same harness against the same corpus with only `RERANK_ENABLED` changed:

| Metric | Reranking OFF | Reranking ON |
|---|---|---|
| Hit-rate | 96.8% (30/31) | **100.0%** (31/31) |
| Keyword pass-rate | 90.3% | 90.3% |

The one miss, every time: `nearmiss-crane-loto-vs-osha-standard` — *"What is the lockout/tagout procedure for the AS/RS stacker crane?"* With reranking off, the correct document is findable but buried at rank 18 of the 20-candidate pool (dense search's top-5 never sees it). The cross-encoder promotes it straight to rank 1. Keyword pass-rate is identical on both sides — reranking's effect here is purely on *which document wins*, not on answer quality once it does.

```bash
python -m scripts.evaluate data/eval/dataset.yaml --skip-judge                       # reranking off (default)
RERANK_ENABLED=true python -m scripts.evaluate data/eval/dataset.yaml --skip-judge    # reranking on
```

Reported honestly, not oversold: 37 questions across six documents is enough to validate the harness and the reranker's real effect, and it's already caught two live issues before they'd have been embarrassing anywhere else — an LLM-judge that contradicted its own verdict, and a keyword check too brittle to tell a correct, reworded answer from a wrong one. It still isn't a large-scale statistical benchmark. Growing the dataset further as more documentation gets ingested is on the [roadmap](#roadmap).

## Configuration

All settings load from environment variables / `.env` and are validated at startup — see `app/core/config.py` for every knob (models, chunk size, top-k, paths). Secrets never live in code.

## Roadmap

- [x] Pluggable LLM/embedding providers — OpenAI, Ollama (fully local / air-gapped), sentence-transformers
- [x] CI-enforced quality gates — ruff, mypy, and a hermetic pytest suite required on every PR
- [x] Evaluation harness — retrieval hit-rate (pre/post-rerank), score distribution, keyword + LLM-judge faithfulness (see [Evaluation & Results](#evaluation--results))
- [x] Configurable relevance threshold — refuses to answer, and skips the LLM call, when nothing retrieved clears the bar
- [x] Cross-encoder reranking stage — retrieve-then-rerank, off by default, its contribution measured by the eval harness rather than assumed
- [x] Grow the evaluation dataset alongside the ingested corpus — 37 cases across six documents, including cross-document near-misses
- [ ] Hybrid retrieval (BM25 + dense)
- [ ] pgvector backend behind the existing `VectorStore` interface
- [ ] Conversation memory / multi-turn queries
- [ ] Auth + multi-tenant corpora
- [ ] React frontend against the same API
- [ ] Deploy and record a live demo

## License

MIT
