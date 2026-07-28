[![CI](https://github.com/AbhinavM28/Werby/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhinavM28/Werby/actions/workflows/ci.yml)

# 🏗️ Werby — AI Engineering Copilot

**Werby** is a Retrieval-Augmented Generation (RAG) system that lets warehouse and industrial engineers ask natural-language questions against their own engineering documentation — equipment manuals, SOPs, spec sheets, safety procedures — and get grounded, source-cited answers.

Ask *"What is the rated load of the AS/RS crane?"* and Werby retrieves the relevant passages from your ingested manuals, feeds them to an LLM under strict grounding rules, and returns the answer **with citations to the exact chunks it used** — because in an industrial setting, a hallucinated torque spec is a safety incident, not a bug.

> **Status:** actively developed, pre-deployment. The RAG pipeline, provider abstraction, CI quality gates, and evaluation harness are complete and tested — see [Evaluation](#evaluation) for real measured numbers and [Roadmap](#roadmap) for what's next. A hosted demo and walkthrough recording will follow deployment.

## How RAG works here

```
WRITE PATH (ingestion)
  PDF/TXT/MD ──▶ text extraction ──▶ semantic chunking ──▶ OpenAI embeddings ──▶ ChromaDB

READ PATH (query)
  question ──▶ embed ──▶ similarity search (top-k) ──▶ context assembly
           ──▶ LLM (grounded system prompt) ──▶ answer + cited sources
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
│       ├── rag_service.py          # Ingestion & RAG orchestration
│       ├── evaluation.py           # Retrieval hit-rate / faithfulness harness
│       └── providers/              # LLMProvider & EmbeddingProvider ABCs + OpenAI, Ollama, local impls
├── frontend/streamlit_app.py       # Pure HTTP client UI (swappable for React)
├── scripts/
│   ├── ingest.py                   # Bulk CLI ingestion (reuses IngestionService)
│   └── evaluate.py                 # CLI for the evaluation harness
├── data/eval/dataset.yaml          # Hand-curated evaluation question set
├── tests/                          # Unit tests (mocked externals, no network)
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

## Evaluation

Unit tests prove the code behaves correctly against mocks. A separate `EvaluationService` measures whether the *system* actually retrieves the right thing and answers faithfully, run against the real vector store and a real LLM:

- **Retrieval hit-rate** — does the expected source document appear in the top-k results, and at what rank.
- **Score distribution** — the top similarity score per question, split between in-scope questions (a correct source exists) and a deliberately out-of-scope one, to establish where a future relevance threshold would sit.
- **Faithfulness** — a free, deterministic keyword check, plus an optional LLM-judge that flags answer claims unsupported by the retrieved context. The judge produces two signals (a list of unsupported claims, and a summary `yes`/`no`); when they disagree with each other, that's recorded as its own **inconsistent** outcome instead of being silently resolved one way or the other.

Latest run, against the current sample corpus (OSHA 1910.178 + a sample AS/RS crane spec, 63 chunks, 7 hand-curated questions):

| Metric | Result |
|---|---|
| Hit-rate | **100%** (6/6 in-scope questions) |
| Mean in-scope score | 0.8475 |
| Mean out-of-scope score | 0.5469 |
| Score gap (in − out) | 0.3006 |
| Keyword pass-rate | 66.7% |
| Faithfulness pass-rate | 85.7% |
| Faithfulness inconsistent | 1 case |

```bash
python -m scripts.evaluate data/eval/dataset.yaml
python -m scripts.evaluate data/eval/dataset.yaml --skip-judge   # skip the LLM-judge call and its cost
```

Reported honestly, not oversold: 7 questions is enough to validate the harness itself and catch real issues — it already caught a live case of the judge contradicting itself, which is the one "inconsistent" result above — but it isn't yet a statistically meaningful faithfulness benchmark. Growing the dataset alongside the ingested corpus is on the roadmap.

## Configuration

All settings load from environment variables / `.env` and are validated at startup — see `app/core/config.py` for every knob (models, chunk size, top-k, paths). Secrets never live in code.

## Roadmap

- [x] Pluggable LLM/embedding providers — OpenAI, Ollama (fully local / air-gapped), sentence-transformers
- [x] CI-enforced quality gates — ruff, mypy, and a hermetic pytest suite required on every PR
- [x] Evaluation harness — retrieval hit-rate, in-scope/out-of-scope score gap, keyword + LLM-judge faithfulness (see [Evaluation](#evaluation))
- [ ] Grow the evaluation dataset alongside the ingested corpus
- [ ] Hybrid retrieval (BM25 + dense) and a reranking stage
- [ ] pgvector backend behind the existing `VectorStore` interface
- [ ] Conversation memory / multi-turn queries
- [ ] Auth + multi-tenant corpora
- [ ] React frontend against the same API
- [ ] Deploy and record a live demo

## License

MIT
