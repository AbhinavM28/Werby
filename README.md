# 🏗️ Werby — AI Engineering Copilot

**Werby** is a Retrieval-Augmented Generation (RAG) system that lets warehouse and industrial engineers ask natural-language questions against their own engineering documentation — equipment manuals, SOPs, spec sheets, safety procedures — and get grounded, source-cited answers.

Ask *"What is the rated load of the AS/RS crane?"* and Werby retrieves the relevant passages from your ingested manuals, feeds them to an LLM under strict grounding rules, and returns the answer **with citations to the exact chunks it used** — because in an industrial setting, a hallucinated torque spec is a safety incident, not a bug.

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
│       ├── embedding_service.py    # OpenAI embeddings (batched, retried)
│       ├── vector_store.py         # VectorStore ABC + ChromaDB implementation
│       ├── llm_service.py          # Prompt engineering + generation
│       └── rag_service.py          # Ingestion & RAG orchestration
├── frontend/streamlit_app.py       # Pure HTTP client UI (swappable for React)
├── scripts/ingest.py               # Bulk CLI ingestion (reuses same services)
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

## Configuration

All settings load from environment variables / `.env` and are validated at startup — see `app/core/config.py` for every knob (models, chunk size, top-k, paths). Secrets never live in code.

## Roadmap

- [x] Pluggable LLM/embedding providers — OpenAI, Ollama (fully local / air-gapped), sentence-transformers
- [ ] Hybrid retrieval (BM25 + dense) and a reranking stage
- [ ] pgvector backend behind the existing `VectorStore` interface
- [ ] Conversation memory / multi-turn queries
- [ ] Evaluation harness (retrieval hit-rate, answer faithfulness)
- [ ] Auth + multi-tenant corpora
- [ ] React frontend against the same API

## License

MIT
