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
│       ├── pgvector_store.py       # Postgres + pgvector implementation
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

**Dependency inversion at the vector store.** Everything depends on the `VectorStore` interface, not ChromaDB. Proven, not just claimed: `PgVectorStore` (Postgres + pgvector) is a second, real backend, selected via `VECTOR_STORE_BACKEND` — one new subclass and one line of wiring in `app/api/deps.py`, with zero changes anywhere in `RAGService`, `IngestionService`, hybrid retrieval, or reranking. See [Vector Store Backends](#vector-store-backends-chroma-vs-pgvector) for the real, measured proof.

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
docker compose up --build                      # API on :8000, Chroma persisted in a volume
docker compose --profile pgvector up postgres  # optional: Postgres + pgvector, for VECTOR_STORE_BACKEND=pgvector
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

Unit tests prove the code behaves correctly against mocks. `EvaluationService` (`app/services/evaluation.py`) measures whether the *system* — real vector store, real LLM — actually retrieves the right thing and answers faithfully, run against a versioned question set (`data/eval/dataset.yaml`): 38 cases (32 in-scope, 6 deliberately out-of-scope) across the six OSHA standards and equipment manuals currently ingested, including cross-document near-misses designed to expose ranking *confusion* and identifier-only lookups designed to expose dense search's specific blind spot, not just outright retrieval failure.

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

Dense (embedding) search is weak on exact identifiers — a part number or standard number is one token among many in a chunk's vector, easily outweighed by the surrounding prose. BM25 lexical search is the opposite: no notion of paraphrase, but a rare token that matches gets a large weight precisely because it's rare. Hybrid retrieval fuses both result lists by rank (Reciprocal Rank Fusion, not raw score — see `reciprocal_rank_fusion()` in `app/services/rag_service.py` for why the two scores aren't comparable) before reranking. Re-running the same harness with only `HYBRID_ENABLED` changed, in both provider modes, against the current 38-case dataset (32 in-scope):

| Metric | Cloud OFF | Cloud ON | Local OFF | Local ON |
|---|---|---|---|---|
| Hit-rate | 100.0% (32/32) | 100.0% (32/32) | 100.0% (32/32) | 100.0% (32/32) |
| Dense-only hit-rate | 100.0% (32/32) | 100.0% (32/32) | 100.0% (32/32) | 100.0% (32/32) |
| Keyword pass-rate | 90.6% | 87.5% | 90.6% | 84.4% |
| Faithfulness pass-rate | 92.1% | 92.1% | 23.7% | 23.7% |
| Avg. answer latency | 1.90s (median 1.84s) | 1.92s (median 1.68s) | 28.7s (median 27.9s) | 27.8s (median 26.9s) |

Reported honestly: hybrid retrieval shows **no hit-rate improvement** in this production-config table — dense search with reranking already finds every in-scope document, so there's nothing left for BM25 to rescue at the document level. That's a ceiling effect specific to this table's setup (reranking on, and hit-rate measured at document granularity — see below), not proof hybrid retrieval has no value. Latency overhead is negligible (BM25 search over 185 chunks is a low-millisecond operation, dwarfed by LLM generation either way). Keyword pass-rate dips slightly with hybrid on in both modes (90.6% → 87.5% cloud, 90.6% → 84.4% local) — within the same run-to-run LLM-phrasing variance already noted for reranking above, not a hybrid-specific regression (see the isolated, single-case evidence below, where hybrid's effect is unambiguous rather than noise-sized).

**Isolating hybrid's own contribution.** Document-level hit-rate structurally can't see hybrid's real strength: it counts a "hit" if *any* chunk of the expected document is retrieved, but hybrid's actual mechanism is promoting the *specific* chunk containing a rare token dense search underrates. A new case, `hybrid-pump-product-number-lookup` — the bare query `"What is V6001176?"`, no topical words to lean on — makes this concrete. `V6001176` is Grundfos's catalog number for a specific lubricant, appearing in exactly one of the pump manual's 50+ chunks. With `RERANK_ENABLED=false` (isolating hybrid from reranking's own independent wide-pool rescue) and hybrid off, that chunk never reaches the top 5, and the LLM answers: *"The context provided does not contain any information regarding 'V6001176.'"* Turn hybrid on, same settings: the correct chunk is promoted and the model answers *"V6001176 is the product number for Castrol Optimol Paste White T, 0.5 kg."* Across the full 38-case set at these isolated settings, hit-rate moves **96.9% → 100.0% (31/32 → 32/32)** — a real, reproducible fix, not noise.

**Recalibrated lexical-only scoring — mostly fixed, one known gap remains.** BM25 scores are unbounded and not on the `[0, 1]` cosine scale, so chunks found *only* by lexical search were originally exempted from `relevance_threshold` entirely (`RetrievedChunk.bypass_relevance_filter`) rather than compared against a cosine-calibrated bar. That let two deliberately out-of-scope questions (*"a good recipe for sourdough bread"*, *"symptoms of seasonal allergies"*) get answered instead of refused, with reported top scores of 4.11 and 4.67 — nonsense on a similarity scale.

Root cause, found by measuring rather than guessing: the BM25 tokenizer never filtered stopwords, so coincidental overlap on words like "how", "do", "i", "a" could out-score a genuine identifier match — measured for real, *"How do I set up a VPN on my home router?"* scored 10.67 against an unrelated OSHA chunk, higher than `V6001176`'s own 7.27 against the chunk that actually answers it. Filtering stopwords (`app/services/lexical_index.py`) plus a calibrated minimum-score floor (`bm25_min_score`, `app/core/config.py`) fixed every tested false positive but one. Re-running the affected out-of-scope questions with the fix applied:

| Question | Before (raw BM25 score) | After |
|---|---|---|
| sourdough bread | 4.11 (bypassed threshold, answered) | 0.55 (normal cosine score, no bypass) |
| seasonal allergies | 4.67 (bypassed threshold, answered) | 0.55 (normal cosine score, no bypass) |
| tire pressure / VPN / boiling point | not bypassed in this table, but bypassed at the raw `hybrid_retrieve()` level (3.86–4.84) | no bypass (all excluded by the floor) |

**The remaining gap, closed: routing lexical-only hits through the reranker's own judgment.** The scaffolding case above wasn't fixable by adjusting `bm25_min_score` further: it coincidentally shares a genuinely rare word — "requirement" (singular), appearing in exactly 1 of 185 chunks, identical corpus-wide rarity to `V6001176` — with an unrelated fire-extinguisher clause. BM25 has no way to distinguish "rare word reflecting real relevance" from "rare word shared by coincidence"; that's a semantic judgment, not a statistical one, the same kind of blind spot hybrid retrieval exists to fix on dense search's side, showing up as lexical search's mirror-image limitation.

The reranker's cross-encoder reads `(query, chunk)` jointly and can make exactly that semantic judgment — it's the same mechanism already proven against an analogous dense-search failure earlier in this project. Measuring real `(query, chunk)` pairs with `rerank_model`'s default cross-encoder found a clean, wide separation: every genuine identifier-lookup match scored between **-2.47 and +9.68**, every coincidental or irrelevant match (including scaffolding) scored between **-11.07 and -11.30** — an 8.6-point gap. `RetrievedChunk.rerank_score` now carries the cross-encoder's own score (previously computed, used to sort, then discarded), and a `bypass_relevance_filter=True` chunk must clear a calibrated `rerank_relevance_threshold` (-5.0) instead of passing unconditionally — but only when a reranker actually ran; without one, it still falls back to the original unconditional bypass, since there's no semantic judgment available to check. Re-validated against the real corpus: the scaffolding case's bypass is now correctly rejected, every previously-fixed false positive stays fixed, and genuine identifier lookups still resolve correctly.

One real bug this caught before it shipped: an early version of the fix still let a *rejected* bypass chunk back in, because `_filter_by_relevance`'s fallback check (`chunk.score >= relevance_threshold`) doesn't know a bypass chunk's `.score` is a raw BM25 value, not cosine — and 7.3 (a real BM25 score) trivially clears a 0.35 cosine threshold regardless of what the reranker actually judged. Live validation against the real corpus caught it immediately (the scaffolding chunk survived with a rejected `rerank_score` of -10.5); a unit test with a realistic BM25-magnitude score now guards against the same regression, documented in `test_low_rerank_score_excludes_bypass_chunk_despite_the_flag`'s docstring.

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

### Vector Store Backends: Chroma vs. pgvector

The architecture's central claim (see [Key design decisions](#key-design-decisions)) is that swapping the vector database means one new `VectorStore` subclass and one line of wiring — nothing in `RAGService`, `IngestionService`, hybrid retrieval, or reranking changes. That claim had never actually been tested until `PgVectorStore` (Postgres + the pgvector extension) existed as a second, real implementation to test it against.

Re-running the full 38-case harness against the same corpus, ingested fresh into pgvector with zero code changes — only `VECTOR_STORE_BACKEND=pgvector` and `POSTGRES_DSN` set:

| Metric | Chroma (cloud) | pgvector (cloud) |
|---|---|---|
| Hit-rate | 100.0% (32/32) | 100.0% (32/32) |
| Keyword pass-rate | 90.6% | 87.5% |
| Score gap (in - out) | 0.2257 | 0.4515 |

Hit-rate is identical — every question that finds its correct document on Chroma finds it on pgvector too, which is the number that actually matters for "did the abstraction hold." One honest difference: `mean_in_scope_score` is meaningfully lower on pgvector (0.63 vs. Chroma's typical ~0.82) even though relative *ranking* is preserved. The two backends don't compute cosine similarity identically in absolute magnitude — pgvector's `<=>` operator and Chroma's own distance metric are each internally consistent but not numerically interchangeable. Practical consequence: `relevance_threshold` is calibrated against Chroma's score distribution today and would need separate, real recalibration before being trusted as-is on pgvector — the same kind of caveat this project already carries for cloud-vs-local embedding models, now extended to vector store backends too.

**Schema note:** pgvector's `vector(n)` column needs a fixed dimension at table-creation time, unlike Chroma's fully lazy schema. Rather than add a settings knob for a number most users don't know off-hand, `PgVectorStore` creates its table lazily on the first real `upsert()` call, sized from the embeddings ingestion is already computing — no wasted "probe" API call just to learn a dimension. The embedding-compatibility guard (same invariant as Chroma's — see that guard's own description above) is stamped at that same point and checked eagerly on every subsequent connection.

**Tested for real, not mocked.** Unlike Chroma (embedded, no server, trivial to unit test), pgvector needs an actual running Postgres — a mock couldn't meaningfully validate real SQL or a real HNSW similarity index. `tests/test_pgvector_store.py` skips gracefully if no Postgres is reachable locally (`docker compose --profile pgvector up postgres` to enable it), so a plain `pytest -q` stays hermetic and green with zero setup either way — but CI always has a real Postgres+pgvector service container (`.github/workflows/ci.yml`), so this backend gets full, non-mocked coverage on every PR regardless of what's running on a given contributor's machine.

```bash
docker compose --profile pgvector up -d postgres   # local Postgres + pgvector
VECTOR_STORE_BACKEND=pgvector python -m scripts.ingest docs_corpus
VECTOR_STORE_BACKEND=pgvector python -m scripts.evaluate data/eval/dataset.yaml --skip-judge
```

## Configuration

All settings load from environment variables / `.env` and are validated at startup — see `app/core/config.py` for every knob (models, chunk size, top-k, paths). Secrets never live in code.

## Roadmap

- [x] Pluggable LLM/embedding providers — OpenAI, Ollama (fully local / air-gapped), sentence-transformers
- [x] CI-enforced quality gates — ruff, mypy, and a hermetic pytest suite required on every PR
- [x] Evaluation harness — retrieval hit-rate (dense/pre-rerank/post-rerank), score distribution, keyword + LLM-judge faithfulness (see [Evaluation & Results](#evaluation--results))
- [x] Configurable relevance threshold — refuses to answer, and skips the LLM call, when nothing retrieved clears the bar
- [x] Cross-encoder reranking stage — retrieve-then-rerank, off by default, its contribution measured by the eval harness rather than assumed
- [x] Grow the evaluation dataset alongside the ingested corpus — 38 cases across six documents, including cross-document near-misses and identifier-only lookups
- [x] Hybrid retrieval (BM25 + dense, fused via Reciprocal Rank Fusion) — off by default; see [Hybrid Retrieval](#hybrid-retrieval-dense--bm25-measured-not-assumed) for its measured effect and a known scoring limitation
- [x] Recalibrate lexical-only relevance scoring — stopword filtering + a calibrated `bm25_min_score` floor, fixing every tested false positive but one; the remaining gap (a coincidental rare-word match) is a semantic-vs-statistical limit of BM25 itself, not a tunable parameter — see [Hybrid Retrieval](#hybrid-retrieval-dense--bm25-measured-not-assumed)
- [x] Route lexical-only hits through the reranker's own semantic judgment — closes the remaining gap above (measured: an 8.6-point score separation between genuine and coincidental matches); only strengthens the gate when `RERANK_ENABLED=true`, unconditional bypass otherwise — see [Hybrid Retrieval](#hybrid-retrieval-dense--bm25-measured-not-assumed)
- [x] pgvector backend behind the existing `VectorStore` interface — real, CI-tested (not mocked) second backend proving the abstraction; see [Vector Store Backends](#vector-store-backends-chroma-vs-pgvector)
- [ ] Conversation memory / multi-turn queries
- [ ] Auth + multi-tenant corpora
- [ ] React frontend against the same API
- [ ] Deploy and record a live demo

## License

MIT
