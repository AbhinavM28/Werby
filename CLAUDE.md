# CLAUDE.md — Working Agreement for Werby

This file briefs any AI coding assistant on how Werby is built and how to work
in it. Read it fully before proposing changes. When in doubt, ask rather than
assume.

## What Werby is

Werby is an AI Engineering Copilot for warehouse and industrial engineers. It
uses Retrieval-Augmented Generation (RAG) to answer natural-language questions
against ingested engineering documentation (manuals, SOPs, spec sheets, safety
procedures) and returns grounded, source-cited answers. This is a portfolio
project meant to demonstrate professional software and AI engineering, so code
quality, clarity, and sound architecture matter as much as functionality.

## How I want you to work with me

- I am learning. When you make or propose a change, explain the underlying
  concept and the "why" — not just the "what." Favor teaching over speed.
- Explain tradeoffs when a decision has more than one reasonable option.
- Never invent placeholder or stub code when a real implementation is possible.
- Propose changes as diffs I review before applying. Do not accept your own
  changes automatically. If I can't explain a diff, I shouldn't merge it.
- If a request is ambiguous, ask one clarifying question before coding.
- Point out when something I ask for conflicts with the architecture below.

## Architecture — the rules that must not be violated

Werby uses a layered architecture. Dependencies point inward and downward,
never sideways or up.

- **API layer** (`app/api/`): HTTP only. Routes validate input, call a service,
  and map the result to a response schema. Routes contain NO business logic.
- **Service layer** (`app/services/`): all business logic. Services know
  NOTHING about HTTP, FastAPI, or request/response objects. They must be usable
  from the API, the CLI, or a future worker with no changes.
- **Core layer** (`app/core/`): configuration, logging, domain exceptions.
- **Schemas** (`app/schemas/`): Pydantic models that define the public API
  contract. Kept separate from service internals so each can evolve alone.

Key seams (do not collapse these — they are the point of the design):

- **VectorStore** is an abstract base class (`app/services/vector_store.py`).
  Everything depends on the interface, never on ChromaDB directly. A new
  backend (pgvector, Qdrant) means one new subclass, not edits across the app.
- **LLMProvider / EmbeddingProvider** are ABCs (`app/services/providers/`).
  The AI backend (OpenAI vs. local Ollama vs. sentence-transformers) is a
  `.env` choice, not a code change. This enables fully-local, air-gapped
  deployment for proprietary documentation.
- **Domain exceptions**: services raise `WerbyError` subclasses (HTTP-agnostic).
  The single handler in `app/main.py` maps each to an HTTP status. To add an
  error, add the exception and map it once — do not scatter HTTP codes.
- **Composition root**: `app/api/deps.py` is the ONLY place concrete classes
  are chosen and wired. Provider/store selection from settings happens here and
  nowhere else.

Invariant to protect: embeddings from different models occupy incompatible
vector spaces. The Chroma collection is stamped with its embedding model, and
the app refuses to start on a mismatch. Never weaken this guard.

## Conventions

- **Python 3.11+**, full type hints on function signatures.
- **Formatting/linting**: ruff. Code must pass `ruff check app tests scripts`.
- **Docstrings**: explain *why*, not just *what*. Module docstrings state the
  module's role in the architecture.
- **Config**: all settings flow through `app/core/config.py` (pydantic-settings).
  No stray `os.getenv` calls. Secrets only in `.env` (gitignored), never in code.
- **Logging**: `logger = logging.getLogger(__name__)` per module. No `print()`.
- **Tests must be hermetic**: no network, no real API keys, no dependence on a
  developer's local `.env`. Use mocks/fakes and pytest's `tmp_path`/`monkeypatch`.

## Git & workflow

- Never commit directly to `main`. `main` is protected and requires green CI.
- Branch per unit of work: `type/kebab-case` (e.g. `feat/evaluation-harness`,
  `fix/pdf-encoding`, `chore/ci-cache`, `docs/architecture`).
- Conventional Commits: `type(scope): imperative summary`
  (`feat`, `fix`, `chore`, `docs`, `refactor`, `test`).
- Before proposing a commit, run BOTH gates locally and confirm they pass:
  `ruff check app tests scripts` and `pytest -q`.
- Open a PR, self-review the diff in Files Changed, squash-merge, delete branch.

## Environment

- Always work inside the project virtualenv. Activate with
  `source .venv/Scripts/activate` (Windows/Git Bash). The prompt shows `(.venv)`
  when active. `ModuleNotFoundError` almost always means it isn't activated.
- Install everything with `pip install -e ".[dev]"`.
- Only `app` is the installable package (see `pyproject.toml`); `frontend`,
  `scripts`, and `data` are intentionally excluded.

## Commands

- Run API: `uvicorn app.main:app --reload --port 8000` (docs at `/docs`)
- Run frontend: `streamlit run frontend/streamlit_app.py`
- Test: `pytest -q`
- Lint: `ruff check app tests scripts`
- Bulk ingest: `python -m scripts.ingest ./path/to/docs`

## Current roadmap (next up first)

1. Evaluation harness — measure retrieval hit-rate and answer faithfulness.
2. Hybrid retrieval (BM25 + dense) and a reranking stage.
3. pgvector backend behind the existing VectorStore interface.
4. Conversation memory / multi-turn queries.