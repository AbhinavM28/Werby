"""Application configuration.

All runtime configuration flows through this single module. Settings are loaded
from environment variables (and a local ``.env`` file in development) and
validated at startup by Pydantic. This gives us:

* One source of truth for configuration (no scattered ``os.getenv`` calls).
* Type validation at boot -- a missing API key fails fast with a clear error
  instead of a cryptic exception deep inside a request handler.
* Free documentation: this class *is* the list of every knob the app has.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, sourced from environment variables.

    Field names map to env vars case-insensitively, e.g. ``openai_api_key``
    is populated from ``OPENAI_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Werby"
    app_version: str = "0.1.0"
    environment: str = "development"  # development | staging | production
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    # --- Provider selection ---
    # "openai" = hosted frontier models. "ollama"/"local" = fully local
    # inference for proprietary or air-gapped deployments (no data leaves
    # the machine, no API costs). See app/services/providers/.
    llm_provider: Literal["openai", "ollama"] = "openai"
    embedding_provider: Literal["openai", "ollama", "local"] = "openai"

    # --- OpenAI (required only when an "openai" provider is selected) ---
    openai_api_key: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Ollama (local inference server) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- sentence-transformers (in-process local embeddings) ---
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # --- Generation parameters (apply to whichever LLM provider is active) ---
    llm_temperature: float = 0.1  # low temperature: factual, grounded answers
    llm_max_tokens: int = 1024

    # --- Vector store (ChromaDB) ---
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection: str = "werby_documents"

    # --- Ingestion / chunking ---
    chunk_size: int = 1000       # characters per chunk
    chunk_overlap: int = 200     # overlap preserves context across boundaries
    max_upload_mb: int = 25

    # --- Retrieval ---
    retrieval_top_k: int = 5
    max_context_chars: int = 12_000
    # Minimum similarity score (see RetrievedChunk.score, [0, 1]) a chunk must
    # clear to be used as context. Below this, RAGService.query() refuses to
    # answer rather than generate from irrelevant context -- a wrong industrial
    # spec is worse than an honest "I don't know". 0.35 is a deliberately
    # provisional floor: the eval harness measured in-scope questions
    # averaging 0.85 and one out-of-scope question at 0.55, so this default
    # won't filter anything from what's been measured so far. Tune it upward
    # with `python -m scripts.evaluate` once a wider score distribution
    # (including near-miss, not just obviously-unrelated, questions) is
    # available -- see README's Evaluation section.
    relevance_threshold: float = 0.35  # hard cap on prompt context size

    # --- Reranking ---
    # Cross-encoders score (query, chunk) jointly and catch cases pure vector
    # similarity misses -- e.g. a short, specific document losing to a long,
    # vocabulary-dense one on a query about that specific topic (a real,
    # measured failure: see app/services/providers/local_reranker.py). Off
    # by default so existing retrieval behavior is unchanged until explicitly
    # enabled, and enabling it requires the optional 'local' extra.
    rerank_enabled: bool = False
    # Candidate pool size fed to the reranker. Generous on purpose: the
    # reranker can only promote a chunk that's IN the pool -- it can't
    # recover one dense search never surfaced at all. Must be >= top_k (see
    # the validator below) or the pool would be smaller than what's supposed
    # to survive it.
    retrieve_n: int = 20
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Hybrid retrieval (dense + BM25 lexical, fused) ---
    # Off by default, same reasoning as rerank_enabled: existing dense-only
    # behavior is unchanged until explicitly opted in. See
    # app/services/lexical_index.py and reciprocal_rank_fusion() in
    # rag_service.py for the full design.
    hybrid_enabled: bool = False
    # Reciprocal Rank Fusion damping constant -- 60 is the original RRF
    # paper's value, reused as-is by Elasticsearch/Weaviate/etc. Rarely
    # needs tuning; see reciprocal_rank_fusion()'s docstring for why.
    rrf_k: int = 60

    @model_validator(mode="after")
    def _validate_retrieve_n(self) -> "Settings":
        """retrieve_n must be >= retrieval_top_k.

        A candidate pool smaller than the final top_k can't fill top_k slots
        -- RAGService.rerank() would silently return fewer chunks than
        requested instead of the configured amount. Fail loudly at startup
        instead of letting that surface later as an unexplained drop in
        context size.
        """
        if self.retrieve_n < self.retrieval_top_k:
            raise ValueError(
                f"retrieve_n ({self.retrieve_n}) must be >= retrieval_top_k "
                f"({self.retrieval_top_k}): a smaller candidate pool would "
                "silently under-return fewer chunks than requested."
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton.

    ``lru_cache`` ensures the ``.env`` file and environment are parsed exactly
    once per process. In tests, call ``get_settings.cache_clear()`` after
    monkeypatching env vars.
    """
    return Settings()
