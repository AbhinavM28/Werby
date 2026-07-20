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
    max_context_chars: int = 12_000  # hard cap on prompt context size

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
