"""Tests for provider selection and the embedding-compatibility guard.

These tests exercise the composition root with different settings and prove
the vector store refuses to mix embedding spaces -- the silent-corruption
bug this architecture is designed to prevent.
"""

import pytest

import app.api.deps as deps
from app.core.config import get_settings
from app.core.exceptions import ConfigurationError, VectorStoreError
from app.services.providers.ollama_provider import (
    OllamaEmbeddingProvider,
    OllamaLLMProvider,
)
from app.services.vector_store import ChromaVectorStore


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    """Each test gets fresh settings and fresh provider singletons."""
    get_settings.cache_clear()
    deps.get_embedding_provider.cache_clear()
    deps.get_llm_provider.cache_clear()
    deps.get_vector_store.cache_clear()
    yield
    get_settings.cache_clear()
    deps.get_embedding_provider.cache_clear()
    deps.get_llm_provider.cache_clear()
    deps.get_vector_store.cache_clear()


def test_openai_provider_without_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    get_settings.cache_clear()

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        deps.get_llm_provider()


def test_ollama_providers_need_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    get_settings.cache_clear()

    assert isinstance(deps.get_llm_provider(), OllamaLLMProvider)
    provider = deps.get_embedding_provider()
    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.model.startswith("ollama/")


def test_vector_store_rejects_embedding_model_mismatch(tmp_path):
    # First boot: collection is stamped with model A.
    store = ChromaVectorStore(
        persist_dir=tmp_path,
        collection_name="guard_test",
        embedding_model="openai/text-embedding-3-small",
    )
    assert store.count() == 0

    # Same model again: fine (idempotent).
    ChromaVectorStore(
        persist_dir=tmp_path,
        collection_name="guard_test",
        embedding_model="openai/text-embedding-3-small",
    )

    # Different model: must fail loudly, not corrupt retrieval silently.
    with pytest.raises(VectorStoreError, match="mismatch"):
        ChromaVectorStore(
            persist_dir=tmp_path,
            collection_name="guard_test",
            embedding_model="ollama/nomic-embed-text",
        )
