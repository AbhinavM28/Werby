"""Dependency wiring (the "composition root").

This is the ONLY place in the codebase that decides which concrete classes to
use. Routes declare *what* they need via ``Depends(...)``; this module decides
*how* those things are built.

With the provider abstraction, this file is also where ``.env`` becomes
architecture: ``LLM_PROVIDER`` and ``EMBEDDING_PROVIDER`` select which
implementation gets constructed. Note that misconfiguration (e.g. choosing
OpenAI without an API key) fails *here, at startup, with an actionable
message* -- not deep inside the first user request.

Heavy objects (API clients, the Chroma connection, loaded local models) are
cached with ``lru_cache`` so they're constructed once per process. The
lightweight orchestrators (RAGService, IngestionService) are built per request
from those cached singletons, keeping request handling stateless.
"""

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError
from app.services.document_processor import DocumentProcessor
from app.services.llm_service import LLMService
from app.services.providers.base import EmbeddingProvider, LLMProvider
from app.services.rag_service import IngestionService, RAGService
from app.services.vector_store import ChromaVectorStore, VectorStore


def _require_openai_key(settings: Settings, needed_for: str) -> str:
    if not settings.openai_api_key:
        raise ConfigurationError(
            f"OPENAI_API_KEY is required because {needed_for} is set to "
            "'openai'. Add it to your .env, or switch to a local provider "
            "(LLM_PROVIDER=ollama, EMBEDDING_PROVIDER=ollama|local) to run "
            "without any external API."
        )
    return settings.openai_api_key


@lru_cache
def _get_openai_client():  # lazy import: only needed on the OpenAI path
    from openai import OpenAI

    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()

    if settings.embedding_provider == "openai":
        from app.services.providers.openai_provider import OpenAIEmbeddingProvider

        _require_openai_key(settings, "EMBEDDING_PROVIDER")
        return OpenAIEmbeddingProvider(
            client=_get_openai_client(), model=settings.embedding_model
        )

    if settings.embedding_provider == "ollama":
        from app.services.providers.ollama_provider import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )

    # "local" -- in-process sentence-transformers (optional heavy dependency)
    from app.services.providers.local_embeddings import (
        SentenceTransformersEmbeddingProvider,
    )

    return SentenceTransformersEmbeddingProvider(settings.local_embedding_model)


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()

    if settings.llm_provider == "openai":
        from app.services.providers.openai_provider import OpenAILLMProvider

        _require_openai_key(settings, "LLM_PROVIDER")
        return OpenAILLMProvider(
            client=_get_openai_client(),
            model=settings.chat_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    from app.services.providers.ollama_provider import OllamaLLMProvider

    return OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
        # Stamp the collection with the active embedding model so a provider
        # switch fails loudly instead of silently corrupting retrieval.
        embedding_model=get_embedding_provider().model,
    )


def get_llm_service() -> LLMService:
    return LLMService(provider=get_llm_provider())


def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        processor=DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embeddings=get_embedding_provider(),
        store=get_vector_store(),
    )


def get_rag_service() -> RAGService:
    settings = get_settings()
    return RAGService(
        embeddings=get_embedding_provider(),
        store=get_vector_store(),
        llm=get_llm_service(),
        default_top_k=settings.retrieval_top_k,
        max_context_chars=settings.max_context_chars,
    )


def get_app_settings() -> Settings:
    return get_settings()
