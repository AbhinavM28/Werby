"""Provider interfaces for LLM inference and embeddings.

Why this exists (the architectural argument):

Werby's target users -- warehouse and industrial engineers -- often work with
proprietary or export-controlled documentation that must never leave their
network. That makes "which AI backend?" a *deployment* decision, not a code
decision. These ABCs make the backend a configuration detail:

    LLM_PROVIDER=openai          # hosted, frontier quality
    LLM_PROVIDER=ollama          # fully local, air-gapped capable

This mirrors the ``VectorStore`` abstraction exactly: high-level RAG logic
depends on these interfaces; concrete providers live in sibling modules; the
composition root (``app/api/deps.py``) picks the implementation from Settings.
"""

from abc import ABC, abstractmethod

from app.services.vector_store import RetrievedChunk


class EmbeddingProvider(ABC):
    """Converts text into dense vectors for similarity search.

    CRITICAL INVARIANT: embeddings from different models live in different,
    incompatible vector spaces. A query embedded with model A is meaningless
    when searched against a corpus embedded with model B -- retrieval will
    silently return garbage rather than erroring. The vector store guards
    against this by stamping its collection with ``model`` (see
    ``ChromaVectorStore``); switching embedding providers therefore requires
    re-ingesting the corpus.
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the underlying embedding model (used as the
        compatibility stamp on the vector store collection)."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, preserving input order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (default: delegate to embed_texts)."""
        return self.embed_texts([text])[0]


class LLMProvider(ABC):
    """Generates text from a (system, user) prompt pair.

    Deliberately minimal: prompt *construction* is domain logic and stays in
    ``llm_service.py``; providers only own transport, retries, and
    provider-specific parameters (temperature, token limits).
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the underlying chat model (surfaced in responses)."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's completion for the given prompts.

        Raises:
            LLMServiceError: On persistent provider failure.
        """


class Reranker(ABC):
    """Reorders retrieved chunks by relevance to a query.

    Same argument as ``EmbeddingProvider``/``LLMProvider``: which reranking
    backend runs (a local cross-encoder vs. a future hosted API) is a
    deployment decision, not a code decision, so it lives behind an
    interface RAGService depends on rather than a concrete class.
    """

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return candidates reordered by relevance to query, truncated to top_k."""
