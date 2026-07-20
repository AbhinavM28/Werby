"""Orchestration services: ingestion and query.

These two classes are the "use cases" of the application. Each composes the
lower-level services (processor, embeddings, vector store, LLM) into one
business operation. They receive their dependencies through the constructor
(dependency injection) rather than creating them -- which is what makes them
testable with mocks and reconfigurable at the wiring layer.

The full RAG pipeline, end to end:

    INGESTION (write path)
    file bytes -> extract text -> chunk -> embed chunks -> upsert vectors

    QUERY (read path)
    question -> embed question -> similarity search -> assemble context
             -> LLM generates grounded answer -> answer + cited sources
"""

import logging
import time
from dataclasses import dataclass

from app.core.exceptions import EmptyCorpusError
from app.services.document_processor import DocumentProcessor
from app.services.llm_service import LLMService, build_context
from app.services.providers.base import EmbeddingProvider
from app.services.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    document: str
    chunks_created: int
    characters: int


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk]
    model: str
    latency_ms: int


class IngestionService:
    """Write path: turn an uploaded document into searchable vectors."""

    def __init__(
        self,
        processor: DocumentProcessor,
        embeddings: EmbeddingProvider,
        store: VectorStore,
    ) -> None:
        self._processor = processor
        self._embeddings = embeddings
        self._store = store

    def ingest(self, filename: str, content: bytes) -> IngestionResult:
        """Process, embed, and store a document. Idempotent per filename."""
        chunks = self._processor.process(filename, content)
        vectors = self._embeddings.embed_texts([c.text for c in chunks])
        self._store.upsert(chunks, vectors)
        total_chars = sum(len(c.text) for c in chunks)
        logger.info(
            "Ingested '%s' (%d chunks, %d chars)",
            filename, len(chunks), total_chars,
        )
        return IngestionResult(
            document=filename,
            chunks_created=len(chunks),
            characters=total_chars,
        )


class RAGService:
    """Read path: answer a question grounded in the ingested corpus."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        llm: LLMService,
        default_top_k: int = 5,
        max_context_chars: int = 12_000,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._llm = llm
        self._default_top_k = default_top_k
        self._max_context_chars = max_context_chars

    def query(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Run the full retrieve-then-generate pipeline.

        Raises:
            EmptyCorpusError: If no documents have been ingested yet.
        """
        started = time.perf_counter()

        if self._store.count() == 0:
            raise EmptyCorpusError(
                "No documents have been ingested yet. Upload documentation "
                "before querying."
            )

        k = top_k or self._default_top_k
        query_vector = self._embeddings.embed_query(question)
        retrieved = self._store.search(query_vector, top_k=k)
        logger.info(
            "Retrieved %d chunks for question: %.80s", len(retrieved), question
        )

        context = build_context(retrieved, max_chars=self._max_context_chars)
        answer = self._llm.generate_answer(question, context)

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Answered in %d ms using %s", latency_ms, self._llm.model)
        return RAGAnswer(
            answer=answer,
            sources=retrieved,
            model=self._llm.model,
            latency_ms=latency_ms,
        )
