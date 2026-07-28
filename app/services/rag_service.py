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

NO_RELEVANT_CONTEXT_MESSAGE = (
    "No documentation in the corpus is relevant enough to answer this "
    "question with confidence. Try rephrasing, or ingest documentation "
    "that covers this topic."
)


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
    sufficient_context: bool


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
        relevance_threshold: float = 0.35,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._llm = llm
        self._default_top_k = default_top_k
        self._max_context_chars = max_context_chars
        self._relevance_threshold = relevance_threshold

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """The read path's retrieval step, isolated from generation.

        Split out from ``query`` so callers that only need retrieval quality
        (the evaluation harness, see ``app/services/evaluation.py``) can reuse
        the exact same embed-and-search logic without paying for an LLM call.

        Raises:
            EmptyCorpusError: If no documents have been ingested yet.
        """
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
        return retrieved

    def _filter_by_relevance(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Discard chunks scoring below the configured relevance threshold.

        Kept as a separate step from ``retrieve`` (see that method's
        docstring) so the raw retrieval measurement callers like the
        evaluation harness depend on stays untouched by this business-logic
        cutoff. A chunk scoring exactly at the threshold passes -- the bar is
        inclusive, not exclusive.
        """
        survivors = [c for c in chunks if c.score >= self._relevance_threshold]
        if len(survivors) < len(chunks):
            logger.info(
                "Filtered %d of %d retrieved chunks below relevance "
                "threshold %.2f",
                len(chunks) - len(survivors), len(chunks), self._relevance_threshold,
            )
        return survivors

    def query(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Run the full retrieve-then-generate pipeline.

        Chunks below ``relevance_threshold`` are discarded before context is
        built. If none survive, this returns a refusal instead of calling the
        LLM: in an industrial safety context, a wrong answer generated from
        irrelevant context is worse than an honest "I don't know", and
        skipping the call also saves its cost and latency outright.

        Raises:
            EmptyCorpusError: If no documents have been ingested yet.
        """
        started = time.perf_counter()

        retrieved = self.retrieve(question, top_k)
        relevant = self._filter_by_relevance(retrieved)

        if not relevant:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "No chunks cleared the relevance threshold (%.2f) for "
                "question: %.80s",
                self._relevance_threshold, question,
            )
            return RAGAnswer(
                answer=NO_RELEVANT_CONTEXT_MESSAGE,
                sources=[],
                model=self._llm.model,
                latency_ms=latency_ms,
                sufficient_context=False,
            )

        context = build_context(relevant, max_chars=self._max_context_chars)
        answer = self._llm.generate_answer(question, context)

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Answered in %d ms using %s", latency_ms, self._llm.model)
        return RAGAnswer(
            answer=answer,
            sources=relevant,
            model=self._llm.model,
            latency_ms=latency_ms,
            sufficient_context=True,
        )
