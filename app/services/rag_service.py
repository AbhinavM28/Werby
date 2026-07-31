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
from dataclasses import dataclass, replace

from app.core.exceptions import EmptyCorpusError
from app.services.document_processor import DocumentProcessor
from app.services.lexical_index import LexicalIndex
from app.services.llm_service import LLMService, build_context
from app.services.providers.base import EmbeddingProvider, Reranker
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


def reciprocal_rank_fusion(
    dense: list[RetrievedChunk], lexical: list[RetrievedChunk], k: int = 60
) -> list[RetrievedChunk]:
    """Merge two independently-ranked chunk lists by rank position, not score.

    Dense search's ``score`` is a cosine similarity in [0, 1]; BM25's score
    is an unbounded, corpus- and query-dependent sum with no fixed scale.
    Averaging or summing those directly would be meaningless -- whichever
    scale happens to produce numerically larger values would dominate
    regardless of actual relevance, like comparing meters to kilograms. RRF
    sidesteps this by discarding raw scores entirely and fusing on rank
    position (1st, 2nd, 3rd...), which is comparable across any two ranking
    systems no matter how each computes its own scores:

        RRF_score(chunk) = sum, over every list containing it, of 1 / (k + rank)

    ``k`` is a damping constant: small k makes rank 1 dominate sharply over
    rank 2; larger k flattens the curve so a chunk ranked moderately in BOTH
    lists can outscore one ranked #1 in only one. 60 is the original RRF
    paper's value, reused as-is by Elasticsearch, Weaviate, and others --
    empirically robust enough that it's rarely worth tuning per-domain.

    Chunks found by both lists keep dense search's real cosine score (still
    meaningful for relevance filtering). Chunks found ONLY by lexical search
    get ``bypass_relevance_filter=True`` -- see that field's docstring on
    ``RetrievedChunk`` for why.
    """

    def _key(chunk: RetrievedChunk) -> tuple[str, int]:
        return (chunk.source_document, chunk.chunk_index)

    scores: dict[tuple[str, int], float] = {}
    representative: dict[tuple[str, int], RetrievedChunk] = {}

    for rank, chunk in enumerate(dense, start=1):
        chunk_key = _key(chunk)
        scores[chunk_key] = scores.get(chunk_key, 0.0) + 1.0 / (k + rank)
        representative.setdefault(chunk_key, chunk)

    for rank, chunk in enumerate(lexical, start=1):
        chunk_key = _key(chunk)
        scores[chunk_key] = scores.get(chunk_key, 0.0) + 1.0 / (k + rank)
        if chunk_key not in representative:
            representative[chunk_key] = replace(chunk, bypass_relevance_filter=True)

    ranked_keys = sorted(scores, key=lambda ck: scores[ck], reverse=True)
    return [representative[ck] for ck in ranked_keys]


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
        reranker: Reranker | None = None,
        retrieve_n: int = 20,
        lexical_index: LexicalIndex | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._llm = llm
        self._default_top_k = default_top_k
        self._max_context_chars = max_context_chars
        self._relevance_threshold = relevance_threshold
        self._reranker = reranker
        self._retrieve_n = retrieve_n
        self._lexical_index = lexical_index
        self._rrf_k = rrf_k

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

    def hybrid_retrieve(
        self, question: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Dense search fused with BM25 lexical search via ``reciprocal_rank_fusion``.

        Split out from ``retrieve`` for the same reason ``rerank`` is split
        from ``query``: the evaluation harness needs dense-only and
        dense+lexical as two independent, directly-callable checkpoints, to
        measure what hybrid retrieval actually adds -- mirroring how
        pre-/post-rerank are already measured independently.

        A graceful no-op when no lexical index is configured
        (``lexical_index=None``, the default): returns ``retrieve()``'s
        output untouched, so existing dense-only behavior is unaffected
        until hybrid retrieval is explicitly enabled.
        """
        dense = self.retrieve(question, top_k=top_k)
        if self._lexical_index is None:
            return dense

        k = top_k or self._default_top_k
        lexical = self._lexical_index.search(question, top_k=k)
        fused = reciprocal_rank_fusion(dense, lexical, k=self._rrf_k)
        logger.info(
            "Hybrid-fused %d dense + %d lexical candidate(s) into %d for "
            "question: %.80s",
            len(dense), len(lexical), min(len(fused), k), question,
        )
        return fused[:k]

    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Reorder retrieve()'s candidates by cross-encoder relevance.

        Split out from ``query`` for the same reason ``retrieve`` is: the
        evaluation harness needs to measure retrieval quality at two
        checkpoints independently -- pre-rerank (was the right chunk
        anywhere in the wide candidate pool?) and post-rerank (did reranking
        promote it into the final set that actually reaches the LLM?) --
        without reimplementing this logic itself.

        A graceful no-op when no reranker is configured (``reranker=None``,
        the default): just truncates to top_k, so callers see identical
        behavior to before this feature existed. This is deliberately the
        *only* switch -- there's no separate "rerank_enabled" flag on this
        class, because that would allow the nonsensical state of "enabled"
        with nothing to enable.
        """
        k = top_k or self._default_top_k
        if self._reranker is None or not candidates:
            return candidates[:k]
        reranked = self._reranker.rerank(question, candidates, top_k=k)
        logger.info(
            "Reranked %d candidate(s) to top %d for question: %.80s",
            len(candidates), len(reranked), question,
        )
        return reranked

    def _filter_by_relevance(
        self, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Discard chunks scoring below the configured relevance threshold.

        Kept as a separate step from ``retrieve`` (see that method's
        docstring) so the raw retrieval measurement callers like the
        evaluation harness depend on stays untouched by this business-logic
        cutoff. A chunk scoring exactly at the threshold passes -- the bar is
        inclusive, not exclusive.

        Chunks with ``bypass_relevance_filter=True`` (lexical-only hybrid
        hits -- see that field's docstring on ``RetrievedChunk``) always
        survive this step regardless of score, since their score is a raw
        BM25 value, not the cosine similarity this threshold was calibrated
        against.
        """
        survivors = [
            c
            for c in chunks
            if c.bypass_relevance_filter or c.score >= self._relevance_threshold
        ]
        if len(survivors) < len(chunks):
            logger.info(
                "Filtered %d of %d retrieved chunks below relevance "
                "threshold %.2f",
                len(chunks) - len(survivors), len(chunks), self._relevance_threshold,
            )
        return survivors

    def query(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """Run the full hybrid-retrieve-then-rerank-then-generate pipeline.

        Retrieval itself (``hybrid_retrieve``) transparently fuses dense and
        lexical search when a lexical index is configured, or is a pure
        dense-only no-op otherwise -- everything downstream (widening for
        the reranker, reranking, threshold filtering) is unaffected either
        way and unchanged from before this feature existed.

        When a reranker is configured, retrieval widens to ``retrieve_n``
        candidates first -- a cross-encoder can only promote a chunk that's
        in the pool it's given, not recover one dense search never surfaced
        at all -- then reranks down to ``top_k`` before anything else
        happens. Without a reranker, this widening doesn't happen and
        behavior is identical to before this feature existed.

        Chunks below ``relevance_threshold`` are discarded before context is
        built. If none survive, this returns a refusal instead of calling the
        LLM: in an industrial safety context, a wrong answer generated from
        irrelevant context is worse than an honest "I don't know", and
        skipping the call also saves its cost and latency outright.

        Raises:
            EmptyCorpusError: If no documents have been ingested yet.
        """
        started = time.perf_counter()

        k = top_k or self._default_top_k
        pool_size = self._retrieve_n if self._reranker is not None else k
        candidates = self.hybrid_retrieve(question, top_k=pool_size)
        ranked = self.rerank(question, candidates, top_k=k)
        relevant = self._filter_by_relevance(ranked)

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
