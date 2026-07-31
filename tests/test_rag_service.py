"""Unit tests for RAG orchestration using an in-memory fake vector store.

A hand-rolled fake (rather than MagicMock everywhere) keeps tests readable
and verifies the VectorStore *interface* is actually sufficient.
"""

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import EmptyCorpusError
from app.services.document_processor import DocumentChunk
from app.services.lexical_index import LexicalIndex
from app.services.providers.base import Reranker
from app.services.rag_service import (
    NO_RELEVANT_CONTEXT_MESSAGE,
    RAGService,
    reciprocal_rank_fusion,
)
from app.services.vector_store import RetrievedChunk, VectorStore


class FakeVectorStore(VectorStore):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def upsert(self, chunks: list[DocumentChunk], embeddings) -> None:
        raise NotImplementedError

    def search(self, query_embedding, top_k: int) -> list[RetrievedChunk]:
        return self._chunks[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def list_documents(self) -> list[str]:
        return sorted({c.source_document for c in self._chunks})

    def delete_document(self, source_document: str) -> int:
        return 0

    def get_all_chunks(self) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id=f"{c.source_document}::chunk_{c.chunk_index}",
                text=c.text,
                source_document=c.source_document,
                chunk_index=c.chunk_index,
            )
            for c in self._chunks
        ]


class FakeReranker(Reranker):
    """Predictable, deterministic rule (reverse order) -- never loads a
    real cross-encoder. Records each call's (query, pool size, top_k) so
    tests can assert what RAGService actually asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        self.calls.append((query, len(candidates), top_k))
        return list(reversed(candidates))[:top_k]


class FakeLexicalIndex(LexicalIndex):
    """Predictable, canned results -- never loads real rank-bm25. Records
    each call's (query, top_k) so tests can assert what hybrid_retrieve()
    actually asked for."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query_text, top_k))
        return self._results[:top_k]


@pytest.fixture
def embeddings() -> MagicMock:
    mock = MagicMock()
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def llm() -> MagicMock:
    mock = MagicMock()
    mock.generate_answer.return_value = "Rated load is 2000 kg [Source 1]."
    mock.model = "gpt-test"
    return mock


def test_query_returns_answer_and_sources(embeddings, llm) -> None:
    store = FakeVectorStore(
        [RetrievedChunk("load: 2000 kg", "crane.pdf", 0, 0.95)]
    )
    service = RAGService(embeddings, store, llm, default_top_k=5)

    result = service.query("What is the rated load?")

    assert "2000 kg" in result.answer
    assert result.sources[0].source_document == "crane.pdf"
    assert result.model == "gpt-test"
    embeddings.embed_query.assert_called_once_with("What is the rated load?")


def test_query_on_empty_corpus_raises(embeddings, llm) -> None:
    service = RAGService(embeddings, FakeVectorStore([]), llm)
    with pytest.raises(EmptyCorpusError):
        service.query("anything")
    llm.generate_answer.assert_not_called()


def test_top_k_override_limits_retrieval(embeddings, llm) -> None:
    store = FakeVectorStore(
        [RetrievedChunk(f"chunk {i}", "doc.pdf", i, 0.9) for i in range(10)]
    )
    service = RAGService(embeddings, store, llm, default_top_k=5)
    result = service.query("q", top_k=2)
    assert len(result.sources) == 2


def test_query_keeps_chunks_above_threshold(embeddings, llm) -> None:
    store = FakeVectorStore(
        [RetrievedChunk("load: 2000 kg", "crane.pdf", 0, 0.62)]
    )
    service = RAGService(embeddings, store, llm, relevance_threshold=0.5)

    result = service.query("What is the rated load?")

    assert result.sufficient_context is True
    assert result.sources[0].score == 0.62
    assert "2000 kg" in result.answer
    llm.generate_answer.assert_called_once()


def test_query_refuses_when_all_chunks_below_threshold(embeddings, llm) -> None:
    store = FakeVectorStore(
        [
            RetrievedChunk("unrelated", "doc.pdf", 0, 0.41),
            RetrievedChunk("also unrelated", "doc.pdf", 1, 0.30),
        ]
    )
    service = RAGService(embeddings, store, llm, relevance_threshold=0.5)

    result = service.query("What is the rated load?")

    assert result.sufficient_context is False
    assert result.sources == []
    assert result.answer == NO_RELEVANT_CONTEXT_MESSAGE
    llm.generate_answer.assert_not_called()


def test_query_boundary_score_equal_to_threshold_passes(embeddings, llm) -> None:
    """The bar is inclusive: a chunk scoring exactly at the threshold
    clears it, rather than being discarded by a strict '>' comparison."""
    store = FakeVectorStore([RetrievedChunk("exact match", "doc.pdf", 0, 0.5)])
    service = RAGService(embeddings, store, llm, relevance_threshold=0.5)

    result = service.query("q")

    assert result.sufficient_context is True
    assert len(result.sources) == 1


def test_no_reranker_leaves_query_output_unchanged(embeddings, llm) -> None:
    """reranker=None (the default) must be a true no-op: query()'s final
    sources must be identical -- in order, content, and score -- to
    retrieve()'s raw output on the same instance. Not "runs without
    error": a direct equality check against the untouched pre-feature
    method, proving nothing about existing behavior moved."""
    chunks = [
        RetrievedChunk("load spec", "crane.pdf", 0, 0.91),
        RetrievedChunk("training reqs", "pit.pdf", 1, 0.72),
        RetrievedChunk("inspection interval", "crane.pdf", 2, 0.58),
    ]
    store = FakeVectorStore(chunks)
    # reranker not passed -> defaults to None
    service = RAGService(embeddings, store, llm, default_top_k=3)

    baseline = service.retrieve("q")
    result = service.query("q")

    assert result.sources == baseline
    assert [c.chunk_index for c in result.sources] == [0, 1, 2]
    assert [c.score for c in result.sources] == [0.91, 0.72, 0.58]


def test_rerank_reorders_via_configured_reranker(embeddings, llm) -> None:
    chunks = [
        RetrievedChunk(f"chunk {i}", "doc.pdf", i, 0.9 - i * 0.01) for i in range(4)
    ]
    store = FakeVectorStore(chunks)
    reranker = FakeReranker()
    service = RAGService(
        embeddings, store, llm, reranker=reranker, retrieve_n=4, default_top_k=4
    )

    result = service.query("q")

    # FakeReranker reverses order -- chunk 3 (originally last) is now first.
    assert [c.chunk_index for c in result.sources] == [3, 2, 1, 0]
    assert reranker.calls == [("q", 4, 4)]


def test_rerank_widens_pool_then_narrows_to_top_k(embeddings, llm) -> None:
    chunks = [RetrievedChunk(f"chunk {i}", "doc.pdf", i, 0.9) for i in range(10)]
    store = FakeVectorStore(chunks)
    reranker = FakeReranker()
    service = RAGService(
        embeddings, store, llm, reranker=reranker, retrieve_n=8, default_top_k=3
    )

    result = service.query("q")

    # retrieve() was asked for the wide pool (8), not the final top_k (3).
    assert reranker.calls[0][1] == 8
    # and the final output is narrowed to top_k.
    assert len(result.sources) == 3
    llm.generate_answer.assert_called_once()


def test_reciprocal_rank_fusion_merges_by_rank_not_score() -> None:
    dense = [
        RetrievedChunk("dense top", "a.pdf", 0, 0.9),
        RetrievedChunk("dense second", "b.pdf", 0, 0.8),
    ]
    lexical = [
        RetrievedChunk("lexical top", "c.pdf", 0, 15.2),  # unbounded BM25 score
        RetrievedChunk("dense top again", "a.pdf", 0, 15.0),  # same chunk as dense's #1
    ]

    fused = reciprocal_rank_fusion(dense, lexical, k=60)

    keys = [(c.source_document, c.chunk_index) for c in fused]
    # a.pdf/0 was ranked in both lists -- highest combined RRF score, first.
    assert keys[0] == ("a.pdf", 0)
    # every unique chunk survives fusion; none silently dropped.
    assert set(keys) == {("a.pdf", 0), ("b.pdf", 0), ("c.pdf", 0)}

    by_key = {(c.source_document, c.chunk_index): c for c in fused}
    # Found by both sources -- keeps dense's real cosine score, still
    # subject to the relevance threshold.
    assert by_key[("a.pdf", 0)].score == 0.9
    assert by_key[("a.pdf", 0)].bypass_relevance_filter is False
    # Lexical-only -- raw BM25 score preserved as-is (not on the cosine
    # scale), and exempted from the relevance threshold.
    assert by_key[("c.pdf", 0)].score == 15.2
    assert by_key[("c.pdf", 0)].bypass_relevance_filter is True


def test_hybrid_disabled_leaves_retrieve_output_unchanged(embeddings, llm) -> None:
    """lexical_index=None (the default) must be a true no-op: hybrid_retrieve()
    must return exactly what retrieve() returns, proving dense-only behavior
    is unaffected until hybrid retrieval is explicitly enabled."""
    chunks = [
        RetrievedChunk("a", "doc.pdf", 0, 0.9),
        RetrievedChunk("b", "doc.pdf", 1, 0.8),
    ]
    store = FakeVectorStore(chunks)
    service = RAGService(embeddings, store, llm, default_top_k=2)

    dense_only = service.retrieve("q")
    hybrid = service.hybrid_retrieve("q")

    assert hybrid == dense_only


def test_hybrid_retrieve_fuses_lexical_hits_then_rerank_still_works(
    embeddings, llm
) -> None:
    """The full combined pipeline: a chunk dense search never surfaced at
    all, but BM25 did, must reach the reranker's candidate pool -- and the
    rest of query() (widening, reranking, truncation to top_k) still works
    exactly as it did before hybrid retrieval existed."""
    dense_chunks = [
        RetrievedChunk(f"dense {i}", "dense.pdf", i, 0.9 - i * 0.05) for i in range(3)
    ]
    store = FakeVectorStore(dense_chunks)
    lexical_index = FakeLexicalIndex(
        [RetrievedChunk("exact identifier match", "lexical.pdf", 0, 12.0)]
    )
    reranker = FakeReranker()
    service = RAGService(
        embeddings, store, llm,
        default_top_k=2, retrieve_n=3,
        lexical_index=lexical_index, reranker=reranker,
    )

    result = service.query("q")

    # Lexical search was asked for the same wide pool size as dense search.
    assert lexical_index.calls == [("q", 3)]
    # hybrid_retrieve(top_k=N) returns at most N fused candidates, mirroring
    # retrieve()'s own contract -- the reranker's pool size still matches
    # retrieve_n exactly as before hybrid retrieval existed; what changes is
    # which candidates fill it.
    assert reranker.calls[0][1] == 3
    # The lexical-only chunk -- dense search never found it at all -- was
    # promoted into that pool by fusion and survived through to the final
    # result. This is the concrete value hybrid retrieval adds.
    assert "lexical.pdf" in [c.source_document for c in result.sources]
    # Reranker still narrows to the configured top_k.
    assert len(result.sources) == 2
