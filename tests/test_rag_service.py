"""Unit tests for RAG orchestration using an in-memory fake vector store.

A hand-rolled fake (rather than MagicMock everywhere) keeps tests readable
and verifies the VectorStore *interface* is actually sufficient.
"""

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import EmptyCorpusError
from app.services.document_processor import DocumentChunk
from app.services.providers.base import Reranker
from app.services.rag_service import NO_RELEVANT_CONTEXT_MESSAGE, RAGService
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
