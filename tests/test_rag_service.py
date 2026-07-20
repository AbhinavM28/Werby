"""Unit tests for RAG orchestration using an in-memory fake vector store.

A hand-rolled fake (rather than MagicMock everywhere) keeps tests readable
and verifies the VectorStore *interface* is actually sufficient.
"""

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import EmptyCorpusError
from app.services.document_processor import DocumentChunk
from app.services.rag_service import RAGService
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
