"""Integration tests for PgVectorStore against a real Postgres + pgvector.

Unlike Chroma (embedded, no server needed), pgvector needs a real running
Postgres instance -- unlike every other test in this suite, these are
deliberately NOT mocked, because a mock couldn't meaningfully validate real
SQL or a real vector similarity index. Instead: skip gracefully if no
Postgres is reachable locally (run `docker compose --profile pgvector up
postgres` to enable them -- see docker-compose.yml), so a plain `pytest -q`
stays hermetic and green with zero setup. CI always has a real instance via
a service container (see .github/workflows/ci.yml), so this still gets full,
non-mocked coverage on every PR regardless of what's running on a given
contributor's machine.
"""

import os
import uuid
from collections.abc import Callable, Iterator

import psycopg
import pytest

from app.core.exceptions import VectorStoreError
from app.services.document_processor import DocumentChunk
from app.services.pgvector_store import PgVectorStore

_TEST_DSN = os.environ.get(
    "TEST_POSTGRES_DSN", "postgresql://werby:werby@localhost:5432/werby"
)


def _postgres_available() -> bool:
    try:
        with psycopg.connect(_TEST_DSN, connect_timeout=2):
            return True
    except psycopg.Error:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason=(
        "No Postgres reachable at TEST_POSTGRES_DSN (or the local "
        "docker-compose default) -- run `docker compose --profile pgvector "
        "up postgres` to enable these tests locally. Always runs in CI."
    ),
)


@pytest.fixture
def collection_name() -> Iterator[str]:
    """A fresh, unique collection per test -- avoids cross-test
    interference without needing a shared reset step -- cleaned up
    afterward so repeated local runs don't accumulate leftover tables."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    yield name
    with psycopg.connect(_TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS werby_{name}")
        conn.execute(
            "DELETE FROM werby_collections_meta WHERE collection_name = %s",
            (name,),
        )


@pytest.fixture
def make_store(
    collection_name: str,
) -> Iterator[Callable[..., PgVectorStore]]:
    """Factory for PgVectorStore instances scoped to this test's
    collection_name, closing every one created afterward -- without this,
    an unclosed connection pool's background worker threads raise a benign
    but noisy "cannot join current thread" warning at garbage-collection
    time (found the hard way: this file passed clean until a fresh
    database round-trip surfaced it alongside the extension-ordering bug
    above)."""
    stores: list[PgVectorStore] = []

    def _make(embedding_model: str | None = "test-model") -> PgVectorStore:
        store = PgVectorStore(_TEST_DSN, collection_name, embedding_model)
        stores.append(store)
        return store

    yield _make
    for store in stores:
        store.close()


def test_upsert_and_search_roundtrip(make_store: Callable[..., PgVectorStore]) -> None:
    store = make_store()
    chunks = [
        DocumentChunk("a::0", "close match", "a.pdf", 0),
        DocumentChunk("a::1", "far match", "a.pdf", 1),
    ]
    store.upsert(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    results = store.search([0.9, 0.1, 0.0], top_k=2)

    assert results[0].source_document == "a.pdf"
    assert results[0].chunk_index == 0
    assert results[0].score > results[1].score


def test_count_reflects_upserted_chunks(
    make_store: Callable[..., PgVectorStore],
) -> None:
    store = make_store()
    assert store.count() == 0
    store.upsert([DocumentChunk("a::0", "x", "a.pdf", 0)], [[1.0, 0.0]])
    assert store.count() == 1


def test_upsert_is_idempotent_by_chunk_id(
    make_store: Callable[..., PgVectorStore],
) -> None:
    """Re-ingesting the same chunk_id updates in place rather than
    duplicating -- matches ChromaVectorStore's own upsert semantics."""
    store = make_store()
    store.upsert([DocumentChunk("a::0", "original", "a.pdf", 0)], [[1.0, 0.0]])
    store.upsert([DocumentChunk("a::0", "updated", "a.pdf", 0)], [[1.0, 0.0]])

    assert store.count() == 1
    assert store.get_all_chunks()[0].text == "updated"


def test_list_documents_returns_distinct_sorted_names(
    make_store: Callable[..., PgVectorStore],
) -> None:
    store = make_store()
    store.upsert(
        [
            DocumentChunk("b::0", "x", "b.pdf", 0),
            DocumentChunk("a::0", "y", "a.pdf", 0),
            DocumentChunk("a::1", "z", "a.pdf", 1),
        ],
        [[1.0, 0.0]] * 3,
    )
    assert store.list_documents() == ["a.pdf", "b.pdf"]


def test_delete_document_removes_only_its_own_chunks(
    make_store: Callable[..., PgVectorStore],
) -> None:
    store = make_store()
    store.upsert(
        [
            DocumentChunk("a::0", "x", "a.pdf", 0),
            DocumentChunk("b::0", "y", "b.pdf", 0),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    deleted = store.delete_document("a.pdf")

    assert deleted == 1
    assert store.list_documents() == ["b.pdf"]


def test_get_all_chunks_round_trips_upserted_data(
    make_store: Callable[..., PgVectorStore],
) -> None:
    store = make_store()
    store.upsert([DocumentChunk("a::0", "hello", "a.pdf", 0)], [[1.0, 0.0]])

    chunks = store.get_all_chunks()

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "a::0"
    assert chunks[0].text == "hello"
    assert chunks[0].source_document == "a.pdf"
    assert chunks[0].chunk_index == 0


def test_reads_on_never_upserted_collection_are_empty(
    make_store: Callable[..., PgVectorStore],
) -> None:
    """No table has ever been created for this collection -- every read
    method must degrade gracefully to empty, not raise on a missing table
    (mirrors a brand-new Chroma collection's behavior)."""
    store = make_store()
    assert store.count() == 0
    assert store.search([1.0, 0.0], top_k=5) == []
    assert store.list_documents() == []
    assert store.get_all_chunks() == []


def test_embedding_compatibility_guard_persists_across_instances(
    make_store: Callable[..., PgVectorStore],
) -> None:
    """Same invariant as ChromaVectorStore's guard of the same name: a
    stamped collection rejects a different embedding model, checked on a
    later, separate connection/process, not just in-memory."""
    first = make_store(embedding_model="model-a")
    first.upsert([DocumentChunk("a::0", "x", "a.pdf", 0)], [[1.0, 0.0]])

    # Same model again: fine (idempotent).
    make_store(embedding_model="model-a")

    with pytest.raises(VectorStoreError, match="mismatch"):
        make_store(embedding_model="model-b")


def test_upsert_rejects_chunk_embedding_count_mismatch(
    make_store: Callable[..., PgVectorStore],
) -> None:
    store = make_store()
    with pytest.raises(VectorStoreError, match="mismatch"):
        store.upsert([DocumentChunk("a::0", "x", "a.pdf", 0)], [[1.0], [2.0]])


def test_invalid_collection_name_rejected() -> None:
    """Collection names get interpolated directly into table DDL (Postgres
    identifiers can't be parameterized like values) -- an unsafe name must
    be rejected before it ever reaches SQL, not sanitized silently."""
    with pytest.raises(VectorStoreError, match="Invalid collection name"):
        PgVectorStore(_TEST_DSN, "not; safe", embedding_model="test-model")
