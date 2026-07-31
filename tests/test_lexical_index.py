"""Unit tests for BM25LexicalIndex.

rank-bm25 is pure Python, deterministic, and has no model-download cost
(unlike the reranker's sentence-transformers dependency), so these run
against the real BM25Okapi -- only the VectorStore it reads from is faked.
"""

from app.services.document_processor import DocumentChunk
from app.services.lexical_index import BM25LexicalIndex
from app.services.vector_store import VectorStore


class FakeChunkStore(VectorStore):
    """Minimal fake: BM25LexicalIndex only ever calls count() and
    get_all_chunks(), so that's all this needs to implement for real."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks

    def upsert(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int):
        raise NotImplementedError

    def count(self) -> int:
        return len(self.chunks)

    def list_documents(self) -> list[str]:
        raise NotImplementedError

    def delete_document(self, source_document: str) -> int:
        raise NotImplementedError

    def get_all_chunks(self) -> list[DocumentChunk]:
        return self.chunks


CHUNKS = [
    DocumentChunk(
        "crane::chunk_0",
        "The SRM-4000 hoist has a rated load of 2000 kg.",
        "crane.pdf",
        0,
    ),
    DocumentChunk(
        "osha::chunk_0",
        "General lockout/tagout procedures apply to all powered equipment.",
        "osha.pdf",
        0,
    ),
    DocumentChunk(
        "forklift::chunk_0",
        "Daily inspection intervals are required for all forklifts.",
        "forklift.pdf",
        0,
    ),
]


def test_bm25_ranks_exact_identifier_match_first() -> None:
    """The scenario this feature exists for: a distinctive, rare identifier
    that appears in exactly one chunk should be found even by pure term
    overlap, with no embedding involved at all."""
    index = BM25LexicalIndex(store=FakeChunkStore(list(CHUNKS)))

    results = index.search("What is the SRM-4000 rated for?", top_k=3)

    assert results[0].source_document == "crane.pdf"
    assert results[0].chunk_index == 0


def test_bm25_search_respects_top_k() -> None:
    index = BM25LexicalIndex(store=FakeChunkStore(list(CHUNKS)))
    results = index.search("inspection", top_k=1)
    assert len(results) == 1


def test_bm25_empty_corpus_returns_empty_list() -> None:
    index = BM25LexicalIndex(store=FakeChunkStore([]))
    assert index.search("anything", top_k=5) == []


def test_bm25_rebuilds_when_store_count_changes() -> None:
    """The self-checking staleness design: a chunk added to the store after
    the index's first build must become searchable on the very next
    search() call, with no explicit invalidation from the caller -- see
    BM25LexicalIndex._ensure_fresh's docstring for why this is a deliberate
    trade rather than an oversight."""
    store = FakeChunkStore(list(CHUNKS))
    index = BM25LexicalIndex(store=store)
    first = index.search("hydraulic", top_k=3)
    assert first == []  # nothing about hydraulics ingested yet

    store.chunks.append(
        DocumentChunk(
            "hydraulic::chunk_0",
            "Hydraulic pressure must not exceed 3000 psi.",
            "hydraulic.pdf",
            0,
        )
    )

    results = index.search("hydraulic pressure", top_k=1)
    assert results[0].source_document == "hydraulic.pdf"


def test_bm25_pure_stopword_query_returns_no_matches() -> None:
    """A query with no discriminating tokens at all (see _STOPWORDS) must
    not match anything -- function words alone carry no signal, and every
    chunk in the corpus contains some of them."""
    index = BM25LexicalIndex(store=FakeChunkStore(list(CHUNKS)))
    results = index.search("What is this and that?", top_k=3)
    assert results == []


def test_bm25_filters_matches_at_or_below_min_score() -> None:
    """A calibrated min_score excludes weak-but-nonzero matches -- the fix
    for a real measured problem (see min_score's docstring on
    BM25LexicalIndex.__init__): a permissive index finds a weak, single-
    token match; a floor at or above that match's own score excludes it.
    Self-referential on purpose -- doesn't assume exact BM25 magnitudes."""
    store = FakeChunkStore(list(CHUNKS))
    permissive = BM25LexicalIndex(store=store)
    weak_query = "equipment"  # a single, weakly-discriminating token
    baseline = permissive.search(weak_query, top_k=3)
    assert baseline  # sanity: the permissive index does find something

    weak_score = baseline[0].score
    strict = BM25LexicalIndex(store=store, min_score=weak_score)
    assert strict.search(weak_query, top_k=3) == []


def test_bm25_min_score_still_allows_matches_above_it() -> None:
    """A floor doesn't exclude everything -- a match genuinely above it
    still comes through, which is the whole point of a floor rather than
    disabling lexical-only matches outright."""
    store = FakeChunkStore(list(CHUNKS))
    permissive = BM25LexicalIndex(store=store)
    query = "What is the SRM-4000 rated for?"
    strong_score = permissive.search(query, top_k=1)[0].score

    index = BM25LexicalIndex(store=store, min_score=strong_score - 0.01)
    results = index.search(query, top_k=1)
    assert results[0].source_document == "crane.pdf"
