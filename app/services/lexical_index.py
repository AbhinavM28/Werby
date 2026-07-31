"""Lexical (BM25) search -- the other half of hybrid retrieval.

Why this is a separate seam from ``VectorStore`` rather than a new method on
it: ``VectorStore.search()`` takes a query *embedding* (a vector) because
dense search's whole mechanism is comparing vectors. BM25 has no embedding
step at all -- it scores raw query *text* against raw chunk text by weighted
term overlap. Bolting that onto ``VectorStore`` would mean either lying about
the interface's contract or branching inside ``search()`` on which kind of
index it secretly is. Two different input contracts means two different
interfaces, composed side by side in ``RAGService`` instead of nested inside
one another.

Why dense and lexical are complementary, not redundant:

* Dense (embedding) search captures *meaning* -- it can match "how do I stop
  the crane" to a chunk titled "emergency stop procedure" with no shared
  words. But embeddings compress a whole chunk into one fixed-size vector,
  which can blur out a single rare, specific token (a part number, a
  standard number) that's surrounded by a lot of other, more "average"
  prose.
* BM25 captures *exact term overlap*, weighted by how rare each term is
  across the corpus (inverse document frequency). A distinctive token like
  "SRM-4000" or "1910.147" that appears in only one or two chunks gets a
  large BM25 weight when it matches, precisely because it's rare and
  therefore highly discriminating -- the exact case dense search is weakest
  on. BM25 has no notion of paraphrase or meaning at all, which is dense
  search's strength -- the two failure modes don't overlap.

Index lifecycle: ``rank-bm25`` has no persistence and no incremental update
-- ``BM25Okapi(tokenized_corpus)`` always builds a fresh in-memory index from
a full corpus. Rather than hand-roll a second persisted store (and risk it
silently drifting from Chroma, the actual source of truth), ``BM25LexicalIndex``
treats itself as a derived cache: it remembers the chunk count it was last
built from and rebuilds from ``VectorStore.get_all_chunks()`` whenever that
count has changed. This means ingestion and deletion need no awareness that
a lexical index exists at all.
"""

import logging
import re
from abc import ABC, abstractmethod

from rank_bm25 import BM25Okapi

from app.services.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-run tokenizer.

    Deliberately naive -- no stemming or lemmatization. BM25 is being used
    here specifically to catch *exact* identifier matches ("1910.147",
    "SRM-4000") that dense search underrates; aggressive normalization would
    work against that goal by conflating tokens BM25 should keep distinct.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class LexicalIndex(ABC):
    """Interface for term-based (non-embedding) retrieval."""

    @abstractmethod
    def search(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks with the highest lexical overlap with
        ``query_text``."""


class BM25LexicalIndex(LexicalIndex):
    """BM25 search over the vector store's corpus, kept in sync lazily.

    Holds a reference to a ``VectorStore`` purely to read from (``get_all_chunks``,
    ``count``) -- it never writes to it. This keeps the write paths
    (``IngestionService.ingest``, the document-delete route) completely
    unaware that a lexical index exists; this class figures out on its own,
    at query time, whether it needs to rebuild.
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._bm25: BM25Okapi | None = None
        # (source_document, chunk_index, text) per chunk
        self._chunks: list[tuple[str, int, str]] = []
        self._indexed_count: int | None = None

    def _ensure_fresh(self) -> None:
        """Rebuild the in-memory index if the store's chunk count has moved.

        Known, accepted limitation: a same-size swap within one process (N
        chunks deleted, N chunks added, net-zero count change) won't trigger
        a rebuild until the *next* count change. Catching that exactly would
        require explicit rebuild hooks in both IngestionService and the
        delete route instead of this self-checking design -- a real
        precision/simplicity trade, made deliberately in favor of simplicity
        since same-size swaps are a rare sequence for how Werby is actually
        used, and the index self-heals on the next real ingest or delete.
        """
        current_count = self._store.count()
        if current_count == self._indexed_count:
            return

        # Pulls the entire corpus into memory to tokenize it -- fine at
        # Werby's current scale (rank-bm25 itself is in-memory only, so this
        # cost is unavoidable for this library regardless). At a scale where
        # holding every chunk's text in memory stops being trivial, lexical
        # search belongs in the database layer (e.g. Postgres full-text
        # search alongside pgvector) instead of a process-local index like
        # this one.
        chunks = self._store.get_all_chunks()
        self._chunks = [(c.source_document, c.chunk_index, c.text) for c in chunks]
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None
        self._indexed_count = current_count
        logger.info("Rebuilt BM25 lexical index over %d chunk(s)", len(chunks))

    def search(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        self._ensure_fresh()
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # A score of exactly 0 means literally no term overlap with the
        # query at all -- BM25Okapi still returns a number for every
        # document, so without this filter a totally unrelated query would
        # "match" whichever chunks happen to sort first among a sea of
        # zeros. Those chunks would then bypass the relevance threshold
        # entirely as lexical-only hits (see reciprocal_rank_fusion()),
        # which is exactly the wrong behavior this filter prevents.
        matched = [i for i in ranked if scores[i] > 0][:top_k]
        return [
            RetrievedChunk(
                text=self._chunks[i][2],
                source_document=self._chunks[i][0],
                chunk_index=self._chunks[i][1],
                # Raw BM25 score -- unbounded, not on the [0, 1] cosine
                # scale. Only meaningful as a *relative* ranking signal
                # within this one query's results; see
                # reciprocal_rank_fusion()'s docstring in rag_service.py for
                # how fusion avoids ever comparing this number to a dense
                # score directly.
                score=round(float(scores[i]), 4),
            )
            for i in matched
        ]

