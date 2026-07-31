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

Two false-positive-reduction measures, both added after real measurement
caught real failures (see ``search()``'s and ``min_score``'s docstrings):
stopword filtering, and a minimum-score floor. Neither is a complete fix --
see the module-level note in ``search()`` for the known remaining gap and
why closing it further isn't a tunable-parameter problem.
"""

import logging
import re
from abc import ABC, abstractmethod

from rank_bm25 import BM25Okapi

from app.services.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+")

# Common English function words, excluded from BM25 scoring entirely. Without
# this, a query and a chunk that share nothing but "how", "do", "i", "a" can
# out-score a genuine identifier match -- measured for real: "How do I set up
# a VPN on my home router?" scored *higher* against an unrelated OSHA chunk
# (10.67, driven entirely by those four function words) than "What is
# V6001176?" scored against the chunk that actually answers it (7.27).
# Filtering stopwords isn't optional polish here; it's the majority fix for a
# real, measured false-positive class.
_STOPWORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being do does did how what why when "
    "where who which this that these those i you he she it we they my your "
    "his her its our their am will would shall should can could may might "
    "must not no nor so if then than of for to in on at by with from as and "
    "or but up down out about into over under again further here there all "
    "any both each few more most other some such only own same".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-run tokenizer with stopwords removed.

    Deliberately naive otherwise -- no stemming or lemmatization. BM25 is
    being used here specifically to catch *exact* identifier matches
    ("1910.147", "SRM-4000") that dense search underrates; aggressive
    normalization would work against that goal by conflating tokens BM25
    should keep distinct. Stopword removal is different in kind: those
    tokens carry no discriminating signal at all (see ``_STOPWORDS``), so
    dropping them only removes noise, never a real signal.
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


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

    def __init__(self, store: VectorStore, min_score: float = 0.0) -> None:
        """
        Args:
            store: Read-only source of the corpus to index.
            min_score: Chunks scoring at or below this are excluded from
                results entirely (see ``search()``'s docstring for why a
                bare "score > 0" isn't enough). Deliberately defaults to 0.0
                (today's original, purely-nonzero behavior) rather than a
                "safe" nonzero value -- the right floor is corpus-specific
                (BM25 scores scale with corpus size and content via IDF), so
                a hardcoded default here would be meaningless for a corpus
                other than the one it was calibrated against. Production
                wiring passes the calibrated value from
                ``Settings.bm25_min_score`` explicitly; see that field's
                comment for the real measurement behind the number.
        """
        self._store = store
        self._min_score = min_score
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
        # min_score defaults to 0 (exclude only exact-zero, no-overlap-at-all
        # matches -- BM25Okapi returns a number for every chunk, including
        # ones sharing nothing with the query). Production wiring raises
        # this to a calibrated floor (Settings.bm25_min_score) that also
        # excludes *weak* positive-overlap matches, not just zero ones --
        # chunks that would otherwise bypass relevance_threshold entirely as
        # lexical-only hits (see reciprocal_rank_fusion()) on the strength of
        # a coincidental match.
        #
        # Known, accepted gap: no score floor can fully close this. Real
        # measurement found a query about scaffolding fall-protection height
        # scoring *higher* (8.58) against an unrelated fire-extinguisher
        # clause than "What is V6001176?" scores (5.27) against the chunk
        # that actually answers it -- because both share one token with
        # identical corpus-wide rarity ("requirement", singular, appears in
        # exactly 1 of 185 chunks -- same document frequency as "v6001176").
        # BM25 has no way to distinguish "rare word reflecting genuine
        # relevance" from "rare word shared by coincidence"; that's a
        # semantic judgment, not a statistical one. A floor catches the
        # clear-cut cases (weak or zero overlap) but not this one.
        matched = [i for i in ranked if scores[i] > self._min_score][:top_k]
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

