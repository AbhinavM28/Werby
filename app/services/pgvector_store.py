"""PostgreSQL + pgvector backend for the VectorStore interface.

This is the proof that the VectorStore abstraction (app/services/vector_store.py)
actually delivers on its architectural promise: swapping backends means one
new subclass and one line in app/api/deps.py, nothing in RAGService or
IngestionService changes. Chroma stays the default (embedded, zero-config);
this backend is for teams already running Postgres who'd rather not run a
second database just for vectors.

Schema and dimension: pgvector's ``vector(n)`` column type needs a fixed
size ``n`` set at table-creation time -- unlike Chroma, which infers its
schema lazily on first write. Rather than add a new settings knob for a
number most users don't know off-hand, the table is created lazily on the
first real ``upsert()`` call, using ``len(embeddings[0])`` -- ingestion is
computing that embedding anyway, so this costs nothing extra (no dedicated
"probe" API call just to learn a dimension).

Table-per-collection, not a shared table with a collection_name column:
a shared table would need one vector width for every collection sharing
it, which breaks the moment two collections use different embedding
models (different dimensions). Chroma's own model is independent
per-collection namespaces; a table per collection is the direct
Postgres equivalent, and avoids ever needing to reconcile mismatched
widths in one column.

Embedding-compatibility guard: same invariant Chroma enforces (see
ChromaVectorStore's docstring) -- a stamp table
(``werby_collections_meta``) records which embedding model built each
collection, checked eagerly at construction (fail fast, like Chroma), even
though the dimension part of the stamp can't be written until the first
upsert actually reveals it.
"""

import logging
import re
from collections.abc import Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.core.exceptions import VectorStoreError
from app.services.document_processor import DocumentChunk
from app.services.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

# Postgres identifiers can't be parameterized like values, so collection_name
# gets interpolated directly into DDL/table names below. Validating it
# against a strict allowlist first (matches Chroma's own collection-naming
# restrictions closely enough) turns "arbitrary SQL injection via a config
# string" into a loud startup error instead of a silent risk -- even though
# in practice this value only ever comes from trusted Settings, not
# end-user input.
_SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")

_META_TABLE = "werby_collections_meta"


class PgVectorStore(VectorStore):
    """Postgres + pgvector backend, one table per collection."""

    def __init__(
        self,
        dsn: str,
        collection_name: str,
        embedding_model: str | None = None,
    ) -> None:
        """
        Args:
            dsn: Postgres connection string (e.g.
                ``postgresql://user:pass@host:5432/dbname``).
            collection_name: Becomes part of the table name -- must match
                ``_SAFE_IDENTIFIER`` (letters, digits, underscores, starting
                with a letter or underscore).
            embedding_model: See ``_enforce_embedding_compatibility``.
        """
        if not _SAFE_IDENTIFIER.match(collection_name):
            raise VectorStoreError(
                f"Invalid collection name '{collection_name}': must match "
                f"{_SAFE_IDENTIFIER.pattern} (used directly in table DDL)."
            )
        self._collection_name = collection_name
        self._table = f"werby_{collection_name}"
        self._table_ready = False
        # Set by _enforce_embedding_compatibility when this collection has
        # never been stamped before; written into werby_collections_meta at
        # the first real upsert, once the dimension is also known. None
        # once stamped (nothing left to do) or when no embedding_model was
        # given at all.
        self._pending_stamp: str | None = None

        # ConnectionPool(..., open=True) does NOT connect synchronously --
        # it starts opening in a background thread and returns immediately
        # even if the DSN is unreachable (confirmed live: it never raises
        # here). The real failure only surfaces on first actual use, below.
        # Two separate timeouts are needed to fail fast there instead of
        # hanging (confirmed live, the hard way -- neither alone was
        # enough): `timeout` bounds how long a `.connection()` call waits
        # for the pool itself; `connect_timeout` (a libpq/psycopg connection
        # kwarg, not a pool one) bounds each individual TCP connection
        # attempt the pool's background worker makes internally. Matches
        # every other provider in this codebase failing fast on an
        # unreachable backend instead of hanging (see e.g. Ollama's 5s
        # connect timeout).
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=5,
            configure=register_vector,
            open=True,
            timeout=10.0,
            kwargs={"connect_timeout": 5},
        )

        try:
            with self._pool.connection() as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_META_TABLE} (
                        collection_name TEXT PRIMARY KEY,
                        embedding_model TEXT NOT NULL,
                        dimension INTEGER NOT NULL
                    )
                    """
                )
        except psycopg.Error as exc:
            raise VectorStoreError(
                f"Failed to connect to or initialize Postgres/pgvector: {exc}"
            ) from exc

        if embedding_model is not None:
            self._enforce_embedding_compatibility(embedding_model)

        logger.info(
            "pgvector collection '%s' ready (table %s)",
            collection_name, self._table,
        )

    def _enforce_embedding_compatibility(self, embedding_model: str) -> None:
        """Refuse to operate on a collection built with a different embedder.

        Same reasoning as ChromaVectorStore's guard of the same name:
        embeddings from different models occupy incompatible vector spaces,
        and querying one with the wrong model's vectors silently returns
        garbage-relevance results rather than erroring. The check itself
        happens now, eagerly (fail fast on a stale collection); the WRITE
        of a brand-new stamp is deferred to the first upsert, since the
        dimension half of the stamp isn't knowable until then.
        """
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT embedding_model FROM {_META_TABLE} "
                "WHERE collection_name = %s",
                (self._collection_name,),
            ).fetchone()

        if row is None:
            self._pending_stamp = embedding_model
            return

        stamped = row[0]
        if stamped != embedding_model:
            raise VectorStoreError(
                f"Embedding model mismatch: collection '{self._collection_name}' "
                f"was built with '{stamped}', but the configured provider is "
                f"'{embedding_model}'. Vectors from different models are not "
                "comparable. Either switch EMBEDDING_PROVIDER back, point "
                "CHROMA_COLLECTION at a new collection name, or delete the "
                "collection and re-ingest your documents."
            )
        self._pending_stamp = None

    def _ensure_table(self, dimension: int) -> None:
        """Create this collection's table (and its HNSW index) on first
        write, sized for the embeddings actually being upserted -- see the
        module docstring for why this can't happen at construction time."""
        if self._table_ready:
            return

        with self._pool.connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source_document TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding VECTOR({dimension}) NOT NULL
                )
                """
            )
            # HNSW: the same index family Chroma uses internally. Built
            # with cosine ops to match search()'s <=> usage below.
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self._table}_hnsw_idx
                ON {self._table} USING hnsw (embedding vector_cosine_ops)
                """
            )
            if self._pending_stamp is not None:
                conn.execute(
                    f"""
                    INSERT INTO {_META_TABLE}
                        (collection_name, embedding_model, dimension)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (collection_name) DO NOTHING
                    """,
                    (self._collection_name, self._pending_stamp, dimension),
                )
                logger.info(
                    "Stamped pgvector collection '%s' with embedding model '%s'",
                    self._collection_name, self._pending_stamp,
                )
                self._pending_stamp = None
        self._table_ready = True

    def _table_exists(self) -> bool:
        if self._table_ready:
            return True
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
                (self._table,),
            ).fetchone()
        self._table_ready = row is not None
        return self._table_ready

    def upsert(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return
        try:
            self._ensure_table(dimension=len(embeddings[0]))
            with self._pool.connection() as conn:
                conn.cursor().executemany(
                    f"""
                    INSERT INTO {self._table}
                        (id, text, source_document, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        source_document = EXCLUDED.source_document,
                        chunk_index = EXCLUDED.chunk_index,
                        embedding = EXCLUDED.embedding
                    """,
                    [
                        (c.chunk_id, c.text, c.source_document, c.chunk_index, emb)
                        for c, emb in zip(chunks, embeddings, strict=True)
                    ],
                )
            logger.info("Upserted %d chunks into pgvector", len(chunks))
        except psycopg.Error as exc:
            raise VectorStoreError(f"pgvector upsert failed: {exc}") from exc

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        if not self._table_exists():
            return []
        try:
            with self._pool.connection() as conn:
                # <=> is pgvector's cosine *distance* (1 - cosine similarity,
                # per pgvector's own definition) -- similarity is 1 minus
                # that directly, no /2 rescaling needed (that's Chroma's own
                # distance convention, not pgvector's; the two libraries
                # don't share a definition).
                #
                # ::vector casts are required here (found by live testing,
                # not assumed): a raw Python list parameter used inside an
                # expression like `embedding <=> %s` has no column context
                # to infer its type from, so psycopg/Postgres defaults to
                # `double precision[]` and the comparison fails outright.
                # upsert() doesn't need this -- there, the target column's
                # own declared type (`vector(n)`) gives Postgres the type
                # to adapt the parameter to.
                rows = conn.execute(
                    f"""
                    SELECT text, source_document, chunk_index,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {self._table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k),
                ).fetchall()
        except psycopg.Error as exc:
            raise VectorStoreError(f"pgvector query failed: {exc}") from exc

        return [
            RetrievedChunk(
                text=text,
                source_document=source_document,
                chunk_index=chunk_index,
                score=round(float(score), 4),
            )
            for text, source_document, chunk_index, score in rows
        ]

    def count(self) -> int:
        if not self._table_exists():
            return 0
        with self._pool.connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()
        return int(row[0]) if row else 0

    def list_documents(self) -> list[str]:
        if not self._table_exists():
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT source_document FROM {self._table} "
                "ORDER BY source_document"
            ).fetchall()
        return [r[0] for r in rows]

    def delete_document(self, source_document: str) -> int:
        if not self._table_exists():
            return 0
        try:
            with self._pool.connection() as conn:
                cur = conn.execute(
                    f"DELETE FROM {self._table} WHERE source_document = %s",
                    (source_document,),
                )
                deleted = cur.rowcount
            logger.info("Deleted %d chunks for '%s'", deleted, source_document)
            return deleted
        except psycopg.Error as exc:
            raise VectorStoreError(f"pgvector delete failed: {exc}") from exc

    def get_all_chunks(self) -> list[DocumentChunk]:
        # Same "fine at current scale, would need to move to streaming/
        # paginated reads at large scale" caveat as ChromaVectorStore's
        # get_all_chunks() -- see that docstring.
        if not self._table_exists():
            return []
        with self._pool.connection() as conn:
            rows: Sequence[tuple[str, str, str, int]] = conn.execute(
                f"SELECT id, text, source_document, chunk_index FROM {self._table}"
            ).fetchall()
        return [
            DocumentChunk(
                chunk_id=chunk_id,
                text=text,
                source_document=source_document,
                chunk_index=chunk_index,
            )
            for chunk_id, text, source_document, chunk_index in rows
        ]
