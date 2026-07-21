"""Vector store abstraction and ChromaDB implementation.

Architectural decision -- the most important seam in this codebase:

The rest of the app depends on the ``VectorStore`` *interface* (an ABC), never
on ChromaDB directly. Your spec says "ChromaDB (initially)" -- this is how we
honor the "(initially)". When Werby needs to scale to pgvector, Qdrant, or
Pinecone, we write one new subclass and change one line in the dependency
wiring (``app/api/deps.py``). Nothing in the RAG or ingestion logic changes.
This is the Dependency Inversion Principle in practice: high-level policy
(RAG) does not depend on low-level detail (which database).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.core.exceptions import VectorStoreError
from app.services.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from similarity search."""

    text: str
    source_document: str
    chunk_index: int
    score: float  # similarity in [0, 1]; higher = more relevant


class VectorStore(ABC):
    """Interface every vector database backend must implement."""

    @abstractmethod
    def upsert(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        """Insert or update chunks with their embeddings (idempotent by id)."""

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks."""

    @abstractmethod
    def count(self) -> int:
        """Total number of stored chunks."""

    @abstractmethod
    def list_documents(self) -> list[str]:
        """Distinct source document names in the store."""

    @abstractmethod
    def delete_document(self, source_document: str) -> int:
        """Delete all chunks for a document. Returns number deleted."""


class ChromaVectorStore(VectorStore):
    """Persistent local ChromaDB backend.

    Chroma is ideal for this stage: embedded (no separate server), persistent
    to disk, and zero-config. Its limits (single-node, no replication) are
    exactly why the ABC above exists.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_model: str | None = None,
    ) -> None:
        """
        Args:
            persist_dir: Directory for Chroma's on-disk storage.
            collection_name: Name of the collection to use/create.
            embedding_model: Identifier of the embedding model that produced
                (or will produce) this collection's vectors. Used as a
                compatibility stamp -- see ``_enforce_embedding_compatibility``.
        """
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            # cosine distance suits normalized embeddings (OpenAI, nomic,
            # and sentence-transformers with normalize_embeddings=True)
            self._collection = self._client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {exc}") from exc

        if embedding_model is not None:
            self._enforce_embedding_compatibility(collection_name, embedding_model)

        logger.info(
            "Chroma collection '%s' ready (%d chunks) at %s",
            collection_name, self._collection.count(), persist_dir,
        )

    def _enforce_embedding_compatibility(
        self, collection_name: str, embedding_model: str
    ) -> None:
        """Refuse to operate on a collection built with a different embedder.

        Why this guard exists: embeddings from different models occupy
        different, incompatible vector spaces. Querying a corpus embedded
        with model A using vectors from model B doesn't error -- it silently
        returns *garbage-relevance* results, which in Werby's domain could
        surface the wrong safety procedure. We stamp the collection with the
        model that built it and fail loudly on mismatch, converting a silent
        data-corruption bug into an actionable startup error.
        """
        metadata = dict(self._collection.metadata or {})
        stamped = metadata.get("embedding_model")

        if stamped is None:
            # 'hnsw:*' keys are immutable index settings; Chroma rejects them
            # in modify(). Stamp only mutable metadata.
            mutable = {
                k: v for k, v in metadata.items() if not k.startswith("hnsw:")
            }
            mutable["embedding_model"] = embedding_model
            self._collection.modify(metadata=mutable)
            logger.info(
                "Stamped collection '%s' with embedding model '%s'",
                collection_name, embedding_model,
            )
        elif stamped != embedding_model:
            raise VectorStoreError(
                f"Embedding model mismatch: collection '{collection_name}' was "
                f"built with '{stamped}', but the configured provider is "
                f"'{embedding_model}'. Vectors from different models are not "
                "comparable. Either switch EMBEDDING_PROVIDER back, point "
                "CHROMA_COLLECTION at a new collection name, or delete "
                "the collection and re-ingest your documents."
            )

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
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
            logger.info("Upserted %d chunks into Chroma", len(chunks))
        except Exception as exc:
            raise VectorStoreError(f"Chroma upsert failed: {exc}") from exc

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        retrieved: list[RetrievedChunk] = []
        for text, meta, distance in zip(documents, metadatas, distances, strict=False):
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    source_document=str(meta.get("source_document", "unknown")),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    # cosine distance in [0, 2] -> similarity in [0, 1]
                    score=round(1.0 - (distance / 2.0), 4),
                )
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[str]:
        data = self._collection.get(include=["metadatas"])
        names = {
            str(meta.get("source_document"))
            for meta in data.get("metadatas") or []
            if meta.get("source_document")
        }
        return sorted(names)

    def delete_document(self, source_document: str) -> int:
        try:
            matching = self._collection.get(
                where={"source_document": source_document}
            )
            ids = matching.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
            logger.info("Deleted %d chunks for '%s'", len(ids), source_document)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete failed: {exc}") from exc
