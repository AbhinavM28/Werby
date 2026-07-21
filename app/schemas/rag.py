"""Pydantic schemas: the public API contract.

Schemas live in their own package, separate from services, because they serve
a different master: they define what the *outside world* sends and receives.
Services can evolve internally without breaking API consumers, as long as
these schemas stay stable (and vice versa).
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A user question against the ingested document corpus."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language question about the engineering docs.",
        examples=["What is the maximum rated load for the AS/RS crane?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Override for how many chunks to retrieve (default from config).",
    )


class SourceChunk(BaseModel):
    """A retrieved document chunk that grounded the answer."""

    document: str = Field(description="Source filename the chunk came from.")
    chunk_index: int = Field(description="Position of the chunk within its document.")
    score: float = Field(description="Similarity score (higher = more relevant).")
    excerpt: str = Field(description="Text of the retrieved chunk.")


class QueryResponse(BaseModel):
    """Grounded answer plus the evidence used to produce it."""

    answer: str
    sources: list[SourceChunk]
    model: str = Field(description="Chat model that generated the answer.")
    retrieved_chunks: int
    latency_ms: int


class IngestResponse(BaseModel):
    """Result of ingesting one uploaded document."""

    document: str
    chunks_created: int
    characters: int
    status: str = "ingested"


class CorpusStats(BaseModel):
    """Snapshot of what's currently in the vector store."""

    collection: str
    total_chunks: int
    documents: list[str]


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str = "ok"
    app: str
    version: str
    environment: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
