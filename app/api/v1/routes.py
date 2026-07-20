"""API v1 routes.

Routes are intentionally thin: validate input (Pydantic does this for free),
call a service, map the result to a response schema. All business logic lives
in the service layer; all HTTP-error mapping lives in ``app.main``'s exception
handlers. A route function you can read in ten seconds is a feature.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import (
    get_app_settings,
    get_ingestion_service,
    get_rag_service,
    get_vector_store,
)
from app.core.config import Settings
from app.schemas.rag import (
    CorpusStats,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.services.rag_service import IngestionService, RAGService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Liveness probe for load balancers, Docker healthchecks, and humans."""
    return HealthResponse(
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def ingest_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_app_settings),
) -> IngestResponse:
    """Upload and ingest one document (.pdf, .txt, .md) into the corpus."""
    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_mb} MB limit.",
        )
    result = service.ingest(file.filename or "unnamed", content)
    return IngestResponse(
        document=result.document,
        chunks_created=result.chunks_created,
        characters=result.characters,
    )


@router.get("/documents", response_model=CorpusStats, tags=["documents"])
def corpus_stats(
    store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_app_settings),
) -> CorpusStats:
    """List what's currently ingested."""
    return CorpusStats(
        collection=settings.chroma_collection,
        total_chunks=store.count(),
        documents=store.list_documents(),
    )


@router.delete("/documents/{document_name}", tags=["documents"])
def delete_document(
    document_name: str,
    store: VectorStore = Depends(get_vector_store),
) -> dict[str, int | str]:
    """Remove a document and all of its chunks from the corpus."""
    deleted = store.delete_document(document_name)
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document named '{document_name}' in the corpus.",
        )
    return {"document": document_name, "chunks_deleted": deleted}


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(
    payload: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Ask a question; get a grounded answer with cited sources."""
    result = service.query(payload.question, top_k=payload.top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(
                document=s.source_document,
                chunk_index=s.chunk_index,
                score=s.score,
                excerpt=s.text,
            )
            for s in result.sources
        ],
        model=result.model,
        retrieved_chunks=len(result.sources),
        latency_ms=result.latency_ms,
    )
