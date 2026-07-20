"""Application entrypoint and factory.

The ``create_app`` factory pattern (instead of a module-level ``app = ...``
with side effects everywhere) makes it possible to build a fresh, isolated
app in tests, and keeps startup order explicit: configure logging -> build
app -> register routes -> register error handlers.

The exception handlers here are the bridge between the two worlds this
codebase keeps separate: services raise *domain* exceptions (WerbyError
subclasses, HTTP-agnostic), and this layer maps each to an appropriate HTTP
status. Add a new domain error, map it once here, and every route benefits.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routes import router as v1_router
from app.core.config import get_settings
from app.core.exceptions import (
    DocumentProcessingError,
    EmptyCorpusError,
    LLMServiceError,
    UnsupportedFileTypeError,
    VectorStoreError,
    WerbyError,
)
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

_ERROR_STATUS_MAP: dict[type[WerbyError], int] = {
    UnsupportedFileTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    DocumentProcessingError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    EmptyCorpusError: status.HTTP_409_CONFLICT,
    LLMServiceError: status.HTTP_502_BAD_GATEWAY,
    VectorStoreError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI Engineering Copilot for warehouse and industrial engineers. "
            "Retrieval-Augmented Generation over your engineering documentation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS: permissive in development so a local React/Streamlit frontend can
    # call the API. Lock allow_origins down to your real domain in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(v1_router, prefix=settings.api_v1_prefix)

    @app.exception_handler(WerbyError)
    async def handle_domain_error(request: Request, exc: WerbyError) -> JSONResponse:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        for error_type, mapped in _ERROR_STATUS_MAP.items():
            if isinstance(exc, error_type):
                status_code = mapped
                break
        logger.error(
            "%s on %s %s: %s",
            type(exc).__name__, request.method, request.url.path, exc.message,
        )
        return JSONResponse(
            status_code=status_code,
            content={"error": type(exc).__name__, "detail": exc.message},
        )

    logger.info(
        "%s v%s started (env=%s)",
        settings.app_name, settings.app_version, settings.environment,
    )
    return app


app = create_app()
