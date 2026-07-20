"""Domain exceptions.

The service layer raises *domain* exceptions -- it knows nothing about HTTP.
The API layer translates them into HTTP responses (see ``app.main``). This
separation means the same services could later back a CLI, a worker queue, or
a gRPC server without touching this file.
"""


class WerbyError(Exception):
    """Base class for all application errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ConfigurationError(WerbyError):
    """Raised when settings are invalid or inconsistent (e.g. a provider
    is selected but its required credentials/packages are missing)."""


class DocumentProcessingError(WerbyError):
    """Raised when a document cannot be parsed, chunked, or ingested."""


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when an uploaded file's type is not supported."""


class VectorStoreError(WerbyError):
    """Raised when the vector database fails (connection, query, upsert)."""


class LLMServiceError(WerbyError):
    """Raised when the LLM provider fails after retries are exhausted."""


class EmptyCorpusError(WerbyError):
    """Raised when a query is made but no documents have been ingested."""
