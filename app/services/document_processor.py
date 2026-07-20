"""Document loading and chunking.

This module is deliberately *pure*: it takes bytes in and returns chunks out,
with no knowledge of HTTP, OpenAI, or ChromaDB. That makes it trivially unit
testable (see ``tests/test_document_processor.py``) and reusable from the API,
a CLI script, or a background worker.

Why chunking matters (RAG fundamentals):
    Embedding models and LLM context windows can't handle whole manuals, and
    retrieval works best when each vector represents ONE coherent idea. We use
    LangChain's ``RecursiveCharacterTextSplitter``, which tries to split on
    paragraph breaks first, then sentences, then words -- preserving semantic
    boundaries far better than naive fixed-width slicing. Overlap between
    chunks ensures a sentence straddling a boundary is fully present in at
    least one chunk.
"""

import io
import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.exceptions import DocumentProcessingError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"})


@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk of a source document, ready for embedding.

    Frozen dataclass: chunks are immutable value objects. ``chunk_id`` is
    deterministic (filename + index), so re-ingesting the same file *updates*
    existing vectors instead of duplicating them -- idempotent ingestion.
    """

    chunk_id: str
    text: str
    source_document: str
    chunk_index: int

    @property
    def metadata(self) -> dict[str, str | int]:
        """Metadata stored alongside the vector for filtering and citation."""
        return {
            "source_document": self.source_document,
            "chunk_index": self.chunk_index,
        }


class DocumentProcessor:
    """Turns raw uploaded files into embedding-ready chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, filename: str, content: bytes) -> list[DocumentChunk]:
        """Extract text from a file and split it into chunks.

        Args:
            filename: Original filename (used for type detection and metadata).
            content: Raw file bytes.

        Returns:
            Ordered list of chunks.

        Raises:
            UnsupportedFileTypeError: If the extension isn't supported.
            DocumentProcessingError: If the file can't be parsed or is empty.
        """
        text = self._extract_text(filename, content)
        if not text.strip():
            raise DocumentProcessingError(
                f"'{filename}' contained no extractable text. "
                "If this is a scanned PDF, it needs OCR before ingestion."
            )

        pieces = self._splitter.split_text(text)
        chunks = [
            DocumentChunk(
                chunk_id=f"{filename}::chunk_{i}",
                text=piece,
                source_document=filename,
                chunk_index=i,
            )
            for i, piece in enumerate(pieces)
        ]
        logger.info(
            "Processed '%s': %d characters -> %d chunks",
            filename, len(text), len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _extract_text(self, filename: str, content: bytes) -> str:
        extension = self._extension_of(filename)
        if extension == ".pdf":
            return self._extract_pdf(filename, content)
        # .txt / .md
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="replace")

    @staticmethod
    def _extension_of(filename: str) -> str:
        dot = filename.rfind(".")
        extension = filename[dot:].lower() if dot != -1 else ""
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension or filename}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return extension

    @staticmethod
    def _extract_pdf(filename: str, content: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception as exc:  # pypdf raises many exception types
            raise DocumentProcessingError(
                f"Failed to parse PDF '{filename}': {exc}"
            ) from exc
