"""Unit tests for the pure chunking pipeline (no network, no DB)."""

import pytest

from app.core.exceptions import DocumentProcessingError, UnsupportedFileTypeError
from app.services.document_processor import DocumentProcessor


@pytest.fixture
def processor() -> DocumentProcessor:
    return DocumentProcessor(chunk_size=200, chunk_overlap=50)


def test_text_file_is_chunked(processor: DocumentProcessor) -> None:
    text = ("Conveyor belt maintenance procedure. " * 30).encode()
    chunks = processor.process("manual.txt", text)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].source_document == "manual.txt"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic(processor: DocumentProcessor) -> None:
    content = b"Safety clearance is 1.5 meters around the palletizer."
    first = processor.process("sop.md", content)
    second = processor.process("sop.md", content)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_unsupported_extension_rejected(processor: DocumentProcessor) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        processor.process("firmware.bin", b"\x00\x01")


def test_empty_file_rejected(processor: DocumentProcessor) -> None:
    with pytest.raises(DocumentProcessingError):
        processor.process("empty.txt", b"   \n  ")


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        DocumentProcessor(chunk_size=100, chunk_overlap=100)
