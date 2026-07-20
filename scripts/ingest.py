"""Bulk-ingest documents from a local directory, bypassing HTTP.

Usage:
    python -m scripts.ingest path/to/docs_folder

Why this exists: uploading 40 PDFs through a web form is miserable. Because
the service layer is decoupled from FastAPI, this script reuses the exact
same IngestionService the API uses -- one pipeline, two entry points.
"""

import argparse
import logging
import sys
from pathlib import Path

from app.api.deps import get_ingestion_service
from app.core.config import get_settings
from app.core.exceptions import WerbyError
from app.core.logging import configure_logging
from app.services.document_processor import SUPPORTED_EXTENSIONS

logger = logging.getLogger("scripts.ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-ingest documents into Werby.")
    parser.add_argument("directory", type=Path, help="Folder containing documents")
    args = parser.parse_args()

    configure_logging(get_settings().log_level)

    if not args.directory.is_dir():
        logger.error("Not a directory: %s", args.directory)
        return 1

    service = get_ingestion_service()
    files = sorted(
        p for p in args.directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.warning("No supported files found in %s", args.directory)
        return 1

    succeeded = failed = 0
    for path in files:
        try:
            result = service.ingest(path.name, path.read_bytes())
            logger.info("OK  %s (%d chunks)", path.name, result.chunks_created)
            succeeded += 1
        except WerbyError as exc:
            logger.error("FAIL %s: %s", path.name, exc.message)
            failed += 1

    logger.info("Done: %d ingested, %d failed", succeeded, failed)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
