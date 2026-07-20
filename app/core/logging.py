"""Structured logging configuration.

We configure the *root* logger once at startup (see ``app.main``). Every module
then does ``logger = logging.getLogger(__name__)`` -- the standard library
pattern -- so log lines carry their module path (e.g. ``app.services.rag``),
which makes production debugging dramatically easier than bare ``print()``.
"""

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger for the whole application.

    Args:
        level: Standard logging level name (DEBUG, INFO, WARNING, ...).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(level.upper())
    # Avoid duplicate handlers if configure_logging is called twice (e.g. tests)
    root.handlers.clear()
    root.addHandler(handler)

    # Third-party libraries are chatty at INFO; keep them at WARNING.
    for noisy in ("httpx", "chromadb", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
