"""In-process local embeddings via sentence-transformers.

A third embedding option that needs no server at all: the model runs inside
the API process on CPU (or GPU if available). ``all-MiniLM-L6-v2`` is the
classic workhorse -- 384 dimensions, ~80 MB, fast on CPU, surprisingly good
retrieval quality for its size.

The import is *lazy* and the dependency is an optional extra
(``pip install -e ".[local]"``): sentence-transformers pulls in PyTorch
(~2 GB), and users on the OpenAI or Ollama paths shouldn't pay that cost.
This "optional heavy dependency behind an extra + lazy import + actionable
error" pattern is standard practice in production Python libraries.
"""

import logging

from app.core.exceptions import ConfigurationError
from app.services.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Embeddings computed in-process with a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=local requires the 'sentence-transformers' "
                "package. Install it with: pip install -e \".[local]\""
            ) from exc

        logger.info("Loading local embedding model '%s'...", model_name)
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        logger.info("Local embedding model ready")

    @property
    def model(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # normalize_embeddings=True -> unit vectors, so cosine distance in
        # Chroma behaves identically to the hosted providers' embeddings.
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]
