"""Local cross-encoder reranking via sentence-transformers.

Why a cross-encoder catches what dense search (a bi-encoder) misses: a
bi-encoder -- what ``EmbeddingProvider`` does -- embeds the query and each
chunk *independently* into fixed vectors, then compares them with a single
cosine similarity. The query and chunk text never actually interact, which is
exactly how a long, vocabulary-dense document can outscore a short, specific
one that's the *actually correct* answer (the real failure this feature
fixes: a query about one specific machine's lockout/tagout procedure
retrieved five chunks from the general OSHA standard and zero from that
machine's own spec, because the general standard has far more matching
vocabulary). A cross-encoder takes ``(query, chunk)`` as one joint input and
lets the model attend across both simultaneously, scoring "does this text
answer this question" directly instead of "are these two vectors close" --
far more accurate, but too slow to run against an entire corpus. That's why
this is retrieve-*then*-rerank: cheap dense search narrows the corpus to a
candidate pool (``retrieve_n``, see ``app/core/config.py``), and only that
pool pays the cross-encoder's cost.

Optional extra (``pip install -e ".[local]"``), same as ``local_embeddings.py``
and the same lazy-import pattern -- ``CrossEncoder`` ships in the same
``sentence-transformers`` package local embeddings already use, so this rides
that existing extra rather than adding a new one.
"""

import logging
from dataclasses import replace

from app.core.exceptions import ConfigurationError
from app.services.providers.base import Reranker
from app.services.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class LocalCrossEncoderReranker(Reranker):
    """Reranks retrieved chunks in-process with a sentence-transformers CrossEncoder."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ConfigurationError(
                "rerank_enabled=true requires the 'sentence-transformers' "
                "package. Install it with: pip install -e \".[local]\""
            ) from exc

        logger.info("Loading cross-encoder reranker model '%s'...", model_name)
        self._model_name = model_name
        self._model = CrossEncoder(model_name)
        logger.info("Reranker model ready")

    @property
    def model(self) -> str:
        return self._model_name

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        # Original bi-encoder .score is left untouched -- only order changes.
        # See rag_service.py's rerank() docstring for why. The cross-encoder's
        # own score is attached separately as rerank_score, unbounded model
        # logits rather than the [0, 1] cosine scale -- see that field's
        # docstring on RetrievedChunk for what it's used for.
        return [
            replace(chunk, rerank_score=float(score))
            for chunk, score in ranked[:top_k]
        ]
