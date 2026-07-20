"""Ollama providers: fully local inference for proprietary documentation.

Ollama (https://ollama.com) runs open-weight models (Llama, Mistral, Qwen,
nomic-embed-text, ...) on your own hardware and exposes a small HTTP API on
localhost. With ``LLM_PROVIDER=ollama`` and ``EMBEDDING_PROVIDER=ollama``,
Werby runs with **zero external network calls** -- no API keys, no per-token
cost, and documentation never leaves the machine. This is the deployment mode
for air-gapped industrial sites and export-controlled documentation.

We call Ollama's REST API directly with ``httpx`` rather than pulling in an
SDK: it's two endpoints, and owning the transport keeps the dependency
surface small and the error messages precise.

Setup (one-time, on the host):
    ollama pull llama3.1:8b        # chat model  (~4.7 GB)
    ollama pull nomic-embed-text   # embedding model (~270 MB)
"""

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import LLMServiceError
from app.services.providers.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

# Local inference can be slow on CPU; give generation generous headroom.
_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
_RETRYABLE = (httpx.TransportError,)  # connection resets, timeouts mid-load
_EMBED_BATCH_SIZE = 64


def _connection_hint(base_url: str) -> str:
    return (
        f"Could not reach Ollama at {base_url}. Is it running? "
        "Start it with 'ollama serve' (or open the Ollama app) and ensure "
        "the model is pulled, e.g. 'ollama pull llama3.1:8b'."
    )


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeddings via a local Ollama server (e.g. nomic-embed-text)."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def model(self) -> str:
        return f"ollama/{self._model}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + _EMBED_BATCH_SIZE]))
        return vectors

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": batch},
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMServiceError(_connection_hint(self._base_url)) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMServiceError(
                f"Ollama embedding failed ({exc.response.status_code}): "
                f"{exc.response.text[:300]}"
            ) from exc

        embeddings = response.json().get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise LLMServiceError(
                f"Ollama returned {len(embeddings or [])} embeddings "
                f"for {len(batch)} inputs (model '{self._model}' pulled?)"
            )
        return embeddings


class OllamaLLMProvider(LLMProvider):
    """Chat generation via a local Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return f"ollama/{self._model}"

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                    },
                },
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMServiceError(_connection_hint(self._base_url)) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMServiceError(
                f"Ollama generation failed ({exc.response.status_code}): "
                f"{exc.response.text[:300]}"
            ) from exc

        content = (response.json().get("message") or {}).get("content", "")
        if not content.strip():
            raise LLMServiceError(
                f"Ollama returned an empty response from '{self._model}'"
            )
        return content.strip()
