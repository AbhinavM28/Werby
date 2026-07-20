"""OpenAI-hosted providers.

This is the previous ``EmbeddingService``/``LLMService`` transport code,
relocated behind the provider interfaces. Nothing about the retry, batching,
or error-translation behavior changed -- only where it lives. That's the
essence of the "extract interface" refactoring: behavior-preserving.
"""

import logging

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import LLMServiceError
from app.services.providers.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APIConnectionError)
_EMBED_BATCH_SIZE = 100


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via the OpenAI API, batched and retried."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return f"openai/{self._model}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + _EMBED_BATCH_SIZE]))
        logger.debug("Embedded %d texts with %s", len(texts), self.model)
        return vectors

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self._model, input=batch)
        except _RETRYABLE:
            logger.warning("Transient OpenAI embedding error; retrying...")
            raise
        except APIStatusError as exc:
            raise LLMServiceError(
                f"OpenAI embedding request failed ({exc.status_code}): {exc.message}"
            ) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]


class OpenAILLMProvider(LLMProvider):
    """Chat completions via the OpenAI API, retried on transient errors."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return f"openai/{self._model}"

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except _RETRYABLE:
            logger.warning("Transient OpenAI chat error; retrying...")
            raise
        except APIStatusError as exc:
            raise LLMServiceError(
                f"OpenAI chat completion failed ({exc.status_code}): {exc.message}"
            ) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMServiceError("OpenAI returned an empty response")
        return content.strip()
