"""LLM answer service: prompt engineering, decoupled from transport.

After the provider refactor, this module's responsibility narrowed to what it
was always really about: *what we say to the model*. The system prompt,
context assembly, and prompt formatting are Werby's domain logic and apply
identically whether the model is GPT-4o-mini in OpenAI's cloud or Llama 3.1
on an air-gapped workstation. Transport, retries, and provider parameters
now live behind the ``LLMProvider`` interface.

Prompt-engineering decisions, and why:

* **Grounding instruction**: the model is told to answer ONLY from provided
  context and to say so when the context is insufficient. This is the main
  defense against hallucination -- critical in an industrial setting where a
  wrong torque spec or rated load is a safety issue, not a UX bug.
* **Numbered source blocks**: each chunk is wrapped in ``[Source N]`` markers
  so the model can cite which document supported each claim.
* **Low temperature** (configured per provider): deterministic, factual
  answers, not creativity.
"""

import logging

from app.services.providers.base import LLMProvider
from app.services.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Werby, an AI Engineering Copilot for warehouse and \
industrial engineers. You answer questions using ONLY the documentation \
excerpts provided in the context below.

Rules:
1. Ground every claim in the provided context. Never invent specifications, \
part numbers, tolerances, load ratings, or procedures.
2. If the context does not contain the answer, say exactly that and suggest \
what documentation the engineer should consult or ingest.
3. When you use information from a source, reference it inline like [Source 2].
4. For safety-critical values (loads, pressures, clearances, lockout/tagout \
steps), quote the value precisely as written in the context.
5. Be concise and technical. Your audience consists of professional engineers.
"""


def build_context(chunks: list[RetrievedChunk], max_chars: int) -> str:
    """Assemble retrieved chunks into a numbered context block.

    Chunks arrive sorted by relevance; we add them in that order and stop
    before exceeding ``max_chars`` so the prompt never blows past our budget.
    """
    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        block = (
            f"[Source {i}] (document: {chunk.source_document}, "
            f"chunk {chunk.chunk_index})\n{chunk.text}\n"
        )
        if used + len(block) > max_chars and parts:
            logger.debug("Context budget reached at source %d", i)
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts)


def build_user_prompt(question: str, context: str) -> str:
    """Combine context and question into the final user message."""
    return (
        f"Documentation context:\n\n{context}\n\n"
        f"Engineer's question: {question}"
    )


class LLMService:
    """Generates grounded answers through whichever LLMProvider is configured."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @property
    def model(self) -> str:
        return self._provider.model

    def generate_answer(self, question: str, context: str) -> str:
        """Produce a grounded answer for a question given retrieved context.

        Raises:
            LLMServiceError: If the provider fails after its retries.
        """
        return self._provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(question, context),
        )
