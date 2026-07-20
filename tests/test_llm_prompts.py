"""Unit tests for prompt construction (pure functions, zero API calls)."""

from app.services.llm_service import SYSTEM_PROMPT, build_context, build_user_prompt
from app.services.vector_store import RetrievedChunk


def make_chunk(i: int, text: str = "torque spec 45 Nm") -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source_document=f"doc{i}.pdf", chunk_index=i, score=0.9
    )


def test_context_numbers_sources_in_order() -> None:
    context = build_context([make_chunk(0), make_chunk(1)], max_chars=10_000)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert context.index("[Source 1]") < context.index("[Source 2]")


def test_context_respects_char_budget() -> None:
    chunks = [make_chunk(i, text="x" * 500) for i in range(20)]
    context = build_context(chunks, max_chars=1200)
    assert len(context) <= 1300  # budget + separators
    assert "[Source 1]" in context


def test_context_always_includes_first_chunk_even_if_over_budget() -> None:
    context = build_context([make_chunk(0, text="y" * 5000)], max_chars=100)
    assert "[Source 1]" in context


def test_user_prompt_contains_question_and_context() -> None:
    prompt = build_user_prompt("What is the rated load?", "[Source 1] 2000 kg")
    assert "What is the rated load?" in prompt
    assert "2000 kg" in prompt


def test_system_prompt_enforces_grounding() -> None:
    assert "ONLY" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT
