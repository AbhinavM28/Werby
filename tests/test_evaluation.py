"""Unit tests for the evaluation harness.

Hermetic throughout: retrieval is driven by a hand-rolled fake VectorStore
keyed by question marker (no real vector math, no network), the LLM is a
MagicMock exactly as in test_rag_service.py, and the faithfulness judge is a
hand-rolled fake implementing FaithfulnessJudge -- never a real model call.
"""

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import EmptyCorpusError, EvaluationDatasetError
from app.services.document_processor import DocumentChunk
from app.services.evaluation import (
    EvalCase,
    EvaluationService,
    FaithfulnessJudge,
    FaithfulnessResult,
    FaithfulnessVerdict,
    LLMFaithfulnessJudge,
    format_report,
    load_dataset,
)
from app.services.rag_service import RAGService
from app.services.vector_store import RetrievedChunk, VectorStore


class ScriptedVectorStore(VectorStore):
    """Fake store returning canned results keyed by a question marker.

    embeddings.embed_query is mocked to return [question_text] below, so this
    store can look up the right canned chunks per question without any real
    similarity computation.
    """

    def __init__(self, script: dict[str, list[RetrievedChunk]], count: int = 1) -> None:
        self._script = script
        self._count = count

    def upsert(self, chunks: list[DocumentChunk], embeddings) -> None:
        raise NotImplementedError

    def search(self, query_embedding, top_k: int) -> list[RetrievedChunk]:
        marker = query_embedding[0]
        return self._script.get(marker, [])[:top_k]

    def count(self) -> int:
        return self._count

    def list_documents(self) -> list[str]:
        docs = {c.source_document for chunks in self._script.values() for c in chunks}
        return sorted(docs)

    def delete_document(self, source_document: str) -> int:
        return 0


class FakeJudge(FaithfulnessJudge):
    """Hand-rolled fake -- proves FaithfulnessJudge is a sufficient interface."""

    def __init__(self, verdict: FaithfulnessResult) -> None:
        self.verdict = verdict
        self.calls: list[str] = []

    def judge(self, question: str, context: str, answer: str) -> FaithfulnessResult:
        self.calls.append(question)
        return self.verdict


@pytest.fixture
def embeddings() -> MagicMock:
    mock = MagicMock()
    # Marker trick: embed_query returns the question text itself wrapped in a
    # list, so ScriptedVectorStore can key retrieval results off the question
    # without any real vector math.
    mock.embed_query.side_effect = lambda text: [text]
    return mock


@pytest.fixture
def llm() -> MagicMock:
    mock = MagicMock()
    mock.generate_answer.return_value = "Rated load is 2000 kg, per the spec."
    mock.model = "gpt-test"
    return mock


PASSING = FaithfulnessResult(
    verdict=FaithfulnessVerdict.FAITHFUL, unsupported_claims=[], rationale="ok"
)


@pytest.fixture
def judge() -> "FakeJudge":
    return FakeJudge(PASSING)


HIT_Q = "What is the rated load?"
MISS_Q = "What is the wire rope inspection interval?"
OOS_Q = "What's a good sourdough recipe?"

SCRIPT = {
    HIT_Q: [
        RetrievedChunk("load: 2000 kg", "crane_spec.pdf", 0, 0.91),
        RetrievedChunk("other", "osha_1910_178.pdf", 0, 0.40),
    ],
    MISS_Q: [
        RetrievedChunk("unrelated chunk", "osha_1910_178.pdf", 0, 0.55),
    ],
    OOS_Q: [
        RetrievedChunk("unrelated chunk", "osha_1910_178.pdf", 0, 0.08),
    ],
}


def make_service(
    embeddings: MagicMock, llm: MagicMock, judge: FaithfulnessJudge
) -> EvaluationService:
    store = ScriptedVectorStore(SCRIPT)
    rag = RAGService(embeddings, store, llm, default_top_k=5)
    return EvaluationService(rag=rag, judge=judge, default_top_k=5)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def test_load_dataset_parses_valid_file(tmp_path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: case-1\n"
        "  question: What is the rated load?\n"
        "  expected_source: crane_spec.pdf\n"
        "  expected_keywords: [load]\n"
        "- id: case-2\n"
        "  question: Unrelated question\n"
        "  expected_source: null\n"
    )
    cases = load_dataset(path)
    assert len(cases) == 2
    assert cases[0].in_scope is True
    assert cases[1].in_scope is False


def test_load_dataset_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(EvaluationDatasetError, match="Cannot read"):
        load_dataset(tmp_path / "missing.yaml")


def test_load_dataset_rejects_malformed_yaml(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- id: case-1\n  question: [unterminated\n")
    with pytest.raises(EvaluationDatasetError, match="Malformed YAML"):
        load_dataset(path)


def test_load_dataset_rejects_non_list(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("id: case-1\nquestion: hi\n")
    with pytest.raises(EvaluationDatasetError, match="must contain a YAML list"):
        load_dataset(path)


def test_load_dataset_rejects_unknown_field(tmp_path) -> None:
    # extra="forbid" turns a typo like `expcted_source` into a loud load-time
    # error instead of a silently-wrong test case.
    path = tmp_path / "typo.yaml"
    path.write_text("- id: case-1\n  question: hi\n  expcted_source: doc.pdf\n")
    with pytest.raises(EvaluationDatasetError, match="Invalid eval case"):
        load_dataset(path)


def test_load_dataset_rejects_empty_list(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("[]\n")
    with pytest.raises(EvaluationDatasetError, match="no cases"):
        load_dataset(path)


# --------------------------------------------------------------------------- #
# Retrieval hit-rate and score distribution
# --------------------------------------------------------------------------- #


def test_run_computes_hit_with_rank(embeddings, llm, judge) -> None:
    service = make_service(embeddings, llm, judge)
    case = EvalCase(id="hit", question=HIT_Q, expected_source="crane_spec.pdf")

    report = service.run([case], use_judge=False)

    result = report.results[0]
    assert result.hit is True
    assert result.rank == 1
    assert result.top_score == pytest.approx(0.91)


def test_run_computes_miss(embeddings, llm, judge) -> None:
    service = make_service(embeddings, llm, judge)
    case = EvalCase(id="miss", question=MISS_Q, expected_source="crane_spec.pdf")

    report = service.run([case], use_judge=False)

    result = report.results[0]
    assert result.hit is False
    assert result.rank is None


def test_hit_rate_excludes_out_of_scope_cases(embeddings, llm, judge) -> None:
    service = make_service(embeddings, llm, judge)
    cases = [
        EvalCase(id="hit", question=HIT_Q, expected_source="crane_spec.pdf"),
        EvalCase(id="miss", question=MISS_Q, expected_source="crane_spec.pdf"),
        EvalCase(id="oos", question=OOS_Q, expected_source=None),
    ]

    report = service.run(cases, use_judge=False)

    # 1 hit / 2 in-scope cases -- the out-of-scope case must not dilute this.
    assert report.hit_rate == 50.0


def test_score_gap_separates_in_scope_from_out_of_scope(embeddings, llm, judge) -> None:
    service = make_service(embeddings, llm, judge)
    cases = [
        EvalCase(id="hit", question=HIT_Q, expected_source="crane_spec.pdf"),
        EvalCase(id="oos", question=OOS_Q, expected_source=None),
    ]

    report = service.run(cases, use_judge=False)

    assert report.mean_in_scope_score == pytest.approx(0.91)
    assert report.mean_out_of_scope_score == pytest.approx(0.08)
    assert report.score_gap == pytest.approx(0.83)


def test_score_gap_none_without_both_populations(embeddings, llm, judge) -> None:
    service = make_service(embeddings, llm, judge)
    report = service.run(
        [EvalCase(id="hit", question=HIT_Q, expected_source="crane_spec.pdf")],
        use_judge=False,
    )
    assert report.score_gap is None
    assert report.mean_out_of_scope_score is None


def test_run_raises_on_empty_corpus(embeddings, llm, judge) -> None:
    store = ScriptedVectorStore(SCRIPT, count=0)
    rag = RAGService(embeddings, store, llm, default_top_k=5)
    service = EvaluationService(rag=rag, judge=judge)

    with pytest.raises(EmptyCorpusError):
        case = EvalCase(id="hit", question=HIT_Q, expected_source="crane_spec.pdf")
        service.run([case])


# --------------------------------------------------------------------------- #
# Keyword check and cost-avoidance
# --------------------------------------------------------------------------- #


def test_keyword_check_reports_partial_match(embeddings, llm, judge) -> None:
    llm.generate_answer.return_value = "Rated load is 2000 kg."
    service = make_service(embeddings, llm, judge)
    case = EvalCase(
        id="hit", question=HIT_Q, expected_source="crane_spec.pdf",
        expected_keywords=["load", "boom"],
    )

    report = service.run([case], use_judge=False)

    check = report.results[0].keyword_check
    assert check.matched == ["load"]
    assert check.passed is False


def test_no_answer_generated_when_not_needed(embeddings, llm, judge) -> None:
    """A case with no expected_keywords and use_judge=False should only
    retrieve, never pay for an LLM answer-generation call."""
    service = make_service(embeddings, llm, judge)
    case = EvalCase(id="hit", question=HIT_Q, expected_source="crane_spec.pdf")

    report = service.run([case], use_judge=False)

    llm.generate_answer.assert_not_called()
    assert report.results[0].answer is None


def test_answer_generated_when_keywords_present(embeddings, llm, judge) -> None:
    service = make_service(embeddings, llm, judge)
    case = EvalCase(
        id="hit", question=HIT_Q, expected_source="crane_spec.pdf",
        expected_keywords=["load"],
    )

    report = service.run([case], use_judge=False)

    llm.generate_answer.assert_called_once()
    assert report.results[0].answer is not None


# --------------------------------------------------------------------------- #
# Faithfulness (LLM judge)
# --------------------------------------------------------------------------- #


def test_faithfulness_populated_when_judge_used(embeddings, llm) -> None:
    verdict = FaithfulnessResult(FaithfulnessVerdict.FAITHFUL, [], "grounded")
    judge = FakeJudge(verdict)
    service = make_service(embeddings, llm, judge)
    case = EvalCase(id="hit", question=HIT_Q, expected_source="crane_spec.pdf")

    report = service.run([case], use_judge=True)

    assert report.results[0].faithfulness == verdict
    assert judge.calls == [HIT_Q]
    assert report.faithfulness_pass_rate == 100.0
    assert report.faithfulness_inconsistent_count == 0


def test_faithfulness_none_when_judge_skipped(embeddings, llm) -> None:
    judge = FakeJudge(FaithfulnessResult(FaithfulnessVerdict.FAITHFUL, [], "grounded"))
    service = make_service(embeddings, llm, judge)
    case = EvalCase(id="hit", question=HIT_Q, expected_source="crane_spec.pdf")

    report = service.run([case], use_judge=False)

    assert report.results[0].faithfulness is None
    assert judge.calls == []
    assert report.faithfulness_pass_rate is None
    assert report.faithfulness_inconsistent_count is None


def test_aggregate_counts_inconsistent_verdicts_separately(embeddings, llm) -> None:
    """End-to-end through EvaluationService.run(): an inconsistent verdict
    shows up in its own count, not silently folded into the pass-rate."""
    verdicts = {
        HIT_Q: FaithfulnessResult(FaithfulnessVerdict.FAITHFUL, [], "ok"),
        MISS_Q: FaithfulnessResult(
            FaithfulnessVerdict.INCONSISTENT, [], "self-contradictory"
        ),
    }

    class SequencedJudge(FaithfulnessJudge):
        def judge(self, question: str, context: str, answer: str) -> FaithfulnessResult:
            return verdicts[question]

    service = make_service(embeddings, llm, SequencedJudge())
    cases = [
        EvalCase(id="a", question=HIT_Q, expected_source="crane_spec.pdf"),
        EvalCase(id="b", question=MISS_Q, expected_source="crane_spec.pdf"),
    ]

    report = service.run(cases, use_judge=True)

    # 1 clean pass / 2 judged -- the inconsistent case is not a pass, but it
    # also isn't hidden: it shows up in its own count below.
    assert report.faithfulness_pass_rate == 50.0
    assert report.faithfulness_inconsistent_count == 1


def test_llm_judge_agreement_produces_unfaithful_verdict() -> None:
    provider = MagicMock()
    provider.generate.return_value = (
        "FAITHFUL: no\nUNSUPPORTED: 3000 kg rating\nRATIONALE: not in context\n"
    )
    judge = LLMFaithfulnessJudge(provider=provider)

    result = judge.judge("q", "context", "answer")

    assert result.verdict is FaithfulnessVerdict.UNFAITHFUL
    assert result.unsupported_claims == ["3000 kg rating"]
    assert result.rationale == "not in context"


def test_llm_judge_agreement_produces_faithful_verdict() -> None:
    provider = MagicMock()
    provider.generate.return_value = (
        "FAITHFUL: yes\nUNSUPPORTED: none\nRATIONALE: fully grounded\n"
    )
    judge = LLMFaithfulnessJudge(provider=provider)

    result = judge.judge("q", "context", "answer")

    assert result.verdict is FaithfulnessVerdict.FAITHFUL
    assert result.unsupported_claims == []


def test_llm_judge_flags_contradictory_response_as_inconsistent() -> None:
    """The real bug we hit: FAITHFUL says no, but the claims list -- the
    more reliable signal -- says there's nothing unsupported, and the
    rationale reads as an endorsement. Neither signal alone is trustworthy
    here, so this must come back INCONSISTENT, not silently pass or fail."""
    provider = MagicMock()
    provider.generate.return_value = (
        "FAITHFUL: no  \n"
        "UNSUPPORTED: none  \n"
        "RATIONALE: The answer accurately reflects the conditions under which "
        "a powered industrial truck must be removed from service as stated "
        "in the provided documentation."
    )
    judge = LLMFaithfulnessJudge(provider=provider)

    result = judge.judge("q", "context", "answer")

    assert result.verdict is FaithfulnessVerdict.INCONSISTENT
    assert result.unsupported_claims == []
    assert "accurately reflects" in result.rationale


def test_llm_judge_flags_reverse_contradiction_as_inconsistent() -> None:
    """The other direction of disagreement: FAITHFUL says yes, but the
    claims list isn't actually empty. Confirms INCONSISTENT triggers on
    disagreement in general, not just the one direction we happened to hit."""
    provider = MagicMock()
    provider.generate.return_value = (
        "FAITHFUL: yes\nUNSUPPORTED: made-up torque spec\nRATIONALE: looks fine\n"
    )
    judge = LLMFaithfulnessJudge(provider=provider)

    result = judge.judge("q", "context", "answer")

    assert result.verdict is FaithfulnessVerdict.INCONSISTENT
    assert result.unsupported_claims == ["made-up torque spec"]


def test_llm_judge_handles_malformed_response_gracefully() -> None:
    provider = MagicMock()
    provider.generate.return_value = "the model rambled instead of following format"
    judge = LLMFaithfulnessJudge(provider=provider)

    result = judge.judge("q", "context", "answer")

    assert result.verdict is FaithfulnessVerdict.UNFAITHFUL
    assert "Could not parse" in result.rationale


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #


def test_format_report_surfaces_score_gap(embeddings, llm, judge) -> None:
    service = make_service(embeddings, llm, judge)
    cases = [
        EvalCase(id="hit-case", question=HIT_Q, expected_source="crane_spec.pdf"),
        EvalCase(id="oos-case", question=OOS_Q, expected_source=None),
    ]

    report = service.run(cases, use_judge=False)
    text = format_report(report)

    assert "hit-case" in text
    assert "oos-case" in text
    assert "score=0.9100" in text
    assert "score=0.0800" in text
    assert "Score gap (in - out):     0.8300" in text
    assert "Hit-rate:                 100.0%" in text
