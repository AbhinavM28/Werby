"""Retrieval and faithfulness evaluation harness.

This is a third "use case" alongside ``IngestionService`` and ``RAGService``:
it composes an existing ``RAGService`` (never re-implements embedding or
search) to answer a fixed question, and reports whether retrieval found the
right document and whether the generated answer stayed grounded in it.

Three metrics, three different costs:

* **Hit-rate** and **score distribution** cost nothing extra -- they read
  straight off ``RAGService.retrieve()``, which every query already computes.
  The score distribution specifically separates in-scope questions (where a
  correct source is expected) from out-of-scope ones (where none is), because
  the *gap* between those two populations -- not either number alone -- is
  what a future relevance threshold would sit between. Hit-rate is measured
  twice, pre- and post-rerank (via ``RAGService.retrieve()`` and ``rerank()``
  independently) -- the gap between *those* two numbers is what a reranker is
  actually worth, separate from whatever a relevance threshold contributes.
* **Keyword faithfulness** is free and deterministic: a case-insensitive
  substring check against the generated answer. Shallow (lexical, not
  semantic) but has zero cost and zero flakiness, so it always runs when a
  case supplies ``expected_keywords``. Some questions have more than one
  textually-correct answer -- different true clauses of the same source --
  where a single fixed keyword set would fail a correct answer that simply
  led with a different (but equally true) clause than the dataset author
  wrote first. ``expected_keyword_groups`` handles that: one or more
  AND-groups, passing if the answer fully satisfies ANY one. This is a
  dataset-authoring fix, not a matching-algorithm one -- a fuzzy/synonym
  matcher would only help if the model reworded the *same* fact, and this is
  about two *different* true facts both being acceptable.
* **LLM-judge faithfulness** is the only metric that costs a real model call
  and is never deterministic. It is entirely optional (``use_judge=False``
  skips it) and injected through ``FaithfulnessJudge`` -- an ABC mirroring
  ``LLMProvider`` -- so unit tests supply a fake judge and never touch a
  network, while ``scripts/evaluate.py`` wires a real one.

The judge is asked for two structured signals: a list of unsupported claims,
and a summary ``FAITHFUL: yes/no`` field. The claims list is treated as the
more reliable one, because it's what the model is actually asked to
*produce* -- concrete, individually-checkable items -- while the yes/no field
is a self-reported summary layered on top of that work, and self-reported
summaries are exactly where a model tends to drift from its own evidence. We
hit this for real: a judge response with ``FAITHFUL: no``, an empty
unsupported-claims list, and a rationale that read as an endorsement. Rather
than trust either signal blindly, a disagreement between them becomes its
own ``INCONSISTENT`` verdict -- visible in the report as its own count --
instead of being silently resolved one way or the other.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.exceptions import EvaluationDatasetError
from app.services.llm_service import build_context
from app.services.providers.base import LLMProvider
from app.services.rag_service import RAGService
from app.services.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class EvalCase(BaseModel):
    """One test case from a human-edited dataset file.

    ``extra="forbid"`` is deliberate: this file is hand-edited YAML, and a
    typo'd field name (``expcted_source``) would otherwise be silently
    dropped by pydantic's default "ignore extra fields" behavior -- turning a
    typo into a silently-wrong test case instead of a loud load-time error.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_source: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    expected_keyword_groups: list[list[str]] | None = None
    notes: str | None = None

    @property
    def in_scope(self) -> bool:
        """Whether this case expects retrieval to find a specific source.

        A case with no ``expected_source`` (e.g. a deliberately unrelated
        question) is out-of-scope by construction -- there is nothing for
        hit-rate to check, only a score to observe.
        """
        return self.expected_source is not None

    @property
    def keyword_groups(self) -> list[list[str]]:
        """One or more AND-groups; the case passes if the answer fully
        satisfies ANY one group -- see ``expected_keyword_groups``'s
        docstring above for why. A plain ``expected_keywords`` list is just
        the single-group case, normalized here so ``_check_keywords`` never
        needs to branch on which field a case actually used.
        """
        if self.expected_keyword_groups is not None:
            return self.expected_keyword_groups
        return [self.expected_keywords]

    @model_validator(mode="after")
    def _validate_keyword_fields(self) -> "EvalCase":
        """expected_keywords and expected_keyword_groups are mutually
        exclusive (ambiguous which should apply if both are set), and every
        group in expected_keyword_groups must be non-empty -- an empty
        group would trivially satisfy len(matched) == len(group) (0 == 0),
        silently auto-passing the case regardless of any other group.
        """
        if self.expected_keyword_groups is not None:
            if self.expected_keywords:
                raise ValueError(
                    "Set either expected_keywords or expected_keyword_groups, "
                    "not both -- ambiguous which should apply."
                )
            if not self.expected_keyword_groups or any(
                not group for group in self.expected_keyword_groups
            ):
                raise ValueError(
                    "expected_keyword_groups must be a non-empty list of "
                    "non-empty keyword groups -- an empty group would "
                    "trivially auto-pass every case."
                )
        return self


def load_dataset(path: Path) -> list[EvalCase]:
    """Load and validate an evaluation dataset from a YAML file.

    Raises:
        EvaluationDatasetError: If the file is missing, isn't valid YAML,
            isn't a list of cases, or a case fails schema validation.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationDatasetError(f"Cannot read dataset '{path}': {exc}") from exc

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise EvaluationDatasetError(f"Malformed YAML in '{path}': {exc}") from exc

    if not isinstance(raw, list):
        raise EvaluationDatasetError(
            f"'{path}' must contain a YAML list of cases, got "
            f"{type(raw).__name__}"
        )

    try:
        cases = [EvalCase.model_validate(entry) for entry in raw]
    except ValidationError as exc:
        raise EvaluationDatasetError(f"Invalid eval case in '{path}': {exc}") from exc

    if not cases:
        raise EvaluationDatasetError(f"'{path}' contains no cases")

    logger.info("Loaded %d eval case(s) from '%s'", len(cases), path)
    return cases


@dataclass(frozen=True)
class KeywordCheck:
    """Deterministic, free faithfulness signal: keyword presence in the answer.

    ``expected``/``matched`` describe the single best-matching AND-group out
    of the case's ``keyword_groups`` (see ``EvalCase``) -- whichever group is
    fully matched, or failing that, whichever came closest. For a case using
    the plain ``expected_keywords`` field there's only ever one group, so
    this is exactly today's behavior; for ``expected_keyword_groups`` cases
    it's the group the report should actually show, not an arbitrary one.
    """

    expected: list[str]
    matched: list[str]

    @property
    def passed(self) -> bool | None:
        """None when the case supplied no keywords -- there's nothing to check."""
        if not self.expected:
            return None
        return len(self.matched) == len(self.expected)


class FaithfulnessVerdict(Enum):
    """Outcome of judging one answer against its retrieved context.

    A plain bool can't represent "the judge's own signals disagreed" without
    silently picking a side -- INCONSISTENT exists so that disagreement stays
    a visible, distinct outcome instead of being folded into faithful or
    unfaithful by default.
    """

    FAITHFUL = "faithful"
    UNFAITHFUL = "unfaithful"
    INCONSISTENT = "inconsistent"


_VERDICT_LABELS: dict[FaithfulnessVerdict, str] = {
    FaithfulnessVerdict.FAITHFUL: "yes",
    FaithfulnessVerdict.UNFAITHFUL: "no",
    FaithfulnessVerdict.INCONSISTENT: "inconsistent",
}


@dataclass(frozen=True)
class FaithfulnessResult:
    """Verdict from a ``FaithfulnessJudge``."""

    verdict: FaithfulnessVerdict
    unsupported_claims: list[str]
    rationale: str


class FaithfulnessJudge(ABC):
    """Judges whether an answer is grounded in its retrieved context.

    Deliberately mirrors ``LLMProvider``: a narrow interface so real
    implementations depend on a model, and tests depend on nothing.
    """

    @abstractmethod
    def judge(self, question: str, context: str, answer: str) -> FaithfulnessResult:
        """Return whether every claim in ``answer`` is supported by ``context``."""


_JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker reviewing an AI \
assistant's answer against the documentation it was given. Identify any \
claim in the answer that is NOT directly supported by the context.

Respond in exactly this format, with no extra commentary:
FAITHFUL: yes|no
UNSUPPORTED: comma-separated unsupported claims, or "none"
RATIONALE: one sentence explaining the verdict
"""


class LLMFaithfulnessJudge(FaithfulnessJudge):
    """Faithfulness judge backed by any ``LLMProvider``.

    Reuses the exact provider abstraction ``LLMService`` already uses for
    answer generation -- judging needs no new provider infrastructure, only
    a different, narrower prompt.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def judge(self, question: str, context: str, answer: str) -> FaithfulnessResult:
        user_prompt = (
            f"Documentation context:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Generated answer: {answer}"
        )
        raw = self._provider.generate(_JUDGE_SYSTEM_PROMPT, user_prompt)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> FaithfulnessResult:
        """Parse the judge's fixed-format response into a verdict.

        The judge is an LLM producing free text at a system boundary we
        don't control -- malformed output must not crash the eval run, so a
        total parse failure (no usable FAITHFUL field at all) degrades to a
        conservative UNFAITHFUL verdict with the raw text preserved, rather
        than raising.

        When the response *does* parse, the unsupported-claims list is
        treated as the primary signal and FAITHFUL yes/no as a cross-check,
        not the other way around -- see the module docstring for why. If the
        two disagree (e.g. FAITHFUL: no but zero unsupported claims listed),
        neither is trustworthy alone, so the case is recorded as
        INCONSISTENT instead of silently picking one side.
        """
        fields: dict[str, str] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key.strip().upper()] = value.strip()

        rationale = fields.get("RATIONALE", "")
        faithful_field = fields.get("FAITHFUL", "").strip().lower()

        if "UNSUPPORTED" not in fields or faithful_field not in ("yes", "no"):
            # Total parse failure: there's no second signal to disagree with,
            # just a response that didn't follow the format -- a different
            # failure mode than a genuine disagreement between two real
            # signals, so it stays UNFAITHFUL rather than INCONSISTENT.
            logger.warning(
                "Unparseable judge response, treating as unfaithful: %.200s", raw
            )
            fallback = f"Could not parse judge response: {raw.strip()[:200]}"
            return FaithfulnessResult(
                verdict=FaithfulnessVerdict.UNFAITHFUL,
                unsupported_claims=[],
                rationale=rationale or fallback,
            )

        unsupported_raw = fields["UNSUPPORTED"]
        unsupported = (
            []
            if unsupported_raw.strip().lower() == "none"
            else [c.strip() for c in unsupported_raw.split(",") if c.strip()]
        )

        claims_say_faithful = not unsupported
        field_says_faithful = faithful_field == "yes"

        if claims_say_faithful == field_says_faithful:
            verdict = (
                FaithfulnessVerdict.FAITHFUL
                if claims_say_faithful
                else FaithfulnessVerdict.UNFAITHFUL
            )
        else:
            verdict = FaithfulnessVerdict.INCONSISTENT
            logger.warning(
                "Judge gave contradictory signals (FAITHFUL: %s, unsupported "
                "claims: %s) -- recording as inconsistent: %.200s",
                faithful_field, unsupported or "none", raw,
            )

        return FaithfulnessResult(
            verdict=verdict, unsupported_claims=unsupported, rationale=rationale
        )


@dataclass(frozen=True)
class CaseResult:
    """Outcome of running one ``EvalCase`` through the pipeline."""

    case: EvalCase
    retrieved: list[RetrievedChunk]  # final, post-rerank candidates
    rank: int | None  # post-rerank 1-indexed position of expected_source; None = miss
    top_score: float | None  # post-rerank score of the #1 chunk, if any
    pre_rerank_rank: int | None  # rank within the wide dense-search pool
    pre_rerank_top_score: float | None  # top score within that wide pool
    keyword_check: KeywordCheck
    faithfulness: FaithfulnessResult | None  # None if the judge wasn't run
    answer: str | None  # None if no answer needed to be generated

    @property
    def hit(self) -> bool | None:
        """True/False for in-scope cases; None for out-of-scope (no target)."""
        if not self.case.in_scope:
            return None
        return self.rank is not None

    @property
    def pre_rerank_hit(self) -> bool | None:
        """Same as ``hit``, but measured against the wide pre-rerank pool.

        The gap between this and ``hit`` across a whole report is the value
        reranking adds -- see ``EvaluationReport.pre_rerank_hit_rate``.
        """
        if not self.case.in_scope:
            return None
        return self.pre_rerank_rank is not None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values), 4)


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate results across an evaluation run."""

    results: list[CaseResult] = field(default_factory=list)

    @property
    def in_scope_results(self) -> list[CaseResult]:
        return [r for r in self.results if r.case.in_scope]

    @property
    def out_of_scope_results(self) -> list[CaseResult]:
        return [r for r in self.results if not r.case.in_scope]

    @property
    def hit_rate(self) -> float | None:
        """Percentage of in-scope cases whose expected source was retrieved.

        None (not 0.0) when there are no in-scope cases -- "no data" and
        "zero percent" are different facts and shouldn't be conflated.
        """
        in_scope = self.in_scope_results
        if not in_scope:
            return None
        hits = sum(1 for r in in_scope if r.hit)
        return round(hits / len(in_scope) * 100, 2)

    @property
    def pre_rerank_hit_rate(self) -> float | None:
        """Same as ``hit_rate``, measured against the wide pre-rerank pool.

        Without a reranker configured, this comes out ~equal to ``hit_rate``
        (reranking is a no-op truncation of the same pool). Once one is
        configured, the difference between the two rates is the reranker's
        measured contribution -- the number this feature exists to produce.
        """
        in_scope = self.in_scope_results
        if not in_scope:
            return None
        hits = sum(1 for r in in_scope if r.pre_rerank_hit)
        return round(hits / len(in_scope) * 100, 2)

    @property
    def mean_in_scope_score(self) -> float | None:
        return _mean(
            r.top_score for r in self.in_scope_results if r.top_score is not None
        )

    @property
    def mean_out_of_scope_score(self) -> float | None:
        return _mean(
            r.top_score for r in self.out_of_scope_results if r.top_score is not None
        )

    @property
    def score_gap(self) -> float | None:
        """Mean in-scope top score minus mean out-of-scope top score.

        This is the number a future relevance threshold would sit between: a
        large positive gap means in-domain and out-of-domain questions are
        cleanly separable by score alone. None if either population is empty.
        """
        in_scope, out_of_scope = self.mean_in_scope_score, self.mean_out_of_scope_score
        if in_scope is None or out_of_scope is None:
            return None
        return round(in_scope - out_of_scope, 4)

    @property
    def keyword_pass_rate(self) -> float | None:
        checked = [r for r in self.results if r.keyword_check.passed is not None]
        if not checked:
            return None
        passed = sum(1 for r in checked if r.keyword_check.passed)
        return round(passed / len(checked) * 100, 2)

    @property
    def faithfulness_pass_rate(self) -> float | None:
        """Percentage of judged cases with a clean FAITHFUL verdict.

        Inconsistent cases count toward the denominator (they were judged)
        but not the numerator (an inconsistent verdict is not a pass) -- the
        same treatment an unfaithful verdict gets for this specific metric.
        See ``faithfulness_inconsistent_count`` for how many of those
        "not a pass" cases were inconsistent rather than genuinely unfaithful.
        """
        judged = [r for r in self.results if r.faithfulness is not None]
        if not judged:
            return None
        passed = sum(
            1
            for r in judged
            if r.faithfulness and r.faithfulness.verdict is FaithfulnessVerdict.FAITHFUL
        )
        return round(passed / len(judged) * 100, 2)

    @property
    def faithfulness_inconsistent_count(self) -> int | None:
        """Count of judged cases where the judge's own signals disagreed.

        Surfaced separately from the pass-rate so a disagreement stays
        visible instead of being silently counted as a pass or a fail --
        it's a comment on judge reliability for that case, not a verdict on
        the answer itself. None (not 0) when nothing was judged.
        """
        judged = [r for r in self.results if r.faithfulness is not None]
        if not judged:
            return None
        return sum(
            1
            for r in judged
            if r.faithfulness
            and r.faithfulness.verdict is FaithfulnessVerdict.INCONSISTENT
        )


class EvaluationService:
    """Runs an evaluation dataset against a real ``RAGService``.

    Deliberately depends on ``RAGService`` itself, not on its constituent
    parts -- composing the existing use case is what "reuse RAGService, don't
    duplicate retrieval logic" means in practice, mirroring how
    ``scripts/ingest.py`` composes ``IngestionService`` rather than calling
    the document processor and vector store directly.
    """

    def __init__(
        self,
        rag: RAGService,
        judge: FaithfulnessJudge,
        default_top_k: int = 5,
        max_context_chars: int = 12_000,
        retrieve_n: int = 20,
    ) -> None:
        self._rag = rag
        self._judge = judge
        self._default_top_k = default_top_k
        self._max_context_chars = max_context_chars
        self._retrieve_n = retrieve_n

    def run(
        self,
        cases: list[EvalCase],
        top_k: int | None = None,
        use_judge: bool = True,
    ) -> EvaluationReport:
        """Run every case and return an aggregate report.

        Raises:
            EmptyCorpusError: If no documents have been ingested yet.
        """
        k = top_k or self._default_top_k
        results = [self._run_case(case, k, use_judge) for case in cases]
        logger.info("Evaluation complete: %d case(s) run", len(results))
        return EvaluationReport(results=results)

    def _run_case(self, case: EvalCase, top_k: int, use_judge: bool) -> CaseResult:
        # Rank/top_score are always measured via direct retrieve()/rerank()
        # calls, never via query()'s -- RAGService.query() applies a
        # relevance threshold before generating an answer, and if rank/score
        # came from it, a case that happens to need an answer (has
        # expected_keywords or the judge is on) would silently measure
        # different retrieval than every other case. That would make
        # hit-rate depend on an unrelated flag instead of on retrieval
        # quality, which is the one thing this harness exists to measure
        # honestly.
        #
        # Two checkpoints, both from RAGService itself (never reimplemented
        # here): the wide pre-rerank pool, and what reranking narrows it to.
        # Without a reranker configured, RAGService.rerank() is a no-op
        # truncation, so the two will come out ~equal -- that's expected, not
        # a bug, and is itself useful confirmation that nothing changed.
        pool = self._rag.retrieve(case.question, top_k=self._retrieve_n)
        pre_rank = (
            self._rank_of(case.expected_source, pool) if case.expected_source else None
        )
        pre_top_score = pool[0].score if pool else None

        retrieved = self._rag.rerank(case.question, pool, top_k=top_k)
        rank = (
            self._rank_of(case.expected_source, retrieved)
            if case.expected_source
            else None
        )
        top_score = retrieved[0].score if retrieved else None

        # Only pay for an LLM answer when a metric actually needs one -- an
        # out-of-scope case with no keyword requirement and no judge only
        # needs retrieval, which is free of LLM cost.
        needs_answer = any(case.keyword_groups) or use_judge
        answer: str | None = None
        answer_context: list[RetrievedChunk] = []
        if needs_answer:
            result = self._rag.query(case.question, top_k=top_k)
            answer, answer_context = result.answer, result.sources

        keyword_check = self._check_keywords(case.keyword_groups, answer)

        faithfulness = None
        if use_judge and answer is not None:
            # Judge against exactly what the LLM saw (post-filter), not the
            # raw retrieval above -- otherwise a chunk the threshold filtered
            # out could make a correct refusal look "unfaithful" to context
            # the model never actually used.
            context = build_context(answer_context, max_chars=self._max_context_chars)
            faithfulness = self._judge.judge(case.question, context, answer)

        return CaseResult(
            case=case,
            retrieved=retrieved,
            rank=rank,
            top_score=top_score,
            pre_rerank_rank=pre_rank,
            pre_rerank_top_score=pre_top_score,
            keyword_check=keyword_check,
            faithfulness=faithfulness,
            answer=answer,
        )

    @staticmethod
    def _rank_of(expected_source: str, retrieved: list[RetrievedChunk]) -> int | None:
        for i, chunk in enumerate(retrieved, start=1):
            if chunk.source_document == expected_source:
                return i
        return None

    @staticmethod
    def _check_keywords(groups: list[list[str]], answer: str | None) -> KeywordCheck:
        """Check each AND-group against the answer; report whichever group
        is fully matched (first one, in declaration order), or failing that,
        whichever came closest -- see KeywordCheck's docstring for why."""
        if answer is None:
            first = groups[0] if groups else []
            return KeywordCheck(expected=first, matched=[])

        lowered = answer.lower()
        scored = [
            (group, [kw for kw in group if kw.lower() in lowered]) for group in groups
        ]
        for group, matched in scored:
            if group and len(matched) == len(group):
                return KeywordCheck(expected=group, matched=matched)

        best_group, best_matched = max(
            scored, key=lambda pair: len(pair[1]), default=([], [])
        )
        return KeywordCheck(expected=best_group, matched=best_matched)


def format_report(report: EvaluationReport) -> str:
    """Render a report as a plain, readable console summary.

    A pure function from data to string -- not print statements inside the
    service -- so writing to a file later is `path.write_text(format_report(r))`
    with no change to EvaluationService, and this function is unit-testable
    without capturing stdout.
    """
    lines: list[str] = []
    in_scope_n = len(report.in_scope_results)
    out_scope_n = len(report.out_of_scope_results)

    lines.append("Werby Evaluation Report")
    lines.append("=" * 24)
    lines.append(
        f"Cases: {len(report.results)} ({in_scope_n} in-scope, "
        f"{out_scope_n} out-of-scope)"
    )
    lines.append("")
    lines.append("Per-question results")
    lines.append("-" * 21)
    for r in report.results:
        if not r.case.in_scope:
            status = "  --  "
        elif r.hit:
            status = f"HIT r{r.rank}"
        else:
            status = " MISS "

        score = f"{r.top_score:.4f}" if r.top_score is not None else "n/a"
        kw = (
            "n/a"
            if r.keyword_check.passed is None
            else f"{len(r.keyword_check.matched)}/{len(r.keyword_check.expected)}"
        )
        faithful = "n/a"
        if r.faithfulness is not None:
            faithful = _VERDICT_LABELS[r.faithfulness.verdict]

        rerank_note = ""
        if r.case.in_scope and r.pre_rerank_rank != r.rank:
            pre_label = (
                f"r{r.pre_rerank_rank}" if r.pre_rerank_rank is not None else "miss"
            )
            rerank_note = f"  [pre-rerank: {pre_label}]"

        lines.append(
            f"[{status}] {r.case.id:<28} score={score}  "
            f"keywords={kw}  faithful={faithful}{rerank_note}"
        )
        lines.append(f"    Q: {r.case.question}")

    lines.append("")
    lines.append("Aggregate")
    lines.append("-" * 9)

    def fmt_pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f}%"

    def fmt_score(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    lines.append(f"Hit-rate:                 {fmt_pct(report.hit_rate)}")
    lines.append(f"Pre-rerank hit-rate:      {fmt_pct(report.pre_rerank_hit_rate)}")
    if report.hit_rate is not None and report.pre_rerank_hit_rate is not None:
        delta = round(report.hit_rate - report.pre_rerank_hit_rate, 2)
        lines.append(f"Reranking delta:          {delta:+.1f}%")
    lines.append(f"Mean in-scope score:      {fmt_score(report.mean_in_scope_score)}")
    lines.append(
        f"Mean out-of-scope score:  {fmt_score(report.mean_out_of_scope_score)}"
    )
    lines.append(f"Score gap (in - out):     {fmt_score(report.score_gap)}")
    lines.append(f"Keyword pass-rate:        {fmt_pct(report.keyword_pass_rate)}")
    lines.append(f"Faithfulness pass-rate:   {fmt_pct(report.faithfulness_pass_rate)}")

    inconsistent = report.faithfulness_inconsistent_count
    inconsistent_text = "n/a" if inconsistent is None else str(inconsistent)
    lines.append(f"Faithfulness inconsistent: {inconsistent_text}")

    return "\n".join(lines)
