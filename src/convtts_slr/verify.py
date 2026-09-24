"""Independent verification (kept separate from the extractor on purpose).

1. Grounding: every System-Two quote must actually occur in the parsed paper.
2. Second opinion: a different backend re-answers the highest-stakes typed questions
   on the same state; disagreements go to the human queue instead of into the data.
"""

import re

from rapidfuzz import fuzz

from .models import DatasetFacts, ExtractionResult, VerificationResult
from .protocol import Protocol, Question
from .screening import Decider

DEFAULT_CROSS_CHECK = [
    "ic4_pitch_predictor",
    "ic4_energy_predictor",
    "ic4_duration_predictor",
    "rq1_sample_unit",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"-\s+", "", s)).strip().lower()


def quote_found(quote: str, text: str, threshold: float = 90.0) -> bool:
    q, t = _norm(quote), _norm(text)
    if not q:
        return True
    if q in t:
        return True
    return fuzz.partial_ratio(q, t, score_cutoff=threshold) >= threshold


def missing_quotes(facts: DatasetFacts | None, text: str) -> list[str]:
    if facts is None:
        return []
    bad = []
    for name in DatasetFacts.model_fields:
        item = getattr(facts, name)
        if item.value not in (None, "", []) and not item.quote:
            bad.append(f"{name}:no_quote")
        elif item.quote and not quote_found(item.quote, text):
            bad.append(f"{name}:quote_not_in_text")
    return bad


def _disagree(a, b) -> bool:
    if a.kind == "noul":
        return (a.score >= 0.5) != (b.score >= 0.5)
    return a.choice != b.choice


def verify(
    ex: ExtractionResult,
    state: dict,
    text: str,
    protocol: Protocol,
    checker: Decider | None,
    screening_answers: dict,
    cross_check: list[str] = DEFAULT_CROSS_CHECK,
) -> VerificationResult:
    missing = missing_quotes(ex.facts, text)
    disagreements: list[str] = []
    if checker is not None:
        all_q: dict[str, Question] = {q.id: q for c in protocol.criteria for q in c.questions}
        all_q |= {q.id: q for q in protocol.extraction}
        primary = {**screening_answers, **ex.typed.answers}
        qs = [all_q[i] for i in cross_check if i in all_q and i in primary]
        if qs:
            second = checker.evaluate(state, qs, ex.paper_id)
            disagreements = [q.id for q in qs if _disagree(primary[q.id], second.answers[q.id])]
    return VerificationResult(
        paper_id=ex.paper_id,
        missing_quotes=missing,
        disagreements=disagreements,
        passed=not missing and not disagreements,
    )
