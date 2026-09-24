"""Eligibility screening.

One batched backend call per paper per stage (all IC/EC propositions share the same
state). Answers are cached by (state, exact question text), so changing thresholds after
calibration re-routes papers without paying for new Jev calls; changing a question's
wording triggers new calls for the batches that contain it.
"""

import hashlib
import json
import threading
from typing import Any

from .backends import DecisionBackend, state_sha
from .fulltext import select_sections
from .models import CriterionOutcome, DecisionBatch, Paper, ScreeningResult, Verdict
from .protocol import Criterion, Protocol, Question, Stage
from .store import Store


class Decider:
    """Backend + answer cache persisted in the event log."""

    def __init__(self, backend: DecisionBackend, store: Store):
        self.backend = backend
        self.store = store
        self._cache: dict[str, DecisionBatch] = {}
        self._lock = threading.Lock()
        for ev in store.events("backend_call"):
            if ev.payload.get("backend") == backend.name:
                self._cache[ev.payload["key"]] = DecisionBatch.model_validate(ev.payload["batch"])

    @staticmethod
    def key(backend: str, state: dict[str, Any], questions: list[Question]) -> str:
        qs = json.dumps([q.model_dump() for q in questions], sort_keys=True)
        return hashlib.sha256(f"{backend}|{state_sha(state)}|{qs}".encode()).hexdigest()[:24]

    def evaluate(
        self, state: dict[str, Any], questions: list[Question], paper_id: str | None = None
    ) -> DecisionBatch:
        k = self.key(self.backend.name, state, questions)
        with self._lock:
            hit = self._cache.get(k)
        if hit is not None:
            return hit
        batch = self.backend.evaluate(state, questions)  # outside the lock: calls run in parallel
        self.store.append(
            "backend_call",
            paper_id,
            key=k,
            backend=self.backend.name,
            batch=batch.model_dump(mode="json"),
        )
        with self._lock:
            self._cache[k] = batch
        return batch


def build_state(paper: Paper, stage: Stage, protocol: Protocol, text: str | None) -> dict[str, Any]:
    if stage == Stage.TITLE_ABSTRACT:
        return {
            "title": paper.title,
            "abstract": paper.abstract or "(no abstract available)",
            "year": paper.year,
            "venue": paper.venue,
        }
    return {
        "title": paper.title,
        "fulltext": select_sections(text or "", protocol.fulltext_char_budget),
    }


def _relevant(c: Criterion, protocol: Protocol) -> list[str]:
    ids = [q.id for q in c.questions]
    if c.id == "IC4" and not protocol.duration_alone_satisfies_ic4:
        ids = [i for i in ids if i != "ic4_duration_predictor"]
    return ids


def route(c: Criterion, scores: dict[str, float], protocol: Protocol) -> CriterionOutcome:
    vals = [scores[i] for i in _relevant(c, protocol)]
    combined = max(vals) if c.rule == "any" else min(vals)
    t = c.thresholds
    status = "pass" if combined >= t.high else "fail" if combined <= t.low else "uncertain"
    return CriterionOutcome(
        criterion_id=c.id,
        polarity=c.polarity,
        status=status,
        question_scores={q.id: scores[q.id] for q in c.questions},
    )


def verdict_from(outcomes: list[CriterionOutcome]) -> tuple[Verdict, list[str]]:
    hard = [
        o.criterion_id
        for o in outcomes
        if (o.polarity == "include" and o.status == "fail")
        or (o.polarity == "exclude" and o.status == "pass")
    ]
    if hard:
        return "exclude", hard
    unsure = [o.criterion_id for o in outcomes if o.status == "uncertain"]
    if unsure:
        return "needs_human", unsure
    return "include", []


def screen(
    paper: Paper, stage: Stage, protocol: Protocol, decider: Decider, text: str | None = None
) -> ScreeningResult:
    criteria = protocol.criteria_for(stage)
    questions: list[Question] = [q for c in criteria for q in c.questions]
    state = build_state(paper, stage, protocol, text)
    batch = decider.evaluate(state, questions, paper.id)
    scores = {qid: a.score for qid, a in batch.answers.items() if a.score is not None}
    outcomes = [route(c, scores, protocol) for c in criteria]
    verdict, reasons = verdict_from(outcomes)
    if paper.is_seed and verdict == "exclude":
        # Seeds are known positives: a model exclusion is a recall alarm, not a decision.
        verdict, reasons = "needs_human", ["SEED_EXCLUDED_BY_MODEL", *reasons]
    return ScreeningResult(
        paper_id=paper.id,
        stage=stage.value,
        verdict=verdict,
        reasons=reasons,
        outcomes=outcomes,
        batch=batch,
    )
