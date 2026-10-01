from collections.abc import Callable
from typing import Any

from ..models import DecisionBatch, QuestionAnswer
from ..protocol import ChoiceSpec, Question
from .interface import EvaluationRequest

Script = Callable[[dict[str, Any], Question], float | str]


class ScriptedBackend:
    """`script(state, question)` returns a score for Nouls or an option key for Choices."""

    name = "scripted"

    def __init__(self, script: Script):
        self.script = script
        self.calls = 0

    def evaluate(self, request: EvaluationRequest) -> DecisionBatch:
        self.calls += 1
        answers = {}
        for q in request.questions:
            v = self.script(request.state, q)
            if isinstance(q, ChoiceSpec):
                answers[q.id] = QuestionAnswer(
                    question_id=q.id,
                    kind="choice",
                    choice=str(v),
                    distribution={k: float(k == v) for k in q.options},
                    confidence=1.0,
                )
            else:
                answers[q.id] = QuestionAnswer(question_id=q.id, kind="noul", score=float(v))
        return DecisionBatch(
            backend=self.name,
            model="script",
            answers=answers,
            state_sha=request.sha,
        )
