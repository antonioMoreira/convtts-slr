import hashlib
import json
from collections.abc import Iterator
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from ..models import DecisionBatch
from ..protocol import Question


def state_sha(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


class EvaluationRequest(BaseModel):
    """What a backend is asked: typed questions against one shared JSON state, with no
    answer used as hidden context for another (Jev semantics)."""

    state: dict[str, Any]
    questions: list[Question]

    model_config = ConfigDict(frozen=True)

    @property
    def sha(self) -> str:
        return state_sha(self.state)

    def chunks(self, size: int) -> Iterator["EvaluationRequest"]:
        """The same state with at most `size` questions each, for providers with batch limits."""
        for i in range(0, len(self.questions), size):
            yield EvaluationRequest(state=self.state, questions=self.questions[i : i + size])


class DecisionBackend(Protocol):
    """Answers an `EvaluationRequest` with a `DecisionBatch`: implement this for System One.

    `Decider` caches by `name`, so a backend must change its `name` when its model changes.
    Raise `BackendError` subclasses on failure.
    """

    name: str

    def evaluate(self, request: EvaluationRequest) -> DecisionBatch: ...
