import os
import random
import time
from typing import Any

from ..models import DecisionBatch, QuestionAnswer
from ..protocol import NoulSpec, Question
from .exceptions import (
    BackendConfigurationError,
    BackendRequestError,
    BackendResponseError,
)
from .interface import EvaluationRequest


class JevBackend:
    """Wraps `TypeSafeClient.system_one(state=..., questions=...)`.

    Needs TYPESAFE_API_KEY. Model defaults to TYPESAFE_DEFAULT_MODEL or 'jev-latest'
    (the default used by Jev-Mem's config)."""

    name = "jev"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        batch_size: int = 16,
        max_retries: int = 4,
        timeout: float = 60.0,
    ):
        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.name = f"jev:{self.model}"  # cache keys change when the model changes
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        try:
            from typesafe_sdk import RetryPolicy, TypeSafeClient  # optional dependency
        except ImportError as exc:
            raise BackendConfigurationError(
                self.__init__,
                "typesafe-sdk is not installed (pip install 'convtts-slr[jev]')",
                backend_name=self.name,
            ) from exc
        key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise BackendConfigurationError(
                self.__init__, "TYPESAFE_API_KEY is not set", backend_name=self.name
            )
        self._client = TypeSafeClient(
            api_key=key,
            model=self.model,
            base_url=base_url or os.getenv("TYPESAFE_BASE_URL"),
            retry=RetryPolicy(max_retries=0, timeout=None),  # we own retries below
        )

    @staticmethod
    def _to_sdk(q: Question):
        from typesafe_sdk import Choice, Noul

        if isinstance(q, NoulSpec):
            return Noul(instructions=q.instructions, criteria={"true": q.true, "false": q.false})
        return Choice(instructions=q.instructions, criteria=dict(q.options))

    def _call(self, request: EvaluationRequest):
        from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError

        payload = {q.id: self._to_sdk(q) for q in request.questions}
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.system_one(
                    state=request.state, questions=payload, timeout=self.timeout
                )
            except TypeSafeAPIError as exc:
                if exc.status not in (408, 429) and exc.status < 500 or attempt == self.max_retries:
                    raise BackendRequestError(
                        self.evaluate,
                        f"HTTP {exc.status}",
                        backend_name=self.name,
                        status_code=exc.status,
                    ) from exc
            except TypeSafeAPIConnectionError as exc:
                if attempt == self.max_retries:
                    raise BackendRequestError(
                        self.evaluate, f"connection failed: {exc}", backend_name=self.name
                    ) from exc
            time.sleep(min(30.0, 2**attempt) * random.uniform(0.75, 1.0))
        raise RuntimeError("unreachable")

    def _answer(self, q: Question, resp: Any) -> QuestionAnswer:
        try:
            a = resp.answers[q.id]
            if isinstance(q, NoulSpec):
                return QuestionAnswer(question_id=q.id, kind="noul", score=float(a.noul))
            return QuestionAnswer(
                question_id=q.id,
                kind="choice",
                choice=a.choice,
                distribution=dict(a.probabilities),
                confidence=float(a.confidence),
            )
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise BackendResponseError(
                self.evaluate,
                f"no usable answer for question {q.id!r}: {type(exc).__name__}: {exc}",
                backend_name=self.name,
            ) from exc

    def evaluate(self, request: EvaluationRequest) -> DecisionBatch:
        answers: dict[str, QuestionAnswer] = {}
        model = self.model
        for chunk in request.chunks(self.batch_size):
            resp = self._call(chunk)
            model = resp.model
            for q in chunk.questions:
                answers[q.id] = self._answer(q, resp)
        return DecisionBatch(backend=self.name, model=model, answers=answers, state_sha=request.sha)
