"""System-One decision backends.

All backends answer the same thing: a batch of typed questions against one shared
JSON state, with no answer used as hidden context for another (Jev semantics).

- JevBackend     : TypeSafe's Jev via `typesafe-sdk` (the API used by Jev-Mem).
- LLMBackend     : pydantic-ai structured output. Used as the fallback before you have
                   Jev access, and as the *independent* checker in verify.py.
- ScriptedBackend: deterministic answers for tests and dry runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol

from pydantic import Field, create_model

from .models import DecisionBatch, QuestionAnswer
from .protocol import ChoiceSpec, NoulSpec, Question


def state_sha(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


class DecisionBackend(Protocol):
    name: str

    def evaluate(self, state: dict[str, Any], questions: list[Question]) -> DecisionBatch: ...


# --------------------------------------------------------------------------- #
# Jev
# --------------------------------------------------------------------------- #


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
        from typesafe_sdk import RetryPolicy, TypeSafeClient  # optional dependency

        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self.name = f"jev:{self.model}"  # cache keys change when the model changes
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = TypeSafeClient(
            api_key=api_key or os.environ["TYPESAFE_API_KEY"],
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

    def _call(self, state: dict[str, Any], questions: list[Question]):
        from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError

        payload = {q.id: self._to_sdk(q) for q in questions}
        for attempt in range(self.max_retries + 1):
            try:
                return self._client.system_one(state=state, questions=payload, timeout=self.timeout)
            except TypeSafeAPIError as exc:
                if exc.status not in (408, 429) and exc.status < 500 or attempt == self.max_retries:
                    raise
            except TypeSafeAPIConnectionError:
                if attempt == self.max_retries:
                    raise
            time.sleep(min(30.0, 2**attempt) * random.uniform(0.75, 1.0))
        raise RuntimeError("unreachable")

    def evaluate(self, state: dict[str, Any], questions: list[Question]) -> DecisionBatch:
        answers: dict[str, QuestionAnswer] = {}
        model = self.model
        for i in range(0, len(questions), self.batch_size):
            chunk = questions[i : i + self.batch_size]
            resp = self._call(state, chunk)
            model = resp.model
            for q in chunk:
                a = resp.answers[q.id]
                if isinstance(q, NoulSpec):
                    answers[q.id] = QuestionAnswer(
                        question_id=q.id, kind="noul", score=float(a.noul)
                    )
                else:
                    answers[q.id] = QuestionAnswer(
                        question_id=q.id,
                        kind="choice",
                        choice=a.choice,
                        distribution=dict(a.probabilities),
                        confidence=float(a.confidence),
                    )
        return DecisionBatch(
            backend=self.name, model=model, answers=answers, state_sha=state_sha(state)
        )


# --------------------------------------------------------------------------- #
# LLM fallback / independent checker
# --------------------------------------------------------------------------- #

_LLM_INSTRUCTIONS = (
    "You are screening papers for a systematic literature review. Answer every question "
    "independently, using only the provided state. For each yes/no proposition return the "
    "probability (0-1) that it is TRUE under its TRUE/FALSE criteria. For each choice return "
    "exactly one option key. Do not use outside knowledge about the paper."
)


class LLMBackend:
    name = "llm"

    def __init__(self, model: str = "anthropic:claude-sonnet-5", batch_size: int = 12):
        self.model = model
        self.name = f"llm:{model}"
        self.batch_size = batch_size

    def _output_type(self, questions: list[Question]):
        fields: dict[str, Any] = {}
        for q in questions:
            if isinstance(q, NoulSpec):
                fields[q.id] = (float, Field(ge=0.0, le=1.0))
            else:
                fields[q.id] = (Literal[tuple(q.options)], ...)  # type: ignore[valid-type]
        return create_model("Answers", **fields)

    @staticmethod
    def _render(state: dict[str, Any], questions: list[Question]) -> str:
        lines = [
            "STATE (JSON):",
            json.dumps(state, ensure_ascii=False, indent=1),
            "",
            "QUESTIONS:",
        ]
        for q in questions:
            lines.append(f"- [{q.id}] {q.instructions}")
            if isinstance(q, NoulSpec):
                lines += [f"    TRUE if: {q.true}", f"    FALSE if: {q.false}"]
            else:
                lines += [f"    option `{k}`: {v}" for k, v in q.options.items()]
        return "\n".join(lines)

    def evaluate(self, state: dict[str, Any], questions: list[Question]) -> DecisionBatch:
        from pydantic_ai import Agent  # optional dependency

        answers: dict[str, QuestionAnswer] = {}
        for i in range(0, len(questions), self.batch_size):
            chunk = questions[i : i + self.batch_size]
            agent = Agent(
                self.model,
                output_type=self._output_type(chunk),
                instructions=_LLM_INSTRUCTIONS,
            )
            out = agent.run_sync(self._render(state, chunk)).output
            for q in chunk:
                v = getattr(out, q.id)
                if isinstance(q, NoulSpec):
                    answers[q.id] = QuestionAnswer(question_id=q.id, kind="noul", score=float(v))
                else:  # an LLM gives no distribution; record a one-hot, confidence unknown
                    answers[q.id] = QuestionAnswer(
                        question_id=q.id,
                        kind="choice",
                        choice=v,
                        distribution={k: float(k == v) for k in q.options},
                    )
        return DecisionBatch(
            backend=self.name,
            model=str(self.model),
            answers=answers,
            state_sha=state_sha(state),
        )


# --------------------------------------------------------------------------- #
# Scripted (tests / dry runs)
# --------------------------------------------------------------------------- #

Script = Callable[[dict[str, Any], Question], float | str]


class ScriptedBackend:
    """`script(state, question)` returns a score for Nouls or an option key for Choices."""

    name = "scripted"

    def __init__(self, script: Script):
        self.script = script
        self.calls = 0

    def evaluate(self, state: dict[str, Any], questions: list[Question]) -> DecisionBatch:
        self.calls += 1
        answers = {}
        for q in questions:
            v = self.script(state, q)
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
            state_sha=state_sha(state),
        )
