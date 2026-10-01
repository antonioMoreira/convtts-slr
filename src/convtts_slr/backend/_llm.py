"""Shared logic for backends that go through pydantic-ai structured output.

`AnthropicBackend` and `GeminiBackend` differ only in provider prefix, default model and
credentials; everything else (prompt, output type, batching, error mapping) lives here.
"""

import json
from typing import Any, ClassVar, Literal

from pydantic import Field, ValidationError, create_model

from ..models import DecisionBatch, QuestionAnswer
from ..protocol import NoulSpec, Question
from .exceptions import (
    BackendConfigurationError,
    BackendRequestError,
    BackendResponseError,
)
from .interface import EvaluationRequest

_INSTRUCTIONS = (
    "You are screening papers for a systematic literature review. Answer every question "
    "independently, using only the provided state. For each yes/no proposition return the "
    "probability (0-1) that it is TRUE under its TRUE/FALSE criteria. For each choice return "
    "exactly one option key. Do not use outside knowledge about the paper."
)


class PydanticAIBackend:
    provider: ClassVar[str]  # label used in `name`, e.g. "anthropic"
    prefix: ClassVar[str]  # pydantic-ai model-string prefix, e.g. "anthropic"
    default_model: ClassVar[str]
    extra: ClassVar[str]  # pip extra that installs the provider SDK

    def __init__(self, model: str | None = None, batch_size: int = 12):
        self.model = model or self.default_model
        self.name = f"{self.provider}:{self.model}"  # cache keys change with the model
        self.batch_size = batch_size

    @property
    def model_id(self) -> str:
        return f"{self.prefix}:{self.model}"

    def _output_type(self, questions: list[Question]):
        fields: dict[str, Any] = {}
        for q in questions:
            if isinstance(q, NoulSpec):
                fields[q.id] = (float, Field(ge=0.0, le=1.0))
            else:
                # Literal's members are runtime option keys, not literal syntax, so its
                # argument can't be statically checked; the values are still validated
                # at model-construction time by pydantic.
                fields[q.id] = (Literal[tuple(q.options)], ...)  # ty: ignore[invalid-type-form]
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

    def _run(self, chunk: EvaluationRequest) -> dict[str, QuestionAnswer]:
        try:
            from pydantic_ai import Agent  # optional dependency
            from pydantic_ai.exceptions import (
                ModelAPIError,
                ModelHTTPError,
                UnexpectedModelBehavior,
                UserError,
            )
        except ImportError as exc:
            raise BackendConfigurationError(
                self.evaluate,
                f"pydantic-ai is not installed (pip install 'convtts-slr[{self.extra}]')",
                backend_name=self.name,
            ) from exc
        try:
            agent = Agent(
                self.model_id,
                output_type=self._output_type(chunk.questions),
                instructions=_INSTRUCTIONS,
            )
            out = agent.run_sync(self._render(chunk.state, chunk.questions)).output
        except UserError as exc:  # missing credential, missing provider SDK
            raise BackendConfigurationError(
                self.evaluate, str(exc), backend_name=self.name
            ) from exc
        except ModelHTTPError as exc:
            raise BackendRequestError(
                self.evaluate,
                f"HTTP {exc.status_code}",
                backend_name=self.name,
                status_code=exc.status_code,
            ) from exc
        except ModelAPIError as exc:
            raise BackendRequestError(
                self.evaluate, f"request failed: {exc}", backend_name=self.name
            ) from exc
        except (UnexpectedModelBehavior, ValidationError) as exc:
            raise BackendResponseError(
                self.evaluate, f"unusable model output: {exc}", backend_name=self.name
            ) from exc

        answers: dict[str, QuestionAnswer] = {}
        for q in chunk.questions:
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
        return answers

    def evaluate(self, request: EvaluationRequest) -> DecisionBatch:
        answers: dict[str, QuestionAnswer] = {}
        for chunk in request.chunks(self.batch_size):
            answers |= self._run(chunk)
        return DecisionBatch(
            backend=self.name, model=self.model_id, answers=answers, state_sha=request.sha
        )
