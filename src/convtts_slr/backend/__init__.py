"""System-One decision backends.

All backends answer the same thing: an `EvaluationRequest` (a batch of typed questions
against one shared JSON state, no answer used as hidden context for another, Jev semantics)
with a `DecisionBatch`.

- JevBackend      : TypeSafe's Jev via `typesafe-sdk` (the API used by Jev-Mem).
- AnthropicBackend: Claude via pydantic-ai structured output. Also the *independent* checker
                    in verify.py.
- GeminiBackend   : Gemini via pydantic-ai structured output.
- ScriptedBackend : deterministic answers for tests and dry runs.

Failures raise `BackendError` subclasses (configuration, request, response).
"""

from ..models import DecisionBatch, QuestionAnswer
from . import exceptions
from .anthropic import AnthropicBackend
from .exceptions import (
    BackendConfigurationError,
    BackendError,
    BackendRequestError,
    BackendResponseError,
)
from .gemini import GeminiBackend
from .interface import DecisionBackend, EvaluationRequest, state_sha
from .jev import JevBackend
from .scripted import Script, ScriptedBackend

PROVIDERS = ("jev", "anthropic", "gemini")


def create_backend(spec: str, model: str | None = None) -> DecisionBackend:
    """Build a backend from `provider` or `provider:model`, e.g. `jev`, `anthropic:claude-sonnet-5`
    or `gemini:gemini-2.5-flash`. An explicit `model` wins over the one in `spec`."""
    provider, _, spec_model = spec.partition(":")
    chosen = model or spec_model or None
    match provider:
        case "jev":
            return JevBackend(model=chosen)
        case "anthropic":
            return AnthropicBackend(model=chosen)
        case "gemini":
            return GeminiBackend(model=chosen)
    raise BackendConfigurationError(
        create_backend,
        f"unknown backend {spec!r}; expected one of {', '.join(PROVIDERS)} "
        "(optionally as provider:model)",
        backend_name=spec,
    )


__all__ = [
    "PROVIDERS",
    "AnthropicBackend",
    "BackendConfigurationError",
    "BackendError",
    "BackendRequestError",
    "BackendResponseError",
    "DecisionBackend",
    "DecisionBatch",
    "EvaluationRequest",
    "GeminiBackend",
    "JevBackend",
    "QuestionAnswer",
    "Script",
    "ScriptedBackend",
    "create_backend",
    "exceptions",
    "state_sha",
]
