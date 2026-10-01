"""Exceptions raised by decision backends.

Every error carries the failing method and the backend's name, formatted as
`[backend_method=evaluate][backend_name=jev:jev-latest] message`, so a log line is enough to
tell which backend and which call failed. Provider exceptions are chained with `from exc`.
"""

from collections.abc import Callable


class BackendError(Exception):
    """Base exception for all decision-backend operations."""

    def __init__(
        self,
        backend_method: Callable,
        message: str | None = None,
        *,
        backend_name: str,
    ) -> None:
        # partials and other wrapped callables have no __name__
        self.backend_method_name = getattr(backend_method, "__name__", str(backend_method))
        self.backend_name = backend_name
        self.message = message
        super().__init__(self._format())

    def _tags(self) -> list[str]:
        return [f"backend_method={self.backend_method_name}", f"backend_name={self.backend_name}"]

    def _format(self) -> str:
        prefix = "".join(f"[{tag}]" for tag in self._tags())
        return f"{prefix} {self.message}" if self.message else prefix


class BackendConfigurationError(BackendError):
    """Raised when a backend cannot run as configured: a missing credential, a missing optional
    dependency, an unknown backend spec."""


class BackendRequestError(BackendError):
    """Raised when talking to the provider fails (transport error or HTTP error status, after
    retries). `status_code` is None for failures without an HTTP response."""

    def __init__(
        self,
        backend_method: Callable,
        message: str | None = None,
        *,
        backend_name: str,
        status_code: int | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(backend_method, message, backend_name=backend_name)

    def _tags(self) -> list[str]:
        return [*super()._tags(), f"status_code={self.status_code}"]


class BackendResponseError(BackendError):
    """Raised when a response arrives but cannot be turned into a `DecisionBatch`: an answer
    missing for a question, an invalid choice or score, output the provider SDK rejected."""
