"""Exceptions raised by retrieval sources.

Every error carries the failing method and the source's name, formatted as
`[source_method=search][source_name=openalex] message`, so a log line is enough to tell
which source and which call failed.
"""

from collections.abc import Callable


class SourceError(Exception):
    """Base exception for all retrieval-source operations."""

    def __init__(
        self,
        source_method: Callable,
        message: str | None = None,
        *,
        source_name: str,
    ) -> None:
        # partials and other wrapped callables have no __name__
        self.source_method_name = getattr(source_method, "__name__", str(source_method))
        self.source_name = source_name
        self.message = message
        super().__init__(self._format())

    def _tags(self) -> list[str]:
        return [f"source_method={self.source_method_name}", f"source_name={self.source_name}"]

    def _format(self) -> str:
        prefix = "".join(f"[{tag}]" for tag in self._tags())
        return f"{prefix} {self.message}" if self.message else prefix


class SourceConfigurationError(SourceError):
    """Raised when a source cannot run as configured: an unsupported local file type, a
    corpus that cannot be downloaded, a missing credential."""


class SourceRequestError(SourceError):
    """Raised when talking to the remote service fails (transport error or HTTP error status,
    after retries). `status_code` is None for failures without an HTTP response."""

    def __init__(
        self,
        source_method: Callable,
        message: str | None = None,
        *,
        source_name: str,
        status_code: int | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(source_method, message, source_name=source_name)

    def _tags(self) -> list[str]:
        return [*super()._tags(), f"status_code={self.status_code}"]


class SourceResponseError(SourceError):
    """Raised when a response or input row arrives but cannot be turned into a `Paper`
    (missing required field, unparseable value)."""
