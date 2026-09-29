import time
from collections.abc import Callable

import httpx
from whenever import Instant

from .exceptions import SourceRequestError

_MAX_RETRY_DELAY = 120.0  # a server asking for longer is treated as broken, not waited for


def _retry_delay(r: httpx.Response, attempt: int) -> float:
    """Seconds to wait per `retry-after`, which is either a number of seconds or an HTTP-date
    (RFC 9110). A missing or unparseable header falls back to exponential backoff."""
    backoff = 2**attempt + 1
    value = r.headers.get("retry-after")
    if value is None:
        return backoff
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (Instant.parse_rfc2822(value) - Instant.now()).total("seconds")
        except ValueError:
            return backoff
    return min(max(0.0, seconds), _MAX_RETRY_DELAY)


def get(
    client: httpx.Client,
    url: str,
    params=None,
    headers=None,
    tries: int = 5,
    *,
    source_name: str,
    source_method: Callable | None = None,
) -> httpx.Response:
    """GET with retry on throttling and server errors, honouring `retry-after`.

    `source_method` is the caller (e.g. `self.search`) and is what errors are tagged with;
    it defaults to this function.
    """
    method = source_method or get
    for attempt in range(tries):
        try:
            r = client.get(url, params=params, headers=headers)
        except httpx.TransportError as exc:
            raise SourceRequestError(method, f"{url}: {exc}", source_name=source_name) from exc
        if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            time.sleep(_retry_delay(r, attempt))
            continue
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SourceRequestError(
                method,
                f"{url}: HTTP {r.status_code}",
                source_name=source_name,
                status_code=r.status_code,
            ) from exc
        return r
    raise RuntimeError("unreachable")
