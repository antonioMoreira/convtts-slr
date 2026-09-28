import time
from collections.abc import Callable

import httpx

from .exceptions import SourceRequestError


def _retry_delay(r: httpx.Response, attempt: int) -> float:
    """`retry-after` in seconds; an HTTP-date (also valid per RFC 9110) or a missing header
    falls back to exponential backoff."""
    backoff = 2**attempt + 1
    try:
        return max(0.0, float(r.headers.get("retry-after", backoff)))
    except ValueError:
        return backoff


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
