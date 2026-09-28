import os
import re
from typing import final

import requests
from pyalex import Works
from pyalex import config as oa_config
from pyalex.api import QueryError

from ..models import Paper
from ..protocol import SearchConfig
from .exceptions import SourceRequestError
from .interface import CitationResult, CitationSource, SearchResult, Source

_ARXIV_IN_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})")
_CITATIONS_LIMIT = 2000


def _quote(term: str) -> str:
    return f'"{term}"' if " " in term or "-" in term else term


def openalex_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    pos = {i: w for w, idx in inv.items() for i in idx}
    return " ".join(pos[i] for i in sorted(pos))


@final
class OpenAlexSource(Source, CitationSource):
    """Uses `pyalex`, whose `config` (email/api_key/retry policy) is process-global, not
    per-instance -- constructing this class configures pyalex for the whole process."""

    name = "openalex"
    SELECT = (
        "id,doi,title,publication_year,publication_date,abstract_inverted_index,"
        "primary_location,best_oa_location,locations,referenced_works"
    )

    def __init__(self, email: str | None = None, api_key: str | None = None):
        oa_config.email = email or os.getenv("OPENALEX_EMAIL") or oa_config.email
        oa_config.api_key = api_key or os.getenv("OPENALEX_API_KEY") or oa_config.api_key
        oa_config.max_retries = 5
        oa_config.retry_backoff_factor = 0.5
        oa_config.retry_http_codes = [429, 500, 502, 503, 504]

    @staticmethod
    def render(cfg: SearchConfig) -> str:
        return " AND ".join("(" + " OR ".join(_quote(t) for t in b) + ")" for b in cfg.blocks)

    def _to_paper(self, w: dict) -> Paper:
        arxiv = None
        pdf = None
        for loc in [w.get("best_oa_location") or {}] + (w.get("locations") or []):
            for u in (loc.get("landing_page_url"), loc.get("pdf_url")):
                if u and (m := _ARXIV_IN_URL.search(u)):
                    arxiv = arxiv or m.group(1)
            pdf = pdf or loc.get("pdf_url")
        src = (w.get("primary_location") or {}).get("source") or {}
        return Paper(
            id=w["id"],
            title=w.get("title") or "",
            abstract=openalex_abstract(w.get("abstract_inverted_index")),
            year=w.get("publication_year"),
            publication_date=w.get("publication_date"),
            venue=src.get("display_name"),
            doi=w.get("doi"),
            arxiv_id=arxiv,
            openalex_id=w["id"].rsplit("/", 1)[-1],
            pdf_url=pdf,
            sources=[self.name],
        )

    def _query(self, cfg: SearchConfig) -> Works:
        return (
            Works()
            .search(self.render(cfg))
            .filter(from_publication_date=f"{cfg.from_year}-01-01", to_publication_date=cfg.to_date)
            .select(self.SELECT)
        )

    def _request_error(self, method, exc: requests.exceptions.RequestException | QueryError):
        response = getattr(exc, "response", None)
        return SourceRequestError(
            method,
            str(exc),
            source_name=self.name,
            status_code=response.status_code if response is not None else None,
        )

    def search(self, cfg: SearchConfig) -> SearchResult:
        limit = cfg.max_results_per_source
        out: list[Paper] = []
        try:
            for page in self._query(cfg).paginate(per_page=200, n_max=limit):
                out += [self._to_paper(w) for w in page if w.get("title")]
        except (requests.exceptions.RequestException, QueryError) as exc:
            raise self._request_error(self.search, exc) from exc
        return SearchResult(
            source=self.name,
            query=self.render(cfg),
            papers=out[:limit],
            truncated=len(out) >= limit,
        )

    def _work(self, p: Paper) -> dict | None:
        # a single-id lookup (Works()[key]) drops any .select() chained before it, so this
        # fetches the full record -- larger payload, same fields available.
        key = p.openalex_id or (f"doi:{p.doi}" if p.doi else None)
        if not key:
            return None
        try:
            return Works()[key]
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 404:
                return None
            raise self._request_error(self._work, exc) from exc
        except QueryError:  # malformed key: treated as not found
            return None
        except requests.exceptions.RequestException as exc:
            raise self._request_error(self._work, exc) from exc

    def references(self, p: Paper) -> CitationResult:
        w = self._work(p)
        ids = [u.rsplit("/", 1)[-1] for u in (w or {}).get("referenced_works", [])]
        out = []
        try:
            for i in range(0, len(ids), 100):  # Works()[list] caps at 100 ids per call
                chunk = ids[i : i + 100]
                out += [
                    self._to_paper(x) for x in Works().select(self.SELECT)[chunk] if x.get("title")
                ]
        except (requests.exceptions.RequestException, QueryError) as exc:
            raise self._request_error(self.references, exc) from exc
        return CitationResult(source=self.name, paper_id=p.id, direction="references", papers=out)

    def citations(self, p: Paper) -> CitationResult:
        w = self._work(p)
        out = []
        if w:
            short_id = w["id"].rsplit("/", 1)[-1]
            query = Works().filter(cites=short_id).select(self.SELECT)
            try:
                for page in query.paginate(per_page=200, n_max=_CITATIONS_LIMIT):
                    out += [self._to_paper(x) for x in page if x.get("title")]
            except (requests.exceptions.RequestException, QueryError) as exc:
                raise self._request_error(self.citations, exc) from exc
        return CitationResult(
            source=self.name,
            paper_id=p.id,
            direction="citations",
            papers=out[:_CITATIONS_LIMIT],
        )
