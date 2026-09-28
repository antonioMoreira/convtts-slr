import os
import time
from collections.abc import Callable

import httpx

from ..models import Paper
from ..protocol import SearchConfig
from . import _http
from .exceptions import SourceRequestError
from .interface import CitationResult, SearchResult


class SemanticScholarSource:
    name = "semantic_scholar"
    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,year,publicationDate,venue,externalIds,openAccessPdf"

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)
        key = os.getenv("S2_API_KEY")
        self.headers = {"x-api-key": key} if key else {}
        self.delay = 1.0 if key else 3.5  # unauthenticated traffic is heavily throttled

    @staticmethod
    def render(cfg: SearchConfig) -> str:
        return " + ".join("(" + " | ".join(f'"{t}"' for t in b) + ")" for b in cfg.blocks)

    def _to_paper(self, d: dict) -> Paper:
        ext = d.get("externalIds") or {}
        return Paper(
            id=d.get("paperId") or "",
            title=d.get("title") or "",
            abstract=d.get("abstract") or "",
            year=d.get("year"),
            publication_date=d.get("publicationDate"),
            venue=d.get("venue"),
            doi=ext.get("DOI"),
            arxiv_id=ext.get("ArXiv"),
            s2_id=d.get("paperId"),
            pdf_url=(d.get("openAccessPdf") or {}).get("url") or None,
            sources=[self.name],
        )

    def search(self, cfg: SearchConfig) -> SearchResult:
        limit = cfg.max_results_per_source
        query = self.render(cfg)
        out, token = [], None
        params = {
            "query": query,
            "fields": self.FIELDS,
            "year": f"{cfg.from_year}-{cfg.to_date[:4]}",
        }
        while len(out) < limit:
            r = _http.get(
                self.client,
                f"{self.BASE}/paper/search/bulk",
                {**params, **({"token": token} if token else {})},
                self.headers,
                source_name=self.name,
                source_method=self.search,
            )
            data = r.json()
            out += [self._to_paper(d) for d in data.get("data", []) if d.get("title")]
            token = data.get("token")
            if not token:
                break
            time.sleep(self.delay)
        return SearchResult(
            source=self.name, query=query, papers=out[:limit], truncated=len(out) >= limit
        )

    @staticmethod
    def _key(p: Paper) -> str | None:
        if p.s2_id:
            return p.s2_id
        if p.arxiv_id:
            return f"ARXIV:{p.arxiv_id}"
        return f"DOI:{p.doi}" if p.doi else None

    def _edges(self, p: Paper, edge: str, side: str, method: Callable) -> list[Paper]:
        key = self._key(p)
        if not key:
            return []
        try:
            r = _http.get(
                self.client,
                f"{self.BASE}/paper/{key}/{edge}",
                {"fields": self.FIELDS, "limit": 1000},
                self.headers,
                source_name=self.name,
                source_method=method,
            )
        except SourceRequestError as exc:
            if exc.status_code == 404:  # S2 does not know this paper
                return []
            raise
        time.sleep(self.delay)
        return [
            self._to_paper(e[side])
            for e in r.json().get("data", [])
            if (e.get(side) or {}).get("title")
        ]

    def references(self, p: Paper) -> CitationResult:
        papers = self._edges(p, "references", "citedPaper", self.references)
        return CitationResult(
            source=self.name, paper_id=p.id, direction="references", papers=papers
        )

    def citations(self, p: Paper) -> CitationResult:
        papers = self._edges(p, "citations", "citingPaper", self.citations)
        return CitationResult(source=self.name, paper_id=p.id, direction="citations", papers=papers)
