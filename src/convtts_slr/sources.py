"""Retrieval sources (deterministic, no model calls).

OpenAlex and Semantic Scholar also index ACL Anthology and ISCA (Interspeech) papers,
so those venues are covered without scraping. IEEE Xplore has no free search API:
export results as CSV from the website and load them with LocalSource.
"""

import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import httpx

from .models import Paper
from .protocol import SearchConfig


class Source(Protocol):
    name: str

    def search(self, cfg: SearchConfig) -> list[Paper]: ...


class CitationSource(Protocol):
    name: str

    def references(self, p: Paper) -> list[Paper]: ...
    def citations(self, p: Paper) -> list[Paper]: ...


def _get(
    client: httpx.Client, url: str, params=None, headers=None, tries: int = 5
) -> httpx.Response:
    for attempt in range(tries):
        r = client.get(url, params=params, headers=headers)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            time.sleep(float(r.headers.get("retry-after", 2**attempt + 1)))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("unreachable")


def _quote(term: str) -> str:
    return f'"{term}"' if " " in term or "-" in term else term


# --------------------------------------------------------------------------- #
# OpenAlex
# --------------------------------------------------------------------------- #

_ARXIV_IN_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})")


def openalex_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    pos = {i: w for w, idx in inv.items() for i in idx}
    return " ".join(pos[i] for i in sorted(pos))


class OpenAlexSource:
    name = "openalex"
    BASE = "https://api.openalex.org/works"
    SELECT = (
        "id,doi,title,publication_year,publication_date,abstract_inverted_index,"
        "primary_location,best_oa_location,locations,referenced_works"
    )

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)
        self.base_params = {
            k: v
            for k, v in {
                "mailto": os.getenv("OPENALEX_MAILTO"),
                "api_key": os.getenv("OPENALEX_API_KEY"),
            }.items()
            if v
        }

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

    def _paged(self, params: dict, limit: int) -> list[Paper]:
        out, cursor = [], "*"
        while cursor and len(out) < limit:
            r = _get(
                self.client,
                self.BASE,
                {
                    **self.base_params,
                    **params,
                    "per-page": 200,
                    "cursor": cursor,
                    "select": self.SELECT,
                },
            )
            data = r.json()
            out += [self._to_paper(w) for w in data["results"] if w.get("title")]
            cursor = data["meta"].get("next_cursor")
        return out[:limit]

    def search(self, cfg: SearchConfig) -> list[Paper]:
        flt = f"from_publication_date:{cfg.from_year}-01-01,to_publication_date:{cfg.to_date}"
        return self._paged({"search": self.render(cfg), "filter": flt}, cfg.max_results_per_source)

    def _work(self, p: Paper) -> dict | None:
        key = p.openalex_id or (f"doi:{p.doi}" if p.doi else None)
        if not key:
            return None
        try:
            return _get(
                self.client, f"{self.BASE}/{key}", {**self.base_params, "select": self.SELECT}
            ).json()
        except httpx.HTTPStatusError:
            return None

    def references(self, p: Paper) -> list[Paper]:
        w = self._work(p)
        ids = [u.rsplit("/", 1)[-1] for u in (w or {}).get("referenced_works", [])]
        out = []
        for i in range(0, len(ids), 50):
            out += self._paged({"filter": "openalex_id:" + "|".join(ids[i : i + 50])}, 50)
        return out

    def citations(self, p: Paper) -> list[Paper]:
        w = self._work(p)
        if not w:
            return []
        return self._paged({"filter": f"cites:{w['id'].rsplit('/', 1)[-1]}"}, 2000)


# --------------------------------------------------------------------------- #
# Semantic Scholar
# --------------------------------------------------------------------------- #


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

    def search(self, cfg: SearchConfig) -> list[Paper]:
        out, token = [], None
        params = {
            "query": self.render(cfg),
            "fields": self.FIELDS,
            "year": f"{cfg.from_year}-{cfg.to_date[:4]}",
        }
        while len(out) < cfg.max_results_per_source:
            r = _get(
                self.client,
                f"{self.BASE}/paper/search/bulk",
                {**params, **({"token": token} if token else {})},
                self.headers,
            )
            data = r.json()
            out += [self._to_paper(d) for d in data.get("data", []) if d.get("title")]
            token = data.get("token")
            if not token:
                break
            time.sleep(self.delay)
        return out[: cfg.max_results_per_source]

    @staticmethod
    def _key(p: Paper) -> str | None:
        if p.s2_id:
            return p.s2_id
        if p.arxiv_id:
            return f"ARXIV:{p.arxiv_id}"
        return f"DOI:{p.doi}" if p.doi else None

    def _edges(self, p: Paper, edge: str, side: str) -> list[Paper]:
        key = self._key(p)
        if not key:
            return []
        try:
            r = _get(
                self.client,
                f"{self.BASE}/paper/{key}/{edge}",
                {"fields": self.FIELDS, "limit": 1000},
                self.headers,
            )
        except httpx.HTTPStatusError:
            return []
        time.sleep(self.delay)
        return [
            self._to_paper(e[side])
            for e in r.json().get("data", [])
            if (e.get(side) or {}).get("title")
        ]

    def references(self, p: Paper) -> list[Paper]:
        return self._edges(p, "references", "citedPaper")

    def citations(self, p: Paper) -> list[Paper]:
        return self._edges(p, "citations", "citingPaper")


# --------------------------------------------------------------------------- #
# arXiv
# --------------------------------------------------------------------------- #

_ATOM = {"a": "http://www.w3.org/2005/Atom"}


class ArxivSource:
    name = "arxiv"
    BASE = "https://export.arxiv.org/api/query"

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)

    @staticmethod
    def render(cfg: SearchConfig) -> str:
        blocks = " AND ".join("(" + " OR ".join(f'all:"{t}"' for t in b) + ")" for b in cfg.blocks)
        end = cfg.to_date.replace("-", "")
        return f"{blocks} AND submittedDate:[{cfg.from_year}01010000 TO {end}2359]"

    @staticmethod
    def parse(xml_text: str) -> list[Paper]:
        root = ET.fromstring(xml_text)
        out = []
        for e in root.findall("a:entry", _ATOM):
            aid = e.findtext("a:id", "", _ATOM).rsplit("/abs/", 1)[-1]
            title = re.sub(r"\s+", " ", e.findtext("a:title", "", _ATOM)).strip()
            published = e.findtext("a:published", "", _ATOM)
            doi = e.findtext("{http://arxiv.org/schemas/atom}doi")
            out.append(
                Paper(
                    id=f"arxiv:{aid}",
                    title=title,
                    abstract=re.sub(r"\s+", " ", e.findtext("a:summary", "", _ATOM)).strip(),
                    year=int(published[:4]) if published else None,
                    publication_date=published[:10] or None,
                    venue="arXiv",
                    doi=doi,
                    arxiv_id=aid,
                    pdf_url=f"https://arxiv.org/pdf/{aid}",
                    sources=["arxiv"],
                )
            )
        return out

    def search(self, cfg: SearchConfig) -> list[Paper]:
        out, start = [], 0
        q = self.render(cfg)
        while start < cfg.max_results_per_source:
            r = _get(
                self.client, self.BASE, {"search_query": q, "start": start, "max_results": 200}
            )
            batch = self.parse(r.text)
            out += batch
            if len(batch) < 200:
                break
            start += 200
            time.sleep(3.0)  # arXiv API etiquette
        return out


# --------------------------------------------------------------------------- #
# Local exports (IEEE Xplore CSV, hand-curated JSON/JSONL, seeds)
# --------------------------------------------------------------------------- #

_CSV_MAP = {  # IEEE Xplore export headers first, generic names second
    "title": ["Document Title", "title", "Title"],
    "abstract": ["Abstract", "abstract"],
    "year": ["Publication Year", "year", "Year"],
    "doi": ["DOI", "doi"],
    "venue": ["Publication Title", "venue", "Venue"],
    "pdf_url": ["PDF Link", "pdf_url"],
    "arxiv_id": ["arxiv_id", "arXiv"],
}


class LocalSource:
    def __init__(self, path: str | Path, name: str | None = None):
        self.path = Path(path)
        self.name = name or f"local:{self.path.name}"

    def _rows(self) -> Iterable[dict]:
        if self.path.suffix == ".csv":
            with self.path.open(encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    yield {
                        k: next((row[c] for c in cols if row.get(c)), None)
                        for k, cols in _CSV_MAP.items()
                    }
        elif self.path.suffix == ".jsonl":
            yield from (
                json.loads(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        else:
            yield from json.loads(self.path.read_text(encoding="utf-8"))

    def search(self, cfg: SearchConfig | None = None) -> list[Paper]:
        out = []
        for row in self._rows():
            row = {k: v for k, v in row.items() if v not in (None, "")}
            if "year" in row:
                row["year"] = int(str(row["year"])[:4])
            row.setdefault("id", row.get("doi") or row.get("arxiv_id") or row["title"])
            row["sources"] = sorted(set(row.get("sources", [])) | {self.name})
            out.append(Paper.model_validate(row))
        return out
