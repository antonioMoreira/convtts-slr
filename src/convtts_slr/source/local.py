import csv
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from ..models import Paper
from ..protocol import SearchConfig
from .exceptions import SourceConfigurationError, SourceResponseError
from .interface import SearchResult

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

    def search(self, cfg: SearchConfig | None = None) -> SearchResult:
        out = []
        try:
            for row in self._rows():
                row = {k: v for k, v in row.items() if v not in (None, "")}
                if "year" in row:
                    row["year"] = int(str(row["year"])[:4])
                row.setdefault("id", row.get("doi") or row.get("arxiv_id") or row["title"])
                row["sources"] = sorted(set(row.get("sources", [])) | {self.name})
                out.append(Paper.model_validate(row))
        except FileNotFoundError as exc:
            raise SourceConfigurationError(self.search, str(exc), source_name=self.name) from exc
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError) as exc:
            # ValidationError and JSONDecodeError are ValueError subclasses; listed for clarity
            raise SourceResponseError(
                self.search, f"{self.path}: {exc!r}", source_name=self.name
            ) from exc
        return SearchResult(source=self.name, query="", papers=out)
