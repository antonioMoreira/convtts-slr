import re
from typing import final

from acl_anthology import Anthology

from ..models import Paper
from ..protocol import SearchConfig
from .exceptions import SourceConfigurationError
from .interface import SearchResult, Source


@final
class AclAnthologySource(Source):
    """The `acl-anthology` package has no keyword search index (its own docs say so):
    it loads the whole corpus locally (~120 MB, git-cloned and cached by the package
    itself on first use) and every query filters that in Python. Loading is lazy and
    cached on the instance, so constructing this source is cheap; the clone/parse cost
    is only paid the first time `search()` actually runs.
    """

    name = "acl_anthology"

    def __init__(self, anthology=None):
        self._anthology = anthology

    def _get_anthology(self):
        if self._anthology is None:
            try:
                self._anthology = Anthology.from_repo()
            except Exception as exc:  # git clone / cache failures come from several libraries
                raise SourceConfigurationError(
                    self._get_anthology, str(exc), source_name=self.name
                ) from exc
        return self._anthology

    @staticmethod
    def _matches(text: str, blocks: list[list[str]]) -> bool:
        return all(any(t.lower() in text for t in block) for block in blocks)

    @staticmethod
    def _year(p) -> int | None:
        try:
            return int(p.year)
        except TypeError, ValueError:
            return None

    @classmethod
    def _venue(cls, p) -> str | None:
        try:
            names = [v.name for v in p.parent.venues()]
        except Exception:  # venues.json not loaded, or a dangling venue id
            names = [v.upper() for v in p.venue_ids]
        return ", ".join(names) or None

    @classmethod
    def _to_paper(cls, p) -> Paper:
        title = p.title.as_text()
        abstract = p.abstract.as_text() if p.abstract else ""
        return Paper(
            id=f"acl:{p.full_id}",
            title=re.sub(r"\s+", " ", title).strip(),
            abstract=re.sub(r"\s+", " ", abstract).strip(),
            year=cls._year(p),
            venue=cls._venue(p),
            doi=p.doi or None,
            pdf_url=p.pdf.url if p.pdf else p.web_url,
            sources=[cls.name],
        )

    def search(self, cfg: SearchConfig) -> SearchResult:
        end_year = cfg.to_date.year
        limit = cfg.max_results_per_source
        out: list[Paper] = []
        for p in self._get_anthology().papers():
            if p.is_deleted or p.is_frontmatter:
                continue
            year = self._year(p)
            if year is not None and not (cfg.from_year <= year <= end_year):
                continue
            text = f"{p.title.as_text()} {p.abstract.as_text() if p.abstract else ''}".lower()
            if not self._matches(text, cfg.blocks):
                continue
            out.append(self._to_paper(p))
            if len(out) >= limit:
                break
        return SearchResult(
            source=self.name,
            query=" AND ".join("(" + " OR ".join(b) + ")" for b in cfg.blocks),
            papers=out,
            truncated=len(out) >= limit,
        )
