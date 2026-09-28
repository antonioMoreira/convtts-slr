import re
from typing import final

import arxiv

from ..models import Paper
from ..protocol import SearchConfig
from .exceptions import SourceRequestError
from .interface import SearchResult, Source


@final
class ArxivSource(Source):
    name = "arxiv"

    def __init__(
        self,
        client: arxiv.Client | None = None,
        page_size: int = 100,
        delay_seconds: float = 3.0,
        num_retries: int = 3,
    ):
        self.client = client or arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )

    @staticmethod
    def render(cfg: SearchConfig) -> str:
        blocks = " AND ".join("(" + " OR ".join(f'all:"{t}"' for t in b) + ")" for b in cfg.blocks)
        end = cfg.to_date.replace("-", "")
        return f"{blocks} AND submittedDate:[{cfg.from_year}01010000 TO {end}2359]"

    @classmethod
    def _to_paper(cls, r: arxiv.Result) -> Paper:
        aid = r.get_short_id()
        pub = r.published
        return Paper(
            id=f"arxiv:{aid}",
            title=re.sub(r"\s+", " ", r.title).strip(),
            abstract=re.sub(r"\s+", " ", r.summary).strip(),
            year=pub.year if pub else None,
            publication_date=pub.strftime("%Y-%m-%d") if pub else None,
            venue="arXiv",
            doi=r.doi or None,
            arxiv_id=aid,
            pdf_url=r.pdf_url or f"https://arxiv.org/pdf/{aid}",
            sources=["arxiv"],
        )

    @classmethod
    def parse(cls, xml_text: str | bytes) -> list[Paper]:
        content = xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text
        if b"<updated>" not in content and b"<published>" in content:
            content = re.sub(
                rb"(<published>([^<]+)</published>)",
                rb"\1<updated>\2</updated>",
                content,
            )
        feed = arxiv._feed.parse(content)
        return [cls._to_paper(r) for r in feed.results]

    def search(self, cfg: SearchConfig) -> SearchResult:
        query = self.render(cfg)
        search = arxiv.Search(
            query=query,
            max_results=cfg.max_results_per_source,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            papers = [self._to_paper(r) for r in self.client.results(search)]
        except arxiv.ArxivError as exc:
            raise SourceRequestError(
                self.search,
                str(exc),
                source_name=self.name,
                status_code=getattr(exc, "status", None),  # only arxiv.HTTPError has one
            ) from exc
        return SearchResult(
            source=self.name,
            query=query,
            papers=papers,
            truncated=len(papers) >= cfg.max_results_per_source,
        )
