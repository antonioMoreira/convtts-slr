from abc import abstractmethod
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ..models import Paper
from ..protocol import SearchConfig


class SearchResult(BaseModel):
    """What one source returned for one `SearchConfig`."""

    source: str
    query: str  # the native query sent to the source; empty for sources that do not query
    papers: list[Paper]
    truncated: bool = False  # the source stopped at `max_results_per_source`

    model_config = ConfigDict(frozen=True)


class CitationResult(BaseModel):
    """The papers one citation source found on one side of one paper's citation graph."""

    source: str
    paper_id: str
    direction: Literal["references", "citations"]
    papers: list[Paper]

    model_config = ConfigDict(frozen=True)


class Source(Protocol):
    name: str

    @abstractmethod
    def search(self, cfg: SearchConfig) -> SearchResult:
        """Run the protocol's query against this source.

        Raises `SourceError` on failure.
        """
        ...


class CitationSource(Protocol):
    name: str

    @abstractmethod
    def references(self, p: Paper) -> CitationResult:
        """Papers cited by `p`. Empty when `p` has no usable identifier or is not found."""
        ...

    @abstractmethod
    def citations(self, p: Paper) -> CitationResult:
        """Papers citing `p`. Empty when `p` has no usable identifier or is not found."""
        ...
