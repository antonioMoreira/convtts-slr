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
    """Finds papers from a query: implement this to feed the `identify` stage.

    A `Source` answers "which papers match the protocol's search blocks?". It receives the
    whole `SearchConfig` and translates it to whatever the backend understands (a remote API
    query, a filter over a local corpus, or nothing at all for a pre-made export).

    Use `Source` for anything that retrieves papers by keywords or from a file: OpenAlex,
    Semantic Scholar, arXiv, the ACL Anthology, and `LocalSource` exports (IEEE Xplore CSV,
    seeds). `Identify` runs every entry of `Context.sources` once per protocol version.

    A class that can also walk the citation graph should additionally implement
    `CitationSource`; the two protocols are independent.
    """

    name: str

    @abstractmethod
    def search(self, cfg: SearchConfig) -> SearchResult:
        """Run the protocol's query against this source.

        Raises `SourceError` on failure.
        """
        ...


class CitationSource(Protocol):
    """Expands from a known paper along its citation links: implement this for `snowball`.

    A `CitationSource` answers "which papers does this one cite, and which cite it?". It
    takes a `Paper` (not a `SearchConfig`) and looks it up by identifier (OpenAlex id, S2 id,
    arXiv id or DOI), so it only works for backends that index citation edges. `Snowball`
    calls it for every included paper in `Context.citation_sources`, one round at a time.

    Use `CitationSource` only when the backend exposes references and citations: currently
    OpenAlex and Semantic Scholar, which implement both protocols. Keyword-only sources
    (arXiv, ACL Anthology, local files) should implement `Source` alone.

    Lookups for a paper with no usable identifier, or one the backend does not know, return
    an empty `CitationResult` rather than raising; real failures raise `SourceError`.
    """

    name: str

    @abstractmethod
    def references(self, p: Paper) -> CitationResult:
        """Papers cited by `p`. Empty when `p` has no usable identifier or is not found."""
        ...

    @abstractmethod
    def citations(self, p: Paper) -> CitationResult:
        """Papers citing `p`. Empty when `p` has no usable identifier or is not found."""
        ...
