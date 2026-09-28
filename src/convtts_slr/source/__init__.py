"""Retrieval sources (deterministic, no model calls).

OpenAlex and Semantic Scholar also index ACL Anthology and ISCA (Interspeech) papers, so
those venues are covered even without a dedicated scraper. AclAnthologySource hits the
ACL Anthology directly anyway: it is a completeness/precision cross-check against
OpenAlex/S2's indexing lag, not the only way ACL papers are found. ISCA/Interspeech still
has no free API of its own, so that venue still relies solely on OpenAlex/S2 coverage.
IEEE Xplore also has no free search API: export results as CSV from the website and load
them with LocalSource.
"""

from . import exceptions
from .acl_anthology import AclAnthologySource
from .arxiv import ArxivSource
from .exceptions import (
    SourceConfigurationError,
    SourceError,
    SourceRequestError,
    SourceResponseError,
)
from .interface import CitationResult, CitationSource, SearchResult, Source
from .local import LocalSource
from .openalex import OpenAlexSource, openalex_abstract
from .semantic_scholar import SemanticScholarSource

__all__ = [
    "AclAnthologySource",
    "ArxivSource",
    "CitationResult",
    "CitationSource",
    "LocalSource",
    "OpenAlexSource",
    "SearchResult",
    "SemanticScholarSource",
    "Source",
    "SourceConfigurationError",
    "SourceError",
    "SourceRequestError",
    "SourceResponseError",
    "exceptions",
    "openalex_abstract",
]
