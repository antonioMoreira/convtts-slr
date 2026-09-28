import httpx
import pytest
import requests

from convtts_slr.models import Paper
from convtts_slr.protocol import SearchConfig
from convtts_slr.sources import (
    AclAnthologySource,
    LocalSource,
    OpenAlexSource,
    SemanticScholarSource,
    _get,
)


def test_local_source_reads_ieee_xplore_style_csv(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text(
        "Document Title,Abstract,Publication Year,DOI,Publication Title\n"
        '"A Dialogue Corpus","We collect...",2023,10.1/abc,ICASSP\n'
    )
    papers = LocalSource(path).search()
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "A Dialogue Corpus"
    assert p.year == 2023
    assert p.doi == "10.1/abc"
    assert p.venue == "ICASSP"
    assert p.id == "10.1/abc"
    assert f"local:{path.name}" in p.sources


def test_local_source_reads_jsonl(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_text(
        '{"title": "Paper One", "doi": "10.1/one"}\n{"title": "Paper Two", "doi": "10.1/two"}\n'
    )
    papers = LocalSource(path).search()
    assert [p.title for p in papers] == ["Paper One", "Paper Two"]


def test_local_source_falls_back_to_title_as_id(tmp_path):
    path = tmp_path / "export.json"
    path.write_text('[{"title": "No Identifiers Here"}]')
    papers = LocalSource(path).search()
    assert papers[0].id == "No Identifiers Here"


def test_local_source_truncates_a_full_date_to_the_year(tmp_path):
    path = tmp_path / "export.json"
    path.write_text('[{"title": "t", "doi": "10.1/x", "year": "2022-05-01"}]')
    assert LocalSource(path).search()[0].year == 2022


def test_openalex_to_paper_extracts_arxiv_id_from_a_landing_page_url():
    w = {
        "id": "https://openalex.org/W123",
        "title": "A Paper",
        "best_oa_location": {"landing_page_url": "https://arxiv.org/abs/2207.01063"},
        "locations": [],
    }
    p = OpenAlexSource()._to_paper(w)
    assert p.arxiv_id == "2207.01063"
    assert p.openalex_id == "W123"


# --------------------------------------------------------------------------- #
# OpenAlex via pyalex: fake `Works`-shaped stand-ins, matching the same
# duck-typed-stub pattern used for ACL Anthology below -- no real HTTP, no
# coupling to pyalex's exact object construction.
# --------------------------------------------------------------------------- #


def test_openalex_search_builds_filter_and_paginates(monkeypatch):
    pages = [
        [
            {"id": "https://openalex.org/W1", "title": "A"},
            {"id": "https://openalex.org/W2", "title": "B"},
        ],
        [{"id": "https://openalex.org/W3", "title": "C"}],
    ]

    class _FakeWorks:
        def __init__(self):
            self.captured = {"filter": {}}

        def search(self, s):
            self.captured["search"] = s
            return self

        def filter(self, **kw):
            self.captured["filter"].update(kw)
            return self

        def select(self, s):
            self.captured["select"] = s
            return self

        def paginate(self, per_page=None, n_max=None):
            self.captured["per_page"] = per_page
            self.captured["n_max"] = n_max
            return iter(pages)

    fake = _FakeWorks()
    monkeypatch.setattr("convtts_slr.sources.Works", lambda: fake)
    cfg = SearchConfig(
        blocks=[["dialogue"]], from_year=2020, to_date="2024-06-01", max_results_per_source=2
    )
    papers = OpenAlexSource().search(cfg)
    assert fake.captured["filter"] == {
        "from_publication_date": "2020-01-01",
        "to_publication_date": "2024-06-01",
    }
    assert fake.captured["per_page"] == 200
    assert fake.captured["n_max"] == 2
    # a full page can overshoot n_max; search() truncates to the requested limit
    assert [p.title for p in papers] == ["A", "B"]


def test_openalex_work_returns_none_on_http_error(monkeypatch):
    class _FakeWorks:
        def __getitem__(self, key):
            raise requests.exceptions.HTTPError("404")

    monkeypatch.setattr("convtts_slr.sources.Works", lambda: _FakeWorks())
    p = Paper(id="x", title="x", openalex_id="W0")
    assert OpenAlexSource()._work(p) is None


def test_openalex_references_chunks_lookups_at_100_ids(monkeypatch):
    ids = [f"W{i}" for i in range(150)]
    work = {"referenced_works": [f"https://openalex.org/{i}" for i in ids]}
    calls: list[list[str]] = []

    class _FakeWorks:
        def select(self, s):
            return self

        def __getitem__(self, key):
            calls.append(list(key))
            return [{"id": f"https://openalex.org/{i}", "title": f"t{i}"} for i in key]

    src = OpenAlexSource()
    monkeypatch.setattr(src, "_work", lambda p: work)
    monkeypatch.setattr("convtts_slr.sources.Works", lambda: _FakeWorks())
    papers = src.references(Paper(id="x", title="x", openalex_id="W0"))
    assert [len(c) for c in calls] == [100, 50]
    assert len(papers) == 150


def test_openalex_citations_filters_by_cites_id_and_caps_at_2000(monkeypatch):
    class _FakeWorks:
        def __init__(self):
            self.captured = {}

        def filter(self, **kw):
            self.captured.update(kw)
            return self

        def select(self, s):
            return self

        def paginate(self, per_page=None, n_max=None):
            self.captured["n_max"] = n_max
            page = [{"id": f"https://openalex.org/W{i}", "title": f"t{i}"} for i in range(200)]
            return iter([page])

    fake = _FakeWorks()
    src = OpenAlexSource()
    monkeypatch.setattr(src, "_work", lambda p: {"id": "https://openalex.org/W0"})
    monkeypatch.setattr("convtts_slr.sources.Works", lambda: fake)
    papers = src.citations(Paper(id="x", title="x"))
    assert fake.captured["cites"] == "W0"
    assert fake.captured["n_max"] == 2000
    assert len(papers) == 200


def test_openalex_citations_returns_empty_when_work_not_found(monkeypatch):
    src = OpenAlexSource()
    monkeypatch.setattr(src, "_work", lambda p: None)
    assert src.citations(Paper(id="x", title="x")) == []


def test_semantic_scholar_to_paper_maps_external_ids():
    d = {
        "paperId": "abc123",
        "title": "A Paper",
        "externalIds": {"DOI": "10.1/xyz", "ArXiv": "2207.01063"},
        "openAccessPdf": {"url": "https://example.org/x.pdf"},
    }
    p = SemanticScholarSource()._to_paper(d)
    assert p.doi == "10.1/xyz"
    assert p.arxiv_id == "2207.01063"
    assert p.s2_id == "abc123"
    assert p.pdf_url == "https://example.org/x.pdf"


def test_get_retries_a_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    r = _get(client, "https://example.org/x")
    assert r.status_code == 200
    assert calls["n"] == 2


def test_get_raises_after_persistent_server_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    with pytest.raises(httpx.HTTPStatusError):
        _get(client, "https://example.org/x", tries=2)


# --------------------------------------------------------------------------- #
# ACL Anthology: duck-typed stubs, so none of this touches the network or the ~120 MB
# corpus -- AclAnthologySource only calls `Anthology.from_repo()` inside
# _get_anthology(), and only when no anthology instance is injected.
# --------------------------------------------------------------------------- #


class _FakeText:
    def __init__(self, text: str):
        self._text = text

    def as_text(self) -> str:
        return self._text


class _FakeVenue:
    def __init__(self, name: str):
        self.name = name


class _FakeVolume:
    def __init__(self, venue_names=(), raises: bool = False):
        self._venue_names = venue_names
        self._raises = raises

    def venues(self):
        if self._raises:
            raise RuntimeError("venues.json not loaded")
        return [_FakeVenue(n) for n in self._venue_names]


class _FakePDF:
    def __init__(self, url: str):
        self.url = url


class _FakePaper:
    def __init__(
        self,
        full_id="2022.acl-long.1",
        title="A Dialogue Dataset",
        abstract="We introduce a dataset.",
        year="2022",
        doi="10.18653/v1/2022.acl-long.1",
        pdf_url="https://aclanthology.org/2022.acl-long.1.pdf",
        venue_ids=("acl",),
        venue_names=("Annual Meeting of the ACL",),
        venues_raise=False,
        is_deleted=False,
        is_frontmatter=False,
    ):
        self.full_id = full_id
        self.title = _FakeText(title)
        self.abstract = _FakeText(abstract) if abstract is not None else None
        self.year = year
        self.doi = doi
        self.pdf = _FakePDF(pdf_url) if pdf_url else None
        self.web_url = f"https://aclanthology.org/{full_id}/"
        self.venue_ids = venue_ids
        self.parent = _FakeVolume(venue_names, raises=venues_raise)
        self.is_deleted = is_deleted
        self.is_frontmatter = is_frontmatter


class _FakeAnthology:
    def __init__(self, papers):
        self._papers = papers

    def papers(self):
        return iter(self._papers)


def test_acl_anthology_to_paper_maps_fields():
    paper = AclAnthologySource._to_paper(_FakePaper())
    assert paper.id == "acl:2022.acl-long.1"
    assert paper.title == "A Dialogue Dataset"
    assert paper.abstract == "We introduce a dataset."
    assert paper.year == 2022
    assert paper.doi == "10.18653/v1/2022.acl-long.1"
    assert paper.pdf_url == "https://aclanthology.org/2022.acl-long.1.pdf"
    assert paper.venue == "Annual Meeting of the ACL"
    assert paper.sources == ["acl_anthology"]


def test_acl_anthology_to_paper_falls_back_to_web_url_when_no_pdf():
    p = _FakePaper(pdf_url=None)
    assert AclAnthologySource._to_paper(p).pdf_url == p.web_url


def test_acl_anthology_venue_falls_back_to_venue_ids_when_lookup_fails():
    p = _FakePaper(venue_ids=("acl", "emnlp"), venues_raise=True)
    assert AclAnthologySource._to_paper(p).venue == "ACL, EMNLP"


def test_acl_anthology_to_paper_handles_missing_abstract_and_unparseable_year():
    paper = AclAnthologySource._to_paper(_FakePaper(abstract=None, year="n/a"))
    assert paper.abstract == ""
    assert paper.year is None


def test_acl_anthology_search_filters_by_year_and_keywords_and_skips_deleted_or_frontmatter():
    papers = [
        _FakePaper(full_id="2018.x-1", title="Old dialogue corpus", year="2018"),  # too old
        _FakePaper(full_id="2022.x-1", title="A conversational speech corpus", year="2022"),
        _FakePaper(full_id="2022.x-2", title="Unrelated parsing paper", year="2022"),
        _FakePaper(
            full_id="2022.x-3", title="A deleted dialogue corpus", year="2022", is_deleted=True
        ),
        _FakePaper(
            full_id="2022.x-4",
            title="Front matter dialogue notice",
            year="2022",
            is_frontmatter=True,
        ),
    ]
    src = AclAnthologySource(anthology=_FakeAnthology(papers))
    cfg = SearchConfig(blocks=[["dialogue", "conversational"]], from_year=2020)
    assert [p.id for p in src.search(cfg)] == ["acl:2022.x-1"]


def test_acl_anthology_search_caps_at_max_results_per_source():
    papers = [_FakePaper(full_id=f"2022.x-{i}", title="dialogue corpus") for i in range(5)]
    src = AclAnthologySource(anthology=_FakeAnthology(papers))
    cfg = SearchConfig(blocks=[["dialogue"]], max_results_per_source=2)
    assert len(src.search(cfg)) == 2
