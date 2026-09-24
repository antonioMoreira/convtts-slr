from datetime import datetime, timezone

import arxiv
import httpx
import pytest

from convtts_slr.sources import (
    ArxivSource,
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


def test_arxiv_to_paper_maps_fields():
    dt = datetime(2023, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
    res = arxiv.Result(
        entry_id="http://arxiv.org/abs/2304.12345v1",
        updated=dt,
        published=dt,
        title="  A Neural  Speech \n Model ",
        summary="  We present a model\nfor dialogue. ",
        doi="10.1234/test.doi",
    )
    p = ArxivSource._to_paper(res)
    assert p.arxiv_id == "2304.12345v1"
    assert p.doi == "10.1234/test.doi"
    assert p.id == "arxiv:2304.12345v1"
    assert p.title == "A Neural Speech Model"
    assert p.abstract == "We present a model for dialogue."
    assert p.year == 2023
    assert p.publication_date == "2023-04-15"
    assert p.venue == "arXiv"
    assert p.sources == ["arxiv"]
    assert p.pdf_url == "https://arxiv.org/pdf/2304.12345v1"


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
