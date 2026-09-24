import httpx
import pymupdf
import pytest

from convtts_slr.fulltext import download_pdf, pdf_to_text, select_sections
from convtts_slr.models import Paper
from convtts_slr.store import safe_name


def test_select_sections_returns_short_text_unchanged():
    text = "Short paper, well under budget."
    assert select_sections(text, 5000) == text


def test_download_pdf_uses_a_manually_dropped_file_without_any_network(tmp_path):
    p = Paper(id="doi:10.1/x", title="t")
    manual = tmp_path / "pdfs" / f"{safe_name(p.id)}.pdf"
    manual.parent.mkdir(parents=True)
    manual.write_bytes(b"%PDF-fake")
    # No client/urls involved at all: this must return before touching the network.
    assert download_pdf(p, tmp_path) == manual


def test_download_pdf_returns_none_when_there_is_nothing_to_try(tmp_path):
    p = Paper(id="doi:10.1/x", title="t")  # no arxiv_id, no pdf_url, no manual file
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: pytest.fail("no request expected"))
    )
    assert download_pdf(p, tmp_path, client=client) is None


def test_download_pdf_returns_none_on_persistent_failure(tmp_path):
    p = Paper(id="arxiv:1111.11111", title="t", arxiv_id="1111.11111")
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    assert download_pdf(p, tmp_path, client=client) is None


class _FakePage:
    def get_text(self, _mode):
        return "dia-\nlogue with  extra   spaces"


class _FakeDoc:
    def __enter__(self):
        return [_FakePage()]

    def __exit__(self, *exc):
        return False


def test_pdf_to_text_dehyphenates_and_collapses_whitespace(tmp_path, monkeypatch):
    # pdf_to_text only depends on page.get_text(), so drive it through a fake doc
    # instead of needing a real PDF with a genuine hyphenated line break.
    monkeypatch.setattr(pymupdf, "open", lambda _path: _FakeDoc())
    out = pdf_to_text(tmp_path / "whatever.pdf")
    assert out == "dialogue with extra spaces"
