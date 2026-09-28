from convtts_slr.backends import ScriptedBackend
from convtts_slr.models import Paper, Role
from convtts_slr.protocol import DEFAULT_PROTOCOL, Stage
from convtts_slr.screening import Decider
from convtts_slr.source import (
    CitationResult,
    LocalSource,
    SearchResult,
    SourceRequestError,
)
from convtts_slr.stages import Context, FetchFullText, Identify, Snowball, _parallel
from convtts_slr.store import Store, safe_name


def _ctx(tmp_path, **kw) -> Context:
    store = Store(tmp_path / "work", DEFAULT_PROTOCOL.version)
    decider = Decider(ScriptedBackend(lambda state, q: 0.5), store)
    return Context(protocol=DEFAULT_PROTOCOL, store=store, decider=decider, **kw)


def test_parallel_isolates_one_papers_exception_from_the_rest(tmp_path):
    ctx = _ctx(tmp_path)
    processed = []

    def fn(p: Paper) -> None:
        if p.id == "bad":
            raise ValueError("boom")
        processed.append(p.id)

    n = _parallel(ctx, "testnode", [Paper(id="bad", title="B"), Paper(id="ok", title="O")], fn)
    assert n == 2
    assert processed == ["ok"]
    errors = list(ctx.store.events("error"))
    assert len(errors) == 1
    assert errors[0].paper_id == "bad"
    assert errors[0].payload["node"] == "testnode"


def test_snowball_stops_immediately_with_no_citation_sources(tmp_path):
    ctx = _ctx(tmp_path, citation_sources=[])
    assert Snowball().run(ctx) == "stop"
    assert list(ctx.store.events("snowball_round")) == []


def test_snowball_stops_at_max_rounds(tmp_path):
    ctx = _ctx(tmp_path)
    for i in range(DEFAULT_PROTOCOL.snowball_max_rounds):
        ctx.store.append("snowball_round", None, round=i + 1, frontier=[], identified=0, new=0)
    assert Snowball().run(ctx) == "stop"


def test_snowball_stops_when_there_is_no_frontier(tmp_path):
    class FakeCitations:
        name = "fake"

        def references(self, p):
            return CitationResult(
                source=self.name, paper_id=p.id, direction="references", papers=[]
            )

        def citations(self, p):
            return CitationResult(source=self.name, paper_id=p.id, direction="citations", papers=[])

    ctx = _ctx(tmp_path, citation_sources=[FakeCitations()])
    assert Snowball().run(ctx) == "stop"  # no included papers yet: nothing to expand


def test_fetch_full_text_retries_a_reference_baseline_once_a_manual_pdf_appears(tmp_path):
    p = Paper(id="ref1", title="R", role=Role.REFERENCE_BASELINE)
    calls = []

    def fetch(paper, workdir):
        calls.append(paper.id)
        return None

    ctx = _ctx(tmp_path, fetch=fetch)
    ctx.store.append("paper_added", p.id, paper=p.model_dump(mode="json"))
    ctx.store.append("fulltext", p.id, status="unavailable")

    FetchFullText().run(ctx)
    assert calls == []  # not retried yet: no manual PDF dropped

    manual = ctx.store.dir / "pdfs" / f"{safe_name(p.id)}.pdf"
    manual.parent.mkdir(parents=True, exist_ok=True)
    manual.write_bytes(b"%PDF-fake")
    FetchFullText().run(ctx)
    assert calls == [p.id]


class _FailingSource:
    name = "failing"

    def search(self, cfg):
        raise SourceRequestError(self.search, "boom", source_name=self.name)


class _OkSource:
    name = "ok"

    def search(self, cfg):
        return SearchResult(
            source=self.name,
            query="native q",
            papers=[Paper(id="p1", title="A dialogue corpus")],
            truncated=True,
        )


def test_identify_skips_a_failing_source_and_leaves_it_to_be_retried(tmp_path):
    ctx = _ctx(tmp_path, sources=[_FailingSource(), _OkSource()])
    assert Identify().run(ctx) == "ok"
    done = [ev.payload["source"] for ev in ctx.store.events("search_done")]
    assert done == ["ok"]  # the failing source has no `search_done`, so a resume retries it
    (ev,) = ctx.store.events("search_done")
    assert ev.payload["native_query"] == "native q" and ev.payload["truncated"] is True
    assert [p.title for p in ctx.store.papers().values()] == ["A dialogue corpus"]


class _FailingCitations:
    name = "failing"

    def references(self, p):
        raise SourceRequestError(self.references, "boom", source_name=self.name)

    def citations(self, p):
        return CitationResult(source=self.name, paper_id=p.id, direction="citations", papers=[])


def test_snowball_keeps_a_paper_on_the_frontier_when_a_lookup_fails(tmp_path):
    ctx = _ctx(tmp_path, citation_sources=[_FailingCitations()])
    ctx.store.append("paper_added", "p1", paper=Paper(id="p1", title="P").model_dump(mode="json"))
    ctx.store.append("human_decision", "p1", stage=Stage.FULL_TEXT.value, verdict="include")
    assert Snowball().run(ctx) == "stop"  # nothing new found
    assert list(ctx.store.events("snowballed")) == []  # so p1 is retried in a later round
    assert len(list(ctx.store.events("snowball_round"))) == 1


def test_a_local_source_that_is_missing_raises_a_source_error(tmp_path):
    ctx = _ctx(tmp_path, sources=[LocalSource(tmp_path / "missing.json", name="gone")])
    assert Identify().run(ctx) == "ok"  # logged and skipped, not raised
    assert list(ctx.store.events("search_done")) == []
