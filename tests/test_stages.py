from convtts_slr.backends import ScriptedBackend
from convtts_slr.models import Paper, Role
from convtts_slr.protocol import DEFAULT_PROTOCOL
from convtts_slr.screening import Decider
from convtts_slr.stages import Context, FetchFullText, Snowball, _parallel
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
            return []

        def citations(self, p):
            return []

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
