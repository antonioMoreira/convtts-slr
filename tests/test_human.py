import csv

from convtts_slr.human import export_calibration_sample, import_queue, pending
from convtts_slr.models import (
    CriterionOutcome,
    DecisionBatch,
    Paper,
    Role,
    ScreeningResult,
    VerificationResult,
)
from convtts_slr.protocol import DEFAULT_PROTOCOL, Stage
from convtts_slr.store import Store

_BATCH = DecisionBatch(backend="scripted", model="script", answers={}, state_sha="")


def _add_paper(store: Store, pid: str, **kw) -> None:
    store.append("paper_added", pid, paper=Paper(id=pid, title=pid, **kw).model_dump(mode="json"))


def _screen(store: Store, pid: str, stage: Stage, verdict: str) -> None:
    res = ScreeningResult(
        paper_id=pid,
        stage=stage.value,
        verdict=verdict,
        reasons=[] if verdict != "needs_human" else ["uncertain"],
        outcomes=[
            CriterionOutcome(
                criterion_id="IC1", polarity="include", status="uncertain", question_scores={}
            )
        ],
        batch=_BATCH,
    )
    store.append("screened", pid, result=res.model_dump(mode="json"))


def test_pending_lists_needs_human_title_abstract_rows(tmp_path):
    store = Store(tmp_path / "work", DEFAULT_PROTOCOL.version)
    _add_paper(store, "p1")
    _screen(store, "p1", Stage.TITLE_ABSTRACT, "needs_human")
    rows = pending(store)
    assert [r["paper_id"] for r in rows] == ["p1"]
    assert rows[0]["queue"] == "title_abstract"


def test_pending_lists_unavailable_fulltext_for_an_included_paper(tmp_path):
    store = Store(tmp_path / "work", DEFAULT_PROTOCOL.version)
    _add_paper(store, "p1")
    _screen(store, "p1", Stage.TITLE_ABSTRACT, "include")
    store.append("fulltext", "p1", status="unavailable")
    rows = pending(store)
    assert any(r["queue"] == "fulltext" and r["paper_id"] == "p1" for r in rows)


def test_pending_lists_failed_verification_for_reference_baselines(tmp_path):
    store = Store(tmp_path / "work", DEFAULT_PROTOCOL.version)
    _add_paper(store, "p1", role=Role.REFERENCE_BASELINE)
    ver = VerificationResult(
        paper_id="p1", missing_quotes=["x:no_quote"], disagreements=[], passed=False
    )
    store.append("verified", "p1", result=ver.model_dump(mode="json"))
    rows = pending(store)
    assert any(r["queue"] == "verification" and r["paper_id"] == "p1" for r in rows)


def test_import_queue_rejects_an_unknown_decision(tmp_path):
    store = Store(tmp_path / "work", DEFAULT_PROTOCOL.version)
    csv_path = tmp_path / "q.csv"
    csv_path.write_text("queue,paper_id,decision\ntitle_abstract,p1,maybe\n")
    try:
        import_queue(store, csv_path)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_import_queue_verification_reject_also_excludes_at_full_text(tmp_path):
    store = Store(tmp_path / "work", DEFAULT_PROTOCOL.version)
    csv_path = tmp_path / "q.csv"
    csv_path.write_text("queue,paper_id,decision\nverification,p1,reject\n")
    n = import_queue(store, csv_path)
    assert n == 1
    assert store.verdicts(Stage.FULL_TEXT.value)["p1"] == "exclude"


def test_export_calibration_sample_covers_every_verdict_bucket(tmp_path):
    store = Store(tmp_path / "work", DEFAULT_PROTOCOL.version)
    verdicts = ["include"] * 3 + ["exclude"] * 3 + ["needs_human"] * 3
    for i, verdict in enumerate(verdicts):
        pid = f"p{i}"
        _add_paper(store, pid, abstract="abstract text")
        _screen(store, pid, Stage.TITLE_ABSTRACT, verdict)
    out = tmp_path / "sample.csv"
    n = export_calibration_sample(store, DEFAULT_PROTOCOL, out, Stage.TITLE_ABSTRACT, n=9, seed=1)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == n
    assert {"paper_id", "title", "abstract", "IC1"} <= set(rows[0].keys())
    assert n >= 3  # at least one from each of the three verdict buckets
