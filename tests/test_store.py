from convtts_slr.models import CriterionOutcome, DecisionBatch, ScreeningResult
from convtts_slr.protocol import Stage
from convtts_slr.store import Store, safe_name

_BATCH = DecisionBatch(backend="scripted", model="script", answers={}, state_sha="")


def test_safe_name_keeps_only_filesystem_safe_characters():
    assert safe_name("doi:10.1/abc def!") == "doi_10.1_abc_def_"


def _screened(pid: str, verdict: str) -> ScreeningResult:
    return ScreeningResult(
        paper_id=pid,
        stage=Stage.TITLE_ABSTRACT.value,
        verdict=verdict,
        reasons=[],
        outcomes=[
            CriterionOutcome(
                criterion_id="IC1", polarity="include", status="pass", question_scores={}
            )
        ],
        batch=_BATCH,
    )


def test_verdicts_lets_a_human_decision_override_the_model(tmp_path):
    store = Store(tmp_path / "work", "v1")
    store.append("screened", "p1", result=_screened("p1", "exclude").model_dump(mode="json"))
    store.append("human_decision", "p1", stage=Stage.TITLE_ABSTRACT.value, verdict="include")
    assert store.verdicts(Stage.TITLE_ABSTRACT.value)["p1"] == "include"


def test_screening_is_scoped_to_the_current_protocol_version(tmp_path):
    workdir = tmp_path / "work"
    v1 = Store(workdir, "v1")
    v1.append("screened", "p1", result=_screened("p1", "include").model_dump(mode="json"))

    v2 = Store(workdir, "v2")
    assert v2.screening(Stage.TITLE_ABSTRACT.value) == {}  # filtered out: wrong version
    assert len(list(v2.events("screened"))) == 1  # but the raw event is still on the log


def test_prior_facts_are_reused_across_a_protocol_version_bump(tmp_path):
    workdir = tmp_path / "work"
    facts = {"total_hours": {"value": 20.0, "quote": "We recorded 20 hours"}}
    v1 = Store(workdir, "v1")
    v1.append(
        "extracted",
        "p1",
        result={"facts_model": "llm:m", "facts": facts},
        text_sha="sha123",
    )
    v2 = Store(workdir, "v2")
    assert v2.prior_facts("p1", "sha123", "llm:m") == facts
    assert v2.prior_facts("p1", "different-sha", "llm:m") is None
    assert v2.prior_facts("p1", "sha123", "llm:other-model") is None
