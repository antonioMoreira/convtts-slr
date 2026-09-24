from convtts_slr.backends import ScriptedBackend
from convtts_slr.models import (
    DatasetFacts,
    DecisionBatch,
    ExtractionResult,
    QuestionAnswer,
    Quoted,
    Role,
)
from convtts_slr.protocol import DEFAULT_PROTOCOL
from convtts_slr.screening import Decider
from convtts_slr.store import Store
from convtts_slr.verify import _disagree, verify

_TEXT = "We recorded 20 hours of dialogue with a pitch predictor."


def _extraction(facts=None) -> ExtractionResult:
    return ExtractionResult(
        paper_id="p1",
        role=Role.CANDIDATE,
        typed=DecisionBatch(backend="scripted", model="script", answers={}, state_sha=""),
        facts=facts,
    )


def test_disagree_noul_compares_the_half_threshold():
    a = QuestionAnswer(question_id="q", kind="noul", score=0.6)
    b = QuestionAnswer(question_id="q", kind="noul", score=0.4)
    assert _disagree(a, b)
    assert not _disagree(a, QuestionAnswer(question_id="q", kind="noul", score=0.9))


def test_disagree_choice_compares_the_chosen_option():
    a = QuestionAnswer(question_id="q", kind="choice", choice="x")
    b = QuestionAnswer(question_id="q", kind="choice", choice="y")
    assert _disagree(a, b)
    assert not _disagree(a, QuestionAnswer(question_id="q", kind="choice", choice="x"))


def test_verify_fails_when_a_fact_has_no_supporting_quote():
    ex = _extraction(facts=DatasetFacts(total_hours=Quoted(value=20.0, quote=None)))
    res = verify(ex, {}, _TEXT, DEFAULT_PROTOCOL, checker=None, screening_answers={})
    assert not res.passed
    assert "total_hours:no_quote" in res.missing_quotes
    assert res.disagreements == []  # no checker: never even considered


def test_verify_passes_with_grounded_facts_and_no_checker():
    ex = _extraction(
        facts=DatasetFacts(total_hours=Quoted(value=20.0, quote="We recorded 20 hours"))
    )
    res = verify(ex, {}, _TEXT, DEFAULT_PROTOCOL, checker=None, screening_answers={})
    assert res.passed


def test_verify_flags_a_checker_disagreement(tmp_path):
    ex = _extraction(facts=None)
    screening_answers = {
        "ic4_pitch_predictor": QuestionAnswer(
            question_id="ic4_pitch_predictor", kind="noul", score=0.9
        )
    }
    checker = Decider(ScriptedBackend(lambda state, q: 0.05), Store(tmp_path / "work", "v1"))
    res = verify(
        ex,
        {"fulltext": _TEXT},
        _TEXT,
        DEFAULT_PROTOCOL,
        checker,
        screening_answers,
        cross_check=["ic4_pitch_predictor"],
    )
    assert not res.passed
    assert res.disagreements == ["ic4_pitch_predictor"]
