from convtts_slr.models import DecisionBatch, ExtractionResult, QuestionAnswer, Role
from convtts_slr.protocol import DEFAULT_PROTOCOL
from convtts_slr.synthesis import (
    _level,
    _md_table,
    _noul_level,
    dimension_table,
    prevalence_by_period,
)

_CHOICE_ID = "rq1_sample_unit"
_NOUL_ID = "rq2_scripted_recorded"


def _extraction(pid: str, answers: dict) -> ExtractionResult:
    return ExtractionResult(
        paper_id=pid,
        role=Role.CANDIDATE,
        typed=DecisionBatch(backend="x", model="m", answers=answers, state_sha=""),
    )


def _noul(pid: str, score: float) -> ExtractionResult:
    return _extraction(
        pid, {_NOUL_ID: QuestionAnswer(question_id=_NOUL_ID, kind="noul", score=score)}
    )


def _choice(pid: str, choice: str) -> ExtractionResult:
    return _extraction(
        pid, {_CHOICE_ID: QuestionAnswer(question_id=_CHOICE_ID, kind="choice", choice=choice)}
    )


def test_level_boundaries_use_the_protocol_configured_cuts():
    assert _level(0.75, DEFAULT_PROTOCOL) == "strong"
    assert _level(0.5, DEFAULT_PROTOCOL) == "partial"
    assert _level(0.49, DEFAULT_PROTOCOL) == "none"


def test_noul_level_four_tiers():
    assert _noul_level(0.9, DEFAULT_PROTOCOL) == "consensus: used"
    assert _noul_level(0.6, DEFAULT_PROTOCOL) == "majority"
    assert _noul_level(0.3, DEFAULT_PROTOCOL) == "minority"
    assert _noul_level(0.1, DEFAULT_PROTOCOL) == "rare or unreported"


def test_dimension_table_practice_field_with_no_reports_has_no_data():
    rows = dimension_table([], DEFAULT_PROTOCOL)
    row = next(r for r in rows if r["field"] == _NOUL_ID)
    assert row["n"] == 0
    assert row["level"] == "no data"


def test_dimension_table_choice_field_all_unclear_is_not_reported():
    exs = [_choice("p1", "unclear"), _choice("p2", "unclear")]
    rows = dimension_table(exs, DEFAULT_PROTOCOL)
    row = next(r for r in rows if r["field"] == _CHOICE_ID)
    assert row["level"] == "not reported"
    assert row["unclear"] == 2
    assert row["share"] == 0.0


def test_dimension_table_choice_field_reports_the_modal_option_and_share():
    exs = [_choice("p1", "a"), _choice("p2", "a"), _choice("p3", "b")]
    rows = dimension_table(exs, DEFAULT_PROTOCOL)
    row = next(r for r in rows if r["field"] == _CHOICE_ID)
    assert row["modal"] == "a"
    assert row["share"] == round(2 / 3, 3)
    assert row["unclear"] == 0


def test_prevalence_by_period_buckets_years():
    exs = [_noul("old", 1.0), _noul("new", 0.0)]
    years = {"old": 2020, "new": 2024}
    out = prevalence_by_period(exs, years, DEFAULT_PROTOCOL)
    assert out[_NOUL_ID]["<=2022"] == 1.0
    assert out[_NOUL_ID]["2024"] == 0.0


def test_prevalence_by_period_buckets_missing_years_as_unknown():
    out = prevalence_by_period([_noul("p1", 1.0)], {"p1": None}, DEFAULT_PROTOCOL)
    assert out[_NOUL_ID]["unknown"] == 1.0


def test_md_table_renders_a_pipe_table():
    out = _md_table([{"a": "1", "b": "2"}], ["a", "b"])
    assert out.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |"]
