from datetime import datetime

from convtts_slr.models import (
    DatasetFacts,
    DecisionBatch,
    Event,
    Paper,
    QuestionAnswer,
    Quoted,
    Role,
)


def test_paper_defaults():
    p = Paper(id="p1", title="A Paper")
    assert p.abstract == ""
    assert p.year is None
    assert p.sources == []
    assert p.role == Role.CANDIDATE
    assert p.is_seed is False
    assert p.found_in_round == 0


def test_role_is_a_str_enum_with_expected_values():
    assert Role.CANDIDATE.value == "candidate"
    assert Role.REFERENCE_BASELINE.value == "reference_baseline"
    assert isinstance(Role.CANDIDATE, str)


def test_quoted_defaults_to_no_value_and_no_quote():
    q = Quoted()
    assert q.value is None
    assert q.quote is None


def test_dataset_facts_every_field_defaults_to_an_empty_quoted():
    facts = DatasetFacts()
    for name in DatasetFacts.model_fields:
        assert getattr(facts, name) == Quoted()


def test_event_autogenerates_a_timezone_aware_timestamp():
    ev = Event(kind="test", protocol_version="v1")
    parsed = datetime.fromisoformat(ev.ts)
    assert parsed.tzinfo is not None
    assert ev.paper_id is None
    assert ev.payload == {}


def test_decision_batch_json_round_trips():
    batch = DecisionBatch(
        backend="scripted",
        model="script",
        answers={"q1": QuestionAnswer(question_id="q1", kind="noul", score=0.5)},
        state_sha="deadbeef",
    )
    again = DecisionBatch.model_validate_json(batch.model_dump_json())
    assert again == batch
