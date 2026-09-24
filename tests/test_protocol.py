from convtts_slr.protocol import (
    DEFAULT_PROTOCOL,
    Protocol,
    Stage,
    Thresholds,
    _signal_source,
)


def test_all_question_ids_are_unique_across_criteria_and_extraction():
    ids = [q.id for c in DEFAULT_PROTOCOL.criteria for q in c.questions]
    ids += [q.id for q in DEFAULT_PROTOCOL.extraction]
    assert len(ids) == len(set(ids))


def test_criteria_for_filters_by_stage():
    ta_ids = {c.id for c in DEFAULT_PROTOCOL.criteria_for(Stage.TITLE_ABSTRACT)}
    ft_ids = {c.id for c in DEFAULT_PROTOCOL.criteria_for(Stage.FULL_TEXT)}
    assert ta_ids == {"IC1", "IC2", "IC3", "EC1", "EC3"}
    assert ft_ids == {"IC4", "IC6"}
    assert ta_ids.isdisjoint(ft_ids)


def test_version_is_a_stable_deterministic_hash():
    v1 = DEFAULT_PROTOCOL.version
    v2 = DEFAULT_PROTOCOL.version
    assert v1 == v2
    assert len(v1) == 12
    # A freshly-parsed copy with identical content must hash the same way.
    rebuilt = Protocol.model_validate_json(DEFAULT_PROTOCOL.model_dump_json())
    assert rebuilt.version == v1


def test_version_changes_when_a_criterion_changes():
    edited = DEFAULT_PROTOCOL.model_copy(
        update={
            "criteria": [
                c.model_copy(update={"rule": "all"}) if c.id == "IC3" else c
                for c in DEFAULT_PROTOCOL.criteria
            ]
        }
    )
    assert edited.version != DEFAULT_PROTOCOL.version


def test_thresholds_defaults():
    t = Thresholds()
    assert t.low == 0.15
    assert t.high == 0.85


def test_signal_source_builds_a_four_option_choice():
    q = _signal_source("pitch")
    assert q.id == "rq3_pitch_signal_source"
    assert set(q.options) == {"extracted_from_audio", "annotated_labels", "not_used", "unclear"}
