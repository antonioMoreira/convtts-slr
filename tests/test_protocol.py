from whenever import Date

from convtts_slr.protocol import (
    DEFAULT_PROTOCOL,
    Protocol,
    SearchConfig,
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


def test_search_config_dates_do_not_change_the_protocol_version():
    # `to_date` is a whenever.Date but serializes to the same ISO string as the old `str`,
    # so existing work dirs keep resuming; a changed hash here means logs stop matching.
    assert DEFAULT_PROTOCOL.version == "5a334f33b7ab"
    assert DEFAULT_PROTOCOL.search.model_dump(mode="json")["to_date"] == "2026-09-30"


def test_search_config_from_date_is_the_first_of_january():
    cfg = SearchConfig(blocks=[["x"]], from_year=2021, to_date="2024-06-01")
    assert cfg.from_date == Date(2021, 1, 1)
    assert cfg.to_date == Date(2024, 6, 1)
