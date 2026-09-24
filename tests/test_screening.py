from convtts_slr.backends import ScriptedBackend
from convtts_slr.models import Paper
from convtts_slr.protocol import DEFAULT_PROTOCOL, Criterion, NoulSpec, Stage
from convtts_slr.screening import Decider, build_state, route, screen
from convtts_slr.store import Store


def test_decider_caches_and_never_calls_the_backend_twice_for_the_same_key(tmp_path):
    store = Store(tmp_path / "work", "v1")
    backend = ScriptedBackend(lambda state, q: 0.5)
    decider = Decider(backend, store)
    q = DEFAULT_PROTOCOL.extraction[0]
    state = {"title": "t"}
    decider.evaluate(state, [q], "p1")
    decider.evaluate(state, [q], "p1")
    assert backend.calls == 1


def test_decider_reloads_its_cache_from_the_store_on_reopen(tmp_path):
    store = Store(tmp_path / "work", "v1")
    backend = ScriptedBackend(lambda state, q: 0.5)
    q = DEFAULT_PROTOCOL.extraction[0]
    state = {"title": "t"}
    Decider(backend, store).evaluate(state, [q], "p1")

    reopened = Store(tmp_path / "work", "v1")
    backend2 = ScriptedBackend(lambda state, q: 0.5)
    Decider(backend2, reopened).evaluate(state, [q], "p1")
    assert backend2.calls == 0  # served from the log, not a fresh call


def test_build_state_shape_per_stage():
    p = Paper(id="p1", title="T", abstract="A", year=2024, venue="V")
    ta = build_state(p, Stage.TITLE_ABSTRACT, DEFAULT_PROTOCOL, None)
    assert set(ta) == {"title", "abstract", "year", "venue"}
    ft = build_state(p, Stage.FULL_TEXT, DEFAULT_PROTOCOL, "full text body")
    assert set(ft) == {"title", "fulltext"}


def test_route_all_rule_requires_every_question_to_pass():
    c = Criterion(
        id="TEST",
        label="test",
        polarity="include",
        stage=Stage.TITLE_ABSTRACT,
        rule="all",
        questions=[
            NoulSpec(id="a", instructions="a", true="t", false="f"),
            NoulSpec(id="b", instructions="b", true="t", false="f"),
        ],
    )
    both_high = route(c, {"a": 0.9, "b": 0.9}, DEFAULT_PROTOCOL)
    assert both_high.status == "pass"
    one_low = route(c, {"a": 0.9, "b": 0.1}, DEFAULT_PROTOCOL)
    assert one_low.status == "fail"  # min() drags the combined score down


def test_screen_raises_a_seed_alarm_instead_of_silently_excluding(tmp_path):
    p = Paper(id="seed1", title="A Known Positive With No Abstract", is_seed=True)
    # Everything scores as a hard exclude; for a seed that must become a human alarm.
    backend = ScriptedBackend(lambda state, q: 0.02)
    decider = Decider(backend, Store(tmp_path / "work", "v1"))
    result = screen(p, Stage.TITLE_ABSTRACT, DEFAULT_PROTOCOL, decider)
    assert result.verdict == "needs_human"
    assert "SEED_EXCLUDED_BY_MODEL" in result.reasons
