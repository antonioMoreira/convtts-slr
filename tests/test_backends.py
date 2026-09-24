import pytest
import typesafe_sdk
from pydantic import ValidationError

from convtts_slr.backends import LLMBackend, ScriptedBackend, state_sha
from convtts_slr.protocol import ChoiceSpec, NoulSpec


def test_state_sha_is_deterministic_and_key_order_independent():
    a = state_sha({"title": "t", "abstract": "a"})
    b = state_sha({"abstract": "a", "title": "t"})
    assert a == b
    assert a == state_sha({"title": "t", "abstract": "a"})
    assert a != state_sha({"title": "different"})


def test_scripted_backend_counts_calls_and_maps_kinds():
    noul = NoulSpec(id="n1", instructions="i", true="t", false="f")
    choice = ChoiceSpec(id="c1", instructions="i", options={"x": "X", "y": "Y"})
    backend = ScriptedBackend(lambda state, q: 0.7 if q.id == "n1" else "x")
    batch = backend.evaluate({"title": "t"}, [noul, choice])
    assert backend.calls == 1
    assert batch.answers["n1"].kind == "noul" and batch.answers["n1"].score == 0.7
    assert batch.answers["c1"].kind == "choice" and batch.answers["c1"].choice == "x"


def test_llm_backend_output_type_enforces_noul_range_and_choice_membership():
    backend = LLMBackend()
    noul = NoulSpec(id="n1", instructions="i", true="t", false="f")
    choice = ChoiceSpec(id="c1", instructions="i", options={"x": "X", "y": "Y"})
    model = backend._output_type([noul, choice])
    ok = model(n1=0.5, c1="x")
    assert ok.n1 == 0.5 and ok.c1 == "x"
    with pytest.raises(ValidationError):
        model(n1=1.5, c1="x")  # out of the [0, 1] range
    with pytest.raises(ValidationError):
        model(n1=0.5, c1="not-an-option")


def test_llm_backend_render_includes_state_and_every_question():
    backend = LLMBackend()
    noul = NoulSpec(id="n1", instructions="do the thing", true="T", false="F")
    choice = ChoiceSpec(id="c1", instructions="pick one", options={"x": "X option"})
    text = backend._render({"title": "hello"}, [noul, choice])
    assert '"title": "hello"' in text
    assert "[n1] do the thing" in text and "TRUE if: T" in text
    assert "[c1] pick one" in text and "option `x`: X option" in text


def test_jev_backend_retries_a_connection_error_then_succeeds(monkeypatch):
    from convtts_slr.backends import JevBackend

    monkeypatch.setattr("time.sleep", lambda _s: None)
    attempts = {"n": 0}

    def fake_system_one(self, state, questions, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise typesafe_sdk.TypeSafeAPIConnectionError("boom")
        return typesafe_sdk.SystemOneResponse.model_validate(
            {
                "model": "jev-latest",
                "usage": {},
                "answers": {"n1": {"type": "noul", "noul": 0.6}},
            }
        )

    monkeypatch.setattr(typesafe_sdk.TypeSafeClient, "system_one", fake_system_one)
    backend = JevBackend(api_key="test-key")
    q = NoulSpec(id="n1", instructions="i", true="t", false="f")
    batch = backend.evaluate({"title": "t"}, [q])
    assert attempts["n"] == 2
    assert batch.answers["n1"].score == 0.6


def test_jev_backend_gives_up_after_max_retries(monkeypatch):
    from convtts_slr.backends import JevBackend

    monkeypatch.setattr("time.sleep", lambda _s: None)

    def always_fails(self, state, questions, **kw):
        raise typesafe_sdk.TypeSafeAPIConnectionError("still down")

    monkeypatch.setattr(typesafe_sdk.TypeSafeClient, "system_one", always_fails)
    backend = JevBackend(api_key="test-key", max_retries=1)
    q = NoulSpec(id="n1", instructions="i", true="t", false="f")
    with pytest.raises(typesafe_sdk.TypeSafeAPIConnectionError):
        backend.evaluate({"title": "t"}, [q])


def test_jev_backend_does_not_retry_a_non_retryable_client_error(monkeypatch):
    from convtts_slr.backends import JevBackend

    monkeypatch.setattr("time.sleep", lambda _s: None)
    attempts = {"n": 0}

    def bad_request(self, state, questions, **kw):
        attempts["n"] += 1
        raise typesafe_sdk.TypeSafeAPIError(status=400, body={}, headers={})

    monkeypatch.setattr(typesafe_sdk.TypeSafeClient, "system_one", bad_request)
    backend = JevBackend(api_key="test-key", max_retries=4)
    q = NoulSpec(id="n1", instructions="i", true="t", false="f")
    with pytest.raises(typesafe_sdk.TypeSafeAPIError):
        backend.evaluate({"title": "t"}, [q])
    assert attempts["n"] == 1  # a 400 is not retryable: no point burning retries on it
