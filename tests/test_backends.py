import pytest
import typesafe_sdk
from pydantic import ValidationError

from convtts_slr.backend import (
    AnthropicBackend,
    BackendConfigurationError,
    BackendError,
    BackendRequestError,
    BackendResponseError,
    EvaluationRequest,
    GeminiBackend,
    JevBackend,
    ScriptedBackend,
    create_backend,
    state_sha,
)
from convtts_slr.protocol import ChoiceSpec, NoulSpec

NOUL = NoulSpec(id="n1", instructions="i", true="t", false="f")
CHOICE = ChoiceSpec(id="c1", instructions="i", options={"x": "X", "y": "Y"})


def request(*questions, state=None) -> EvaluationRequest:
    return EvaluationRequest(state=state or {"title": "t"}, questions=list(questions))


def test_state_sha_is_deterministic_and_key_order_independent():
    a = state_sha({"title": "t", "abstract": "a"})
    b = state_sha({"abstract": "a", "title": "t"})
    assert a == b
    assert a == state_sha({"title": "t", "abstract": "a"})
    assert a != state_sha({"title": "different"})


def test_evaluation_request_sha_and_chunks():
    req = request(NOUL, CHOICE, state={"title": "t"})
    assert req.sha == state_sha({"title": "t"})
    chunks = list(req.chunks(1))
    assert [[q.id for q in c.questions] for c in chunks] == [["n1"], ["c1"]]
    assert all(c.state == req.state for c in chunks)
    assert len(list(req.chunks(5))) == 1


def test_evaluation_request_is_a_validated_frozen_message():
    req = EvaluationRequest.model_validate(
        {"state": {"a": 1}, "questions": [NOUL.model_dump(), CHOICE.model_dump()]}
    )
    assert isinstance(req.questions[0], NoulSpec) and isinstance(req.questions[1], ChoiceSpec)
    with pytest.raises(ValidationError):
        req.state = {}  # ty: ignore[invalid-assignment]


def test_scripted_backend_counts_calls_and_maps_kinds():
    backend = ScriptedBackend(lambda state, q: 0.7 if q.id == "n1" else "x")
    batch = backend.evaluate(request(NOUL, CHOICE))
    assert backend.calls == 1
    assert batch.answers["n1"].kind == "noul" and batch.answers["n1"].score == 0.7
    assert batch.answers["c1"].kind == "choice" and batch.answers["c1"].choice == "x"


# --- exceptions -----------------------------------------------------------------------


def test_backend_error_message_carries_method_and_backend_tags():
    def evaluate(): ...

    err = BackendError(evaluate, "boom", backend_name="jev:x")
    assert str(err) == "[backend_method=evaluate][backend_name=jev:x] boom"
    assert err.backend_name == "jev:x" and err.message == "boom"
    assert str(BackendError(evaluate, backend_name="b")) == (
        "[backend_method=evaluate][backend_name=b]"
    )


def test_backend_request_error_carries_status_code():
    err = BackendRequestError(print, "HTTP 503", backend_name="b", status_code=503)
    assert err.status_code == 503 and "[status_code=503]" in str(err)
    assert isinstance(err, BackendError)


# --- factory --------------------------------------------------------------------------


def test_create_backend_parses_provider_and_model(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert create_backend("jev").name == "jev:jev-latest"
    assert create_backend("jev", "m1").name == "jev:m1"
    assert create_backend("anthropic").name == "anthropic:claude-sonnet-5"
    assert create_backend("anthropic:claude-x").name == "anthropic:claude-x"
    assert create_backend("gemini:gemini-y").name == "gemini:gemini-y"
    assert create_backend("gemini:a", "b").name == "gemini:b"  # explicit model wins


def test_create_backend_rejects_an_unknown_provider():
    with pytest.raises(BackendConfigurationError, match="unknown backend 'llm'"):
        create_backend("llm")


# --- pydantic-ai backends (Anthropic, Gemini) -------------------------------------------


@pytest.mark.parametrize("cls", [AnthropicBackend, GeminiBackend])
def test_llm_backend_output_type_enforces_noul_range_and_choice_membership(cls):
    model = cls()._output_type([NOUL, CHOICE])
    ok = model(n1=0.5, c1="x")
    assert ok.n1 == 0.5 and ok.c1 == "x"
    with pytest.raises(ValidationError):
        model(n1=1.5, c1="x")  # out of the [0, 1] range
    with pytest.raises(ValidationError):
        model(n1=0.5, c1="not-an-option")


@pytest.mark.parametrize("cls", [AnthropicBackend, GeminiBackend])
def test_llm_backend_render_includes_state_and_every_question(cls):
    noul = NoulSpec(id="n1", instructions="do the thing", true="T", false="F")
    choice = ChoiceSpec(id="c1", instructions="pick one", options={"x": "X option"})
    text = cls._render({"title": "hello"}, [noul, choice])
    assert '"title": "hello"' in text
    assert "[n1] do the thing" in text and "TRUE if: T" in text
    assert "[c1] pick one" in text and "option `x`: X option" in text


def test_llm_backends_name_and_pydantic_ai_model_id():
    assert AnthropicBackend().model_id == "anthropic:claude-sonnet-5"
    assert GeminiBackend().model_id == "google:gemini-2.5-flash"
    assert GeminiBackend("gemini-2.5-pro").name == "gemini:gemini-2.5-pro"


class _FakeAgent:
    """Stands in for pydantic_ai.Agent: records the model and returns scripted output."""

    seen: list[str] = []
    result: object = None
    error: Exception | None = None

    def __init__(self, model, output_type, instructions):
        self.output_type = output_type
        type(self).seen.append(model)

    def run_sync(self, prompt):
        if self.error:
            raise self.error
        out = self.result or self.output_type(n1=0.8, c1="y")
        return type("Run", (), {"output": out})()


@pytest.fixture
def fake_agent(monkeypatch):
    import pydantic_ai

    _FakeAgent.seen, _FakeAgent.result, _FakeAgent.error = [], None, None
    monkeypatch.setattr(pydantic_ai, "Agent", _FakeAgent)
    return _FakeAgent


@pytest.mark.parametrize(
    ("cls", "model_id"),
    [(AnthropicBackend, "anthropic:claude-sonnet-5"), (GeminiBackend, "google:gemini-2.5-flash")],
)
def test_llm_backend_evaluate_builds_a_decision_batch(fake_agent, cls, model_id):
    batch = cls().evaluate(request(NOUL, CHOICE))
    assert fake_agent.seen == [model_id]
    assert batch.model == model_id and batch.state_sha == state_sha({"title": "t"})
    assert batch.answers["n1"].score == 0.8
    assert batch.answers["c1"].choice == "y"
    assert batch.answers["c1"].distribution == {"x": 0.0, "y": 1.0}


def test_llm_backend_batches_questions(fake_agent):
    class Scripted(_FakeAgent):
        def run_sync(self, prompt):
            fields = {n: (0.1 if n == "n1" else "x") for n in self.output_type.model_fields}
            return type("Run", (), {"output": self.output_type(**fields)})()

    import pydantic_ai

    pydantic_ai.Agent = Scripted  # undone with the rest of fake_agent's monkeypatching
    batch = AnthropicBackend(batch_size=1).evaluate(request(NOUL, CHOICE))
    assert len(Scripted.seen) == 2  # one provider call per chunk
    assert set(batch.answers) == {"n1", "c1"}


def test_llm_backend_maps_provider_errors(fake_agent):
    from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior

    backend = AnthropicBackend()
    fake_agent.error = ModelHTTPError(status_code=529, model_name="m", body=None)
    with pytest.raises(BackendRequestError) as http:
        backend.evaluate(request(NOUL))
    assert http.value.status_code == 529 and isinstance(http.value.__cause__, ModelHTTPError)

    fake_agent.error = ModelAPIError(model_name="m", message="connection reset")
    with pytest.raises(BackendRequestError) as net:
        backend.evaluate(request(NOUL))
    assert net.value.status_code is None

    fake_agent.error = UnexpectedModelBehavior("invalid output")
    with pytest.raises(BackendResponseError):
        backend.evaluate(request(NOUL))


def test_llm_backend_without_credentials_raises_a_configuration_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(BackendConfigurationError, match="backend_name=anthropic:claude"):
        AnthropicBackend().evaluate(request(NOUL))


def test_gemini_backend_accepts_the_legacy_env_var(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-key")
    from pydantic_ai.models import infer_model

    assert infer_model(GeminiBackend().model_id).model_name == "gemini-2.5-flash"


# --- Jev ------------------------------------------------------------------------------


def test_jev_backend_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(BackendConfigurationError, match="TYPESAFE_API_KEY"):
        JevBackend()


def test_jev_backend_retries_a_connection_error_then_succeeds(monkeypatch):
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
    batch = backend.evaluate(request(NOUL))
    assert attempts["n"] == 2
    assert batch.answers["n1"].score == 0.6


def test_jev_backend_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def always_fails(self, state, questions, **kw):
        raise typesafe_sdk.TypeSafeAPIConnectionError("still down")

    monkeypatch.setattr(typesafe_sdk.TypeSafeClient, "system_one", always_fails)
    backend = JevBackend(api_key="test-key", max_retries=1)
    with pytest.raises(BackendRequestError) as exc:
        backend.evaluate(request(NOUL))
    assert exc.value.status_code is None
    assert isinstance(exc.value.__cause__, typesafe_sdk.TypeSafeAPIConnectionError)


def test_jev_backend_does_not_retry_a_non_retryable_client_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    attempts = {"n": 0}

    def bad_request(self, state, questions, **kw):
        attempts["n"] += 1
        raise typesafe_sdk.TypeSafeAPIError(status=400, body={}, headers={})

    monkeypatch.setattr(typesafe_sdk.TypeSafeClient, "system_one", bad_request)
    backend = JevBackend(api_key="test-key", max_retries=4)
    with pytest.raises(BackendRequestError) as exc:
        backend.evaluate(request(NOUL))
    assert exc.value.status_code == 400
    assert isinstance(exc.value.__cause__, typesafe_sdk.TypeSafeAPIError)
    assert attempts["n"] == 1  # a 400 is not retryable: no point burning retries on it


def test_jev_backend_reports_a_missing_answer_as_a_response_error(monkeypatch):
    def partial(self, state, questions, **kw):
        return typesafe_sdk.SystemOneResponse.model_validate(
            {"model": "jev-latest", "usage": {}, "answers": {"n1": {"type": "noul", "noul": 0.6}}}
        )

    monkeypatch.setattr(typesafe_sdk.TypeSafeClient, "system_one", partial)
    other = NoulSpec(id="n2", instructions="i", true="t", false="f")
    with pytest.raises(BackendResponseError, match="n2"):
        JevBackend(api_key="test-key").evaluate(request(NOUL, other))
