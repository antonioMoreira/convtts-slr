# Decision backends (`src/convtts_slr/backend/`)

System One answers batches of typed questions (Nouls and Choices) against one shared JSON
state. The package mirrors `source/`: one interface module, one exceptions module, one module
per backend, and an `__init__.py` that re-exports them.

```
backend/
  __init__.py     re-exports + create_backend(spec)
  interface.py    DecisionBackend (Protocol), EvaluationRequest (BaseModel), state_sha
  exceptions.py   BackendError hierarchy
  _llm.py         PydanticAIBackend: prompt, output type, batching, error mapping
  jev.py          JevBackend      (typesafe-sdk, own retry loop)
  anthropic.py    AnthropicBackend (pydantic-ai, "anthropic:" models)
  gemini.py       GeminiBackend    (pydantic-ai, "google:" models)
  scripted.py     ScriptedBackend  (tests and dry runs)
```

## Why a Protocol

`Decider` (`screening.py`) and `Context.checker` only need an object with `name` and
`evaluate`. A structural `Protocol` (like `source.interface.Source`) keeps the provider SDKs
optional and lazily imported, and lets tests plug in `ScriptedBackend` without inheritance. The
shared pydantic-ai logic is a plain base class instead, because Anthropic and Gemini differ only
in model prefix, default model and credentials.

## Message passing

- In: `EvaluationRequest(state, questions)`, a frozen `BaseModel`. `.sha` is the state hash and
  `.chunks(n)` splits the questions for providers with batch limits.
- Out: `DecisionBatch` / `QuestionAnswer` (`models.py`), unchanged, because they are persisted
  in `backend_call` events.
- `Decider.evaluate(state, questions, paper_id)` keeps its signature and builds the request.

## Exceptions

Messages look like `[backend_method=evaluate][backend_name=jev:jev-latest] message`.
Provider exceptions are chained (`__cause__`).

| Exception | Raised when |
|---|---|
| `BackendError` | base class |
| `BackendConfigurationError` | missing credential or optional dependency, unknown backend spec |
| `BackendRequestError` (`status_code`) | transport or HTTP failure after retries |
| `BackendResponseError` | answer missing or unusable, provider output rejected |

A `BackendConfigurationError` aborts the run (`stages._parallel` re-raises it and `slr run`
exits with status 2), because it would fail every paper. Every other error is recorded as a
per-paper `error` event and the review continues.

## Choosing a backend

`slr run --backend jev|anthropic|gemini [--model M]`. `--checker` takes `provider[:model]`
(or `none`), so the independent checker can come from a different provider than System One:

```bash
slr run work --backend anthropic --checker gemini:gemini-2.5-flash
```

| Backend | Install | Credential | Default model |
|---|---|---|---|
| `jev` | `.[jev]` | `TYPESAFE_API_KEY` | `jev-latest` |
| `anthropic` | `.[llm]` | `ANTHROPIC_API_KEY` | `claude-sonnet-5` |
| `gemini` | `.[gemini]` | `GOOGLE_API_KEY` (legacy `GEMINI_API_KEY` also works) | `gemini-2.5-flash` |

Backend names (`jev:<model>`, `anthropic:<model>`, `gemini:<model>`) are part of the answer
cache key. Workdirs created with the old `llm:<model>` name re-query the model on resume.
`system_two.py` (`--facts`) is a facts extractor, not a decision backend, and still takes a
pydantic-ai model string.

## Adding a backend

1. Add `backend/<name>.py` with a class that has `name` (include the model) and
   `evaluate(request) -> DecisionBatch`.
2. Wrap provider failures in the `Backend*Error` classes with `raise ... from exc`.
3. Export it in `__init__.py` and add a `case` to `create_backend`.
4. Add an optional extra in `pyproject.toml` if it needs an SDK.

## Verification done

Offline suite: 168 passed. `ruff` and `ty check` are clean. Live: `RUN_LIVE=1 pytest -m live`
passes with Jev (see `e2e-live-test.md`), and `slr run --backend gemini --until screen_ta`
against the real Gemini API screened both seeds (`backend_call` events carry
`gemini:gemini-2.5-flash`). Anthropic was not run live (no key available here).
