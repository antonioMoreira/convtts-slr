# Live end-to-end test

`tests/test_e2e_live.py` drives the real `slr` CLI (`CliRunner`) against the real retrieval
sources and the real LLM backend. Nothing is faked or monkeypatched.

It complements the offline tests: `tests/test_pipeline.py::test_end_to_end` covers the graph
with scripted fakes, and `tests/test_cli.py` only validates `slr run` arguments. This test is
the only one that exercises real network sources, the real System One backend and real PDFs.

## Running it

```bash
RUN_LIVE=1 uv run pytest -m live -s tests/test_e2e_live.py
```

A plain `uv run pytest` deselects it (`addopts = "-m 'not live'"` in `pyproject.toml`), so the
default suite stays offline and fast.

| Variable | Required | Effect |
|---|---|---|
| `RUN_LIVE=1` | yes | Enables the test; otherwise it is skipped. |
| `TYPESAFE_API_KEY` | yes | System One (`--backend jev`); `typesafe-sdk` must be installed. |
| `ANTHROPIC_API_KEY` | no | When set, `--facts` and `--checker` use `anthropic:claude-sonnet-5`; otherwise both are `none`. |
| `RUN_LIVE_ACL=1` | no | Adds the ACL Anthology source (about 120 MB download on first use). |
| `OPENALEX_EMAIL`, `OPENALEX_API_KEY`, `S2_API_KEY` | no | Better rate limits. |

## Setup the test builds

- **Protocol**: `DEFAULT_PROTOCOL` with `from_year=2024`, `max_results_per_source=10` and
  `snowball_max_rounds=1`, written to a JSON file and passed with `--protocol`. The narrowed
  search gives it its own protocol version, so it never mixes with a real review.
- **Seeds**: two entries from `data/seeds.json`, selected by arXiv id: DailyTalk
  (`2207.01063`, candidate) and the speech-dataset toolbox (`2104.04896`, reference baseline).
  Both are on arXiv, so PDF fetching is real.
- **Sources**: `openalex`, `s2`, `arxiv` (plus `acl` with `RUN_LIVE_ACL=1`). Snowballing uses
  the OpenAlex and Semantic Scholar citation sources that `run_cmd` always wires in.
- **Workdir**: a pytest temp directory shared by all tests in the module.

## Test sequence

The tests in `TestLivePipeline` run in file order and share one workdir, like the stages of a
real review. A failure in an early test makes the later ones fail too.

1. `test_identify_and_screen_title_abstract`: `slr run --until screen_ta`.
   - At least two sources produced a `search_done` event and together identified papers. A
     source that did not complete only produces a warning.
   - Both seeds are in the store; at least one paper has a title/abstract screening result;
     `backend_call` events exist (the LLM was hit); `human_queue.csv` was written.
   - Stage `error` events produce a warning, and fewer than half the papers may have errored.
2. `test_calibration_sample`: `slr sample --stage title_abstract --n 10` writes 1 to 10 rows.
3. `test_resume_to_synthesis`: the same `slr run` without `--until`.
   - At least one full text has status `ok`, and every `ok` paper has non-empty parsed text.
   - Extractions exist, and verifications cover the same papers.
   - `prisma.json`, `consensus.json`, `extraction.csv` and `report.md` exist, and
     `prisma.json` reports at least one reference baseline.
   - If any paper is a final include, `snowballed` events exist and at least one snowball
     round is recorded.
   - No paper is screened twice at one stage under the current protocol version.
4. `test_resume_is_idempotent`: a third `slr run` adds no `backend_call`, `screened` or
   `extracted` events.
5. `test_human_queue_roundtrip_and_report`: `slr queue export`, set a decision on the first row
   (`exclude` for screening and full-text rows, `accept` for verification rows), `slr queue
   import`, and check that a screening verdict reflects it. Then `report.md` is deleted and
   `slr report` rebuilds it. If the queue is empty, the import path is skipped with a warning.

Assertions are structural (events, files, invariants), never about which papers are found,
because live results change.

## Result of the first run

Run on 2026-09-30 with `TYPESAFE_API_KEY` set, no `ANTHROPIC_API_KEY`, and the ACL source off:

```
tests/test_e2e_live.py::TestLivePipeline::test_identify_and_screen_title_abstract PASSED
tests/test_e2e_live.py::TestLivePipeline::test_calibration_sample PASSED
tests/test_e2e_live.py::TestLivePipeline::test_resume_to_synthesis PASSED
tests/test_e2e_live.py::TestLivePipeline::test_resume_is_idempotent PASSED
tests/test_e2e_live.py::TestLivePipeline::test_human_queue_roundtrip_and_report PASSED
5 passed in 15.65s
```

No warnings were printed, so all three sources completed and no stage errors were logged. The
offline suite was unaffected: 148 passed, 5 deselected, and `ruff check` was clean.

## Limitations

- Without `ANTHROPIC_API_KEY`, System Two (`--facts`) and the independent checker (`--checker`)
  are not exercised live. The scripted `test_pipeline.py` still covers them.
- The ACL Anthology source was not run (opt-in because of the download).
- The snowball check is conditional on at least one final include. With the narrowed protocol
  that is likely but not guaranteed.
- The test spends LLM credits and depends on third-party availability and rate limits, hence
  the opt-in gating and the small `max_results_per_source`.
