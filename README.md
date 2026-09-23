# convtts-slr

This is an agentic pipeline for the systematic review *Conversational datasets for controllable TTS*. The requirements it implements are in [PROTOCOL.md](PROTOCOL.md).

## Design in one paragraph

Bounded, high-frequency decisions are handled by **System One**: every eligibility criterion is a Jev Noul and every categorical extraction field is a Jev Noul or Choice, answered in one batched call per paper. **System Two** (an LLM through pydantic-ai) is used only for values that cannot be bounded, such as hours, speaker counts, tool names and rationale, and every value must carry a verbatim quote. Both are orchestrated by an explicit graph with a bounded snowball cycle. All runtime state lives in an append-only event log, and a separate verifier checks the extractor's work.

| Graph Engineering concern | Where it lives |
|---|---|
| Task organization (DAG, bounded loop) | `graph.py`, `stages.build_graph()` |
| Agent coordination (screener / extractor / verifier / human) | `screening.py`, `system_two.py`, `verify.py`, `human.py` |
| Runtime state management (provenance, resume, recovery) | `store.py` (event sourcing) |

```
identify → screen_ta → fetch_fulltext → screen_ft → extract → verify → snowball
              ↑                                                            │
              └──────────────────────── new_records ───────────────────────┘
                                                             stop → synthesize → END
```

## Setup

```bash
pip install -e ".[jev,llm,dev]"
export TYPESAFE_API_KEY=...        # Jev (System One)
export ANTHROPIC_API_KEY=...       # System Two + independent checker (any pydantic-ai model works)
export OPENALEX_MAILTO=you@uni.br  # OpenAlex polite pool
export S2_API_KEY=...              # optional, strongly recommended for snowballing
pytest                             # offline end-to-end test, no keys needed
```

Without Jev access yet, run with `--backend llm`. In that case, pass a *different* model to `--checker`, otherwise the second opinion is not independent.

## Workflow

The run is split in three phases so that thresholds are calibrated before they affect decisions.

**1. Identify and screen titles/abstracts, then pause.**

```bash
slr run work --until screen_ta
```

**2. Calibrate the title/abstract stage.** Label a stratified sample: in each criterion column, write 1 if the proposition holds and 0 if it doesn't.

```bash
slr sample work cal_ta.csv --stage title_abstract --n 80
slr calibrate work cal_ta.csv --stage title_abstract --out thresholds.json
```

**3. Continue with calibrated thresholds, then calibrate the full-text stage.**

```bash
slr run work --thresholds thresholds.json --until screen_ft
slr sample work cal_ft.csv --stage full_text --n 60 --thresholds thresholds.json
slr calibrate work cal_ft.csv --stage full_text --out thresholds.json --thresholds thresholds.json
```

**4. Full run, with human queue rounds until the queue is empty.**

```bash
slr run work --thresholds thresholds.json
slr queue import work work/human_queue.csv --thresholds thresholds.json   # after filling `decision`
slr run work --thresholds thresholds.json                                  # resumes; no repeated calls
```

The outputs are written to `work/`:

- `report.md`: PRISMA flow, the consensus table per dimension compared against the reference baseline, and trends by year.
- `consensus.json` and `extraction.csv`: the same results in machine-readable form, for your own analysis.
- `human_queue.csv`: pending items for you to decide.
- `events.jsonl`: the full audit trail.

## Things to know

- **Re-running is cheap.** Backend answers are cached by the exact state and question text. Changing thresholds re-routes papers without new Jev calls. Changing a question's wording creates a new protocol version, and only the batches containing that question are re-asked.
- **Papers whose full text can't be fetched** are listed in the queue. To add one, drop the PDF at `work/pdfs/<file_stub>.pdf` and re-run.
- **Editing the protocol without touching code:** `slr protocol --dump p.json`, edit the JSON, then run with `--protocol p.json`.
- **The Jev adapter** follows the `typesafe-sdk` API as used by Jev-Mem (`TypeSafeClient.system_one`, `Noul`, `Choice`). The test suite covers the translation with a mocked client, but it has not been run against the live API from here.
- **Retrieval clients** (OpenAlex, Semantic Scholar, arXiv) were written against the public API docs. Their parsers are unit-tested, but live calls have not been exercised in this sandbox, so do a small first run (`--until identify`) and check `work/prisma.json`.
- **Full-text parsing** uses PyMuPDF. Section selection is heuristic and keeps the opening plus data/model sections within `fulltext_char_budget`; two-column PDFs occasionally interleave. GROBID would be the upgrade if extraction quality on IC4 is weak.
