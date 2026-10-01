"""Live end-to-end test: drives the real `slr` CLI against the real sources and LLM backend.

Opt-in, because it needs the network, API keys and spends LLM credits:

    RUN_LIVE=1 uv run pytest -m live tests/test_e2e_live.py -s

    TYPESAFE_API_KEY    required: System One (`--backend jev`)
    ANTHROPIC_API_KEY   optional: also exercises System Two (`--facts`) and the checker
    RUN_LIVE_ACL=1      optional: add the ACL Anthology source (~120 MB download on first use)
    OPENALEX_EMAIL, OPENALEX_API_KEY, S2_API_KEY   optional: better rate limits

The tests in `TestLivePipeline` run in file order and share one workdir, like the stages of
a real review. Assertions are structural (events, files, invariants), never about which
papers were found: live results change.
"""

import csv
import json
import os
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from convtts_slr.cli import SEEDS, app
from convtts_slr.protocol import DEFAULT_PROTOCOL
from convtts_slr.stages import final_includes
from convtts_slr.store import Store

pytest.importorskip("typesafe_sdk")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to run live tests"
    ),
    pytest.mark.skipif("TYPESAFE_API_KEY" not in os.environ, reason="TYPESAFE_API_KEY not set"),
]

SEED_ARXIV_IDS = ("2207.01063", "2104.04896")  # DailyTalk (candidate), speech toolbox (baseline)
SOURCE_NAMES = {
    "openalex": "openalex",
    "s2": "semantic_scholar",
    "arxiv": "arxiv",
    "acl": "acl_anthology",
}

runner = CliRunner()


def invoke(*args: str | Path):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, f"slr {' '.join(map(str, args))} failed:\n{result.output}"
    return result


@pytest.fixture(scope="module")
def sources() -> list[str]:
    chosen = ["openalex", "s2", "arxiv"]
    if os.environ.get("RUN_LIVE_ACL") == "1":
        chosen.append("acl")
    return chosen


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("live")


@pytest.fixture(scope="module")
def protocol_file(root: Path) -> Path:
    """The default protocol, narrowed so a live run stays small (and gets its own version)."""
    search = DEFAULT_PROTOCOL.search.model_copy(
        update={"from_year": 2024, "max_results_per_source": 10}
    )
    proto = DEFAULT_PROTOCOL.model_copy(update={"search": search, "snowball_max_rounds": 1})
    path = root / "protocol.json"
    path.write_text(proto.model_dump_json(indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def seeds_file(root: Path) -> Path:
    """One candidate seed and one reference baseline, both on arXiv, so fetching is real."""
    seeds = [
        s
        for s in json.loads(SEEDS.read_text(encoding="utf-8"))
        if s.get("arxiv_id") in SEED_ARXIV_IDS
    ]
    assert len(seeds) == 2
    path = root / "seeds.json"
    path.write_text(json.dumps(seeds), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def workdir(root: Path) -> Path:
    return root / "work"


@pytest.fixture(scope="module")
def run_args(workdir, protocol_file, seeds_file, sources) -> list[str | Path]:
    llm = "ANTHROPIC_API_KEY" in os.environ
    return [
        "run",
        workdir,
        "--protocol",
        protocol_file,
        "--seeds",
        seeds_file,
        "--sources",
        ",".join(sources),
        "--workers",
        "4",
        "--facts",
        "anthropic:claude-sonnet-5" if llm else "none",
        "--checker",
        "anthropic:claude-sonnet-5" if llm else "none",
    ]


@pytest.fixture(scope="module")
def protocol_args(protocol_file) -> list[str | Path]:
    return ["--protocol", protocol_file]


def open_store(workdir: Path, protocol_file: Path) -> Store:
    from convtts_slr.cli import load_protocol

    return Store(workdir, load_protocol(protocol_file, None).version)


def kinds(store: Store) -> list[str]:
    return [ev.kind for ev in store.events()]


class TestLivePipeline:
    def test_identify_and_screen_title_abstract(self, workdir, protocol_file, run_args, sources):
        invoke(*run_args, "--until", "screen_ta")
        store = open_store(workdir, protocol_file)

        searched = {ev.payload["source"]: ev.payload for ev in store.events("search_done")}
        wanted = {SOURCE_NAMES[s] for s in sources}
        for missing in sorted(wanted - searched.keys()):
            warnings.warn(
                f"source {missing} did not complete (down or rate limited?)", stacklevel=2
            )
        assert len(searched) >= 2, f"only {sorted(searched)} completed"
        assert sum(p["identified"] for p in searched.values()) > 0

        papers = store.papers()
        seeds = [p for p in papers.values() if p.is_seed]
        assert len(seeds) == 2

        screened = store.screening("title_abstract")
        assert screened, "no paper was screened"
        assert len(list(store.events("backend_call"))) > 0, "the LLM backend was never called"
        errors = list(store.events("error"))
        if errors:
            warnings.warn(f"{len(errors)} stage errors, first: {errors[0].payload}", stacklevel=2)
        assert len(errors) < max(1, len(papers) // 2)

        assert (workdir / "human_queue.csv").exists()

    def test_calibration_sample(self, workdir, protocol_args, tmp_path):
        out = tmp_path / "sample.csv"
        invoke("sample", workdir, out, "--stage", "title_abstract", "--n", "10", *protocol_args)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert 0 < len(rows) <= 10

    def test_resume_to_synthesis(self, workdir, protocol_file, run_args):
        invoke(*run_args)
        store = open_store(workdir, protocol_file)

        fulltext = store.fulltext_status()
        assert fulltext, "no full text was attempted"
        assert "ok" in fulltext.values(), f"no PDF could be fetched and parsed: {fulltext}"
        ok_ids = [pid for pid, s in fulltext.items() if s == "ok"]
        assert all((store.read_text(pid) or "").strip() for pid in ok_ids)

        assert store.extractions(), "nothing was extracted"
        assert store.verifications().keys() == store.extractions().keys()

        for name in ("prisma.json", "consensus.json", "extraction.csv", "report.md"):
            assert (workdir / name).exists(), name
        flow = json.loads((workdir / "prisma.json").read_text(encoding="utf-8"))
        assert flow["reference_baselines"] >= 1

        if final_includes(store):
            assert list(store.events("snowballed")), "includes exist but nothing was snowballed"
            assert store.snowball_rounds_done() >= 1

        stages = [
            (ev.payload["result"]["stage"], ev.payload["result"]["paper_id"])
            for ev in store.events("screened", current_version_only=True)
        ]
        assert len(stages) == len(set(stages)), "a paper was screened twice at one stage"

    def test_resume_is_idempotent(self, workdir, protocol_file, run_args):
        before = kinds(open_store(workdir, protocol_file))
        invoke(*run_args)
        after = kinds(open_store(workdir, protocol_file))
        assert after.count("backend_call") == before.count("backend_call")
        assert after.count("screened") == before.count("screened")
        assert after.count("extracted") == before.count("extracted")

    def test_human_queue_roundtrip_and_report(
        self, workdir, protocol_file, protocol_args, tmp_path
    ):
        queue = tmp_path / "queue.csv"
        invoke("queue", "export", workdir, queue, *protocol_args)
        rows = list(csv.DictReader(queue.open(encoding="utf-8")))

        if rows:
            choice = {
                "title_abstract": "exclude",
                "full_text": "exclude",
                "fulltext": "exclude",
                "verification": "accept",
            }
            row = rows[0]
            row["decision"] = choice[row["queue"]]
            with queue.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            invoke("queue", "import", workdir, queue, *protocol_args)
            store = open_store(workdir, protocol_file)
            if row["queue"] in ("title_abstract", "full_text"):
                assert store.verdicts(row["queue"])[row["paper_id"]] == "exclude"
        else:
            warnings.warn("human queue was empty: import path not exercised", stacklevel=2)

        (workdir / "report.md").unlink()
        invoke("report", workdir, *protocol_args)
        assert (workdir / "report.md").read_text(encoding="utf-8").strip()
