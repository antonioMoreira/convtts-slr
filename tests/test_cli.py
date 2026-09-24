from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from convtts_slr.cli import app, main
from convtts_slr.protocol import DEFAULT_PROTOCOL
from convtts_slr.store import Store

runner = CliRunner()


def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "slr: command-line entry point." in res.stdout

    res_short = runner.invoke(app, ["-h"])
    assert res_short.exit_code == 0
    assert "slr: command-line entry point." in res_short.stdout

    for sub in ["protocol", "graph", "run", "sample", "calibrate", "report"]:
        res_sub = runner.invoke(app, [sub, "--help"])
        assert res_sub.exit_code == 0, f"Failed for {sub}"

    assert runner.invoke(app, ["queue", "--help"]).exit_code == 0
    assert runner.invoke(app, ["queue", "export", "--help"]).exit_code == 0
    assert runner.invoke(app, ["queue", "import", "--help"]).exit_code == 0


def test_cli_protocol(tmp_path: Path):
    res = runner.invoke(app, ["protocol"])
    assert res.exit_code == 0
    assert f"version {DEFAULT_PROTOCOL.version}" in res.stdout
    assert f"{len(DEFAULT_PROTOCOL.criteria)} criteria" in res.stdout

    dump_path = tmp_path / "proto_dump.json"
    res_dump = runner.invoke(app, ["protocol", "--dump", str(dump_path)])
    assert res_dump.exit_code == 0
    assert dump_path.exists()
    loaded_proto = DEFAULT_PROTOCOL.model_validate_json(dump_path.read_text(encoding="utf-8"))
    assert loaded_proto.version == DEFAULT_PROTOCOL.version

    # Read back with --protocol
    res_custom = runner.invoke(app, ["protocol", "--protocol", str(dump_path)])
    assert res_custom.exit_code == 0
    assert f"version {DEFAULT_PROTOCOL.version}" in res_custom.stdout


def test_cli_graph():
    res = runner.invoke(app, ["graph"])
    assert res.exit_code == 0
    assert "flowchart TD" in res.stdout
    assert "identify" in res.stdout
    assert "synthesize" in res.stdout


def test_cli_queue_commands(tmp_path: Path):
    workdir = tmp_path / "work"
    store = Store(workdir, DEFAULT_PROTOCOL.version)
    store.append("identified", paper_id="p1", source="test")

    # Export queue
    queue_csv = tmp_path / "q.csv"
    res_exp = runner.invoke(app, ["queue", "export", str(workdir), str(queue_csv)])
    assert res_exp.exit_code == 0
    assert "0 items exported" in res_exp.stdout or "items exported" in res_exp.stdout

    # Import queue (create empty queue file with header)
    queue_csv.write_text(
        "stage,paper_id,decision,rationale,title,abstract,fulltext_path,criteria_status\n",
        encoding="utf-8",
    )
    res_imp = runner.invoke(app, ["queue", "import", str(workdir), str(queue_csv)])
    assert res_imp.exit_code == 0
    assert "0 decisions imported" in res_imp.stdout


def test_cli_sample_validation(tmp_path: Path):
    workdir = tmp_path / "work"
    out_csv = tmp_path / "sample.csv"
    # Stage option is required
    res = runner.invoke(app, ["sample", str(workdir), str(out_csv)])
    assert res.exit_code != 0
    assert "Missing option" in res.output or "--stage" in res.output

    # With valid stage
    res_ok = runner.invoke(
        app,
        ["sample", str(workdir), str(out_csv), "--stage", "title_abstract", "--n", "10"],
    )
    assert res_ok.exit_code == 0
    assert "papers to label" in res_ok.stdout


def test_cli_report(tmp_path: Path):
    workdir = tmp_path / "work"
    Store(workdir, DEFAULT_PROTOCOL.version)
    res = runner.invoke(app, ["report", str(workdir)])
    assert res.exit_code == 0
    assert f"written {workdir / 'report.md'}" in res.stdout
    assert (workdir / "report.md").exists()


def test_cli_calibrate_validation(tmp_path: Path):
    workdir = tmp_path / "work"
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("paper_id,IC1\n", encoding="utf-8")
    out_json = tmp_path / "thresholds.json"

    # Missing --stage and --out
    res = runner.invoke(app, ["calibrate", str(workdir), str(labels_csv)])
    assert res.exit_code != 0
    assert "Missing option" in res.output or "--stage" in res.output

    # With required options
    res_ok = runner.invoke(
        app,
        [
            "calibrate",
            str(workdir),
            str(labels_csv),
            "--stage",
            "title_abstract",
            "--out",
            str(out_json),
        ],
    )
    assert res_ok.exit_code == 0
    assert out_json.exists()


def test_cli_verbose():
    res = runner.invoke(app, ["-v", "protocol"])
    assert res.exit_code == 0
    assert f"version {DEFAULT_PROTOCOL.version}" in res.stdout


def test_cli_run_validation(tmp_path: Path):
    workdir = tmp_path / "work"
    # Invalid backend choice
    res = runner.invoke(app, ["run", str(workdir), "--backend", "invalid"])
    assert res.exit_code != 0

    # Invalid until stage
    res_until = runner.invoke(app, ["run", str(workdir), "--until", "not_a_stage"])
    assert res_until.exit_code != 0


def test_cli_main_entrypoint(capsys):
    # Test main function directly
    try:
        main(["protocol"])
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_local_requires_repeated_flag(tmp_path: Path):
    # Since the argparse -> Typer migration, `--local` takes one value per
    # occurrence (Click semantics). Space-separated values after a single
    # `--local`, as the old argparse `nargs="*"` accepted, must now fail
    # loudly instead of silently dropping files.
    workdir = tmp_path / "work"
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("", encoding="utf-8")
    b.write_text("", encoding="utf-8")

    res_old_style = runner.invoke(app, ["run", str(workdir), "--local", str(a), str(b)])
    assert res_old_style.exit_code != 0
    assert "unexpected extra argument" in res_old_style.output.lower()

    # Repeating the flag is the supported way to pass multiple files. Both
    # values reach argument parsing (proven by the unrelated --backend
    # validation error firing, rather than a parsing error on --local).
    res_repeated = runner.invoke(
        app,
        ["run", str(workdir), "--local", str(a), "--local", str(b), "--backend", "bogus"],
    )
    assert res_repeated.exit_code != 0
    assert "not one of" in res_repeated.output.lower()
