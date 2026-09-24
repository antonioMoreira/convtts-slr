"""slr: command-line entry point.

slr protocol  [--protocol P] [--dump FILE]      show version / dump editable JSON
slr graph                                        print the pipeline graph (Mermaid)
slr run       WORKDIR [options]                  run / resume the review
slr queue     export|import WORKDIR FILE         human-in-the-loop queue
slr sample    WORKDIR FILE --stage S [--n 80]    calibration sample to label
slr calibrate WORKDIR LABELS --stage S --out F   fit thresholds (then pass --thresholds F)
slr report    WORKDIR                            rebuild report.md from the log
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from .calibration import apply_thresholds, calibrate
from .human import export_calibration_sample, export_queue, import_queue
from .protocol import DEFAULT_PROTOCOL, Protocol, Stage
from .screening import Decider
from .sources import (
    ArxivSource,
    CitationSource,
    LocalSource,
    OpenAlexSource,
    SemanticScholarSource,
    Source,
)
from .stages import Context, build_graph
from .store import Store

SEEDS = Path(__file__).resolve().parents[2] / "data" / "seeds.json"


class BackendKind(str, Enum):
    jev = "jev"
    llm = "llm"


class UntilStage(str, Enum):
    identify = "identify"
    screen_ta = "screen_ta"
    fetch_fulltext = "fetch_fulltext"
    screen_ft = "screen_ft"
    extract = "extract"
    verify = "verify"
    snowball = "snowball"
    synthesize = "synthesize"


app = typer.Typer(
    name="slr",
    help=__doc__,
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

queue_app = typer.Typer(
    name="queue",
    help="Human-in-the-loop review queue.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(queue_app, name="queue")


@app.callback()
def main_callback(
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Enable verbose logging (INFO level)."),
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )


def load_protocol(path: str | Path | None, thresholds: str | Path | None) -> Protocol:
    p = (
        Protocol.model_validate_json(Path(path).read_text(encoding="utf-8"))
        if path
        else DEFAULT_PROTOCOL
    )
    return apply_thresholds(p, thresholds)


def _backend(kind: str | BackendKind, model: str | None):
    kind_val = kind.value if isinstance(kind, BackendKind) else kind
    if kind_val == "jev":
        from .backends import JevBackend

        return JevBackend(model=model)
    if kind_val == "llm":
        from .backends import LLMBackend

        return LLMBackend(model=model or "anthropic:claude-sonnet-5")
    raise typer.BadParameter(f"unknown backend {kind!r}")


@app.command("protocol")
def protocol_cmd(
    protocol: Annotated[
        Path | None,
        typer.Option("--protocol", help="Protocol JSON file (from `slr protocol --dump`)."),
    ] = None,
    thresholds: Annotated[
        Path | None,
        typer.Option("--thresholds", help="Thresholds JSON file from `slr calibrate`."),
    ] = None,
    dump: Annotated[
        Path | None,
        typer.Option("--dump", help="Dump editable protocol JSON to file."),
    ] = None,
) -> None:
    """Show protocol version / dump editable JSON."""
    pr = load_protocol(protocol, thresholds)
    if dump:
        dump.write_text(pr.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        f"version {pr.version}: {len(pr.criteria)} criteria, "
        f"{len(pr.extraction)} extraction questions"
    )


@app.command("graph")
def graph_cmd() -> None:
    """Print the pipeline graph (Mermaid)."""
    typer.echo(build_graph().to_mermaid())


@app.command("run")
def run_cmd(
    workdir: Annotated[
        Path,
        typer.Argument(help="Working directory for the review."),
    ],
    protocol: Annotated[
        Path | None,
        typer.Option("--protocol", help="Protocol JSON (from `slr protocol --dump`)."),
    ] = None,
    thresholds: Annotated[
        Path | None,
        typer.Option("--thresholds", help="Thresholds JSON from `slr calibrate`."),
    ] = None,
    backend: Annotated[
        BackendKind,
        typer.Option("--backend", help="Screening backend."),
    ] = BackendKind.jev,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Jev model (default jev-latest) or pydantic-ai model string.",
        ),
    ] = None,
    sources: Annotated[
        str,
        typer.Option(
            "--sources",
            help="Comma-separated retrieval sources: openalex,s2,arxiv.",
        ),
    ] = "openalex,s2,arxiv",
    local: Annotated[
        list[Path] | None,
        typer.Option(
            "--local",
            help=(
                "CSV/JSON exports, e.g. IEEE Xplore CSV. Repeat the flag for each file "
                "(--local a.csv --local b.csv); space-separated values after one --local "
                "are not accepted."
            ),
        ),
    ] = None,
    seeds: Annotated[
        Path | None,
        typer.Option("--seeds", help="Seed papers JSON file."),
    ] = SEEDS,
    snowball: Annotated[
        bool,
        typer.Option("--snowball/--no-snowball", help="Enable or disable snowballing."),
    ] = True,
    facts: Annotated[
        str,
        typer.Option("--facts", help="System-Two model, or 'none'."),
    ] = "anthropic:claude-sonnet-5",
    checker: Annotated[
        str,
        typer.Option("--checker", help="Independent checker model, or 'none'."),
    ] = "anthropic:claude-sonnet-5",
    workers: Annotated[
        int,
        typer.Option("--workers", help="Number of worker threads."),
    ] = 4,
    until: Annotated[
        UntilStage | None,
        typer.Option(
            "--until",
            help="Pause after this node (e.g. screen_ta, to calibrate before going on).",
        ),
    ] = None,
) -> None:
    """Run / resume the review."""
    proto = load_protocol(protocol, thresholds)
    store = Store(workdir, proto.version)
    sources_map: dict[str, type[Source]] = {
        "openalex": OpenAlexSource,
        "s2": SemanticScholarSource,
        "arxiv": ArxivSource,
    }
    chosen: list[Source] = (
        [sources_map[s.strip()]() for s in sources.split(",") if s.strip()] if sources else []
    )
    if local:
        chosen.extend([LocalSource(f) for f in local])
    cites: list[CitationSource] = [OpenAlexSource(), SemanticScholarSource()] if snowball else []
    seeds_papers = LocalSource(seeds, name="seeds").search() if seeds else []
    facts_extractor = None
    if facts != "none":
        from .system_two import LLMFactsExtractor

        facts_extractor = LLMFactsExtractor(model=facts)
    checker_decider = Decider(_backend("llm", checker), store) if checker != "none" else None
    ctx = Context(
        protocol=proto,
        store=store,
        decider=Decider(_backend(backend, model), store),
        sources=chosen,
        citation_sources=cites,
        seeds=seeds_papers,
        facts=facts_extractor,
        checker=checker_decider,
        workers=workers,
    )
    build_graph().run("identify", ctx, stop_after=until.value if until else None)
    if until and until != UntilStage.synthesize:
        from .synthesis import write_report

        write_report(store, proto)
    n = export_queue(store, workdir / "human_queue.csv")
    typer.echo(
        f"protocol {proto.version}: report at {workdir / 'report.md'}; "
        f"{n} items in {workdir / 'human_queue.csv'}"
    )


@queue_app.command("export")
def queue_export_cmd(
    workdir: Annotated[
        Path,
        typer.Argument(help="Working directory."),
    ],
    file: Annotated[
        Path,
        typer.Argument(help="Destination file path for queue CSV."),
    ],
    protocol: Annotated[
        Path | None,
        typer.Option("--protocol", help="Protocol JSON file path."),
    ] = None,
    thresholds: Annotated[
        Path | None,
        typer.Option("--thresholds", help="Thresholds JSON file path."),
    ] = None,
) -> None:
    """Export human-in-the-loop queue to CSV."""
    proto = load_protocol(protocol, thresholds)
    store = Store(workdir, proto.version)
    count = export_queue(store, file)
    typer.echo(f"{count} items exported")


@queue_app.command("import")
def queue_import_cmd(
    workdir: Annotated[
        Path,
        typer.Argument(help="Working directory."),
    ],
    file: Annotated[
        Path,
        typer.Argument(help="Source CSV file path."),
    ],
    protocol: Annotated[
        Path | None,
        typer.Option("--protocol", help="Protocol JSON file path."),
    ] = None,
    thresholds: Annotated[
        Path | None,
        typer.Option("--thresholds", help="Thresholds JSON file path."),
    ] = None,
    reviewer: Annotated[
        str,
        typer.Option("--reviewer", help="Reviewer identifier."),
    ] = "human",
) -> None:
    """Import human-in-the-loop decisions from CSV."""
    proto = load_protocol(protocol, thresholds)
    store = Store(workdir, proto.version)
    count = import_queue(store, file, reviewer)
    typer.echo(f"{count} decisions imported; re-run `slr run` to continue")


@app.command("sample")
def sample_cmd(
    workdir: Annotated[
        Path,
        typer.Argument(help="Working directory."),
    ],
    file: Annotated[
        Path,
        typer.Argument(help="Destination CSV file path for calibration sample."),
    ],
    stage: Annotated[
        Stage,
        typer.Option("--stage", help="Stage to sample for calibration."),
    ],
    n: Annotated[
        int,
        typer.Option("--n", help="Number of papers to sample."),
    ] = 80,
    protocol: Annotated[
        Path | None,
        typer.Option("--protocol", help="Protocol JSON file path."),
    ] = None,
    thresholds: Annotated[
        Path | None,
        typer.Option("--thresholds", help="Thresholds JSON file path."),
    ] = None,
) -> None:
    """Generate a calibration sample to label."""
    proto = load_protocol(protocol, thresholds)
    store = Store(workdir, proto.version)
    count = export_calibration_sample(store, proto, file, stage, n)
    typer.echo(f"{count} papers to label")


@app.command("calibrate")
def calibrate_cmd(
    workdir: Annotated[
        Path,
        typer.Argument(help="Working directory."),
    ],
    labels: Annotated[
        Path,
        typer.Argument(help="CSV file with labeled calibration sample."),
    ],
    stage: Annotated[
        Stage,
        typer.Option("--stage", help="Stage to calibrate."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output JSON file for fitted thresholds."),
    ],
    precision: Annotated[
        float,
        typer.Option("--precision", help="Target precision."),
    ] = 0.95,
    max_fn: Annotated[
        float,
        typer.Option("--max-fn", help="Maximum false negative rate."),
    ] = 0.02,
    protocol: Annotated[
        Path | None,
        typer.Option("--protocol", help="Protocol JSON file path."),
    ] = None,
    thresholds: Annotated[
        Path | None,
        typer.Option("--thresholds", help="Thresholds JSON file path."),
    ] = None,
) -> None:
    """Fit thresholds (then pass --thresholds to run)."""
    proto = load_protocol(protocol, thresholds)
    store = Store(workdir, proto.version)
    rep = calibrate(store, proto, labels, stage, precision, max_fn)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    out.write_text(json.dumps(existing | rep["thresholds"], indent=2), encoding="utf-8")
    typer.echo(json.dumps(rep["criteria"], indent=2))


@app.command("report")
def report_cmd(
    workdir: Annotated[
        Path,
        typer.Argument(help="Working directory."),
    ],
    protocol: Annotated[
        Path | None,
        typer.Option("--protocol", help="Protocol JSON file path."),
    ] = None,
    thresholds: Annotated[
        Path | None,
        typer.Option("--thresholds", help="Thresholds JSON file path."),
    ] = None,
) -> None:
    """Rebuild report.md from the log."""
    from .synthesis import write_report

    proto = load_protocol(protocol, thresholds)
    store = Store(workdir, proto.version)
    write_report(store, proto)
    typer.echo(f"written {workdir / 'report.md'}")


def main(argv: list[str] | None = None) -> None:
    app(args=argv)


if __name__ == "__main__":
    main()
