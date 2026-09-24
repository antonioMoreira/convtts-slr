"""slr: command-line entry point.

slr protocol  [--protocol P] [--dump FILE]      show version / dump editable JSON
slr graph                                        print the pipeline graph (Mermaid)
slr run       WORKDIR [options]                  run / resume the review
slr queue     export|import WORKDIR FILE         human-in-the-loop queue
slr sample    WORKDIR FILE --stage S [--n 80]    calibration sample to label
slr calibrate WORKDIR LABELS --stage S --out F   fit thresholds (then pass --thresholds F)
slr report    WORKDIR                            rebuild report.md from the log
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .calibration import apply_thresholds, calibrate
from .human import export_calibration_sample, export_queue, import_queue
from .protocol import DEFAULT_PROTOCOL, Protocol, Stage
from .screening import Decider
from .sources import ArxivSource, LocalSource, OpenAlexSource, SemanticScholarSource, Source
from .stages import Context, build_graph
from .store import Store

SEEDS = Path(__file__).resolve().parents[2] / "data" / "seeds.json"


def load_protocol(path: str | None, thresholds: str | None) -> Protocol:
    p = (
        Protocol.model_validate_json(Path(path).read_text(encoding="utf-8"))
        if path
        else DEFAULT_PROTOCOL
    )
    return apply_thresholds(p, thresholds)


def _backend(kind: str, model: str | None):
    if kind == "jev":
        from .backends import JevBackend

        return JevBackend(model=model)
    if kind == "llm":
        from .backends import LLMBackend

        return LLMBackend(model=model or "anthropic:claude-sonnet-5")
    raise SystemExit(f"unknown backend {kind!r}")


def cmd_run(a) -> None:
    protocol = load_protocol(a.protocol, a.thresholds)
    store = Store(a.workdir, protocol.version)
    sources = {
        "openalex": OpenAlexSource,
        "s2": SemanticScholarSource,
        "arxiv": ArxivSource,
    }
    chosen: list[Source] = [sources[s]() for s in a.sources.split(",") if s] if a.sources else []
    chosen += [LocalSource(f) for f in a.local or []]
    cites = [OpenAlexSource(), SemanticScholarSource()] if a.snowball else []
    seeds = LocalSource(a.seeds, name="seeds").search() if a.seeds else []
    facts = None
    if a.facts != "none":
        from .system_two import LLMFactsExtractor

        facts = LLMFactsExtractor(model=a.facts)
    checker = Decider(_backend("llm", a.checker), store) if a.checker != "none" else None
    ctx = Context(
        protocol=protocol,
        store=store,
        decider=Decider(_backend(a.backend, a.model), store),
        sources=chosen,
        citation_sources=cites,
        seeds=seeds,
        facts=facts,
        checker=checker,
        workers=a.workers,
    )
    build_graph().run("identify", ctx, stop_after=a.until)
    if a.until and a.until != "synthesize":
        from .synthesis import write_report

        write_report(store, protocol)
    n = export_queue(store, Path(a.workdir) / "human_queue.csv")
    print(
        f"protocol {protocol.version}: report at {Path(a.workdir) / 'report.md'}; "
        f"{n} items in {Path(a.workdir) / 'human_queue.csv'}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="slr",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("protocol")
    p.add_argument("--protocol")
    p.add_argument("--thresholds")
    p.add_argument("--dump")

    sub.add_parser("graph")

    r = sub.add_parser("run")
    r.add_argument("workdir")
    r.add_argument("--protocol", help="protocol JSON (from `slr protocol --dump`)")
    r.add_argument("--thresholds", help="thresholds JSON from `slr calibrate`")
    r.add_argument("--backend", default="jev", choices=["jev", "llm"])
    r.add_argument("--model", help="Jev model (default jev-latest) or pydantic-ai model string")
    r.add_argument("--sources", default="openalex,s2,arxiv")
    r.add_argument("--local", nargs="*", help="CSV/JSON exports, e.g. IEEE Xplore CSV")
    r.add_argument("--seeds", default=str(SEEDS))
    r.add_argument("--no-snowball", dest="snowball", action="store_false")
    r.add_argument(
        "--facts",
        default="anthropic:claude-sonnet-5",
        help="System-Two model, or 'none'",
    )
    r.add_argument(
        "--checker",
        default="anthropic:claude-sonnet-5",
        help="independent checker model, or 'none'",
    )
    r.add_argument("--workers", type=int, default=4)
    r.add_argument(
        "--until",
        choices=[
            "identify",
            "screen_ta",
            "fetch_fulltext",
            "screen_ft",
            "extract",
            "verify",
            "snowball",
            "synthesize",
        ],
        help="pause after this node (e.g. screen_ta, to calibrate before going on)",
    )

    q = sub.add_parser("queue")
    q.add_argument("action", choices=["export", "import"])
    q.add_argument("workdir")
    q.add_argument("file")
    q.add_argument("--protocol")
    q.add_argument("--thresholds")
    q.add_argument("--reviewer", default="human")

    s = sub.add_parser("sample")
    s.add_argument("workdir")
    s.add_argument("file")
    s.add_argument("--stage", required=True, choices=[x.value for x in Stage])
    s.add_argument("--n", type=int, default=80)
    s.add_argument("--protocol")
    s.add_argument("--thresholds")

    c = sub.add_parser("calibrate")
    c.add_argument("workdir")
    c.add_argument("labels")
    c.add_argument("--stage", required=True, choices=[x.value for x in Stage])
    c.add_argument("--out", required=True)
    c.add_argument("--precision", type=float, default=0.95)
    c.add_argument("--max-fn", type=float, default=0.02)
    c.add_argument("--protocol")
    c.add_argument("--thresholds")

    rp = sub.add_parser("report")
    rp.add_argument("workdir")
    rp.add_argument("--protocol")
    rp.add_argument("--thresholds")

    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if a.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    if a.cmd == "protocol":
        pr = load_protocol(a.protocol, a.thresholds)
        if a.dump:
            Path(a.dump).write_text(pr.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"version {pr.version}: {len(pr.criteria)} criteria, "
            f"{len(pr.extraction)} extraction questions"
        )
    elif a.cmd == "graph":
        print(build_graph().to_mermaid())
    elif a.cmd == "run":
        cmd_run(a)
    else:
        pr = load_protocol(a.protocol, a.thresholds)
        store = Store(a.workdir, pr.version)
        if a.cmd == "queue":
            if a.action == "export":
                print(f"{export_queue(store, a.file)} items exported")
            else:
                print(
                    f"{import_queue(store, a.file, a.reviewer)} decisions imported; "
                    "re-run `slr run` to continue"
                )
        elif a.cmd == "sample":
            n = export_calibration_sample(store, pr, a.file, Stage(a.stage), a.n)
            print(f"{n} papers to label")
        elif a.cmd == "calibrate":
            rep = calibrate(store, pr, a.labels, Stage(a.stage), a.precision, a.max_fn)
            existing = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
            Path(a.out).write_text(
                json.dumps(existing | rep["thresholds"], indent=2), encoding="utf-8"
            )
            print(json.dumps(rep["criteria"], indent=2))
        elif a.cmd == "report":
            from .synthesis import write_report

            write_report(store, pr)
            print(f"written {Path(a.workdir) / 'report.md'}")


if __name__ == "__main__":
    main()
