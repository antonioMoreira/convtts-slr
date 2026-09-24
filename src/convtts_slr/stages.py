"""Pipeline nodes and the review graph.

    identify -> screen_ta -> fetch_fulltext -> screen_ft -> extract -> verify -> snowball
                    ^                                                               |
                    +----------------------- new_records ---------------------------+
                                                                        stop -> synthesize -> END

Every node is idempotent: it only processes papers that have no event for its step
under the current protocol version, so re-running after a crash, after importing human
decisions, or after re-calibrating thresholds continues where the log left off.
Papers routed to `needs_human` simply wait; the rest of the graph keeps moving.
"""

import hashlib
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import dedup
from .backends import state_sha
from .fulltext import download_pdf, pdf_to_text
from .graph import Graph
from .models import (
    CriterionOutcome,
    DatasetFacts,
    DecisionBatch,
    ExtractionResult,
    Paper,
    Role,
    ScreeningResult,
)
from .protocol import Protocol, Stage
from .screening import Decider, build_state, screen
from .sources import CitationSource, Source
from .store import Store, safe_name
from .system_two import FactsExtractor
from .verify import verify

log = logging.getLogger("convtts_slr")


@dataclass
class Context:
    protocol: Protocol
    store: Store
    decider: Decider
    sources: list[Source] = field(default_factory=list)
    citation_sources: list[CitationSource] = field(default_factory=list)
    seeds: list[Paper] = field(default_factory=list)
    facts: FactsExtractor | None = None
    checker: Decider | None = None
    workers: int = 4
    fetch: Callable[[Paper, Path], Path | None] = download_pdf
    to_text: Callable[[Path], str] = pdf_to_text


def _parallel(ctx: Context, node: str, items: Iterable[Paper], fn: Callable[[Paper], None]) -> int:
    items = list(items)

    def safe(p: Paper) -> None:
        try:
            fn(p)
        except Exception as exc:  # one bad paper must not stop the review
            log.exception("%s failed on %s", node, p.id)
            ctx.store.append("error", p.id, node=node, error=f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=max(1, ctx.workers)) as pool:
        list(pool.map(safe, items))
    return len(items)


def _integrate(ctx: Context, found: list[Paper], round_: int = 0) -> tuple[int, int]:
    found = [p.model_copy(update={"found_in_round": round_}) if round_ else p for p in found]
    new, updated = dedup.integrate(ctx.store.papers(), found)
    for p in new:
        ctx.store.append("paper_added", p.id, paper=p.model_dump(mode="json"))
    for p in updated:
        ctx.store.append("paper_updated", p.id, paper=p.model_dump(mode="json"))
    return len(new), len(updated)


def final_includes(store: Store) -> set[str]:
    return {pid for pid, v in store.verdicts(Stage.FULL_TEXT.value).items() if v == "include"}


# --------------------------------------------------------------------------- #


class Identify:
    id = "identify"

    def run(self, ctx: Context) -> str:
        if ctx.seeds:
            _integrate(ctx, ctx.seeds)
        done = {ev.payload["key"] for ev in ctx.store.events("search_done")}
        for src in ctx.sources:
            key = hashlib.sha256(
                f"{src.name}|{ctx.protocol.search.model_dump_json()}".encode()
            ).hexdigest()[:16]
            if key in done:
                continue
            hits = src.search(ctx.protocol.search)
            new, merged = _integrate(ctx, hits)
            ctx.store.append(
                "search_done",
                None,
                key=key,
                source=src.name,
                identified=len(hits),
                new=new,
                merged=merged,
                query=ctx.protocol.search.model_dump(mode="json"),
            )
            log.info("%s: %d hits, %d new", src.name, len(hits), new)
        return "ok"


class ScreenTitleAbstract:
    id = "screen_ta"

    def run(self, ctx: Context) -> str:
        stage = Stage.TITLE_ABSTRACT
        done = ctx.store.screening(stage.value)
        todo = [
            p for p in ctx.store.papers().values() if p.role == Role.CANDIDATE and p.id not in done
        ]
        cfg = ctx.protocol.search

        def one(p: Paper) -> None:
            if p.year and not (cfg.from_year <= p.year <= int(cfg.to_date[:4])) and not p.is_seed:
                res = ScreeningResult(  # IC5 is a metadata rule: no model call
                    paper_id=p.id,
                    stage=stage.value,
                    verdict="exclude",
                    reasons=["IC5"],
                    outcomes=[
                        CriterionOutcome(
                            criterion_id="IC5",
                            polarity="include",
                            status="fail",
                            question_scores={},
                        )
                    ],
                    batch=DecisionBatch(
                        backend="deterministic", model="IC5", answers={}, state_sha=""
                    ),
                )
            else:
                res = screen(p, stage, ctx.protocol, ctx.decider)
            ctx.store.append("screened", p.id, result=res.model_dump(mode="json"))

        _parallel(ctx, self.id, todo, one)
        return "ok"


class FetchFullText:
    id = "fetch_fulltext"

    def run(self, ctx: Context) -> str:
        ta = ctx.store.verdicts(Stage.TITLE_ABSTRACT.value)
        status = ctx.store.fulltext_status()
        manual = ctx.store.dir / "pdfs"

        def wanted(p: Paper) -> bool:
            if status.get(p.id) == "ok":
                return False
            if (
                status.get(p.id) == "unavailable"
                and not (manual / f"{safe_name(p.id)}.pdf").exists()
            ):
                return False  # retried only once a PDF has been dropped in manually
            return p.role == Role.REFERENCE_BASELINE or ta.get(p.id) == "include"

        def one(p: Paper) -> None:
            pdf = ctx.fetch(p, ctx.store.dir)
            if pdf is None:
                ctx.store.append("fulltext", p.id, status="unavailable")
                return
            text = ctx.to_text(pdf)
            ctx.store.text_path(p.id).write_text(text, encoding="utf-8")
            ctx.store.append("fulltext", p.id, status="ok", chars=len(text), pdf=str(pdf))

        _parallel(ctx, self.id, [p for p in ctx.store.papers().values() if wanted(p)], one)
        return "ok"


class ScreenFullText:
    id = "screen_ft"

    def run(self, ctx: Context) -> str:
        stage = Stage.FULL_TEXT
        ta = ctx.store.verdicts(Stage.TITLE_ABSTRACT.value)
        done = ctx.store.screening(stage.value)
        ok = {pid for pid, s in ctx.store.fulltext_status().items() if s == "ok"}
        todo = [
            p
            for p in ctx.store.papers().values()
            if p.role == Role.CANDIDATE
            and ta.get(p.id) == "include"
            and p.id in ok
            and p.id not in done
        ]

        def one(p: Paper) -> None:
            res = screen(p, stage, ctx.protocol, ctx.decider, ctx.store.read_text(p.id))
            ctx.store.append("screened", p.id, result=res.model_dump(mode="json"))

        _parallel(ctx, self.id, todo, one)
        return "ok"


class Extract:
    id = "extract"

    def run(self, ctx: Context) -> str:
        inc = final_includes(ctx.store)
        done = ctx.store.extractions()
        ok = {pid for pid, s in ctx.store.fulltext_status().items() if s == "ok"}
        todo = [
            p
            for p in ctx.store.papers().values()
            if p.id not in done
            and p.id in ok
            and (p.id in inc or p.role == Role.REFERENCE_BASELINE)
        ]

        def one(p: Paper) -> None:
            text = ctx.store.read_text(p.id) or ""
            state = build_state(p, Stage.FULL_TEXT, ctx.protocol, text)
            typed = ctx.decider.evaluate(state, list(ctx.protocol.extraction), p.id)
            tsha = state_sha(state)
            facts = None
            if ctx.facts:
                cached = ctx.store.prior_facts(p.id, tsha, ctx.facts.name)
                facts = (
                    DatasetFacts.model_validate(cached)
                    if cached
                    else ctx.facts.extract(p, state["fulltext"])
                )
            res = ExtractionResult(
                paper_id=p.id,
                role=p.role,
                typed=typed,
                facts=facts,
                facts_model=ctx.facts.name if ctx.facts else None,
            )
            ctx.store.append("extracted", p.id, result=res.model_dump(mode="json"), text_sha=tsha)

        _parallel(ctx, self.id, todo, one)
        return "ok"


class Verify:
    id = "verify"

    def run(self, ctx: Context) -> str:
        exs = ctx.store.extractions()
        done = ctx.store.verifications()
        papers = ctx.store.papers()
        ft = ctx.store.screening(Stage.FULL_TEXT.value)

        def one(pid: str) -> None:
            text = ctx.store.read_text(pid) or ""
            state = build_state(papers[pid], Stage.FULL_TEXT, ctx.protocol, text)
            answers = ft[pid].batch.answers if pid in ft else {}
            res = verify(exs[pid], state, text, ctx.protocol, ctx.checker, answers)
            ctx.store.append("verified", pid, result=res.model_dump(mode="json"))

        todo = [pid for pid in exs if pid not in done and pid in papers]
        with ThreadPoolExecutor(max_workers=max(1, ctx.workers)) as pool:
            list(pool.map(one, todo))
        return "ok"


class Snowball:
    """Backward + forward snowballing from confirmed includes that have not been
    snowballed yet. The cycle ends when a round produces no new includes to expand
    (saturation) or after `snowball_max_rounds`."""

    id = "snowball"

    def run(self, ctx: Context) -> str:
        rounds = ctx.store.snowball_rounds_done()
        if rounds >= ctx.protocol.snowball_max_rounds or not ctx.citation_sources:
            return "stop"
        expanded = {ev.paper_id for ev in ctx.store.events("snowballed")}
        papers = ctx.store.papers()
        frontier = [
            papers[pid]
            for pid in final_includes(ctx.store)
            if pid not in expanded and pid in papers
        ]
        if not frontier:
            return "stop"
        found: list[Paper] = []
        for p in frontier:
            for src in ctx.citation_sources:
                refs, cits = src.references(p), src.citations(p)
                found += [
                    q.model_copy(update={"sources": [f"snowball:{src.name}"]}) for q in refs + cits
                ]
            ctx.store.append("snowballed", p.id)
        new, _ = _integrate(ctx, found, round_=rounds + 1)
        ctx.store.append(
            "snowball_round",
            None,
            round=rounds + 1,
            frontier=[p.id for p in frontier],
            identified=len(found),
            new=new,
        )
        return "new_records" if new else "stop"


class Synthesize:
    id = "synthesize"

    def run(self, ctx: Context) -> str:
        from .synthesis import write_report

        write_report(ctx.store, ctx.protocol)
        return "ok"


def build_graph() -> Graph:
    g = Graph()
    for node in (
        Identify(),
        ScreenTitleAbstract(),
        FetchFullText(),
        ScreenFullText(),
        Extract(),
        Verify(),
        Snowball(),
        Synthesize(),
    ):
        g.add(node)
    (
        g.edge("identify", "screen_ta")
        .edge("screen_ta", "fetch_fulltext")
        .edge("fetch_fulltext", "screen_ft")
        .edge("screen_ft", "extract")
        .edge("extract", "verify")
        .edge("verify", "snowball")
        .edge("snowball", "screen_ta", on="new_records")
        .edge("snowball", "synthesize", on="stop")
        .edge("synthesize", "END")
    )
    return g
