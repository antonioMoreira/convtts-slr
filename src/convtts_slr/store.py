"""Event-sourced runtime state.

Every decision is an immutable event in `events.jsonl`; current state is a fold over
the log. This gives resumability (re-running skips work already logged for the same
protocol version), provenance (backend, model, state hash per decision), and the
PRISMA flow for free. Human decisions are events too, and they win over model ones.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Iterator

from .models import Event, ExtractionResult, Paper, ScreeningResult, VerificationResult


def safe_name(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)


class Store:
    def __init__(self, workdir: str | Path, protocol_version: str):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "fulltext").mkdir(exist_ok=True)
        self.log = self.dir / "events.jsonl"
        self.version = protocol_version
        self._cache: list[Event] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ log --
    def append(self, kind: str, paper_id: str | None = None, **payload) -> Event:
        ev = Event(kind=kind, paper_id=paper_id, protocol_version=self.version, payload=payload)
        with self._lock:
            with self.log.open("a", encoding="utf-8") as f:
                f.write(ev.model_dump_json() + "\n")
            if self._cache is not None:
                self._cache.append(ev)
        return ev

    def events(
        self, kind: str | None = None, current_version_only: bool = False
    ) -> Iterator[Event]:
        if self._cache is None:
            self._cache = []
            if self.log.exists():
                with self.log.open(encoding="utf-8") as f:
                    self._cache = [Event.model_validate_json(line) for line in f if line.strip()]
        for ev in list(self._cache):
            if kind and ev.kind != kind:
                continue
            if current_version_only and ev.protocol_version != self.version:
                continue
            yield ev

    # --------------------------------------------------------------- papers --
    def papers(self) -> dict[str, Paper]:
        out: dict[str, Paper] = {}
        for ev in self.events():
            if ev.kind in ("paper_added", "paper_updated"):
                p = Paper.model_validate(ev.payload["paper"])
                out[p.id] = p
            elif ev.kind == "paper_merged":
                out.pop(ev.payload["duplicate_id"], None)
        return out

    # ------------------------------------------------------------ screening --
    def screening(self, stage: str) -> dict[str, ScreeningResult]:
        out = {}
        for ev in self.events("screened", current_version_only=True):
            r = ScreeningResult.model_validate(ev.payload["result"])
            if r.stage == stage:
                out[r.paper_id] = r
        return out

    def human_decisions(self, stage: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for ev in self.events("human_decision"):  # human labels survive protocol edits
            if ev.paper_id is not None and ev.payload["stage"] == stage:
                out[ev.paper_id] = ev.payload["verdict"]
        return out

    def verdicts(self, stage: str) -> dict[str, str]:
        """Final verdict per paper at a stage: human decision if any, else model."""
        v: dict[str, str] = {pid: str(r.verdict) for pid, r in self.screening(stage).items()}
        v.update(self.human_decisions(stage))
        return v

    # ------------------------------------------------------------ full text --
    def text_path(self, paper_id: str) -> Path:
        return self.dir / "fulltext" / f"{safe_name(paper_id)}.txt"

    def fulltext_status(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for ev in self.events("fulltext"):
            if ev.paper_id is not None:
                out[ev.paper_id] = ev.payload["status"]
        return out

    def read_text(self, paper_id: str) -> str | None:
        p = self.text_path(paper_id)
        return p.read_text(encoding="utf-8") if p.exists() else None

    # ----------------------------------------------------------- extraction --
    def extractions(self) -> dict[str, ExtractionResult]:
        return {
            ev.paper_id: ExtractionResult.model_validate(ev.payload["result"])
            for ev in self.events("extracted", current_version_only=True)
            if ev.paper_id is not None
        }

    def prior_facts(self, paper_id: str, text_sha: str, model: str) -> dict | None:
        """Reuse System-Two output across protocol versions when text and model are unchanged."""
        for ev in reversed(list(self.events("extracted"))):
            r = ev.payload["result"]
            if (
                ev.paper_id == paper_id
                and ev.payload.get("text_sha") == text_sha
                and r.get("facts_model") == model
            ):
                return r.get("facts")
        return None

    def verifications(self) -> dict[str, VerificationResult]:
        return {
            ev.paper_id: VerificationResult.model_validate(ev.payload["result"])
            for ev in self.events("verified", current_version_only=True)
            if ev.paper_id is not None
        }

    def snowball_rounds_done(self) -> int:
        return sum(1 for _ in self.events("snowball_round"))  # rounds are global, not per version

    def dump_json(self, name: str, obj) -> Path:
        path = self.dir / name
        path.write_text(
            json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return path
