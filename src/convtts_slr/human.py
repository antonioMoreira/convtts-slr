"""Human-in-the-loop queue. Humans are nodes of the graph too: their decisions are
events that override model verdicts and survive protocol-version changes.

Export writes one CSV row per pending item; fill the `decision` column and import it.
  screening rows:    decision in {include, exclude}
  fulltext rows:     drop the PDF in <workdir>/pdfs/<file_stub>.pdf, or decision=exclude
  verification rows: decision in {accept, reject}; reject removes the paper from synthesis
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .models import Role
from .protocol import Protocol, Stage
from .stages import final_includes
from .store import Store, safe_name

FIELDS = ["queue", "paper_id", "file_stub", "title", "year", "detail", "decision", "note"]


def pending(store: Store) -> list[dict]:
    papers = store.papers()
    rows = []
    for stage in (Stage.TITLE_ABSTRACT, Stage.FULL_TEXT):
        human = store.human_decisions(stage.value)
        for pid, r in store.screening(stage.value).items():
            if r.verdict == "needs_human" and pid not in human and pid in papers:
                scores = "; ".join(
                    f"{o.criterion_id}={o.status}:{max(o.question_scores.values(), default=0):.2f}"
                    for o in r.outcomes
                )
                rows.append(
                    {
                        "queue": stage.value,
                        "paper_id": pid,
                        "title": papers[pid].title,
                        "year": papers[pid].year,
                        "detail": f"reasons={r.reasons} | {scores}",
                    }
                )
    ta = store.verdicts(Stage.TITLE_ABSTRACT.value)
    for pid, s in store.fulltext_status().items():
        p = papers.get(pid)
        if (
            s == "unavailable"
            and p
            and (ta.get(pid) == "include" or p.role == Role.REFERENCE_BASELINE)
        ):
            rows.append(
                {
                    "queue": "fulltext",
                    "paper_id": pid,
                    "title": p.title,
                    "year": p.year,
                    "detail": f"doi={p.doi} arxiv={p.arxiv_id}",
                }
            )
    accepted = {ev.paper_id for ev in store.events("human_verified")}
    inc = final_includes(store)
    for pid, v in store.verifications().items():
        p = papers.get(pid)
        if (
            p
            and not v.passed
            and pid not in accepted
            and (pid in inc or p.role == Role.REFERENCE_BASELINE)
        ):
            rows.append(
                {
                    "queue": "verification",
                    "paper_id": pid,
                    "title": papers[pid].title,
                    "year": papers[pid].year,
                    "detail": f"missing_quotes={v.missing_quotes} disagreements={v.disagreements}",
                }
            )
    for r in rows:
        r["file_stub"] = safe_name(r["paper_id"])
    return rows


def export_queue(store: Store, path: str | Path) -> int:
    rows = pending(store)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def import_queue(store: Store, path: str | Path, reviewer: str = "human") -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = (row.get("decision") or "").strip().lower()
            if not d:
                continue
            q, pid = row["queue"], row["paper_id"]
            if q in (Stage.TITLE_ABSTRACT.value, Stage.FULL_TEXT.value) and d in (
                "include",
                "exclude",
            ):
                store.append(
                    "human_decision",
                    pid,
                    stage=q,
                    verdict=d,
                    reviewer=reviewer,
                    note=row.get("note"),
                )
            elif q == "fulltext" and d == "exclude":
                store.append(
                    "human_decision",
                    pid,
                    stage=Stage.FULL_TEXT.value,
                    verdict="exclude",
                    reviewer=reviewer,
                    note=row.get("note") or "full text not retrievable",
                )
            elif q == "verification" and d in ("accept", "reject"):
                store.append(
                    "human_verified",
                    pid,
                    accepted=d == "accept",
                    reviewer=reviewer,
                    note=row.get("note"),
                )
                if d == "reject":
                    store.append(
                        "human_decision",
                        pid,
                        stage=Stage.FULL_TEXT.value,
                        verdict="exclude",
                        reviewer=reviewer,
                        note="rejected at verification",
                    )
            else:
                raise ValueError(f"bad decision {d!r} for queue {q!r} (paper {pid})")
            n += 1
    return n


def export_calibration_sample(
    store: Store, protocol: Protocol, path: str | Path, stage: Stage, n: int = 80, seed: int = 13
) -> int:
    """Random sample (stratified over model verdicts) for humans to label per criterion.
    Label columns: 1 = the criterion's proposition holds, 0 = it does not."""
    papers = store.papers()
    res = store.screening(stage.value)
    by_v: dict[str, list[str]] = {}
    for pid, r in res.items():
        by_v.setdefault(r.verdict, []).append(pid)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by_v)))
    ids = [pid for v in sorted(by_v) for pid in rng.sample(by_v[v], min(per, len(by_v[v])))]
    crit = [c.id for c in protocol.criteria_for(stage)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["paper_id", "title", "abstract"] + crit)
        w.writeheader()
        for pid in ids:
            w.writerow(
                {
                    "paper_id": pid,
                    "title": papers[pid].title,
                    "abstract": papers[pid].abstract[:1500]
                    if stage == Stage.TITLE_ABSTRACT
                    else "",
                }
            )
    return len(ids)
