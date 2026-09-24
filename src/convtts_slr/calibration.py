"""Fit per-criterion routing thresholds on a human-labelled sample.

Jev/LLM scores are not calibrated, so `low`/`high` are chosen so that automated
decisions meet a precision target on the sample, and so that include-criteria never
auto-reject more than `max_fn` of the true positives (recall matters most in an SLR).
Everything in between goes to the human queue; the report shows how big that queue is.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .protocol import Protocol, Stage, Thresholds
from .screening import _relevant
from .store import Store


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fit(
    pairs: list[tuple[float, int]], polarity: str, precision: float = 0.95, max_fn: float = 0.02
) -> Thresholds:
    grid = sorted({s for s, _ in pairs} | {0.0, 1.0})
    positives = sum(label for _, label in pairs) or 1

    def ok_low(t: float) -> bool:
        below = [label for s, label in pairs if s <= t]
        if not below:
            return True
        prec = sum(1 - label for label in below) / len(below)
        fn = sum(below) / positives
        return prec >= precision and (polarity == "exclude" or fn <= max_fn)

    def ok_high(t: float) -> bool:
        above = [label for s, label in pairs if s >= t]
        return not above or sum(above) / len(above) >= precision

    low = max((t for t in grid if ok_low(t)), default=0.0)
    high = min((t for t in grid if ok_high(t) and t > low), default=1.0)
    return Thresholds(low=round(low, 4), high=round(high, 4))


def calibrate(
    store: Store,
    protocol: Protocol,
    labels_csv: str | Path,
    stage: Stage,
    precision: float = 0.95,
    max_fn: float = 0.02,
) -> dict:
    res = store.screening(stage.value)
    with open(labels_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    report, thresholds = {}, {}
    for c in protocol.criteria_for(stage):
        pairs = []
        for row in rows:
            lab = (row.get(c.id) or "").strip()
            r = res.get(row["paper_id"])
            if lab not in ("0", "1") or r is None:
                continue
            o = next((o for o in r.outcomes if o.criterion_id == c.id), None)
            if o is None:
                continue
            vals = [o.question_scores[i] for i in _relevant(c, protocol)]
            pairs.append((max(vals) if c.rule == "any" else min(vals), int(lab)))
        if not pairs:
            continue
        t = fit(pairs, c.polarity, precision, max_fn)
        thresholds[c.id] = t.model_dump()
        auto = [(s, label) for s, label in pairs if s <= t.low or s >= t.high]
        errors = sum((s >= t.high) != bool(label) for s, label in auto)
        report[c.id] = {
            "n": len(pairs),
            "thresholds": t.model_dump(),
            "kappa_at_0.5": round(
                cohen_kappa([label for _, label in pairs], [int(s >= 0.5) for s, _ in pairs]), 3
            ),
            "automated_share": round(len(auto) / len(pairs), 3),
            "errors_on_automated": errors,
        }
    return {"stage": stage.value, "criteria": report, "thresholds": thresholds}


def apply_thresholds(protocol: Protocol, path: str | Path | None) -> Protocol:
    if not path or not Path(path).exists():
        return protocol
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    crit = [
        c.model_copy(update={"thresholds": Thresholds(**data[c.id])}) if c.id in data else c
        for c in protocol.criteria
    ]
    return protocol.model_copy(update={"criteria": crit})
