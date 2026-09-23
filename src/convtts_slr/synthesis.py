"""Deterministic synthesis. No model is involved in computing the answer to the RQ:
consensus is a function of the extracted data, so it is reproducible from the log.

Consensus for a dimension = share of included datasets that follow its most common
practice, among datasets that report the dimension ('unclear' is counted separately,
since unreported practice is itself a finding about the literature).
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict

from .models import DatasetFacts, ExtractionResult, Role
from .protocol import ChoiceSpec, NoulSpec, Protocol, Stage
from .stages import final_includes
from .store import Store

NOUL_CUT = 0.5  # extraction Nouls are binarised here; replace with calibrated cuts if you fit them


def _level(share: float, protocol: Protocol) -> str:
    lv = protocol.consensus_levels
    return "strong" if share >= lv["strong"] else "partial" if share >= lv["partial"] else "none"


def _value(ex: ExtractionResult, q) -> str | None:
    a = ex.typed.answers.get(q.id)
    if a is None:
        return None
    if isinstance(q, ChoiceSpec):
        return a.choice
    return "yes" if (a.score or 0) >= NOUL_CUT else "no"


def _noul_level(p: float, protocol: Protocol) -> str:
    strong = protocol.consensus_levels["strong"]
    if p >= strong:
        return "consensus: used"
    if p >= protocol.consensus_levels["partial"]:
        return "majority"
    if p > 1 - strong:
        return "minority"
    return "rare or unreported"  # absence of reporting is not evidence of absence of practice


def dimension_table(exs: list[ExtractionResult], protocol: Protocol) -> list[dict]:
    """Choice fields: modal option and its share among datasets that report the field.
    Noul fields (practices that can co-occur): prevalence = share of datasets using it."""
    rows = []
    for q in protocol.extraction:
        vals = [v for ex in exs if (v := _value(ex, q)) is not None]
        if isinstance(q, NoulSpec):
            p = sum(v == "yes" for v in vals) / len(vals) if vals else 0.0
            rows.append(
                {
                    "field": q.id,
                    "type": "practice",
                    "n": len(vals),
                    "unclear": "-",
                    "modal": "used",
                    "share": round(p, 3),
                    "level": _noul_level(p, protocol) if vals else "no data",
                }
            )
            continue
        reported = [v for v in vals if v != "unclear"]
        counts = Counter(reported)
        if not reported:
            rows.append(
                {
                    "field": q.id,
                    "type": "choice",
                    "n": len(vals),
                    "unclear": len(vals),
                    "modal": "-",
                    "share": 0.0,
                    "level": "not reported",
                    "counts": {},
                }
            )
            continue
        modal, k = counts.most_common(1)[0]
        share = k / len(reported)
        rows.append(
            {
                "field": q.id,
                "type": "choice",
                "n": len(vals),
                "unclear": len(vals) - len(reported),
                "modal": modal,
                "share": round(share, 3),
                "level": _level(share, protocol),
                "counts": dict(counts),
            }
        )
    return rows


def prevalence_by_period(
    exs: list[ExtractionResult], years: dict[str, int | None], protocol: Protocol
) -> dict:
    def period(y):
        return "unknown" if y is None else "<=2022" if y <= 2022 else str(y)

    def order(p):
        return (p != "<=2022", p == "unknown", p)

    out: dict[str, dict[str, float]] = defaultdict(dict)
    by_p: dict[str, list[ExtractionResult]] = defaultdict(list)
    for ex in exs:
        by_p[period(years.get(ex.paper_id))].append(ex)
    for q in protocol.extraction:
        if not isinstance(q, NoulSpec):
            continue
        for per, group in sorted(by_p.items(), key=lambda kv: order(kv[0])):
            vals = [_value(ex, q) for ex in group]
            out[q.id][per] = round(sum(v == "yes" for v in vals) / len(vals), 2) if vals else 0.0
    return dict(out)


def prisma(store: Store) -> dict:
    papers = store.papers()
    searches = list(store.events("search_done"))
    rounds = list(store.events("snowball_round"))
    ta = store.verdicts(Stage.TITLE_ABSTRACT.value)
    ft = store.verdicts(Stage.FULL_TEXT.value)
    fts = store.fulltext_status()
    cands = {pid for pid, p in papers.items() if p.role == Role.CANDIDATE}
    ft_reasons = Counter()
    for pid, r in store.screening(Stage.FULL_TEXT.value).items():
        if ft.get(pid) == "exclude":
            ft_reasons.update(r.reasons or ["human"])
    identified = sum(e.payload["identified"] for e in searches)
    return {
        "identified_by_source": {e.payload["source"]: e.payload["identified"] for e in searches},
        "identified_by_snowballing": sum(e.payload["identified"] for e in rounds),
        "seeds": sum(p.is_seed for p in papers.values()),
        "unique_records": len(papers),
        "duplicates_removed": identified
        + sum(e.payload["identified"] for e in rounds)
        + sum(p.is_seed for p in papers.values())
        - len(papers),
        "screened_title_abstract": sum(1 for pid in cands if pid in ta),
        "excluded_title_abstract": sum(1 for pid in cands if ta.get(pid) == "exclude"),
        "pending_human_title_abstract": sum(1 for pid in cands if ta.get(pid) == "needs_human"),
        "sought_full_text": sum(1 for pid in cands if ta.get(pid) == "include"),
        "not_retrieved": sum(
            1 for pid in cands if ta.get(pid) == "include" and fts.get(pid) == "unavailable"
        ),
        "assessed_full_text": sum(1 for pid in cands if pid in ft),
        "excluded_full_text": sum(1 for pid in cands if ft.get(pid) == "exclude"),
        "excluded_full_text_reasons": dict(ft_reasons),
        "pending_human_full_text": sum(1 for pid in cands if ft.get(pid) == "needs_human"),
        "included": len(final_includes(store) & cands),
        "reference_baselines": sum(p.role == Role.REFERENCE_BASELINE for p in papers.values()),
        "snowball_rounds": len(rounds),
    }


def _md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def write_report(store: Store, protocol: Protocol) -> None:
    papers = store.papers()
    exs = store.extractions()
    ver = store.verifications()
    inc = final_includes(store)
    corpus = [ex for pid, ex in exs.items() if pid in inc and ex.role == Role.CANDIDATE]
    ref = [ex for ex in exs.values() if ex.role == Role.REFERENCE_BASELINE]
    flow = prisma(store)
    dims = dimension_table(corpus, protocol)
    ref_dims = {r["field"]: r for r in dimension_table(ref, protocol)}
    for r in dims:  # conversational vs general-TTS construction practice
        rr = ref_dims.get(r["field"], {})
        r["reference_modal"] = rr.get("modal", "-")
        r["reference_share"] = rr.get("share", "-")
    trend = prevalence_by_period(corpus, {pid: p.year for pid, p in papers.items()}, protocol)
    unverified = [pid for pid in inc if pid in ver and not ver[pid].passed]

    store.dump_json("prisma.json", flow)
    store.dump_json("consensus.json", {"dimensions": dims, "trend": trend})
    _write_csv(store, protocol, list(exs.values()), papers, ver)

    md = [
        f"# {protocol.title}",
        "",
        f"Protocol version `{protocol.version}`.",
        "",
        "## PRISMA flow",
        "",
        "```json",
        __import__("json").dumps(flow, indent=2),
        "```",
        "",
        f"## Consensus by dimension (n = {len(corpus)} included datasets)",
        "",
        "For `choice` fields, `share` is the share of reporting datasets that follow the modal option "
        f"(strong >= {protocol.consensus_levels['strong']}, partial >= {protocol.consensus_levels['partial']}). "
        "For `practice` fields, `share` is the share of datasets that report using the practice. "
        "`reference_*` = the same field in the non-conversational methods papers.",
        "",
        _md_table(
            dims,
            [
                "field",
                "type",
                "n",
                "unclear",
                "modal",
                "share",
                "level",
                "reference_modal",
                "reference_share",
            ],
        ),
        "",
        "## Prevalence of binary practices by period",
        "",
        _md_table(
            [{"field": k, **v} for k, v in trend.items()],
            ["field"]
            + sorted(
                {p for v in trend.values() for p in v},
                key=lambda p: (p != "<=2022", p == "unknown", p),
            ),
        ),
        "",
        "## Verification",
        "",
        f"{len(unverified)} included papers have ungrounded quotes or checker disagreements "
        f"and are listed in the human queue: {', '.join(unverified) or 'none'}.",
    ]
    (store.dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def _write_csv(store: Store, protocol: Protocol, exs: list[ExtractionResult], papers, ver) -> None:
    facts_fields = list(DatasetFacts.model_fields)
    cols = (
        ["paper_id", "title", "year", "role", "verified"]
        + [q.id for q in protocol.extraction]
        + facts_fields
    )
    with (store.dir / "extraction.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ex in exs:
            p = papers.get(ex.paper_id)
            row = {
                "paper_id": ex.paper_id,
                "title": p.title if p else "",
                "year": p.year if p else "",
                "role": ex.role.value,
                "verified": ver[ex.paper_id].passed if ex.paper_id in ver else "",
            }
            row |= {q.id: _value(ex, q) for q in protocol.extraction}
            if ex.facts:
                row |= {k: getattr(ex.facts, k).value for k in facts_fields}
            w.writerow(row)
