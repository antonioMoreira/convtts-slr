"""Offline tests. The end-to-end test runs the whole graph with a keyword-scripted
System One, fake PDFs and a fake citation source, so no network or API key is needed."""

import csv
import json
import re
from pathlib import Path

import pytest

from convtts_slr import dedup
from convtts_slr.backends import JevBackend, ScriptedBackend
from convtts_slr.calibration import apply_thresholds, cohen_kappa, fit
from convtts_slr.fulltext import select_sections
from convtts_slr.human import export_queue, import_queue, pending
from convtts_slr.models import DatasetFacts, Paper, Quoted, Role
from convtts_slr.protocol import DEFAULT_PROTOCOL, ChoiceSpec
from convtts_slr.screening import Decider, route, verdict_from
from convtts_slr.sources import (
    ArxivSource,
    LocalSource,
    OpenAlexSource,
    SemanticScholarSource,
    openalex_abstract,
)
from convtts_slr.stages import Context, build_graph, final_includes
from convtts_slr.store import Store
from convtts_slr.system_two import ScriptedFactsExtractor
from convtts_slr.verify import missing_quotes, quote_found

SEEDS = Path(__file__).resolve().parents[1] / "data" / "seeds.json"


# ------------------------------------------------------------------ units --


def test_dedup_merges_preprint_and_venue_version():
    pre = Paper(
        id="a",
        title="DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech",
        doi="10.48550/arXiv.2207.01063",
        year=2022,
        sources=["arxiv"],
    )
    venue = Paper(
        id="b",
        title="DailyTalk: spoken dialogue dataset for conversational text-to-speech",
        doi="https://doi.org/10.1109/ICASSP49357.2023.10095751",
        year=2023,
        venue="ICASSP",
        sources=["openalex"],
    )
    new, _ = dedup.integrate({}, [pre, venue])
    assert len(new) == 1
    p = new[0]
    assert p.id == "arxiv:2207.01063" and p.arxiv_id == "2207.01063"
    assert p.doi == "10.1109/icassp49357.2023.10095751" and set(p.sources) == {"arxiv", "openalex"}


def test_routing_and_seed_alarm():
    ic4 = next(c for c in DEFAULT_PROTOCOL.criteria if c.id == "IC4")
    scores = {
        "ic4_pitch_predictor": 0.05,
        "ic4_energy_predictor": 0.05,
        "ic4_duration_predictor": 0.95,
    }
    assert route(ic4, scores, DEFAULT_PROTOCOL).status == "pass"  # duration alone counts by default
    strict = DEFAULT_PROTOCOL.model_copy(update={"duration_alone_satisfies_ic4": False})
    assert route(ic4, scores, strict).status == "fail"
    ec = next(c for c in DEFAULT_PROTOCOL.criteria if c.id == "EC1")
    outs = [route(ic4, scores, strict), route(ec, {"ec1_only_uses_dataset": 0.5}, DEFAULT_PROTOCOL)]
    assert verdict_from(outs) == ("exclude", ["IC4"])
    assert verdict_from(outs[1:]) == ("needs_human", ["EC1"])


def test_protocol_version_changes_with_text():
    edited = DEFAULT_PROTOCOL.model_copy(update={"fulltext_char_budget": 1})
    assert edited.version != DEFAULT_PROTOCOL.version


def test_calibration_fit_and_kappa():
    pairs = [(0.02, 0), (0.05, 0), (0.1, 0), (0.3, 1), (0.6, 0), (0.9, 1), (0.95, 1), (0.99, 1)]
    t = fit(pairs, "include", precision=0.95, max_fn=0.0)
    assert t.low == 0.1 and t.high == 0.9
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0


def test_quote_grounding():
    text = "We recorded 20 hours of dia-\nlogue between 10 speakers.  Each dialogue has 2 speakers."
    assert quote_found("we recorded 20 hours of dialogue between 10 speakers", text)
    assert not quote_found("We crawled 3,000 hours of podcasts from YouTube", text)
    facts = DatasetFacts(
        total_hours=Quoted(value=20.0, quote="We recorded 20 hours of dialogue"),
        num_speakers=Quoted(value=10),
        languages=Quoted(value="en", quote="All speech is English."),
    )
    assert missing_quotes(facts, text) == ["languages:quote_not_in_text", "num_speakers:no_quote"]


def test_source_parsers():
    assert openalex_abstract({"world": [1], "hello": [0]}) == "hello world"
    cfg = DEFAULT_PROTOCOL.search
    assert '"text-to-speech"' in OpenAlexSource.render(cfg) and " AND " in OpenAlexSource.render(
        cfg
    )
    assert " + " in SemanticScholarSource.render(cfg)
    assert "submittedDate:[201901010000 TO 202609302359]" in ArxivSource.render(cfg)
    atom = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2207.01063v2</id>
      <title>DailyTalk:  Spoken Dialogue Dataset</title><summary>A dataset.</summary>
      <published>2022-07-03T00:00:00Z</published></entry></feed>"""
    (p,) = ArxivSource.parse(atom)
    assert (
        p.arxiv_id == "2207.01063v2"
        and p.year == 2022
        and p.title == "DailyTalk: Spoken Dialogue Dataset"
    )
    assert dedup.normalize(p).arxiv_id == "2207.01063"


def test_select_sections_keeps_data_and_model_sections():
    filler = "Lorem ipsum dolor sit amet. " * 300
    text = (
        "Title\nAbstract text.\n1 Introduction\n"
        + filler
        + "\n2 Related Work\n"
        + filler
        + "\n3 Dataset Construction\nWe recorded dialogues.\n"
        + "x " * 200
        + "\n4 Baseline Model\nFastSpeech 2 with pitch predictor.\n\nReferences\n[1] ..."
    )
    out = select_sections(text, 5000)
    assert "Dataset Construction" in out and "pitch predictor" in out and "[1] ..." not in out


def test_jev_backend_translates_specs(monkeypatch):
    pytest.importorskip("typesafe_sdk")
    from typesafe_sdk import Choice, Noul, SystemOneResponse

    seen = {}

    def fake_system_one(self, state, questions, **kw):
        seen["q"] = questions
        return SystemOneResponse.model_validate(
            {
                "model": "jev-latest",
                "usage": {},
                "answers": {
                    "ic1_new_dataset": {"type": "noul", "noul": 0.97},
                    "rq1_sample_unit": {
                        "type": "choice",
                        "choice": "turn_with_context",
                        "confidence": 0.8,
                        "probabilities": {
                            "utterance_only": 0.05,
                            "turn_with_context": 0.8,
                            "full_dialogue": 0.1,
                            "session_multichannel": 0.03,
                            "unclear": 0.02,
                        },
                    },
                },
            }
        )

    from typesafe_sdk import TypeSafeClient

    monkeypatch.setattr(TypeSafeClient, "system_one", fake_system_one)
    b = JevBackend(api_key="test-key")
    q1 = DEFAULT_PROTOCOL.criteria[0].questions[0]
    q2 = next(q for q in DEFAULT_PROTOCOL.extraction if q.id == "rq1_sample_unit")
    batch = b.evaluate({"title": "t", "abstract": "a"}, [q1, q2])
    assert isinstance(seen["q"]["ic1_new_dataset"], Noul) and isinstance(
        seen["q"]["rq1_sample_unit"], Choice
    )
    assert seen["q"]["ic1_new_dataset"].criteria == {"true": q1.true, "false": q1.false}
    assert batch.answers["ic1_new_dataset"].score == 0.97
    assert batch.answers["rq1_sample_unit"].choice == "turn_with_context"


# ------------------------------------------------------------ end to end --

CORPUS = [
    {
        "title": "DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech",
        "arxiv_id": "2207.01063",
        "year": 2022,
        "abstract": "We introduce DailyTalk, a high-quality conversational speech dataset for "
        "conversational text-to-speech, recorded by two speakers from dialogues.",
    },
    {
        "title": "EmoDialog: an emotional two-speaker dialogue corpus for conversational TTS",
        "year": 2024,
        "doi": "10.1/emodialog",
        "abstract": "We release a new dialogue speech corpus with emotion labels for "
        "conversational text-to-speech.",
    },
    {
        "title": "A better context encoder for conversational TTS",
        "year": 2023,
        "doi": "10.1/model",
        "abstract": "We propose a model for conversational text-to-speech evaluated on existing "
        "datasets.",
    },
    {
        "title": "CallASR: a telephone dialogue corpus for speech recognition",
        "year": 2023,
        "doi": "10.1/asr",
        "abstract": "We release a dialogue speech corpus for automatic speech recognition only.",
    },
    {
        "title": "ReadBook: a read-speech corpus for text-to-speech",
        "year": 2024,
        "doi": "10.1/read",
        "abstract": "We release a new read speech corpus of a single speaker for text-to-speech.",
    },
    {
        "title": "OldDialog: a 2015 dialogue speech corpus for TTS",
        "year": 2015,
        "doi": "10.1/old",
        "abstract": "We release a dialogue speech corpus for conversational text-to-speech.",
    },
    {
        "title": "PodGen: a new dialogue speech collection",
        "year": 2025,
        "doi": "10.1/podgen",
        "abstract": "We release a dialogue corpus maybe for speech things.",
    },
]

FULLTEXT = {
    "arxiv:2207.01063": "3 Dataset\nEach sample is a turn with its dialogue context. Speech was "
    "recorded from scripts read by two speakers. 4 Model\nFastSpeech 2 with a pitch predictor "
    "and an energy predictor. We recorded 20 hours of dialogue.",
    "doi:10.1/emodialog": "2 Corpus\nEach sample is a turn with its dialogue context. Emotion "
    "labels. Speech was recorded spontaneously. 3 Baseline\nFastSpeech 2 with a pitch "
    "predictor. We recorded 12 hours of dialogue.",
    "doi:10.1/snowkid": "2 Data\nEach sample is a full dialogue. Harvested from podcasts. "
    "3 Model\nA duration predictor only. The corpus has 300 hours of dialogue.",
}
# The 4 reference-baseline methods papers
for sid in ("arxiv:2104.04896", "arxiv:2402.16380", "doi:10.3390/app15041848", "arxiv:2409.03283"):
    FULLTEXT[sid] = (
        "1 Pipeline\nRead speech from a single speaker, forced alignment and quality filtering."
    )


def keyword_script(state: dict, q) -> float | str:
    """A crude but deterministic System One: keyword rules standing in for Jev."""
    t = (
        state.get("abstract", "") + " " + state.get("fulltext", "") + " " + state.get("title", "")
    ).lower()

    def has(*ws: str) -> bool:
        return any(w in t for w in ws)

    if isinstance(q, ChoiceSpec):
        if q.id == "rq1_sample_unit":
            return (
                "turn_with_context"
                if "turn with" in t
                else "full_dialogue"
                if "full dialogue" in t
                else "unclear"
            )
        return "unclear"
    rules = {
        "ic1_new_dataset": 0.95 if has("introduce", "release", "corpus has") else 0.05,
        "ic2_conversational": 0.95
        if has("dialogue", "conversation") and not has("single speaker")
        else 0.05,
        "ic3_turn_level_ctts": 0.95 if has("conversational text-to-speech") else 0.05,
        "ic3_whole_dialogue_generation": 0.95
        if has("podcast dialogues")
        else 0.5
        if has("speech things")
        else 0.05,
        "ec1_only_uses_dataset": 0.95 if has("existing datasets") else 0.05,
        "ec3_recognition_only": 0.95 if has("recognition only") else 0.05,
        "ic4_pitch_predictor": 0.95 if has("pitch predictor") else 0.05,
        "ic4_energy_predictor": 0.95 if has("energy predictor") else 0.05,
        "ic4_duration_predictor": 0.95 if has("duration predictor") else 0.05,
        "ic6_construction_detail": 0.95 if has("each sample") else 0.05,
        "rq2_scripted_recorded": 0.95 if has("scripts read") else 0.05,
        "rq2_spontaneous_recorded": 0.95 if has("spontaneous") else 0.05,
        "rq2_in_the_wild": 0.95 if has("podcasts") else 0.05,
        "rq3_emotion_labels": 0.95 if has("emotion labels") else 0.05,
        "rq4_forced_alignment": 0.95 if has("forced alignment") else 0.05,
    }
    return rules.get(q.id, 0.05)


def facts_fn(paper: Paper, text: str) -> DatasetFacts:
    m = re.search(r"[^.]*\b(\d+) hours[^.]*\.", text)
    hours = Quoted(value=float(m.group(1)), quote=m.group(0).strip()) if m else Quoted()
    return DatasetFacts(total_hours=hours)


class FakeCitations:
    name = "fake"

    def references(self, p):
        if p.id == "doi:10.1/emodialog":
            return [
                Paper(
                    id="x",
                    title="SnowKid: a podcast dialogue corpus for conversational text-to-speech",
                    doi="10.1/snowkid",
                    year=2025,
                    abstract="We release a dialogue corpus for conversational text-to-speech.",
                )
            ]
        return [
            Paper(
                id="y",
                title="DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech",
                arxiv_id="2207.01063",
                year=2022,
            )
        ]  # a duplicate: must not create a new record

    def citations(self, p):
        return []


def make_ctx(tmp_path, protocol=DEFAULT_PROTOCOL, script=keyword_script):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(CORPUS))
    store = Store(tmp_path / "work", protocol.version)
    backend = ScriptedBackend(script)

    def fetch(p, workdir):
        if p.id not in FULLTEXT:
            return None
        f = workdir / f"{p.id.replace(':', '_').replace('/', '_')}.pdf"
        f.write_text(FULLTEXT[p.id])
        return f

    ctx = Context(
        protocol=protocol,
        store=store,
        decider=Decider(backend, store),
        sources=[LocalSource(corpus, name="fixture")],
        citation_sources=[FakeCitations()],
        seeds=LocalSource(SEEDS, name="seeds").search(),
        facts=ScriptedFactsExtractor(facts_fn),
        checker=Decider(ScriptedBackend(script), store),
        workers=3,
        fetch=fetch,
        to_text=lambda f: f.read_text(),
    )
    return ctx, backend


def test_end_to_end(tmp_path):
    ctx, backend = make_ctx(tmp_path)
    trace = build_graph().run("identify", ctx)
    nodes = [n for n, _ in trace]
    assert nodes.count("screen_ta") == 2  # one snowball cycle, then saturation
    assert trace[-2] == ("snowball", "stop") and nodes[-1] == "synthesize"

    store = ctx.store
    inc = final_includes(store)
    assert inc == {"arxiv:2207.01063", "doi:10.1/emodialog", "doi:10.1/snowkid"}

    ta = store.verdicts("title_abstract")
    assert ta["doi:10.1/model"] == "exclude"  # EC1
    assert ta["doi:10.1/asr"] == "exclude"  # EC3
    assert ta["doi:10.1/read"] == "exclude"  # IC2
    assert ta["doi:10.1/old"] == "exclude"  # IC5 (no model call)
    assert ta["doi:10.1/podgen"] == "needs_human"  # uncertain IC3
    behavior = next(pid for pid, p in store.papers().items() if "Behavior-SD" in p.title)
    assert (
        ta[behavior] == "needs_human"
    )  # seed with no abstract: recall alarm, not a silent exclusion
    assert "SEED_EXCLUDED_BY_MODEL" in store.screening("title_abstract")[behavior].reasons

    exs = store.extractions()
    refs = [e for e in exs.values() if e.role == Role.REFERENCE_BASELINE]
    assert len(refs) == 4  # methods papers extracted but never screened

    flow = json.loads((store.dir / "prisma.json").read_text())
    assert (
        flow["included"] == 3 and flow["reference_baselines"] == 4 and flow["snowball_rounds"] == 2
    )
    consensus = json.loads((store.dir / "consensus.json").read_text())
    unit = next(d for d in consensus["dimensions"] if d["field"] == "rq1_sample_unit")
    assert unit["modal"] == "turn_with_context" and unit["share"] == pytest.approx(2 / 3, abs=1e-3)
    assert (store.dir / "report.md").exists() and (store.dir / "extraction.csv").exists()
    assert all(v.passed for v in store.verifications().values())

    # Resume: a second run makes no new backend calls.
    calls = backend.calls
    build_graph().run("identify", ctx)
    assert backend.calls == calls

    # Recalibrated thresholds re-route from cached answers: still no new calls.
    th = tmp_path / "th.json"
    th.write_text(json.dumps({"IC3": {"low": 0.1, "high": 0.4}}))
    proto2 = apply_thresholds(DEFAULT_PROTOCOL, th)
    ctx2, backend2 = make_ctx(tmp_path, proto2)
    build_graph().run("identify", ctx2)
    assert backend2.calls == 0
    assert ctx2.store.verdicts("title_abstract")["doi:10.1/podgen"] == "include"


def test_human_queue_roundtrip(tmp_path):
    ctx, _ = make_ctx(tmp_path)
    build_graph().run("identify", ctx)
    q = tmp_path / "queue.csv"
    n = export_queue(ctx.store, q)
    assert n >= 2
    rows = list(csv.DictReader(q.open()))
    for r in rows:
        if r["paper_id"] == "doi:10.1/podgen":
            r["decision"] = "exclude"
    with q.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    assert import_queue(ctx.store, q) == 1
    assert ctx.store.verdicts("title_abstract")["doi:10.1/podgen"] == "exclude"
    assert "doi:10.1/podgen" not in {r["paper_id"] for r in pending(ctx.store)}
