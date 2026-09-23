from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    """`reference_baseline` papers (the 4 non-conversational methods papers) get the
    same extraction but never count toward consensus."""

    CANDIDATE = "candidate"
    REFERENCE_BASELINE = "reference_baseline"


class Paper(BaseModel):
    id: str  # canonical key, see dedup.canonical_id
    title: str
    abstract: str = ""
    year: int | None = None
    publication_date: str | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    pdf_url: str | None = None
    sources: list[str] = Field(default_factory=list)  # where it was found
    found_in_round: int = 0  # 0 = database search / seed, n = snowball round n
    role: Role = Role.CANDIDATE
    is_seed: bool = False


Verdict = Literal["include", "exclude", "needs_human"]


class QuestionAnswer(BaseModel):
    """A single typed answer. Nouls carry `score`; Choices carry `choice` + `distribution`."""

    question_id: str
    kind: Literal["noul", "choice"]
    score: float | None = None
    choice: str | None = None
    distribution: dict[str, float] | None = None
    confidence: float | None = None


class DecisionBatch(BaseModel):
    """What a backend returned for one call, with enough provenance to reproduce it."""

    backend: str
    model: str
    answers: dict[str, QuestionAnswer]
    state_sha: str


class CriterionOutcome(BaseModel):
    criterion_id: str
    polarity: Literal["include", "exclude"]
    status: Literal["pass", "fail", "uncertain"]  # 'pass' = proposition holds
    question_scores: dict[str, float]


class ScreeningResult(BaseModel):
    paper_id: str
    stage: str
    verdict: Verdict
    reasons: list[str]
    outcomes: list[CriterionOutcome]
    batch: DecisionBatch


class Quoted(BaseModel):
    """A System-Two value with the verbatim passage supporting it."""

    value: str | float | int | list[str] | None = None
    quote: str | None = Field(
        None, description="Verbatim sentence from the paper supporting the value"
    )


class DatasetFacts(BaseModel):
    """Free-form fields that a bounded Noul/Choice cannot express."""

    dataset_name: Quoted = Quoted()
    languages: Quoted = Quoted()
    total_hours: Quoted = Quoted()
    num_dialogues: Quoted = Quoted()
    num_speakers: Quoted = Quoted()
    speakers_per_dialogue: Quoted = Quoted()
    baseline_models: Quoted = Quoted()
    pitch_energy_extractor: Quoted = Quoted()
    aligner: Quoted = Quoted()
    script_source: Quoted = Quoted()
    prior_method_followed: Quoted = Quoted()
    construction_rationale: Quoted = Quoted()
    stated_limitations: Quoted = Quoted()


class ExtractionResult(BaseModel):
    paper_id: str
    role: Role
    typed: DecisionBatch  # Jev answers to protocol.EXTRACTION
    facts: DatasetFacts | None = None
    facts_model: str | None = None


class VerificationResult(BaseModel):
    paper_id: str
    missing_quotes: list[str]  # DatasetFacts fields whose quote is not in the text
    disagreements: list[str]  # typed fields where the independent checker disagrees
    passed: bool


class Event(BaseModel):
    """Append-only log entry. The log is the single source of truth (state = fold(events))."""

    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    kind: str
    paper_id: str | None = None
    protocol_version: str
    payload: dict[str, Any] = Field(default_factory=dict)
