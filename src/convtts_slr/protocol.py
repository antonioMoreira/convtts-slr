"""The review protocol, expressed as data.

Every eligibility criterion and every categorical extraction field is a typed
question (Noul = binary proposition, Choice = mutually exclusive options).
Instructions name the state fields they read (`title`, `abstract`, `fulltext`),
because question IDs are never shown to the model.

Changing any text here changes `Protocol.version`, which invalidates cached
decisions for that version only. Old decisions stay in the log for audit.
"""

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Stage(str, Enum):
    TITLE_ABSTRACT = "title_abstract"
    FULL_TEXT = "full_text"


class NoulSpec(BaseModel):
    kind: Literal["noul"] = "noul"
    id: str
    instructions: str
    true: str
    false: str


class ChoiceSpec(BaseModel):
    kind: Literal["choice"] = "choice"
    id: str
    instructions: str
    options: dict[str, str]


Question = NoulSpec | ChoiceSpec


class Thresholds(BaseModel):
    """Three-way routing. Scores are NOT calibrated probabilities (Jev-Mem, App. B.1),
    so these defaults must be replaced by `slr calibrate` output before the real run."""

    low: float = Field(0.15, description="score <= low  -> proposition judged false")
    high: float = Field(0.85, description="score >= high -> proposition judged true")


class Criterion(BaseModel):
    """An eligibility rule. `questions` are asked together; `rule` combines them."""

    id: str
    label: str
    polarity: Literal["include", "exclude"]
    stage: Stage
    questions: list[NoulSpec]
    rule: Literal["any", "all"] = "any"
    thresholds: Thresholds = Thresholds()


class SearchConfig(BaseModel):
    """Boolean blocks: OR inside a block, AND across blocks. Control is deliberately
    not a block: it is screened for (IC4), which keeps recall high."""

    blocks: list[list[str]]
    from_year: int = 2019
    to_date: str = "2026-09-30"
    max_results_per_source: int = 2000


class Protocol(BaseModel):
    title: str
    search: SearchConfig
    criteria: list[Criterion]
    extraction: list[Question]
    duration_alone_satisfies_ic4: bool = True
    fulltext_char_budget: int = 24_000
    snowball_max_rounds: int = 3
    consensus_levels: dict[str, float] = {"strong": 0.75, "partial": 0.50}

    @property
    def version(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def criteria_for(self, stage: Stage) -> list[Criterion]:
        return [c for c in self.criteria if c.stage == stage]


# --------------------------------------------------------------------------- #
# Eligibility criteria
# --------------------------------------------------------------------------- #

_TA = "Judge only from `title` and `abstract`. "
_FT = "Judge from `fulltext` (selected sections of the paper). "

CRITERIA: list[Criterion] = [
    Criterion(
        id="IC1",
        label="Dataset is a stated contribution",
        polarity="include",
        stage=Stage.TITLE_ABSTRACT,
        questions=[
            NoulSpec(
                id="ic1_new_dataset",
                instructions=_TA
                + "Does the paper present a new speech dataset, or a substantial new "
                "release or annotation layer of an existing speech dataset, "
                "as one of its contributions?",
                true="Building or releasing the speech dataset/annotation layer is stated or "
                "clearly implied as a contribution.",
                false="The paper only uses existing datasets, or the dataset is text-only.",
            )
        ],
    ),
    Criterion(
        id="IC2",
        label="Conversational (>=2 interlocutors, turn order kept)",
        polarity="include",
        stage=Stage.TITLE_ABSTRACT,
        questions=[
            NoulSpec(
                id="ic2_conversational",
                instructions=_TA
                + "Does the dataset contain speech from two or more interlocutors taking "
                "part in the same exchange, with the order of turns preserved?",
                true="Dialogues, conversations or multi-party exchanges in speech, "
                "recorded or synthesized.",
                false="Monologue, read sentences, isolated utterances, "
                "or single-speaker speech only.",
            )
        ],
    ),
    Criterion(
        id="IC3",
        label="Targets turn-level CTTS or whole-dialogue generation",
        polarity="include",
        stage=Stage.TITLE_ABSTRACT,
        rule="any",
        questions=[
            NoulSpec(
                id="ic3_turn_level_ctts",
                instructions=_TA
                + "Is the dataset intended to train or evaluate a model that synthesizes "
                "speech for one dialogue turn, given that turn's text and the dialogue context?",
                true="Conversational / context-aware text-to-speech at "
                "the turn or utterance level.",
                false="No speech synthesis of individual turns is mentioned as a purpose.",
            ),
            NoulSpec(
                id="ic3_whole_dialogue_generation",
                instructions=_TA
                + "Is the dataset intended to train or evaluate a model that generates the "
                "audio of a whole multi-speaker dialogue from a script or description?",
                true="Generating full spoken dialogues, "
                "podcasts or multi-speaker conversations as audio.",
                false="No generation of whole spoken dialogues is mentioned as a purpose.",
            ),
        ],
    ),
    Criterion(
        id="IC4",
        label="Used with a TTS model having explicit pitch/energy/duration predictors",
        polarity="include",
        stage=Stage.FULL_TEXT,
        rule="any",
        questions=[
            NoulSpec(
                id="ic4_pitch_predictor",
                instructions=_FT
                + "Does a speech synthesis model trained or evaluated on the dataset in this "
                "paper contain a dedicated module that predicts pitch (F0) "
                "as an explicit intermediate variable?",
                true="An explicit pitch/F0 predictor or variance adaptor branch "
                "for pitch (e.g. FastSpeech 2 style).",
                false="Pitch is only modelled implicitly (e.g. in latent codes or tokens), "
                "or no such model is used.",
            ),
            NoulSpec(
                id="ic4_energy_predictor",
                instructions=_FT
                + "Does a speech synthesis model trained or evaluated on the dataset in this "
                "paper contain a dedicated module that predicts energy as an "
                "explicit intermediate variable?",
                true="An explicit energy predictor or variance adaptor branch for energy.",
                false="Energy is only modelled implicitly, or no such model is used.",
            ),
            NoulSpec(
                id="ic4_duration_predictor",
                instructions=_FT
                + "Does a speech synthesis model trained or evaluated on the dataset in this "
                "paper contain a dedicated module that predicts phoneme or "
                "token durations explicitly?",
                true="An explicit duration predictor (deterministic or stochastic) driving "
                "length regulation or alignment.",
                false="Durations are implicit (e.g. autoregressive decoding with attention), "
                "or no such model is used.",
            ),
        ],
    ),
    Criterion(
        id="IC6",
        label="Enough construction detail to extract RQ1 and RQ2",
        polarity="include",
        stage=Stage.FULL_TEXT,
        questions=[
            NoulSpec(
                id="ic6_construction_detail",
                instructions=_FT
                + "Does `fulltext` describe both what one sample of the dataset consists of and "
                "where the speech comes from (recording, harvesting, or synthesis)?",
                true="Both the unit of data and the speech source are described.",
                false="Either the unit of data or the speech source is not described.",
            )
        ],
    ),
    Criterion(
        id="EC1",
        label="Dataset only used, not built",
        polarity="exclude",
        stage=Stage.TITLE_ABSTRACT,
        questions=[
            NoulSpec(
                id="ec1_only_uses_dataset",
                instructions=_TA
                + "Is the paper's contribution a model or method that is merely evaluated on "
                "existing datasets, with no new dataset built?",
                true="A model/method paper using existing corpora only.",
                false="The paper builds or releases a dataset, "
                "or it cannot be told from the abstract.",
            )
        ],
    ),
    Criterion(
        id="EC3",
        label="Purpose is only recognition or understanding",
        polarity="exclude",
        stage=Stage.TITLE_ABSTRACT,
        questions=[
            NoulSpec(
                id="ec3_recognition_only",
                instructions=_TA
                + "Is the dataset's stated purpose limited to recognition or understanding tasks "
                "(ASR, emotion recognition, diarization, spoken language understanding) "
                "or to spoken dialogue "
                "agents that answer a user, with no speech synthesis "
                "or dialogue audio generation purpose?",
                true="Only recognition/understanding/agent-response purposes are stated.",
                false="Speech synthesis or dialogue audio generation is among the stated purposes.",
            )
        ],
    ),
]

# EC2 (text-only / monologue) is the negation of IC2 and EC4 (duplicates) is handled
# deterministically in dedup.py; IC5 (date window) is a metadata filter in stages.py.


# --------------------------------------------------------------------------- #
# Extraction questions (Jev). Sets of Nouls where practices can co-occur,
# Choices where they are mutually exclusive.
# --------------------------------------------------------------------------- #


def _n(id_: str, instructions: str, true: str, false: str) -> NoulSpec:
    return NoulSpec(id=id_, instructions=_FT + instructions, true=true, false=false)


def _signal_source(attr: str) -> ChoiceSpec:
    return ChoiceSpec(
        id=f"rq3_{attr}_signal_source",
        instructions=_FT
        + f"How is the {attr} control signal used for training obtained from the dataset?",
        options={
            "extracted_from_audio": "Computed automatically from the audio "
            f"(e.g. an extractor or aligner) for {attr}.",
            "annotated_labels": f"Given by human or model annotations of {attr} "
            "categories or values.",
            "not_used": f"{attr} is not an explicit control signal in this paper.",
            "unclear": "The paper does not say.",
        },
    )


EXTRACTION: list[Question] = [
    # RQ1 - sample unit
    ChoiceSpec(
        id="rq1_sample_unit",
        instructions=_FT + "What does one training sample of the dataset consist of?",
        options={
            "utterance_only": "A single utterance or turn, without dialogue context attached.",
            "turn_with_context": "A target turn together with a window "
            "of preceding dialogue context.",
            "full_dialogue": "An entire dialogue (all turns) treated as one sample.",
            "session_multichannel": "A full session with one audio channel per speaker.",
            "unclear": "The paper does not say.",
        },
    ),
    ChoiceSpec(
        id="rq1_channel_setup",
        instructions=_FT + "How is the audio of different speakers stored?",
        options={
            "separate_channels": "Each speaker has an isolated channel or track.",
            "single_mixed": "All speakers are mixed into one channel.",
            "per_turn_files": "Each turn is a separate file with one speaker.",
            "unclear": "The paper does not say.",
        },
    ),
    ChoiceSpec(
        id="rq1_overlap_handling",
        instructions=_FT + "How does the dataset treat overlapping speech?",
        options={
            "preserved": "Overlaps are kept and represented.",
            "removed_or_avoided": "Overlaps are removed, trimmed or avoided by design.",
            "unclear": "The paper does not say.",
        },
    ),
    # RQ2 - provenance
    _n(
        "rq2_scripted_recorded",
        "Was any speech recorded by speakers reading a fixed script?",
        "Script reading by speakers or actors.",
        "No scripted reading.",
    ),
    _n(
        "rq2_acted_improvised",
        "Was any speech recorded by speakers improvising within a given scenario?",
        "Role-play or improvised acting from prompts.",
        "No improvisation.",
    ),
    _n(
        "rq2_spontaneous_recorded",
        "Was any speech recorded from spontaneous conversation among participants?",
        "Free, unscripted conversation recorded for the dataset.",
        "No spontaneous recording.",
    ),
    _n(
        "rq2_in_the_wild",
        "Was any speech harvested from existing media (podcasts, video, broadcasts, calls)?",
        "Speech collected from pre-existing media sources.",
        "No harvested media.",
    ),
    _n(
        "rq2_synthetic",
        "Was any speech generated synthetically (e.g. an LLM-written script rendered by TTS)?",
        "Machine-generated speech is part of the dataset.",
        "All speech is human.",
    ),
    _n(
        "rq2_derived",
        "Is the dataset derived from an existing dataset by re-recording or re-annotating it?",
        "Built on top of a named prior dataset.",
        "Built from scratch.",
    ),
    # RQ3 - control space
    _signal_source("pitch"),
    _signal_source("energy"),
    _signal_source("duration"),
    _n(
        "rq3_emotion_labels",
        "Are emotion labels annotated per turn or per utterance?",
        "Categorical or dimensional emotion annotations exist.",
        "No emotion annotation.",
    ),
    _n(
        "rq3_style_description",
        "Are natural-language descriptions of speaking style attached to samples?",
        "Free-text style/prosody descriptions exist.",
        "No natural-language descriptions.",
    ),
    _n(
        "rq3_conversational_behaviors",
        "Are conversational behaviors annotated (backchannels, laughter, "
        "interruptions, fillers, pauses)?",
        "At least one such behavior is annotated.",
        "None annotated.",
    ),
    _n(
        "rq3_annot_human",
        "Were any annotations produced by human annotators?",
        "Humans annotated.",
        "No human annotation.",
    ),
    _n(
        "rq3_annot_model",
        "Were any annotations produced by classifiers or LLMs?",
        "Models annotated.",
        "No model annotation.",
    ),
    _n(
        "rq3_annot_inherited",
        "Were labels inherited from a source dataset?",
        "Labels came from a prior dataset.",
        "No inherited labels.",
    ),
    _n(
        "rq3_agreement_reported",
        "Is inter-annotator agreement or label reliability reported?",
        "An agreement or reliability figure is reported.",
        "No agreement figure.",
    ),
    # RQ4 - processing
    _n(
        "rq4_vad",
        "Is voice activity detection or silence-based segmentation used?",
        "VAD/segmentation is used.",
        "Not mentioned.",
    ),
    _n("rq4_diarization", "Is speaker diarization used?", "Diarization is used.", "Not mentioned."),
    _n(
        "rq4_separation_denoising",
        "Is source separation, denoising or speech enhancement applied?",
        "Applied.",
        "Not mentioned.",
    ),
    _n(
        "rq4_asr",
        "Are transcripts produced by an ASR system?",
        "ASR transcripts are used.",
        "Not mentioned.",
    ),
    _n(
        "rq4_human_transcription",
        "Are transcripts written or corrected by humans?",
        "Human transcription/correction.",
        "Not mentioned.",
    ),
    _n(
        "rq4_forced_alignment",
        "Is forced alignment used to obtain phoneme or word timings?",
        "Forced alignment is used.",
        "Not mentioned.",
    ),
    _n(
        "rq4_quality_filter",
        "Are samples filtered by an automatic quality measure (e.g. SNR, DNSMOS, WER)?",
        "An automatic quality filter is applied.",
        "Not mentioned.",
    ),
    # RQ5 - validation and release
    _n(
        "rq5_baseline_trained",
        "Is a synthesis model trained on the dataset and reported?",
        "A trained baseline is reported.",
        "No baseline.",
    ),
    _n(
        "rq5_subjective_eval",
        "Are subjective listening tests (MOS, CMOS, preference) reported?",
        "Reported.",
        "Not reported.",
    ),
    _n(
        "rq5_objective_eval",
        "Are objective metrics (e.g. MCD, F0 RMSE, WER, speaker similarity) reported?",
        "Reported.",
        "Not reported.",
    ),
    ChoiceSpec(
        id="rq5_release",
        instructions=_FT + "How is the dataset made available?",
        options={
            "open": "Publicly downloadable under an open license.",
            "restricted": "Available on request or under a restrictive license.",
            "not_released": "Not released.",
            "unclear": "The paper does not say.",
        },
    ),
    # RQ6 - lineage
    _n(
        "rq6_follows_prior_method",
        "Does the paper state that its construction follows a previously published "
        "dataset construction method or pipeline?",
        "A prior construction method is explicitly followed.",
        "No prior construction method is named.",
    ),
]


DEFAULT_PROTOCOL = Protocol(
    title="Conversational datasets for controllable TTS: a systematic literature review",
    search=SearchConfig(
        blocks=[
            [
                "dialogue",
                "dialog",
                "conversation",
                "conversational",
                "spoken dialogue",
                "multi-party",
                "multi-speaker",
            ],
            [
                "speech synthesis",
                "text-to-speech",
                "TTS",
                "speech generation",
                "spoken dialogue generation",
            ],
            ["dataset", "corpus", "corpora", "benchmark"],
        ]
    ),
    criteria=CRITERIA,
    extraction=EXTRACTION,
)
