# Review protocol v1

The machine-readable version of this protocol is `src/convtts_slr/protocol.py`. Its hash (`slr protocol`) is stamped on every decision in the log, so this document and the code must be changed together.

## Subject and research question (RQ)

**Subject:** Conversational speech datasets explicitly constructed for training or evaluating controllable conversational speech synthesis.

**Main Research Question (RQ):** Is there a consensus in the literature on how to construct conversational datasets for controllable TTS?

The "how" is decomposed into six sub-questions, each mapped to an extraction block:

| Sub-RQ | Question | Extraction fields |
|---|---|---|
| RQ1 | What is a sample, and how are speakers and overlaps stored? | `rq1_*` |
| RQ2 | Where does the speech come from (scripted, improvised, spontaneous, harvested, synthetic, derived)? | `rq2_*` |
| RQ3 | Where do the control signals come from (extracted from audio, annotated, inherited), and who annotates? | `rq3_*` |
| RQ4 | Which processing steps are applied (VAD, diarization, enhancement, ASR, alignment, filtering)? | `rq4_*` |
| RQ5 | How is fitness shown and how is the dataset released? | `rq5_*` |
| RQ6 | Do papers follow a shared, previously published construction method? | `rq6_*` |

## Operational definitions

**Conversational.** Speech from two or more interlocutors in the same exchange, with turn order preserved. Recorded and synthetic speech both qualify; synthetic and derived datasets are included and tagged (RQ2), because that choice is itself a construction practice.

**Controllable TTS (decision of 22 Sep 2026).** Following Xie et al. (arXiv:2412.06602), a TTS model is controllable when it has a dedicated, explicit architectural module for a control attribute. For this review only explicit **predictors of pitch, energy or duration** count (FastSpeech 2 style variance predictors). Label embeddings, reference-audio style encoders and natural-language prompt encoders do not count.

Consequence 1: controllability is a property of the model, so IC4 is judged on the synthesis model trained or evaluated on the dataset in the paper, at full-text stage.

Consequence 2: pitch, energy and duration targets are normally extracted automatically from audio, so RQ3 records where each control signal comes from rather than whether it exists.

Open point: duration predictors are present in almost every non-autoregressive TTS model (including VITS-family models), so "duration only" makes IC4 weak. It counts by default, as decided; set `duration_alone_satisfies_ic4 = false` to require pitch or energy.

**Downstream task (decision of 22 Sep 2026).** The dataset must target turn-level conversational TTS (next-turn speech from its text plus dialogue context) or whole-dialogue generation (audio for an entire multi-speaker dialogue). Datasets built only for spoken dialogue agents, recognition or understanding are excluded.

**Reference baseline (decision of 22 Sep 2026).** The four non-conversational methods papers (arXiv:2104.04896, arXiv:2402.16380, doi:10.3390/app15041848, arXiv:2409.03283) are extracted with the same schema but never screened and never counted toward consensus. They answer: do conversational datasets follow general TTS-corpus practice, or add dialogue-specific steps?

**Consensus.** For mutually exclusive fields, the share of reporting datasets that follow the most common option: strong at 0.75 or more, partial at 0.50 or more. For co-occurring practices, the share of datasets that use the practice. "Rare or unreported" is not treated as consensus, since a practice that is not reported may still have been used. Trends by year and citation lineage (RQ6) are reported alongside.

## Eligibility criteria

| ID | Proposition (true = passes) | Stage | Evaluated by |
|---|---|---|---|
| IC1 | A new speech dataset or substantial annotation layer is a stated contribution | Title/abstract | Jev |
| IC2 | Two or more interlocutors in the same exchange, turn order kept | Title/abstract | Jev |
| IC3 | Targets turn-level CTTS **or** whole-dialogue generation | Title/abstract | Jev |
| IC4 | A synthesis model trained/evaluated on it has an explicit pitch, energy or duration predictor | Full text | Jev |
| IC5 | Published 2019 to Sept 2026 | Metadata | Code |
| IC6 | Full text describes both the sample unit and the speech source | Full text | Jev |
| EC1 | Dataset only used, not built | Title/abstract | Jev |
| EC2 | Text-only or monologue-only (the negation of IC2) | Title/abstract | Jev |
| EC3 | Purpose limited to recognition, understanding or dialogue agents | Title/abstract | Jev |
| EC4 | Duplicate of an included record (preprint vs venue version) | Identification | Code |

Seeds (DailyTalk, Behavior-SD) are known positives. If the model excludes one, the paper goes to the human queue with a `SEED_EXCLUDED_BY_MODEL` flag instead of being dropped. Behavior-SD is likely to trigger this at IC4, and that decision should be made by you, explicitly.

## Search

Sources are OpenAlex, Semantic Scholar and arXiv. OpenAlex and Semantic Scholar also index ACL Anthology and ISCA. IEEE Xplore is added as a CSV export through `--local`.

The query has three blocks, joined by AND: conversation terms, synthesis terms and dataset terms. Control is deliberately not a query block; it is screened for instead, which keeps recall high.

Backward and forward snowballing from confirmed includes repeats until a round yields no new includes, capped at 3 rounds.

## Validity controls

1. **Calibrated thresholds.** Scores are not calibrated probabilities. You label a stratified sample of about 80 papers per stage. Thresholds are then fitted so that automated decisions reach at least 0.95 precision and auto-reject at most 2% of true positives. Cohen's κ is reported against your labels.
2. **Three-way routing.** Scores above the upper threshold pass, scores below the lower threshold fail, and everything in between goes to the human queue.
3. **Grounded System-Two output.** Every free-text value must carry a verbatim quote, and a quote that cannot be found in the parsed paper sends the paper to human review.
4. **Independent second opinion.** A second model independently re-answers IC4 and `rq1_sample_unit`. Disagreements go to human review.
5. **Audit trail.** Every decision is an event with backend, model, state hash and protocol version. The PRISMA 2020 flow is computed from these events.
