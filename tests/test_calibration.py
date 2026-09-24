import json
import math

from convtts_slr.calibration import apply_thresholds, calibrate, cohen_kappa
from convtts_slr.models import CriterionOutcome, DecisionBatch, ScreeningResult
from convtts_slr.protocol import DEFAULT_PROTOCOL, Stage, Thresholds
from convtts_slr.store import Store


def test_cohen_kappa_empty_input_is_nan():
    assert math.isnan(cohen_kappa([], []))


def test_cohen_kappa_perfect_agreement_all_positive_is_one():
    # pa == pb == 1 makes pe == 1, hitting the special-cased branch.
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_apply_thresholds_without_a_path_returns_the_same_protocol():
    assert apply_thresholds(DEFAULT_PROTOCOL, None) == DEFAULT_PROTOCOL
    assert apply_thresholds(DEFAULT_PROTOCOL, "/no/such/file.json") == DEFAULT_PROTOCOL


def test_apply_thresholds_updates_only_known_criteria(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"IC1": {"low": 0.2, "high": 0.8}, "NOPE": {"low": 0, "high": 1}}))
    updated = apply_thresholds(DEFAULT_PROTOCOL, path)
    ic1 = next(c for c in updated.criteria if c.id == "IC1")
    assert ic1.thresholds == Thresholds(low=0.2, high=0.8)
    other = next(c for c in updated.criteria if c.id == "IC2")
    assert other.thresholds == Thresholds()  # untouched default


def _screening_result(pid: str, score: float) -> ScreeningResult:
    return ScreeningResult(
        paper_id=pid,
        stage=Stage.TITLE_ABSTRACT.value,
        verdict="needs_human",
        reasons=[],
        outcomes=[
            CriterionOutcome(
                criterion_id="IC1",
                polarity="include",
                status="uncertain",
                question_scores={"ic1_new_dataset": score},
            )
        ],
        batch=DecisionBatch(backend="scripted", model="script", answers={}, state_sha=""),
    )


def test_calibrate_fits_thresholds_from_a_labelled_sample(tmp_path):
    store = Store(tmp_path / "work", DEFAULT_PROTOCOL.version)
    scores = {"p1": 0.02, "p2": 0.05, "p3": 0.9, "p4": 0.95}
    labels = {"p1": "0", "p2": "0", "p3": "1", "p4": "1"}
    for pid, score in scores.items():
        store.append("screened", pid, result=_screening_result(pid, score).model_dump(mode="json"))

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "paper_id,IC1\n" + "\n".join(f"{pid},{lab}" for pid, lab in labels.items())
    )
    report = calibrate(store, DEFAULT_PROTOCOL, labels_csv, Stage.TITLE_ABSTRACT)
    assert report["stage"] == "title_abstract"
    assert set(report["thresholds"]) == {"IC1"}
    ic1_report = report["criteria"]["IC1"]
    assert ic1_report["n"] == 4
    assert 0 <= ic1_report["thresholds"]["low"] < ic1_report["thresholds"]["high"] <= 1
