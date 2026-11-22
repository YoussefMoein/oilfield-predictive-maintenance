import numpy as np
import pandas as pd
import pytest

from pdm import config as C
from pdm import evaluation

Y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])
S = np.array([0.1, 0.2, 0.6, 0.3, 0.8, 0.4, 0.7, 0.9, 0.05, 0.65])


def test_threshold_metrics_match_hand_calculation():
    m = evaluation.threshold_metrics(Y, S, 0.5)
    # alerts: 0.6(0) 0.8(1) 0.7(0) 0.9(1) 0.65(1) -> tp 3, fp 2, fn 1, tn 4
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 2, 1, 4)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["alert_rate"] == pytest.approx(0.5)


def test_threshold_metrics_handle_no_alerts_without_dividing_by_zero():
    m = evaluation.threshold_metrics(Y, S, 2.0)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0 and m["tp"] == 0


def test_threshold_is_inclusive():
    assert evaluation.threshold_metrics(Y, S, 0.65)["tp"] == 3
    assert evaluation.threshold_metrics(Y, S, 0.6500001)["tp"] == 2


def test_ranking_metrics_for_perfect_and_inverted_scores():
    perfect = evaluation.ranking_metrics(Y, Y.astype(float))
    assert perfect["pr_auc"] == pytest.approx(1.0) and perfect["roc_auc"] == pytest.approx(1.0)
    inverted = evaluation.ranking_metrics(Y, 1.0 - Y)
    assert inverted["roc_auc"] == pytest.approx(0.0)
    assert perfect["prevalence"] == pytest.approx(0.4)


def test_max_f1_threshold_is_optimal_by_brute_force():
    best = evaluation.threshold_max_f1(Y, S)
    f1_best = evaluation.threshold_metrics(Y, S, best)["f1"]
    assert all(f1_best >= evaluation.threshold_metrics(Y, S, t)["f1"] - 1e-12 for t in np.unique(S))


def test_threshold_for_recall_meets_target_with_highest_possible_threshold():
    t = evaluation.threshold_for_recall(Y, S, 0.75)
    assert evaluation.threshold_metrics(Y, S, t)["recall"] >= 0.75
    higher = [x for x in np.unique(S) if x > t]
    assert all(evaluation.threshold_metrics(Y, S, x)["recall"] < 0.75 for x in higher)


def _scored_frame():
    ts = pd.date_range("2024-10-01", periods=240, freq="h")
    a = pd.DataFrame({C.ID_COL: "A", C.TIME_COL: ts, "score": 0.1, C.TARGET: 0})
    b = pd.DataFrame({C.ID_COL: "B", C.TIME_COL: ts, "score": 0.1, C.TARGET: 0})
    fail_a = pd.Timestamp("2024-10-06 00:00")     # flagged 30 h ahead
    fail_b = pd.Timestamp("2024-10-08 00:00")     # never flagged
    a.loc[(a[C.TIME_COL] >= fail_a - pd.Timedelta(hours=72)) & (a[C.TIME_COL] < fail_a), C.TARGET] = 1
    b.loc[(b[C.TIME_COL] >= fail_b - pd.Timedelta(hours=72)) & (b[C.TIME_COL] < fail_b), C.TARGET] = 1
    a.loc[a[C.TIME_COL] >= fail_a - pd.Timedelta(hours=30), "score"] = 0.9
    a.loc[a[C.TIME_COL] >= fail_a, "score"] = 0.1
    b.loc[b[C.TIME_COL] == pd.Timestamp("2024-10-02 05:00"), "score"] = 0.95   # one false alert day
    fails = pd.DataFrame({C.ID_COL: ["A", "B"], "failure_time": [fail_a, fail_b],
                          "failure_mode": ["bearing_wear", "sudden_no_precursor"]})
    return pd.concat([a, b], ignore_index=True), fails


def test_event_level_metrics_on_handmade_case():
    scored, fails = _scored_frame()
    ev = evaluation.event_level_metrics(scored, fails, threshold=0.5)
    assert ev["n_failures"] == 2 and ev["failures_flagged"] == 1
    assert ev["event_recall"] == pytest.approx(0.5)
    events = ev["events"].set_index(C.ID_COL)
    assert events.loc["A", "lead_time_hours"] == 30
    assert not events.loc["B", "flagged"] and np.isnan(events.loc["B", "lead_time_hours"])
    assert ev["false_alert_unit_days"] == 1
    assert ev["alert_unit_days"] == 3           # A alerts on two calendar days, B on one
    assert ev["unit_day_precision"] == pytest.approx(2 / 3)


def test_event_level_lower_threshold_never_reduces_detection():
    scored, fails = _scored_frame()
    strict = evaluation.event_level_metrics(scored, fails, 0.92)
    loose = evaluation.event_level_metrics(scored, fails, 0.05)
    assert loose["failures_flagged"] >= strict["failures_flagged"]
    assert loose["event_recall"] == 1.0


def test_cluster_bootstrap_interval_brackets_point_estimate():
    rng = np.random.default_rng(1)
    frames = []
    for u in range(12):
        y = (rng.random(200) < 0.2).astype(int)
        frames.append(pd.DataFrame({C.ID_COL: f"U{u}", C.TARGET: y, "s": y * 0.5 + rng.random(200)}))
    df = pd.concat(frames)
    ci = evaluation.cluster_bootstrap_ci(df, "s", n_boot=100)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert 0 < ci["ci_low"] < ci["ci_high"] <= 1


def test_metrics_by_group_skips_single_class_groups():
    df = pd.DataFrame({"g": ["x"] * 6 + ["y"] * 4, C.TARGET: [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                       "s": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.2, 0.3, 0.4]})
    out = evaluation.metrics_by_group(df, "s", "g", 0.5)
    assert out["g"].tolist() == ["x"] and out["recall"].iloc[0] == 1.0
