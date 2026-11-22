import numpy as np
import pandas as pd
import pytest

from pdm import config as C
from pdm import risk


def test_categorize_boundaries_are_inclusive():
    cats = risk.categorize([0.0, 0.299, 0.3, 0.5, 0.699, 0.7, 1.0], 0.3, 0.7)
    assert cats.tolist() == ["Low", "Low", "Medium", "Medium", "Medium", "High", "High"]


def test_categorize_rejects_unordered_thresholds():
    with pytest.raises(ValueError):
        risk.categorize([0.5], 0.7, 0.7)
    with pytest.raises(ValueError):
        risk.categorize([0.5], 0.8, 0.3)


def _scored():
    ts = pd.date_range("2024-12-01", periods=72, freq="h")
    rows = []
    for unit, level in (("PMP-001", 0.9), ("MTR-001", 0.5), ("CMP-001", 0.1), ("CMP-002", 0.95)):
        d = pd.DataFrame({C.ID_COL: unit, C.TIME_COL: ts, C.TYPE_COL: unit[:3], "area": "Area North",
                          "score": level, C.TARGET: int(level > 0.8),
                          "hours_since_last_maintenance": 500.0,
                          "vibration_mm_s_z_mean_24h": 3.2 if level > 0.8 else 0.1,
                          "vibration_mm_s_z_trend_24h_vs_168h": 1.1 if level > 0.8 else 0.0,
                          "temperature_c_z_mean_24h": 0.2, "temperature_c_z_trend_24h_vs_168h": 0.0,
                          "pressure_kpa_z_mean_24h": -1.5 if unit == "MTR-001" else 0.0,
                          "pressure_kpa_z_trend_24h_vs_168h": -0.2,
                          "rotational_speed_rpm_z_mean_24h": 0.0,
                          "rotational_speed_rpm_z_trend_24h_vs_168h": 0.0})
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    # PMP-001 was low a day earlier; CMP-002 stopped reporting 40 h before the snapshot
    out.loc[(out[C.ID_COL] == "PMP-001") & (out[C.TIME_COL] <= ts[50]), "score"] = 0.2
    return out[~((out[C.ID_COL] == "CMP-002") & (out[C.TIME_COL] > ts[31]))], ts[-1]


def test_ranking_order_categories_and_priorities():
    scored, snap = _scored()
    r = risk.build_risk_ranking(scored, snap, 0.3, 0.7)
    assert r["rank"].tolist() == [1, 2, 3, 4]
    assert r[C.ID_COL].tolist() == ["PMP-001", "MTR-001", "CMP-001", "CMP-002"]
    assert r["risk_category"].tolist() == ["High", "Medium", "Low", "No current data"]
    active = r[r["risk_category"] != "No current data"]
    assert active["risk_score"].is_monotonic_decreasing
    assert r.loc[0, "suggested_review_priority"].startswith("Priority 1")
    assert r.loc[1, "suggested_review_priority"].startswith("Priority 2")
    assert r.loc[2, "suggested_review_priority"] == "Routine monitoring"
    assert r.loc[3, "suggested_review_priority"].startswith("Not ranked")
    assert len(r) == scored[C.ID_COL].nunique(), "every unit appears exactly once"


def test_ranking_reports_change_and_sensor_trends():
    scored, snap = _scored()
    r = risk.build_risk_ranking(scored, snap, 0.3, 0.7).set_index(C.ID_COL)
    assert r.loc["PMP-001", "risk_score_change_24h"] == pytest.approx(0.7)
    assert "vibration 3.2 SD above unit baseline" in r.loc["PMP-001", "relevant_sensor_trends"]
    assert "rising" in r.loc["PMP-001", "relevant_sensor_trends"]
    assert "pressure 1.5 SD below unit baseline" in r.loc["MTR-001", "relevant_sensor_trends"]
    assert r.loc["CMP-001", "relevant_sensor_trends"].startswith("no sensor")


def test_ranking_only_uses_information_up_to_the_snapshot():
    scored, snap = _scored()
    early = snap - pd.Timedelta(hours=36)
    r = risk.build_risk_ranking(scored, early, 0.3, 0.7).set_index(C.ID_COL)
    assert r["last_reading"].max() <= early
    assert r.loc["PMP-001", "risk_category"] == "Low"      # the later jump is not visible yet


def test_language_stays_advisory():
    text = " ".join(risk.REVIEW_PRIORITY.values()).lower()
    for word in ("unsafe", "shut down", "replace", "must", "failed", "failing"):
        assert word not in text


def test_weekly_topk_backtest_counts_hits():
    ts = pd.date_range("2024-10-01", "2024-10-31 23:00", freq="h")
    rng = np.random.default_rng(0)
    frames = []
    for i in range(10):
        d = pd.DataFrame({C.ID_COL: f"U{i}", C.TIME_COL: ts, "score": rng.random(len(ts)) * 0.1,
                          C.TARGET: 0})
        if i < 2:   # two units always high and always followed by a failure
            d["score"], d[C.TARGET] = 0.9, 1
        frames.append(d)
    bt = risk.weekly_topk_backtest(pd.concat(frames), k=5)
    assert len(bt) == 4 and (bt["top_k_hits"] == 2).all()
    s = risk.summarize_backtest(bt, k=5)
    assert s["precision_at_k"] == pytest.approx(0.4)
    assert s["base_rate"] == pytest.approx(0.2)
    assert s["lift_over_random"] == pytest.approx(2.0)
    assert s["share_of_upcoming_failures_in_top_k"] == pytest.approx(1.0)
