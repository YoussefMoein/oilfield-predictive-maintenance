import numpy as np
import pandas as pd

from pdm import config as C
from pdm import features


def test_feature_columns_exclude_identifiers_labels_and_hidden_state(small_table):
    cols = features.feature_columns(small_table)
    forbidden = {C.ID_COL, C.TIME_COL, C.TYPE_COL, C.TARGET, "hours_to_failure",
                 "next_failure_mode", "area", "operating_hours"}
    assert not forbidden & set(cols)
    assert not [c for c in cols if "latent" in c or "_abs_" in c]
    assert not set(features.ROLL_SENSORS) & set(cols), "raw absolute levels are ablation-only"
    with_abs = features.feature_columns(small_table, include_absolute=True)
    assert set(cols) < set(with_abs)
    assert all(pd.api.types.is_numeric_dtype(small_table[c]) for c in with_abs)


def test_features_do_not_use_future_information(small_clean, small_condition_model, small_table):
    """Truncation invariance: deleting everything after T must not change any feature at or before T."""
    cutoff = pd.Timestamp("2024-04-01 00:00")
    tele = small_clean["telemetry"]
    truncated = features.build_model_table(
        tele[tele[C.TIME_COL] <= cutoff],
        small_clean["maintenance"][small_clean["maintenance"]["end_time"] <= cutoff],
        small_clean["failures"][small_clean["failures"]["failure_time"] <= cutoff],
        small_clean["registry"], small_condition_model, drop_censored=False)
    cols = features.feature_columns(small_table, include_absolute=True)
    key = [C.ID_COL, C.TIME_COL]
    full = small_table[small_table[C.TIME_COL] <= cutoff].set_index(key)[cols]
    part = truncated.set_index(key)[cols].reindex(full.index)
    assert len(full) > 1000
    pd.testing.assert_frame_equal(full, part, check_exact=False, rtol=1e-9, atol=1e-9)


def test_condition_model_only_uses_training_period(small_clean):
    tele = small_clean["telemetry"]
    fit_end = "2024-03-31 23:00"
    a = features.fit_condition_model(tele, fit_end=fit_end)
    tampered = tele.copy()
    late = tampered[C.TIME_COL] > pd.Timestamp(fit_end)
    tampered.loc[late, "temperature_c"] += 50.0
    b = features.fit_condition_model(tampered, fit_end=fit_end)
    pd.testing.assert_frame_equal(a, b)


def test_condition_model_recovers_simulated_load_response(small_condition_model):
    t = small_condition_model[small_condition_model["sensor"] == "temperature_c"]
    # the simulator uses 0.22 degC per % load and 0.12 degC per degC ambient
    assert abs(t["beta_load"].mean() - 0.22) < 0.04
    assert abs(t["beta_ambient"].mean() - 0.12) < 0.04


def _simple_tele(n=400):
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        C.TIME_COL: ts, C.ID_COL: "PMP-001", C.TYPE_COL: "pump",
        "temperature_c": 60 + rng.normal(0, 1, n), "vibration_mm_s": 2 + rng.normal(0, 0.1, n),
        "pressure_kpa": 2400 + rng.normal(0, 10, n), "rotational_speed_rpm": 1750 + rng.normal(0, 3, n),
        "load_pct": 70.0, "ambient_temp_c": 5.0, "operating_hours": np.arange(n, dtype=float)})


def test_labels_cover_exactly_the_horizon_before_a_failure():
    tele = _simple_tele()
    fail_time = pd.Timestamp("2024-01-10 12:00")
    fails = pd.DataFrame({C.ID_COL: ["PMP-001"], "failure_time": [fail_time],
                          "failure_mode": ["bearing_wear"]})
    out = features.add_labels(tele, fails, horizon_hours=72)
    pos = out[out[C.TARGET] == 1]
    assert pos[C.TIME_COL].min() == fail_time - pd.Timedelta(hours=72)
    assert pos[C.TIME_COL].max() == fail_time - pd.Timedelta(hours=1)
    assert len(pos) == 72
    # the failure hour itself and everything after it look forward to *no* further failure
    assert out.loc[out[C.TIME_COL] >= fail_time, C.TARGET].sum() == 0
    assert out.loc[out[C.TIME_COL] == fail_time - pd.Timedelta(hours=10), "hours_to_failure"].item() == 10


def test_rolling_window_is_time_based_and_backward_looking():
    tele = _simple_tele(n=300)
    tele["vibration_mm_s"] = np.arange(300, dtype=float)
    out = features.add_sensor_features(tele)
    row = out.iloc[200]
    assert np.isclose(row["vibration_mm_s_abs_mean_24h"], np.arange(177, 201).mean())


def test_time_since_maintenance_counts_from_completion_not_start():
    tele = _simple_tele(n=200)
    start, end = pd.Timestamp("2024-01-03 00:00"), pd.Timestamp("2024-01-03 08:00")
    tele = tele[(tele[C.TIME_COL] < start) | (tele[C.TIME_COL] >= end)]
    maint = pd.DataFrame({C.ID_COL: ["PMP-001"], "start_time": [start], "end_time": [end],
                          "maintenance_type": ["preventive"], "downtime_hours": [8]})
    fails = pd.DataFrame({C.ID_COL: pd.Series(dtype=str),
                          "failure_time": pd.Series(dtype="datetime64[ns]")})
    out = features.add_history_features(tele, maint, fails).set_index(C.TIME_COL)
    assert out.loc[end, "hours_since_last_maintenance"] == 0
    assert out.loc[end + pd.Timedelta(hours=5), "hours_since_last_preventive"] == 5
    before = out.loc[start - pd.Timedelta(hours=1)]
    assert before["maintenance_history_censored"] == 1
    assert before["hours_since_last_maintenance"] == 47   # hours since the record started
    assert out.loc[end, "maintenance_history_censored"] == 0


def test_trailing_failure_count_window():
    tele = _simple_tele(n=24 * 120)
    fails = pd.DataFrame({C.ID_COL: ["PMP-001"], "failure_time": [pd.Timestamp("2024-01-05 00:00")]})
    maint = pd.DataFrame({C.ID_COL: ["PMP-001"], "start_time": [pd.Timestamp("2024-01-05 00:00")],
                          "end_time": [pd.Timestamp("2024-01-05 01:00")],
                          "maintenance_type": ["corrective"], "downtime_hours": [1]})
    out = features.add_history_features(tele, maint, fails).set_index(C.TIME_COL)["failures_last_90d"]
    assert out.loc["2024-01-04 23:00"] == 0
    assert out.loc["2024-01-05 00:00"] == 1
    assert out.loc["2024-04-03 23:00"] == 1       # still inside 90 days
    assert out.loc["2024-04-04 01:00"] == 0       # rolled out of the window


def test_engineered_values_are_bounded(small_table):
    z = [c for c in features.feature_columns(small_table) if "_z_" in c or "_relstd_" in c]
    assert len(z) >= 30
    assert small_table[z].abs().max().max() <= features.Z_CLIP + 1e-9
    assert small_table["hours_since_last_maintenance"].max() <= features.HOURS_SINCE_CAP
    assert set(small_table[C.TARGET].unique()) <= {0, 1}
    assert small_table[C.TARGET].sum() > 0


def test_censored_tail_and_warmup_are_removed(small_clean, small_table):
    tele = small_clean["telemetry"]
    assert small_table[C.TIME_COL].max() <= tele[C.TIME_COL].max() - pd.Timedelta(hours=C.HORIZON_HOURS)
    start = tele.groupby(C.ID_COL)[C.TIME_COL].min()
    first = small_table.groupby(C.ID_COL)[C.TIME_COL].min()
    assert ((first - start) >= pd.Timedelta(hours=features.BASELINE_MIN_HOURS)).all()
