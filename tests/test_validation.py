import numpy as np
import pandas as pd

from pdm import config as C
from pdm import validation


def _frame(n=12, unit="PMP-001", eq_type="pump"):
    return pd.DataFrame({
        C.TIME_COL: pd.date_range("2024-01-01", periods=n, freq="h"), C.ID_COL: unit,
        C.TYPE_COL: eq_type, "temperature_c": 60.0, "vibration_mm_s": 2.0, "pressure_kpa": 2400.0,
        "rotational_speed_rpm": 1750.0, "load_pct": 70.0, "ambient_temp_c": 5.0,
        "operating_hours": np.arange(100.0, 100.0 + n)})


REG = pd.DataFrame({C.ID_COL: ["PMP-001"], C.TYPE_COL: ["pump"]})
NO_MAINT = pd.DataFrame({C.ID_COL: pd.Series(dtype=str), "start_time": pd.Series(dtype="datetime64[ns]"),
                         "end_time": pd.Series(dtype="datetime64[ns]"),
                         "maintenance_type": pd.Series(dtype=str), "downtime_hours": pd.Series(dtype=int)})
NO_FAIL = pd.DataFrame({C.ID_COL: pd.Series(dtype=str), "failure_time": pd.Series(dtype="datetime64[ns]"),
                        "failure_mode": pd.Series(dtype=str)})


def test_detects_duplicates_invalid_values_and_type_mismatch():
    df = _frame()
    df.loc[2, "temperature_c"] = -999.0
    df.loc[3, "pressure_kpa"] = -5.0
    df.loc[4, "vibration_mm_s"] = 500.0
    df.loc[5, C.TYPE_COL] = "motor"
    df.loc[6, "load_pct"] = np.nan
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    f = validation.validate_raw(df, REG, NO_MAINT, NO_FAIL)
    assert f["exact_duplicate_rows"] == 2
    assert f["out_of_range"]["temperature_c"] == 1
    assert f["out_of_range"]["pressure_kpa"] == 1
    assert f["out_of_range"]["vibration_mm_s"] == 1
    assert f["equipment_type_mismatches"] == 1
    assert f["missing_values"]["load_pct"] == 1


def test_cleaning_applies_documented_decisions():
    df = _frame()
    df.loc[2, "temperature_c"] = -999.0
    df.loc[5, C.TYPE_COL] = "motor"
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    clean, log = validation.clean_telemetry(df, REG)
    assert len(clean) == 12 and log["exact_duplicates_dropped"] == 1
    assert (clean[C.TYPE_COL] == "pump").all() and log["equipment_type_corrected"] == 1
    assert log["out_of_range_set_missing"]["temperature_c"] == 1
    # the sentinel became missing and was then forward-filled from the previous valid hour
    assert clean.loc[2, "temperature_c"] == 60.0
    assert clean[C.TIME_COL].is_monotonic_increasing


def test_forward_fill_is_limited_and_never_backfills():
    df = _frame(n=20)
    df.loc[0:1, "vibration_mm_s"] = np.nan          # leading gap: nothing earlier to carry forward
    df.loc[5:14, "vibration_mm_s"] = np.nan         # 10 h gap, limit is 6
    df.loc[4, "vibration_mm_s"] = 9.0
    clean, _ = validation.clean_telemetry(df, REG, ffill_limit=6)
    v = clean["vibration_mm_s"]
    assert v.iloc[0:2].isna().all(), "leading gap must stay missing (no back-fill)"
    assert (v.iloc[5:11] == 9.0).all(), "first 6 hours are filled with the last known value"
    assert v.iloc[11:15].isna().all(), "hours beyond the limit stay missing"


def test_operating_hours_reset_is_detected_and_repaired():
    df = _frame(n=10)
    df.loc[6:, "operating_hours"] = np.arange(0.0, 4.0)   # counter restarts at hour 6
    assert validation.find_counter_resets(df)[C.ID_COL].tolist() == ["PMP-001"]
    fixed = validation.repair_operating_hours(df)
    assert (fixed["operating_hours"].diff().dropna() > 0).all()
    assert fixed["operating_hours"].iloc[6] == df["operating_hours"].iloc[5] + 1


def test_coverage_gaps_are_matched_to_maintenance():
    df = _frame(n=30).drop(index=range(10, 15)).drop(index=range(20, 23))
    maint = pd.DataFrame({C.ID_COL: ["PMP-001"], "start_time": [pd.Timestamp("2024-01-01 10:00")],
                          "end_time": [pd.Timestamp("2024-01-01 15:00")],
                          "maintenance_type": ["preventive"], "downtime_hours": [5]})
    gaps = validation.coverage_gaps(df, maint)
    assert len(gaps) == 2
    assert gaps["gap_hours"].tolist() == [5.0, 3.0]
    assert gaps["explained_by_maintenance"].tolist() == [True, False]


def test_failure_label_quality_checks():
    fails = pd.DataFrame({C.ID_COL: ["PMP-001", "PMP-001", "XXX-999"],
                          "failure_time": pd.to_datetime(["2024-01-01 20:00"] * 2 + ["2024-01-01 21:00"]),
                          "failure_mode": ["bearing_wear"] * 3})
    f = validation.validate_raw(_frame(), REG, NO_MAINT, fails)
    assert f["duplicate_failure_records"] == 1
    assert f["failures_unknown_equipment"] == 1
    assert f["failures_without_corrective_record"] == 2
    cleaned = validation.clean_failures(fails, REG)
    assert len(cleaned) == 1


def test_injected_issues_are_all_found(small_raw):
    f = validation.validate_raw(small_raw["telemetry"], small_raw["registry"],
                                small_raw["maintenance"], small_raw["failures"])
    inj = small_raw["injected_issues"]
    assert f["exact_duplicate_rows"] == inj["duplicate_telemetry_rows"]
    assert f["duplicate_failure_records"] == inj["duplicate_failure_records"]
    assert f["equipment_type_mismatches"] >= inj["equipment_type_mismatches"]
    assert f["out_of_range"]["temperature_c"] >= inj["temperature_sentinel_-999"]
    assert set(f["operating_hours_resets"]) == set(inj["operating_hours_counter_resets"])
    assert f["coverage_gaps_unexplained"] == 0


def test_clean_output_is_valid(small_clean):
    tele = small_clean["telemetry"]
    assert not tele.duplicated([C.ID_COL, C.TIME_COL]).any()
    assert sum(validation.count_out_of_range(tele).values()) == 0
    assert (tele.groupby(C.ID_COL)["operating_hours"].diff().dropna() >= 0).all()
    assert not validation.find_type_mismatches(tele, small_clean["registry"]).any()


def test_report_renders_from_measured_numbers(small_raw, small_clean, small_table):
    from pdm import features
    f = validation.validate_raw(small_raw["telemetry"], small_raw["registry"],
                                small_raw["maintenance"], small_raw["failures"])
    text = validation.render_validation_report(f, small_clean["log"], features.label_stats(small_table),
                                               small_raw["injected_issues"])
    assert f"{f['n_rows']:,}" in text and "Class imbalance" in text
