import pandas as pd
import pytest

from pdm import config as C
from pdm import data_generation, data_loading


def test_generation_is_deterministic():
    kw = dict(seed=3, units_per_type=1, start="2024-01-01", end="2024-02-15 23:00")
    a, b = data_generation.generate_dataset(**kw), data_generation.generate_dataset(**kw)
    for key in ("telemetry", "failures", "maintenance", "registry"):
        pd.testing.assert_frame_equal(a[key], b[key])


def test_different_seed_gives_different_data():
    kw = dict(units_per_type=1, start="2024-01-01", end="2024-02-15 23:00")
    a = data_generation.generate_dataset(seed=1, **kw)["telemetry"]
    b = data_generation.generate_dataset(seed=2, **kw)["telemetry"]
    assert not a["temperature_c"].reset_index(drop=True).equals(b["temperature_c"].reset_index(drop=True))


def test_schema_and_fleet(small_raw):
    tele, reg = small_raw["telemetry"], small_raw["registry"]
    assert list(tele.columns) == data_loading.TELEMETRY_COLUMNS
    assert reg[C.ID_COL].is_unique and len(reg) == 9
    assert set(reg[C.TYPE_COL]) == set(C.EQUIPMENT_TYPES)
    assert set(tele[C.ID_COL]) == set(reg[C.ID_COL])


def test_hidden_state_is_not_in_telemetry(small_raw):
    assert not [c for c in small_raw["telemetry"].columns if "latent" in c or "wear" in c]
    assert {"latent_wear", "latent_mode"} <= set(small_raw["ground_truth"].columns)


def test_every_failure_has_corrective_maintenance_and_downtime(small_raw):
    f = small_raw["failures"].drop_duplicates()
    m = small_raw["maintenance"]
    corrective = m[m["maintenance_type"] == "corrective"]
    merged = f.merge(corrective, left_on=[C.ID_COL, "failure_time"], right_on=[C.ID_COL, "start_time"])
    assert len(merged) == len(f) > 0
    # the unit is down from the failure hour onwards: no reading at the failure timestamp
    keys = small_raw["telemetry"][[C.ID_COL, C.TIME_COL]]
    assert f.merge(keys, left_on=[C.ID_COL, "failure_time"], right_on=[C.ID_COL, C.TIME_COL]).empty


def test_clean_generation_has_no_injected_issues():
    d = data_generation.generate_dataset(seed=5, units_per_type=1, start="2024-01-01",
                                         end="2024-02-15 23:00", inject_issues=False)
    tele = d["telemetry"]
    assert not tele.duplicated().any()
    assert tele[C.SENSORS].notna().all().all()
    assert d["injected_issues"] == {}


def test_write_and_load_round_trip(small_raw, tmp_path, monkeypatch):
    for name in ("TELEMETRY_FILE", "REGISTRY_FILE", "MAINTENANCE_FILE", "FAILURES_FILE",
                 "GROUND_TRUTH_FILE", "INJECTED_ISSUES_FILE"):
        monkeypatch.setattr(C, name, tmp_path / getattr(C, name).name)
    monkeypatch.setattr(C, "ensure_dirs", lambda: None)
    data_generation.write_dataset(small_raw)
    tele = data_loading.load_telemetry(C.TELEMETRY_FILE)
    assert len(tele) == len(small_raw["telemetry"])
    assert pd.api.types.is_datetime64_any_dtype(tele[C.TIME_COL])
    assert pd.api.types.is_float_dtype(tele["temperature_c"])
    fails = data_loading.load_failures(C.FAILURES_FILE)
    assert pd.api.types.is_datetime64_any_dtype(fails["failure_time"])
    maint = data_loading.load_maintenance(C.MAINTENANCE_FILE)
    assert (maint["end_time"] > maint["start_time"]).all()


def test_loader_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({C.TIME_COL: ["2024-01-01"], C.ID_COL: ["PMP-001"]}).to_csv(path, index=False)
    with pytest.raises(data_loading.SchemaError):
        data_loading.load_telemetry(path)
