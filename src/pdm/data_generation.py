"""Synthetic sensor-data generator for pumps, motors and compressors.

IMPORTANT: everything produced here is SIMULATED. It does not come from real
oil field equipment or from any company's records. The simulator exists because
no public dataset combines the equipment types, sensor channels, maintenance
history and failure labels this project needs (see documentation/dataset_selection.md).

Design in one paragraph
-----------------------
Each unit carries a hidden "wear" state that grows with running time and load.
Wear is partly reset by time-based preventive maintenance and fully reset by the
corrective maintenance that follows a failure. Failures are drawn from an hourly
hazard that rises steeply with wear, plus a small constant hazard for sudden
failures that have no sensor precursor. Each wear cycle has a dominant
degradation mode that decides which sensors drift (vibration, temperature or
pressure). Because failure *timing* is random given wear, sensors carry real but
imperfect information about the next 72 hours - which is the situation a
predictive-maintenance model actually faces.

The hidden wear state is written to a separate "ground truth" file used only for
diagnostics. It is never used as a model feature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config as C


@dataclass(frozen=True)
class TypeSpec:
    prefix: str
    temp_base: float          # deg C, bearing/casing temperature at 70 % load
    vib_base: float           # mm/s RMS
    pressure_nominal: float   # kPa (discharge pressure; lube-oil pressure for motors)
    rpm_nominal: float
    rated_power_kw: tuple[int, int]
    mode_probs: dict[str, float]


TYPE_SPECS: dict[str, TypeSpec] = {
    "pump": TypeSpec("PMP", 62.0, 2.2, 2400.0, 1780.0, (75, 400),
                     {"bearing_wear": 0.40, "overheating": 0.20, "pressure_loss": 0.40}),
    "motor": TypeSpec("MTR", 70.0, 1.6, 240.0, 3560.0, (150, 900),
                      {"bearing_wear": 0.45, "overheating": 0.40, "pressure_loss": 0.15}),
    "compressor": TypeSpec("CMP", 88.0, 3.4, 6200.0, 990.0, (500, 2500),
                           {"bearing_wear": 0.30, "overheating": 0.30, "pressure_loss": 0.40}),
}

# Sensor response per degradation mode, applied to g = sensitivity * wear**2.
#   temp: additive deg C | vib: fraction of base | pressure: fractional drop
MODE_GAINS = {
    "bearing_wear":  {"temp": 5.0,  "vib": 0.90, "pressure": 0.010},
    "overheating":   {"temp": 11.0, "vib": 0.15, "pressure": 0.005},
    "pressure_loss": {"temp": 1.5,  "vib": 0.35, "pressure": 0.070},
}

# Hazard model (per running hour)
WEAR_RATE_PER_HOUR = 0.00042
HAZARD_SUDDEN = 2.0e-5
HAZARD_WEAR_COEF = 0.020
HAZARD_WEAR_ONSET = 0.70
# Once a wear-driven failure is initiated it takes a random "P-F interval" to complete,
# during which damage accelerates (the classic potential-failure -> functional-failure curve).
PF_INTERVAL_HOURS = (24, 144)
TERMINAL_WEAR_GAIN = 0.35


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) noise with innovation standard deviation ``sigma``."""
    eps = rng.normal(0.0, sigma, n)
    eps[0] /= np.sqrt(1 - phi**2)
    return lfilter([1.0], [1.0, -phi], eps)


def _ambient_series(rng: np.random.Generator, index: pd.DatetimeIndex, offset: float) -> np.ndarray:
    """Hourly outdoor temperature with a continental seasonal + daily cycle."""
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()
    seasonal = 4.0 - 16.0 * np.cos(2 * np.pi * (doy - 15) / 365.0)
    daily = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    return seasonal + daily + offset + _ar1(rng, len(index), 0.97, 1.0)


def _load_series(rng: np.random.Generator, n: int) -> np.ndarray:
    setpoint = rng.uniform(60, 85)
    hour = np.arange(n) % 24
    diurnal = 3.0 * np.sin(2 * np.pi * (hour - 8) / 24.0)
    # piecewise-constant operating regimes that change every few weeks
    regime = np.zeros(n)
    t = 0
    while t < n:
        length = int(rng.uniform(15, 45) * 24)
        regime[t:t + length] = rng.normal(0, 6.0)
        t += length
    load = setpoint + diurnal + regime + _ar1(rng, n, 0.97, 1.2)
    return np.clip(load, 20, 100)


def _draw_mode(rng: np.random.Generator, spec: TypeSpec) -> str:
    modes = list(spec.mode_probs)
    return str(rng.choice(modes, p=[spec.mode_probs[m] for m in modes]))


def _simulate_unit(rng, equipment_id, eq_type, index, ambient):
    """Simulate one unit. Returns telemetry, maintenance rows, failure rows, ground truth."""
    spec = TYPE_SPECS[eq_type]
    n = len(index)
    load = _load_series(rng, n)

    wear_rate = WEAR_RATE_PER_HOUR * rng.lognormal(0.0, 0.35)
    sensitivity = float(np.clip(rng.lognormal(0.0, 0.30), 0.45, 2.0))
    pm_interval_h = rng.uniform(60, 110) * 24
    next_pm = rng.uniform(15 * 24, pm_interval_h)
    wear = rng.uniform(0.0, 0.5)
    mode = _draw_mode(rng, spec)

    running = np.zeros(n, dtype=bool)
    wear_arr = np.full(n, np.nan)
    mode_arr = np.empty(n, dtype=object)
    maintenance, failures = [], []
    down_until = -1

    wear_noise = rng.lognormal(0.0, 0.5, n)
    shock_draw = rng.random(n)
    shock_size = rng.uniform(0.05, 0.20, n)
    fail_draw = rng.random(n)

    terminal_fail_at = -1      # hour index at which an initiated failure completes
    terminal_step = 0.0

    for t in range(n):
        if t < down_until:
            continue
        if t >= next_pm and t != terminal_fail_at:
            # time-based preventive maintenance (blind to wear). If it happens to land
            # inside a developing failure it interrupts that failure.
            downtime = int(rng.integers(4, 13))
            maintenance.append((equipment_id, index[t], index[t] + pd.Timedelta(hours=downtime),
                                "preventive", downtime))
            wear *= 1.0 - rng.uniform(0.5, 0.9)
            if wear < 0.15:
                mode = _draw_mode(rng, spec)
            terminal_fail_at = -1
            down_until = t + downtime
            next_pm = down_until + pm_interval_h * rng.uniform(0.85, 1.15)
            continue

        failed_mode = None
        if t == terminal_fail_at:
            failed_mode = mode
        else:
            running[t] = True
            wear += wear_rate * (0.4 + 0.6 * (load[t] / 75.0) ** 2) * wear_noise[t]
            if shock_draw[t] < 4e-4:
                wear += shock_size[t]
            if terminal_fail_at > 0:
                wear += terminal_step          # accelerating damage after initiation
            wear_arr[t] = wear
            mode_arr[t] = mode
            if fail_draw[t] < HAZARD_SUDDEN and t + 1 < n:
                failed_mode = "sudden_no_precursor"
                t_fail = t + 1
            elif terminal_fail_at < 0:
                hazard_wear = HAZARD_WEAR_COEF * max(0.0, wear - HAZARD_WEAR_ONSET) ** 2
                if fail_draw[t] < HAZARD_SUDDEN + hazard_wear:
                    # failure initiated: it completes after a random P-F interval
                    pf_hours = int(rng.uniform(*PF_INTERVAL_HOURS))
                    terminal_fail_at = t + pf_hours
                    terminal_step = TERMINAL_WEAR_GAIN / pf_hours
        if failed_mode is not None:
            t_fail = t if t == terminal_fail_at else t + 1
            downtime = int(rng.integers(12, 73))
            fail_time = index[t_fail]
            failures.append((equipment_id, fail_time, failed_mode))
            maintenance.append((equipment_id, fail_time, fail_time + pd.Timedelta(hours=downtime),
                                "corrective", downtime))
            wear = rng.uniform(0.0, 0.05)
            mode = _draw_mode(rng, spec)
            terminal_fail_at = -1
            down_until = t_fail + downtime
            next_pm = down_until + pm_interval_h * rng.uniform(0.85, 1.15)

    # ---- sensor signals (vectorised over running hours)
    w = np.nan_to_num(wear_arr, nan=0.0)
    g = sensitivity * w**2
    gain = {k: np.zeros(n) for k in ("temp", "vib", "pressure")}
    for m, gains in MODE_GAINS.items():
        mask = mode_arr == m
        for k, v in gains.items():
            gain[k][mask] = v * g[mask]

    temp_offset = rng.normal(0, 2.5)
    vib_factor = rng.lognormal(0, 0.12)
    pressure_factor = rng.normal(1.0, 0.03)

    temperature = (spec.temp_base + temp_offset + 0.22 * (load - 70) + 0.12 * (ambient - 5)
                   + gain["temp"] + _ar1(rng, n, 0.8, 0.8) + rng.normal(0, 0.7, n))
    vibration = (spec.vib_base * vib_factor * (1 + 0.004 * (load - 70)) * (1 + gain["vib"])
                 * rng.lognormal(0, 0.10, n))
    pressure = (spec.pressure_nominal * pressure_factor * (1 + 0.0012 * (load - 70))
                * (1 - gain["pressure"])
                * (1 + rng.normal(0, 0.012, n) * (1 + 20 * gain["pressure"])))
    rpm = (spec.rpm_nominal * (0.88 + 0.12 * load / 100)
           * (1 + rng.normal(0, 0.003, n) * (1 + 1.5 * gain["vib"])))

    initial_hours = rng.uniform(2_000, 60_000)
    operating_hours = initial_hours + np.cumsum(running)

    tele = pd.DataFrame({
        C.TIME_COL: index, C.ID_COL: equipment_id, C.TYPE_COL: eq_type,
        "temperature_c": temperature.round(2), "vibration_mm_s": vibration.round(3),
        "pressure_kpa": pressure.round(1), "rotational_speed_rpm": rpm.round(1),
        "load_pct": load.round(2), "ambient_temp_c": ambient.round(1),
        "operating_hours": operating_hours.round(0),
    })[running]
    truth = pd.DataFrame({
        C.TIME_COL: index[running], C.ID_COL: equipment_id,
        "latent_wear": wear_arr[running].round(4), "latent_mode": mode_arr[running],
    })
    meta = {"initial_operating_hours": round(initial_hours),
            "rated_power_kw": int(rng.integers(*spec.rated_power_kw))}
    return tele, maintenance, failures, truth, meta


def _inject_quality_issues(rng: np.random.Generator, tele: pd.DataFrame,
                           failures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Deliberately corrupt a small share of records so validation has real work to do."""
    tele = tele.copy()
    n = len(tele)
    log: dict[str, int | list] = {}

    # 1) isolated missing sensor values (~0.4 % per channel)
    isolated = 0
    for col in C.SENSORS:
        idx = rng.choice(n, size=int(0.004 * n), replace=False)
        tele.iloc[idx, tele.columns.get_loc(col)] = np.nan
        isolated += len(idx)
    log["isolated_missing_values"] = isolated

    # 2) sensor dropout blocks (one channel silent for 6-48 h on one unit)
    blocks = []
    ids = tele[C.ID_COL].unique()
    for _ in range(18):
        unit = rng.choice(ids)
        col = str(rng.choice(C.SENSORS[:4]))
        rows = np.flatnonzero((tele[C.ID_COL] == unit).to_numpy())
        length = int(rng.integers(6, 49))
        start = int(rng.integers(0, len(rows) - length))
        tele.iloc[rows[start:start + length], tele.columns.get_loc(col)] = np.nan
        blocks.append({"equipment_id": str(unit), "sensor": col, "hours": length})
    log["dropout_blocks"] = blocks

    # 3) invalid readings: sentinel temperatures, negative pressure, vibration spikes
    idx = rng.choice(n, min(150, n // 100), replace=False)
    tele.iloc[idx, tele.columns.get_loc("temperature_c")] = -999.0
    log["temperature_sentinel_-999"] = len(idx)
    idx = rng.choice(n, min(100, n // 100), replace=False)
    col = tele.columns.get_loc("pressure_kpa")
    tele.iloc[idx, col] = -np.abs(tele.iloc[idx, col].fillna(100.0)) * 0.1 - 1
    log["negative_pressure"] = len(idx)
    idx = rng.choice(n, min(120, n // 100), replace=False)
    col = tele.columns.get_loc("vibration_mm_s")
    tele.iloc[idx, col] = tele.iloc[idx, col].fillna(2.0) * 40 + 60
    log["vibration_spikes_out_of_range"] = len(idx)

    # 4) equipment_type recorded inconsistently with the registry
    idx = rng.choice(n, min(40, n // 100), replace=False)
    col = tele.columns.get_loc(C.TYPE_COL)
    tele.iloc[idx, col] = tele.iloc[idx, col].map(
        {"pump": "motor", "motor": "compressor", "compressor": "pump"})
    log["equipment_type_mismatches"] = len(idx)

    # 5) operating-hours counter reset on two units (e.g. hour-meter replacement)
    resets = []
    for unit in rng.choice(ids, min(2, len(ids)), replace=False):
        rows = np.flatnonzero((tele[C.ID_COL] == unit).to_numpy())
        cut = rows[int(len(rows) * rng.uniform(0.3, 0.7))]
        col = tele.columns.get_loc("operating_hours")
        base = tele.iloc[cut, col]
        tele.iloc[rows[rows >= cut], col] -= base
        resets.append(str(unit))
    log["operating_hours_counter_resets"] = resets

    # 6) exact duplicate rows
    dup = tele.iloc[rng.choice(n, min(300, n // 100), replace=False)]
    tele = pd.concat([tele, dup], ignore_index=True)
    log["duplicate_telemetry_rows"] = len(dup)
    tele = tele.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)

    # 7) duplicate failure-log entries
    dupf = failures.sample(min(3, len(failures)), random_state=int(rng.integers(1 << 30)))
    failures = pd.concat([failures, dupf], ignore_index=True).sort_values(
        ["failure_time", C.ID_COL]).reset_index(drop=True)
    log["duplicate_failure_records"] = len(dupf)
    return tele, failures, log


def generate_dataset(seed: int = C.SEED, units_per_type: int = C.UNITS_PER_TYPE,
                     start: str = C.SIM_START, end: str = C.SIM_END,
                     inject_issues: bool = True) -> dict[str, pd.DataFrame | dict]:
    """Generate the full synthetic dataset in memory (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="h")
    ambient_by_area = {a: _ambient_series(rng, index, off)
                       for a, off in zip(C.AREAS, (-2.0, 0.0, 1.5))}

    tele_parts, truth_parts, maint_rows, fail_rows, registry_rows = [], [], [], [], []
    for eq_type in C.EQUIPMENT_TYPES:
        spec = TYPE_SPECS[eq_type]
        for i in range(1, units_per_type + 1):
            eid = f"{spec.prefix}-{i:03d}"
            area = C.AREAS[int(rng.integers(len(C.AREAS)))]
            tele, maint, fails, truth, meta = _simulate_unit(
                rng, eid, eq_type, index, ambient_by_area[area])
            tele_parts.append(tele)
            truth_parts.append(truth)
            maint_rows += maint
            fail_rows += fails
            registry_rows.append({C.ID_COL: eid, C.TYPE_COL: eq_type, "area": area, **meta})

    telemetry = pd.concat(tele_parts, ignore_index=True)
    truth = pd.concat(truth_parts, ignore_index=True)
    registry = pd.DataFrame(registry_rows)
    maintenance = pd.DataFrame(
        maint_rows, columns=[C.ID_COL, "start_time", "end_time", "maintenance_type",
                             "downtime_hours"]).sort_values(["start_time", C.ID_COL]
                                                            ).reset_index(drop=True)
    failures = pd.DataFrame(fail_rows, columns=[C.ID_COL, "failure_time", "failure_mode"]
                            ).sort_values(["failure_time", C.ID_COL]).reset_index(drop=True)

    issues: dict = {}
    if inject_issues:
        telemetry, failures, issues = _inject_quality_issues(rng, telemetry, failures)
    return {"telemetry": telemetry, "registry": registry, "maintenance": maintenance,
            "failures": failures, "ground_truth": truth, "injected_issues": issues}


def write_dataset(data: dict) -> None:
    C.ensure_dirs()
    # mtime=0 keeps the gzip header free of the current time, so the same seed gives the
    # same file byte for byte
    gz = {"method": "gzip", "mtime": 0}
    data["telemetry"].to_csv(C.TELEMETRY_FILE, index=False, compression=gz)
    data["registry"].to_csv(C.REGISTRY_FILE, index=False)
    data["maintenance"].to_csv(C.MAINTENANCE_FILE, index=False)
    data["failures"].to_csv(C.FAILURES_FILE, index=False)
    data["ground_truth"].to_csv(C.GROUND_TRUTH_FILE, index=False, compression=gz)
    C.INJECTED_ISSUES_FILE.write_text(json.dumps(data["injected_issues"], indent=2))


if __name__ == "__main__":
    write_dataset(generate_dataset())
