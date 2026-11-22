"""Data validation and cleaning.

``validate_raw`` only *measures* problems and returns a findings dictionary.
``clean_telemetry`` / ``clean_failures`` apply the documented decisions.
Keeping the two apart makes every cleaning decision auditable and testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ measurements
def count_out_of_range(tele: pd.DataFrame) -> dict[str, int]:
    out = {}
    for col, (lo, hi) in C.VALID_RANGES.items():
        if col in tele:
            out[col] = int(((tele[col] < lo) | (tele[col] > hi)).sum())
    return out


def find_type_mismatches(tele: pd.DataFrame, registry: pd.DataFrame) -> pd.Series:
    reg_type = tele[C.ID_COL].map(registry.set_index(C.ID_COL)[C.TYPE_COL])
    return tele[C.TYPE_COL] != reg_type


def find_counter_resets(tele: pd.DataFrame) -> pd.DataFrame:
    """Rows where the operating-hours counter goes backwards within a unit."""
    s = tele.sort_values([C.ID_COL, C.TIME_COL])
    diff = s.groupby(C.ID_COL)["operating_hours"].diff()
    return s.loc[diff < 0, [C.ID_COL, C.TIME_COL, "operating_hours"]]


def coverage_gaps(tele: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    """Gaps > 1 h between consecutive readings, flagged if a maintenance outage explains them."""
    s = tele.drop_duplicates([C.ID_COL, C.TIME_COL]).sort_values([C.ID_COL, C.TIME_COL])
    prev = s.groupby(C.ID_COL)[C.TIME_COL].shift()
    gaps = s.loc[(s[C.TIME_COL] - prev) > pd.Timedelta(hours=1), [C.ID_COL, C.TIME_COL]].copy()
    gaps["gap_start"] = prev[gaps.index] + pd.Timedelta(hours=1)
    gaps = gaps.rename(columns={C.TIME_COL: "gap_end"})
    gaps["gap_hours"] = (gaps["gap_end"] - gaps["gap_start"]) / pd.Timedelta(hours=1)
    if gaps.empty:
        gaps["explained_by_maintenance"] = pd.Series(dtype=bool)
        return gaps
    m = maintenance[[C.ID_COL, "start_time", "end_time"]]
    merged = gaps.reset_index().merge(m, on=C.ID_COL, how="left")
    overlap = (merged["start_time"] <= merged["gap_start"]) & (merged["end_time"] >= merged["gap_end"])
    gaps["explained_by_maintenance"] = overlap.groupby(merged["index"]).any().reindex(gaps.index).to_numpy()
    return gaps.reset_index(drop=True)


def validate_raw(tele: pd.DataFrame, registry: pd.DataFrame, maintenance: pd.DataFrame,
                 failures: pd.DataFrame) -> dict:
    """Measure data-quality problems in the raw tables (no modification)."""
    findings: dict = {}
    findings["n_rows"] = int(len(tele))
    findings["n_units"] = int(tele[C.ID_COL].nunique())
    findings["dtypes"] = {c: str(t) for c, t in tele.dtypes.items()}
    findings["time_min"] = str(tele[C.TIME_COL].min())
    findings["time_max"] = str(tele[C.TIME_COL].max())

    findings["exact_duplicate_rows"] = int(tele.duplicated().sum())
    findings["duplicate_keys"] = int(tele.duplicated([C.ID_COL, C.TIME_COL]).sum())
    findings["missing_values"] = {c: int(v) for c, v in tele.isna().sum().items()}
    findings["out_of_range"] = count_out_of_range(tele)
    findings["equipment_type_mismatches"] = int(find_type_mismatches(tele, registry).sum())
    findings["unknown_equipment_ids"] = int((~tele[C.ID_COL].isin(registry[C.ID_COL])).sum())

    resets = find_counter_resets(tele.drop_duplicates())
    findings["operating_hours_resets"] = sorted(resets[C.ID_COL].unique().tolist())

    gaps = coverage_gaps(tele, maintenance)
    findings["coverage_gaps"] = int(len(gaps))
    findings["coverage_gaps_unexplained"] = int((~gaps["explained_by_maintenance"]).sum()) if len(gaps) else 0
    expected = len(pd.date_range(tele[C.TIME_COL].min(), tele[C.TIME_COL].max(), freq="h"))
    per_unit = tele.drop_duplicates([C.ID_COL, C.TIME_COL]).groupby(C.ID_COL).size() / expected
    findings["coverage_fraction_min"] = float(per_unit.min())
    findings["coverage_fraction_mean"] = float(per_unit.mean())

    # failure-label quality
    findings["failure_records"] = int(len(failures))
    findings["duplicate_failure_records"] = int(failures.duplicated().sum())
    f = failures.drop_duplicates()
    findings["failures_unknown_equipment"] = int((~f[C.ID_COL].isin(registry[C.ID_COL])).sum())
    corrective = maintenance[maintenance["maintenance_type"] == "corrective"]
    linked = f.merge(corrective, left_on=[C.ID_COL, "failure_time"],
                     right_on=[C.ID_COL, "start_time"], how="left")
    findings["failures_without_corrective_record"] = int(linked["start_time"].isna().sum())
    key = tele[[C.ID_COL, C.TIME_COL]].drop_duplicates()
    during = f.merge(key, left_on=[C.ID_COL, "failure_time"], right_on=[C.ID_COL, C.TIME_COL])
    findings["failures_with_reading_at_failure_time"] = int(len(during))
    per_unit_failures = f.groupby(C.ID_COL).size().reindex(registry[C.ID_COL]).fillna(0)
    findings["failures_per_unit"] = {
        "min": int(per_unit_failures.min()),
        "max": int(per_unit_failures.max()),
        "units_with_zero": int((~registry[C.ID_COL].isin(f[C.ID_COL])).sum()),
    }
    return findings


# ---------------------------------------------------------------------- cleaning
def repair_operating_hours(tele: pd.DataFrame) -> pd.DataFrame:
    """Re-base the counter after a reset so it is monotonic within each unit.

    Assumes ``tele`` is sorted by unit and time. After a backwards jump the counter
    is continued from the last value before the reset (+1 h for the reset hour).
    """
    out = tele.copy()
    diff = out.groupby(C.ID_COL)["operating_hours"].diff()
    drop = np.where(diff < 0, -diff + 1.0, 0.0)
    offset = pd.Series(drop, index=out.index).groupby(out[C.ID_COL]).cumsum()
    out["operating_hours"] = out["operating_hours"] + offset
    return out


def clean_telemetry(tele: pd.DataFrame, registry: pd.DataFrame,
                    ffill_limit: int = C.FFILL_LIMIT_HOURS) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning decisions. Returns the clean table and a log of actions."""
    log: dict = {}
    n0 = len(tele)
    df = tele.drop_duplicates()
    log["exact_duplicates_dropped"] = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates([C.ID_COL, C.TIME_COL], keep="first")
    log["conflicting_key_duplicates_dropped"] = n1 - len(df)
    n2 = len(df)
    df = df[df[C.ID_COL].isin(registry[C.ID_COL])]
    log["rows_unknown_equipment_dropped"] = n2 - len(df)
    df = df.sort_values([C.ID_COL, C.TIME_COL]).reset_index(drop=True)

    # the registry is the system of record for equipment type
    mism = find_type_mismatches(df, registry)
    df[C.TYPE_COL] = df[C.ID_COL].map(registry.set_index(C.ID_COL)[C.TYPE_COL])
    log["equipment_type_corrected"] = int(mism.sum())

    # out-of-range readings are sensor faults -> missing
    invalid = {}
    for col, (lo, hi) in C.VALID_RANGES.items():
        bad = (df[col] < lo) | (df[col] > hi)
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    log["out_of_range_set_missing"] = invalid

    df = repair_operating_hours(df)
    log["operating_hours_monotonic"] = bool(
        (df.groupby(C.ID_COL)["operating_hours"].diff().dropna() >= 0).all())

    # causal imputation: carry the last valid reading forward for at most `ffill_limit`
    # rows (hours). Never back-fill - that would use future information.
    cols = C.SENSORS + ["ambient_temp_c"]
    missing_before = df[cols].isna().sum()
    df[cols] = df.groupby(C.ID_COL)[cols].ffill(limit=ffill_limit)
    missing_after = df[cols].isna().sum()
    log["forward_filled"] = {c: int(missing_before[c] - missing_after[c]) for c in cols}
    log["still_missing_after_ffill"] = {c: int(missing_after[c]) for c in cols}
    log["rows_out"] = int(len(df))
    return df, log


def clean_failures(failures: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    f = failures.drop_duplicates()
    f = f[f[C.ID_COL].isin(registry[C.ID_COL])]
    return f.sort_values([C.ID_COL, "failure_time"]).reset_index(drop=True)


# ----------------------------------------------------------------------- report
def render_validation_report(findings: dict, clean_log: dict, label_stats: dict,
                             injected: dict | None = None) -> str:
    """Markdown report built only from measured numbers."""
    f, c = findings, clean_log
    mv = "\n".join(f"| `{k}` | {v:,} | {v / f['n_rows']:.2%} |" for k, v in f["missing_values"].items() if v)
    oor = "\n".join(f"| `{k}` | {v:,} | {C.VALID_RANGES[k][0]:g} to {C.VALID_RANGES[k][1]:g} |"
                    for k, v in f["out_of_range"].items())
    ff = "\n".join(f"| `{k}` | {c['forward_filled'][k]:,} | {c['still_missing_after_ffill'][k]:,} |"
                   for k in c["forward_filled"])
    lines = f"""# Data Validation Report

*Generated automatically by `scripts/run_pipeline.py` (stage `validate`). All counts are measured
from the files in `data/raw/`. The data is synthetic; the quality problems below were injected on
purpose by the generator so that this stage has realistic work to do.*

## 1. Shape and temporal coverage

| Item | Value |
|---|---|
| Telemetry rows (raw) | {f['n_rows']:,} |
| Equipment units | {f['n_units']} |
| First / last timestamp | {f['time_min']} / {f['time_max']} |
| Mean share of hours with a reading, per unit | {f['coverage_fraction_mean']:.1%} |
| Lowest unit coverage | {f['coverage_fraction_min']:.1%} |
| Gaps longer than 1 h | {f['coverage_gaps']:,} |
| Gaps **not** explained by a logged maintenance outage | {f['coverage_gaps_unexplained']:,} |

**Decision.** Gaps are equipment downtime (the unit is not running, so there is nothing to
measure). They are kept as gaps - no rows are invented. All rolling features use *time-based*
windows so a 24 h window always means 24 h of clock time, even across a gap.

## 2. Duplicates

| Check | Count | Decision |
|---|---|---|
| Exact duplicate telemetry rows | {f['exact_duplicate_rows']:,} | Dropped ({c['exact_duplicates_dropped']:,} removed) |
| Duplicate (equipment, timestamp) keys after that | {c['conflicting_key_duplicates_dropped']:,} | Keep first |
| Duplicate failure-log records | {f['duplicate_failure_records']:,} | Dropped - a duplicated failure would double-count events |

## 3. Missing values (raw)

| Column | Missing | Share |
|---|---|---|
{mv}

## 4. Invalid sensor readings

| Column | Out-of-range readings | Accepted range |
|---|---|---|
{oor}

**Decision.** Out-of-range values (for example the `-999` temperature sentinel, negative pressure,
vibration far above any plausible level) are treated as instrument or transmission faults and set
to missing. They are not clipped, because a clipped value would look like a real extreme reading.

## 5. Imputation

| Column | Filled by forward-fill (max {C.FFILL_LIMIT_HOURS} h) | Still missing afterwards |
|---|---|---|
{ff}

**Decision.** Short gaps are forward-filled within a unit (last known value), limited to
{C.FFILL_LIMIT_HOURS} hours. Back-fill and interpolation are deliberately not used because both use
future readings, which would leak information into features. Longer dropouts stay missing; rolling
statistics skip them and the model pipelines impute any remaining gaps with training-set medians.

## 6. Equipment-level consistency

| Check | Result | Decision |
|---|---|---|
| Telemetry rows whose `equipment_type` disagrees with the registry | {f['equipment_type_mismatches']:,} | Registry is the system of record; {c['equipment_type_corrected']:,} rows corrected |
| Telemetry rows for equipment not in the registry | {f['unknown_equipment_ids']:,} | Would be dropped |
| Units with an operating-hours counter reset | {len(f['operating_hours_resets'])} ({', '.join(f['operating_hours_resets']) or 'none'}) | Counter re-based so it is monotonic (check passed: {c['operating_hours_monotonic']}) |

## 7. Label quality

| Check | Result |
|---|---|
| Failure records (raw / after de-duplication) | {f['failure_records']:,} / {f['failure_records'] - f['duplicate_failure_records']:,} |
| Failures for unknown equipment | {f['failures_unknown_equipment']} |
| Failures without a matching corrective-maintenance record | {f['failures_without_corrective_record']} |
| Failures that still have a sensor reading at the failure hour | {f['failures_with_reading_at_failure_time']} |
| Failures per unit (min / max) | {f['failures_per_unit']['min']} / {f['failures_per_unit']['max']} |
| Units with no failure in the period | {f['failures_per_unit']['units_with_zero']} |

The target `{C.TARGET}` is 1 when a failure is logged for the unit within the next
{C.HORIZON_HOURS} hours. The final {C.HORIZON_HOURS} hours of the record are excluded from modelling
because their outcome is not yet observable (right-censoring).

## 8. Class imbalance

| Item | Value |
|---|---|
| Modelling rows | {label_stats['n_rows']:,} |
| Positive rows | {label_stats['n_positive']:,} |
| Positive rate | {label_stats['positive_rate']:.2%} |
| Negative : positive ratio | {label_stats['neg_pos_ratio']:.0f} : 1 |

**Decision.** Accuracy is not used as a selection metric - predicting "no failure" everywhere would
score about {1 - label_stats['positive_rate']:.1%}. Models are compared on precision-recall AUC, with
class weighting during training and thresholds chosen on a separate validation period.
"""
    if injected:
        dropout_hours = sum(b["hours"] for b in injected.get("dropout_blocks", []))
        lines += f"""
## 9. Cross-check against what the generator injected

Because the data is synthetic, detection can be checked against the generator's own log
(`data/raw/injected_data_quality_issues.json`).

| Issue | Injected | Detected |
|---|---|---|
| Exact duplicate rows | {injected.get('duplicate_telemetry_rows')} | {f['exact_duplicate_rows']} |
| `-999` temperature sentinels | {injected.get('temperature_sentinel_-999')} | {f['out_of_range']['temperature_c']} |
| Negative pressure readings | {injected.get('negative_pressure')} | {f['out_of_range']['pressure_kpa']} |
| Out-of-range vibration spikes | {injected.get('vibration_spikes_out_of_range')} | {f['out_of_range']['vibration_mm_s']} |
| Equipment-type mismatches | {injected.get('equipment_type_mismatches')} | {f['equipment_type_mismatches']} |
| Operating-hours counter resets (units) | {len(injected.get('operating_hours_counter_resets', []))} | {len(f['operating_hours_resets'])} |
| Duplicate failure records | {injected.get('duplicate_failure_records')} | {f['duplicate_failure_records']} |
| Sensor dropout blocks | {len(injected.get('dropout_blocks', []))} blocks, {dropout_hours} sensor-hours | counted within missing values |

Small differences are expected where two injected problems landed on the same row (for example a
duplicated row that also carries a sentinel value is counted twice in the raw file).
"""
    return lines
