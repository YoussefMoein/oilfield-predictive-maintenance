# Data Validation Report

*Generated automatically by `scripts/run_pipeline.py` (stage `validate`). All counts are measured
from the files in `data/raw/`. The data is synthetic; the quality problems below were injected on
purpose by the generator so that this stage has realistic work to do.*

## 1. Shape and temporal coverage

| Item | Value |
|---|---|
| Telemetry rows (raw) | 519,474 |
| Equipment units | 60 |
| First / last timestamp | 2024-01-01 00:00:00 / 2024-12-31 23:00:00 |
| Mean share of hours with a reading, per unit | 98.5% |
| Lowest unit coverage | 96.3% |
| Gaps longer than 1 h | 292 |
| Gaps **not** explained by a logged maintenance outage | 0 |

**Decision.** Gaps are equipment downtime (the unit is not running, so there is nothing to
measure). They are kept as gaps - no rows are invented. All rolling features use *time-based*
windows so a 24 h window always means 24 h of clock time, even across a gap.

## 2. Duplicates

| Check | Count | Decision |
|---|---|---|
| Exact duplicate telemetry rows | 300 | Dropped (300 removed) |
| Duplicate (equipment, timestamp) keys after that | 0 | Keep first |
| Duplicate failure-log records | 3 | Dropped - a duplicated failure would double-count events |

## 3. Missing values (raw)

| Column | Missing | Share |
|---|---|---|
| `temperature_c` | 2,203 | 0.42% |
| `vibration_mm_s` | 2,224 | 0.43% |
| `pressure_kpa` | 2,176 | 0.42% |
| `rotational_speed_rpm` | 2,258 | 0.43% |
| `load_pct` | 2,077 | 0.40% |

## 4. Invalid sensor readings

| Column | Out-of-range readings | Accepted range |
|---|---|---|
| `temperature_c` | 150 | -50 to 250 |
| `vibration_mm_s` | 120 | 0 to 40 |
| `pressure_kpa` | 100 | 0 to 20000 |
| `rotational_speed_rpm` | 0 | 0 to 6000 |
| `load_pct` | 0 | 0 to 110 |
| `ambient_temp_c` | 0 | -60 to 50 |

**Decision.** Out-of-range values (for example the `-999` temperature sentinel, negative pressure,
vibration far above any plausible level) are treated as instrument or transmission faults and set
to missing. They are not clipped, because a clipped value would look like a real extreme reading.

## 5. Imputation

| Column | Filled by forward-fill (max 6 h) | Still missing afterwards |
|---|---|---|
| `temperature_c` | 2,256 | 96 |
| `vibration_mm_s` | 2,223 | 121 |
| `pressure_kpa` | 2,193 | 82 |
| `rotational_speed_rpm` | 2,105 | 152 |
| `load_pct` | 2,076 | 0 |
| `ambient_temp_c` | 0 | 0 |

**Decision.** Short gaps are forward-filled within a unit (last known value), limited to
6 hours. Back-fill and interpolation are deliberately not used because both use
future readings, which would leak information into features. Longer dropouts stay missing; rolling
statistics skip them and the model pipelines impute any remaining gaps with training-set medians.

## 6. Equipment-level consistency

| Check | Result | Decision |
|---|---|---|
| Telemetry rows whose `equipment_type` disagrees with the registry | 40 | Registry is the system of record; 40 rows corrected |
| Telemetry rows for equipment not in the registry | 0 | Would be dropped |
| Units with an operating-hours counter reset | 2 (CMP-005, PMP-008) | Counter re-based so it is monotonic (check passed: True) |

## 7. Label quality

| Check | Result |
|---|---|
| Failure records (raw / after de-duplication) | 156 / 153 |
| Failures for unknown equipment | 0 |
| Failures without a matching corrective-maintenance record | 0 |
| Failures that still have a sensor reading at the failure hour | 0 |
| Failures per unit (min / max) | 0 / 6 |
| Units with no failure in the period | 3 |

The target `failure_within_72h` is 1 when a failure is logged for the unit within the next
72 hours. The final 72 hours of the record are excluded from modelling
because their outcome is not yet observable (right-censoring).

## 8. Class imbalance

| Item | Value |
|---|---|
| Modelling rows | 510,593 |
| Positive rows | 11,004 |
| Positive rate | 2.16% |
| Negative : positive ratio | 45 : 1 |

**Decision.** Accuracy is not used as a selection metric - predicting "no failure" everywhere would
score about 97.8%. Models are compared on precision-recall AUC, with
class weighting during training and thresholds chosen on a separate validation period.

## 9. Cross-check against what the generator injected

Because the data is synthetic, detection can be checked against the generator's own log
(`data/raw/injected_data_quality_issues.json`).

| Issue | Injected | Detected |
|---|---|---|
| Exact duplicate rows | 300 | 300 |
| `-999` temperature sentinels | 150 | 150 |
| Negative pressure readings | 100 | 100 |
| Out-of-range vibration spikes | 120 | 120 |
| Equipment-type mismatches | 40 | 40 |
| Operating-hours counter resets (units) | 2 | 2 |
| Duplicate failure records | 3 | 3 |
| Sensor dropout blocks | 18 blocks, 558 sensor-hours | counted within missing values |

Small differences are expected where two injected problems landed on the same row (for example a
duplicated row that also carries a sentinel value is counted twice in the raw file).
