# Data Dictionary

> **All data in this project is synthetic.** It is produced by `src/pdm/data_generation.py` and
> does not describe real equipment, a real site or any company's records. Equipment IDs and area
> names are fictional.

## 1. Raw files (`data/raw/`)

### `telemetry.csv.gz` - one row per unit per running hour

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | datetime | hourly | Reading time. No row exists while a unit is down. |
| `equipment_id` | string | - | `PMP-###` pump, `MTR-###` motor, `CMP-###` compressor. |
| `equipment_type` | string | - | `pump`, `motor` or `compressor`. A few rows are deliberately wrong (see validation report); the registry is the system of record. |
| `temperature_c` | float | °C | Bearing / casing temperature. Responds to load, ambient temperature and degradation. |
| `vibration_mm_s` | float | mm/s RMS | Overall vibration velocity. |
| `pressure_kpa` | float | kPa | Primary pressure measurement: discharge pressure for pumps and compressors, bearing lube-oil supply pressure for motors. Scales differ by type by design. |
| `rotational_speed_rpm` | float | rpm | Shaft speed (varies with load). |
| `load_pct` | float | % | Operating load relative to rated duty. |
| `ambient_temp_c` | float | °C | Outdoor temperature for the unit's area (seasonal + daily cycle). |
| `operating_hours` | float | h | Cumulative running-hours counter. Two units contain a counter reset on purpose. |

### `equipment_registry.csv` - one row per unit

| Column | Description |
|---|---|
| `equipment_id` | Primary key. |
| `equipment_type` | Authoritative equipment type. |
| `area` | Fictional operating area (`Area North`, `Area Central`, `Area South`). |
| `initial_operating_hours` | Counter value at the start of the record. |
| `rated_power_kw` | Nameplate power (descriptive only; not used as a model feature). |

### `maintenance_log.csv` - one row per work order

| Column | Description |
|---|---|
| `equipment_id` | Unit worked on. |
| `start_time`, `end_time` | Outage start and return to service. |
| `maintenance_type` | `preventive` (time-based) or `corrective` (after a failure). |
| `downtime_hours` | Length of the outage. |

### `failure_log.csv` - one row per logged failure

| Column | Description |
|---|---|
| `equipment_id` | Unit that failed. |
| `failure_time` | Hour at which the unit stopped. |
| `failure_mode` | `bearing_wear`, `overheating`, `pressure_loss` or `sudden_no_precursor`. Known only after the event - **never a model input**. |

### `simulation_ground_truth.csv.gz` - diagnostics only

| Column | Description |
|---|---|
| `latent_wear` | The simulator's hidden degradation state. |
| `latent_mode` | The hidden dominant degradation mode of the current wear cycle. |

This file would not exist for real equipment. It is used only to sanity-check the finished model
(for example figure 17) and is excluded from features by construction and by a unit test.

### `injected_data_quality_issues.json`

The generator's own log of the data-quality problems it injected, used to cross-check validation.

## 2. Model table (`data/processed/model_table.parquet`)

One row per unit-hour that can be modelled (after a 72 h warm-up and with the final 72 h removed).

### Target and bookkeeping columns (not features)

| Column | Description |
|---|---|
| `failure_within_72h` | **Target.** 1 if a failure is logged for the unit in (t, t + 72 h]. |
| `hours_to_failure` | Hours to the unit's next logged failure (analysis only). |
| `next_failure_mode` | Mode of that next failure (analysis only). |
| `equipment_id`, `timestamp`, `equipment_type`, `area`, `operating_hours` | Identifiers / context. Excluded from features. |

### Features used by the models (47)

For each sensor `S` in {`temperature_c`, `vibration_mm_s`, `pressure_kpa`, `rotational_speed_rpm`}.
"Adjusted" means the unit's fitted response to load and ambient temperature has been removed;
"robust SD" is the unit's own historical IQR / 1.349. All values are clipped to ±8.

| Feature pattern | Count | Meaning |
|---|---|---|
| `S_z_now` | 4 | Current adjusted reading, in robust SDs from the unit's historical median. |
| `S_z_mean_{6,24,72,168}h` | 16 | Rolling mean of the adjusted reading over the last 6 h / 24 h / 72 h / 7 d, same scale. |
| `S_relstd_{24,72}h` | 8 | Rolling standard deviation over the window, relative to the unit's robust SD (signal instability). |
| `S_z_trend_24h_vs_168h` | 4 | Last-day mean minus last-week mean (positive = rising). |
| `S_z_change_24h` | 4 | Last-6 h mean minus the same quantity exactly 24 h earlier. |

| Other features | Meaning |
|---|---|
| `load_pct`, `load_pct_mean_24h` | Current and last-day load. |
| `ambient_temp_c`, `ambient_temp_c_mean_24h` | Current and last-day outdoor temperature. |
| `hours_since_last_maintenance` | Hours since the last *completed* work order of any type (capped at 120 days). |
| `hours_since_last_preventive` | Same, preventive work orders only (capped at 120 days). |
| `maintenance_history_censored` | 1 while no work order has been logged yet for the unit, so the two values above are lower bounds. |
| `failures_last_90d` | Failures logged for the unit in the trailing 90 days. |
| `type_pump`, `type_motor`, `type_compressor` | Equipment type indicators. |

### Ablation-only columns (excluded from the final models)

| Column pattern | Meaning |
|---|---|
| `temperature_c`, `vibration_mm_s`, `pressure_kpa`, `rotational_speed_rpm` | Raw readings in engineering units. |
| `S_abs_mean_{24,168}h`, `S_abs_std_24h` | Rolling statistics of raw readings. |

## 3. Outputs

| File | Contents |
|---|---|
| `reports/metrics.json` | Every number quoted in the reports. |
| `reports/tables/risk_ranking.csv` | Rank, unit, risk score, risk category, sensor-trend summary, suggested review priority. |
| `reports/tables/*.csv` | Model comparison, per-type and per-mode breakdowns, tuning log, ablation, grouped CV, replication, back-test. |
| `models/*.joblib` | Fitted estimators (regenerated by the pipeline; not committed). |
