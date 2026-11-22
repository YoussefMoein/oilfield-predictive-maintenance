"""Central configuration: paths, simulation constants and modelling choices.

Every tunable decision that affects results lives here so that the pipeline,
the notebook and the tests all read the same values.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = ROOT / "figures"

TELEMETRY_FILE = DATA_RAW / "telemetry.csv.gz"
REGISTRY_FILE = DATA_RAW / "equipment_registry.csv"
MAINTENANCE_FILE = DATA_RAW / "maintenance_log.csv"
FAILURES_FILE = DATA_RAW / "failure_log.csv"
GROUND_TRUTH_FILE = DATA_RAW / "simulation_ground_truth.csv.gz"
INJECTED_ISSUES_FILE = DATA_RAW / "injected_data_quality_issues.json"

CLEAN_TELEMETRY_FILE = DATA_PROCESSED / "telemetry_clean.parquet"
FEATURES_FILE = DATA_PROCESSED / "model_table.parquet"

# ---------------------------------------------------------------------- simulation
SEED = 42
SIM_START = "2024-01-01 00:00"
SIM_END = "2024-12-31 23:00"
UNITS_PER_TYPE = 20
AREAS = ("Area North", "Area Central", "Area South")  # fictional

ID_COL = "equipment_id"
TIME_COL = "timestamp"
TYPE_COL = "equipment_type"
EQUIPMENT_TYPES = ("pump", "motor", "compressor")

SENSORS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_kpa",
    "rotational_speed_rpm",
    "load_pct",
]
CONTEXT_COLS = ["ambient_temp_c", "operating_hours"]

# Physically plausible limits used by validation. Readings outside these limits
# are treated as sensor/transmission faults (set to missing), never as real states.
VALID_RANGES = {
    "temperature_c": (-50.0, 250.0),
    "vibration_mm_s": (0.0, 40.0),
    "pressure_kpa": (0.0, 20000.0),
    "rotational_speed_rpm": (0.0, 6000.0),
    "load_pct": (0.0, 110.0),
    "ambient_temp_c": (-60.0, 50.0),
}

# ------------------------------------------------------------------------ modelling
HORIZON_HOURS = 72          # label: failure logged within the next 72 h
FFILL_LIMIT_HOURS = 6       # causal forward-fill limit for short sensor gaps
WARMUP_HOURS = 24           # rows with < 24 h of unit history are not modelled

ROLLING_WINDOWS_H = (6, 24, 72, 168)
ROLLING_MIN_PERIODS = {6: 3, 24: 12, 72: 24, 168: 48}

# Chronological split (inclusive end timestamps). A purge gap equal to the label
# horizon is removed before each boundary so no training label "sees" the next period.
TRAIN_END = "2024-07-31 23:00"
VAL_END = "2024-09-30 23:00"

# Walk-forward folds used for model selection (all before the test period). The last
# pair (TRAIN_END -> VAL_END) is the train / validation split itself.
WALK_FORWARD_BOUNDARIES = ("2024-03-31 23:00", "2024-05-31 23:00", TRAIN_END, VAL_END)
REPLICATION_SEEDS = (7, 2025, 31337)   # fresh simulated fleets for out-of-development checks

TARGET = "failure_within_72h"
RECALL_TARGET_MEDIUM = 0.80  # validation recall used to set the "Medium" risk threshold
TOP_K_REVIEW = 5             # units surfaced per weekly snapshot in the ranking back-test


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
