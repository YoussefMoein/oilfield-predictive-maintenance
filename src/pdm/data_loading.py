"""Loading of the raw files with explicit dtypes and basic schema checks."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C

TELEMETRY_COLUMNS = [C.TIME_COL, C.ID_COL, C.TYPE_COL, *C.SENSORS, *C.CONTEXT_COLS]
REGISTRY_COLUMNS = [C.ID_COL, C.TYPE_COL, "area", "initial_operating_hours", "rated_power_kw"]
MAINTENANCE_COLUMNS = [C.ID_COL, "start_time", "end_time", "maintenance_type", "downtime_hours"]
FAILURE_COLUMNS = [C.ID_COL, "failure_time", "failure_mode"]


class SchemaError(ValueError):
    """Raised when a raw file does not have the expected columns."""


def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(f"{name} is missing columns: {missing}")


def load_telemetry(path: Path = C.TELEMETRY_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[C.TIME_COL])
    _check_columns(df, TELEMETRY_COLUMNS, "telemetry")
    return df[TELEMETRY_COLUMNS]


def load_registry(path: Path = C.REGISTRY_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check_columns(df, REGISTRY_COLUMNS, "equipment_registry")
    return df[REGISTRY_COLUMNS]


def load_maintenance(path: Path = C.MAINTENANCE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start_time", "end_time"])
    _check_columns(df, MAINTENANCE_COLUMNS, "maintenance_log")
    return df[MAINTENANCE_COLUMNS]


def load_failures(path: Path = C.FAILURES_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["failure_time"])
    _check_columns(df, FAILURE_COLUMNS, "failure_log")
    return df[FAILURE_COLUMNS]


def load_ground_truth(path: Path = C.GROUND_TRUTH_FILE) -> pd.DataFrame:
    """Hidden simulator state. Diagnostics only - never a model input."""
    return pd.read_csv(path, parse_dates=[C.TIME_COL])


def load_raw() -> dict[str, pd.DataFrame]:
    return {"telemetry": load_telemetry(), "registry": load_registry(),
            "maintenance": load_maintenance(), "failures": load_failures()}
