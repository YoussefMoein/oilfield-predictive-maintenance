"""Shared fixtures. A small simulated fleet keeps the whole suite fast (well under a minute)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm import config as C  # noqa: E402
from pdm import data_generation, features, validation  # noqa: E402

SMALL_START, SMALL_END = "2024-01-01 00:00", "2024-06-30 23:00"
SMALL_TRAIN_END, SMALL_VAL_END = "2024-04-15 23:00", "2024-05-20 23:00"


@pytest.fixture(scope="session")
def small_raw():
    return data_generation.generate_dataset(seed=11, units_per_type=3, start=SMALL_START, end=SMALL_END)


@pytest.fixture(scope="session")
def small_clean(small_raw):
    tele, log = validation.clean_telemetry(small_raw["telemetry"], small_raw["registry"])
    fails = validation.clean_failures(small_raw["failures"], small_raw["registry"])
    return {"telemetry": tele, "failures": fails, "log": log,
            "maintenance": small_raw["maintenance"], "registry": small_raw["registry"]}


@pytest.fixture(scope="session")
def small_condition_model(small_clean):
    return features.fit_condition_model(small_clean["telemetry"], fit_end=SMALL_TRAIN_END)


@pytest.fixture(scope="session")
def small_table(small_clean, small_condition_model):
    return features.build_model_table(small_clean["telemetry"], small_clean["maintenance"],
                                      small_clean["failures"], small_clean["registry"],
                                      small_condition_model)


@pytest.fixture()
def toy_classification():
    """Separable-ish binary problem shaped like the model table."""
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.7, size=n) > 1.6).astype(int)
    X.loc[rng.choice(n, 30, replace=False), "f2"] = np.nan
    return X.assign(**{C.TARGET: y})
