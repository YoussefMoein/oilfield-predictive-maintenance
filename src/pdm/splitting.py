"""Train / validation / test separation that respects time and equipment structure."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config as C


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("validation", self.val), ("test", self.test)):
            rows.append({"split": name, "start": part[C.TIME_COL].min(), "end": part[C.TIME_COL].max(),
                         "rows": len(part), "positive_rows": int(part[C.TARGET].sum()),
                         "positive_rate": part[C.TARGET].mean(),
                         "units": part[C.ID_COL].nunique()})
        return pd.DataFrame(rows)


def temporal_split(df: pd.DataFrame, train_end: str = C.TRAIN_END, val_end: str = C.VAL_END,
                   purge_hours: int = C.HORIZON_HOURS) -> TemporalSplit:
    """Chronological split with a purge gap before each boundary.

    A row at time ``t`` carries a label that depends on events up to ``t + horizon``.
    Rows in the last ``purge_hours`` before a boundary therefore have labels determined
    by the *next* period; they are removed so no information crosses the boundary.
    """
    train_end_ts, val_end_ts = pd.Timestamp(train_end), pd.Timestamp(val_end)
    purge = pd.Timedelta(hours=purge_hours)
    t = df[C.TIME_COL]
    train = df[t <= train_end_ts - purge]
    val = df[(t > train_end_ts) & (t <= val_end_ts - purge)]
    test = df[t > val_end_ts]
    return TemporalSplit(train.copy(), val.copy(), test.copy())


def walk_forward_folds(df: pd.DataFrame, boundaries: tuple[str, ...] = C.WALK_FORWARD_BOUNDARIES,
                       purge_hours: int = C.HORIZON_HOURS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds: train on everything before a boundary, validate on the next block.

    ``boundaries`` are inclusive end timestamps b0 < b1 < ... < bk. Fold i trains on rows up to
    b(i) and validates on (b(i), b(i+1)], each with the purge gap removed before its end.
    With the default boundaries the last fold equals the train / validation split.
    """
    ts = [pd.Timestamp(b) for b in boundaries]
    purge = pd.Timedelta(hours=purge_hours)
    t = df[C.TIME_COL]
    folds = []
    for start, end in zip(ts[:-1], ts[1:]):
        folds.append((df[t <= start - purge], df[(t > start) & (t <= end - purge)]))
    return folds


def equipment_group_folds(df: pd.DataFrame, n_splits: int = 5):
    """Yield (train_idx, test_idx) with every unit entirely in one fold.

    Answers a different question from the temporal split: does the model transfer
    to equipment it has never seen?
    """
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(df, df[C.TARGET], groups=df[C.ID_COL])
