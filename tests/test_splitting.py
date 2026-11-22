import pandas as pd

from conftest import SMALL_TRAIN_END, SMALL_VAL_END
from pdm import config as C
from pdm import splitting


def test_temporal_split_is_chronological_and_disjoint(small_table):
    sp = splitting.temporal_split(small_table, SMALL_TRAIN_END, SMALL_VAL_END)
    assert len(sp.train) and len(sp.val) and len(sp.test)
    assert sp.train[C.TIME_COL].max() < sp.val[C.TIME_COL].min() < sp.val[C.TIME_COL].max() \
        < sp.test[C.TIME_COL].min()
    keys = [set(map(tuple, p[[C.ID_COL, C.TIME_COL]].to_numpy())) for p in (sp.train, sp.val, sp.test)]
    assert not (keys[0] & keys[1]) and not (keys[1] & keys[2]) and not (keys[0] & keys[2])


def test_purge_gap_keeps_training_labels_inside_training_period(small_table):
    sp = splitting.temporal_split(small_table, SMALL_TRAIN_END, SMALL_VAL_END)
    horizon = pd.Timedelta(hours=C.HORIZON_HOURS)
    # the label of the last training row is fully determined before the validation period begins
    assert sp.train[C.TIME_COL].max() + horizon <= pd.Timestamp(SMALL_TRAIN_END)
    assert sp.val[C.TIME_COL].max() + horizon <= pd.Timestamp(SMALL_VAL_END)
    assert sp.val[C.TIME_COL].min() > pd.Timestamp(SMALL_TRAIN_END)
    # positives in training never refer to a failure that happens in the validation period
    pos = sp.train[sp.train[C.TARGET] == 1]
    fail_time = pos[C.TIME_COL] + pd.to_timedelta(pos["hours_to_failure"], unit="h")
    assert (fail_time <= pd.Timestamp(SMALL_TRAIN_END)).all()


def test_split_summary(small_table):
    summary = splitting.temporal_split(small_table, SMALL_TRAIN_END, SMALL_VAL_END).summary()
    assert summary["split"].tolist() == ["train", "validation", "test"]
    assert summary["rows"].sum() < len(small_table)          # purged rows are really gone
    assert ((summary["positive_rate"] >= 0) & (summary["positive_rate"] < 0.5)).all()


def test_walk_forward_folds_expand_and_never_look_ahead(small_table):
    bounds = ("2024-02-29 23:00", "2024-03-31 23:00", SMALL_TRAIN_END, SMALL_VAL_END)
    folds = splitting.walk_forward_folds(small_table, bounds)
    assert len(folds) == 3
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes) and sizes[0] > 0
    for tr, va in folds:
        assert tr[C.TIME_COL].max() + pd.Timedelta(hours=C.HORIZON_HOURS) < va[C.TIME_COL].min()
    last_train, last_val = folds[-1]
    sp = splitting.temporal_split(small_table, SMALL_TRAIN_END, SMALL_VAL_END)
    assert len(last_train) == len(sp.train) and len(last_val) == len(sp.val)


def test_equipment_group_folds_keep_units_together(small_table):
    seen = set()
    for tr_idx, te_idx in splitting.equipment_group_folds(small_table, n_splits=3):
        tr_units = set(small_table.iloc[tr_idx][C.ID_COL])
        te_units = set(small_table.iloc[te_idx][C.ID_COL])
        assert not tr_units & te_units
        seen |= te_units
    assert seen == set(small_table[C.ID_COL])
