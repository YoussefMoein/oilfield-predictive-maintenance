"""Evaluation utilities: row-level metrics, thresholds, event-level view, uncertainty.

Two views are reported because they answer different questions:

* row level  - "of all unit-hours flagged, how many preceded a failure?"
* event level - "of all failures, how many were flagged beforehand, how early,
  and how much unnecessary reviewing did that cost?"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

from . import config as C


# ----------------------------------------------------------------- row level
def confusion_counts(y_true, y_pred) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def threshold_metrics(y_true, scores, threshold: float) -> dict:
    """Precision, recall, F1 and confusion counts at a decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (c["tp"] + c["tn"]) / len(y_true),
            "alert_rate": float(y_pred.mean()), **c}


def ranking_metrics(y_true, scores) -> dict:
    """Threshold-free metrics. PR-AUC is the headline metric under class imbalance."""
    y_true = np.asarray(y_true)
    return {"pr_auc": float(average_precision_score(y_true, scores)),
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "prevalence": float(y_true.mean())}


def threshold_max_f1(y_true, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def threshold_for_recall(y_true, scores, target_recall: float) -> float:
    """Highest threshold whose recall is still at least ``target_recall``."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.flatnonzero(recall[:-1] >= target_recall)
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


# --------------------------------------------------------------- event level
def event_level_metrics(scored: pd.DataFrame, failures: pd.DataFrame, threshold: float,
                        horizon_hours: int = C.HORIZON_HOURS, score_col: str = "score") -> dict:
    """Failure-level detection, lead time and false-alert burden.

    ``scored`` needs equipment_id, timestamp, the score column and the target.
    A failure counts as *flagged* if at least one reading in the ``horizon_hours``
    before it scored at or above the threshold. Lead time is measured from the
    earliest such reading. A *false-alert unit-day* is a calendar day on which a
    unit raised at least one alert although no failure followed within the horizon.
    """
    start, end = scored[C.TIME_COL].min(), scored[C.TIME_COL].max()
    f = failures[(failures["failure_time"] > start) &
                 (failures["failure_time"] <= end + pd.Timedelta(hours=1))].copy()
    alerts = scored[scored[score_col] >= threshold]

    rows = []
    for fid, ftime, fmode in f[[C.ID_COL, "failure_time", "failure_mode"]].itertuples(index=False):
        window = alerts[(alerts[C.ID_COL] == fid) & (alerts[C.TIME_COL] < ftime) &
                        (alerts[C.TIME_COL] >= ftime - pd.Timedelta(hours=horizon_hours))]
        observed = scored[(scored[C.ID_COL] == fid) & (scored[C.TIME_COL] < ftime) &
                          (scored[C.TIME_COL] >= ftime - pd.Timedelta(hours=horizon_hours))]
        lead = ((ftime - window[C.TIME_COL].min()) / pd.Timedelta(hours=1)) if len(window) else np.nan
        rows.append({C.ID_COL: fid, "failure_time": ftime, "failure_mode": fmode,
                     "flagged": bool(len(window)), "lead_time_hours": lead,
                     "readings_in_window": len(observed)})
    events = pd.DataFrame(rows, columns=[C.ID_COL, "failure_time", "failure_mode", "flagged",
                                         "lead_time_hours", "readings_in_window"])

    day = scored[C.TIME_COL].dt.floor("D")
    unit_days = scored.assign(_day=day).groupby([C.ID_COL, "_day"]).agg(
        alert=(score_col, lambda s: bool((s >= threshold).any())), positive=(C.TARGET, "max"))
    false_days = int((unit_days["alert"] & (unit_days["positive"] == 0)).sum())
    alert_days = int(unit_days["alert"].sum())
    months = max((end - start) / pd.Timedelta(days=30.4375), 1e-9)
    n_units = scored[C.ID_COL].nunique()

    flagged = events["flagged"] if len(events) else pd.Series(dtype=bool)
    return {
        "events": events,
        "n_failures": int(len(events)),
        "failures_flagged": int(flagged.sum()),
        "event_recall": float(flagged.mean()) if len(events) else float("nan"),
        "median_lead_time_hours": float(events["lead_time_hours"].median()) if flagged.any() else float("nan"),
        "alert_unit_days": alert_days,
        "false_alert_unit_days": false_days,
        "unit_days_total": int(len(unit_days)),
        "false_alert_days_per_unit_month": false_days / (n_units * months),
        "unit_day_precision": (alert_days - false_days) / alert_days if alert_days else float("nan"),
    }


# -------------------------------------------------------------- uncertainty
def cluster_bootstrap_ci(df: pd.DataFrame, score_col: str, metric=average_precision_score,
                         n_boot: int = 300, seed: int = C.SEED, alpha: float = 0.05) -> dict:
    """Percentile CI that resamples whole units (rows within a unit are not independent)."""
    rng = np.random.default_rng(seed)
    groups = {k: (g[C.TARGET].to_numpy(), g[score_col].to_numpy())
              for k, g in df.groupby(C.ID_COL)}
    ids = list(groups)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(ids), len(ids), replace=True)
        y = np.concatenate([groups[ids[i]][0] for i in pick])
        s = np.concatenate([groups[ids[i]][1] for i in pick])
        if y.min() == y.max():
            continue
        stats.append(metric(y, s))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": float(metric(df[C.TARGET], df[score_col])), "ci_low": float(lo),
            "ci_high": float(hi), "n_boot": len(stats)}


def metrics_by_group(df: pd.DataFrame, score_col: str, group_col: str, threshold: float) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col):
        if g[C.TARGET].nunique() < 2:
            continue
        rows.append({group_col: key, "rows": len(g), "positive_rate": g[C.TARGET].mean(),
                     **{k: v for k, v in ranking_metrics(g[C.TARGET], g[score_col]).items()
                        if k != "prevalence"},
                     **{k: v for k, v in threshold_metrics(g[C.TARGET], g[score_col], threshold).items()
                        if k in ("precision", "recall", "f1")}})
    return pd.DataFrame(rows)
