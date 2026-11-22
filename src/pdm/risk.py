"""Equipment risk ranking for inspection prioritisation.

Everything here is an *analytical indicator*. A risk category is a statement
about how a model score compares with thresholds chosen on historical data. It
is not a safety assessment, a diagnosis or a maintenance instruction; qualified
personnel decide what, if anything, to do with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CATEGORIES = ["Low", "Medium", "High"]
REVIEW_PRIORITY = {
    "High": "Priority 1 - suggest reviewing first",
    "Medium": "Priority 2 - suggest reviewing when scheduling allows",
    "Low": "Routine monitoring",
}
SENSOR_LABELS = {
    "temperature_c": ("temperature", "°C"),
    "vibration_mm_s": ("vibration", "mm/s"),
    "pressure_kpa": ("pressure", "kPa"),
    "rotational_speed_rpm": ("speed", "rpm"),
}


def categorize(scores, threshold_medium: float, threshold_high: float) -> np.ndarray:
    """Map scores to Low / Medium / High. Boundaries are inclusive on the upper category."""
    if not threshold_medium < threshold_high:
        raise ValueError("threshold_medium must be strictly below threshold_high")
    scores = np.asarray(scores, dtype=float)
    return np.select([scores >= threshold_high, scores >= threshold_medium],
                     ["High", "Medium"], default="Low")


def describe_sensor_trends(row: pd.Series, min_abs_dev: float = 1.0, max_items: int = 2) -> str:
    """Plain-language summary of the sensors furthest from the unit's own baseline."""
    items = []
    for sensor, (label, unit) in SENSOR_LABELS.items():
        dev = row.get(f"{sensor}_z_mean_24h", np.nan)
        trend = row.get(f"{sensor}_z_trend_24h_vs_168h", np.nan)
        if pd.isna(dev) or abs(dev) < min_abs_dev:
            continue
        direction = "above" if dev > 0 else "below"
        text = f"{label} {abs(dev):.1f} SD {direction} unit baseline"
        if not pd.isna(trend):
            state = "rising" if trend > 0.5 else "falling" if trend < -0.5 else "steady"
            text += f", {state} vs last 7 days ({trend:+.1f} SD)"
        items.append((abs(dev), text))
    if not items:
        return "no sensor more than 1 SD from unit baseline"
    return "; ".join(t for _, t in sorted(items, reverse=True)[:max_items])


def build_risk_ranking(scored: pd.DataFrame, snapshot_time: pd.Timestamp,
                       threshold_medium: float, threshold_high: float,
                       score_col: str = "score", max_staleness_hours: int = 24) -> pd.DataFrame:
    """Rank every unit by its most recent score at ``snapshot_time``.

    Units whose latest reading is older than ``max_staleness_hours`` (for example
    because they are down for maintenance) are listed as "No current data" rather
    than being given a stale score.
    """
    snapshot_time = pd.Timestamp(snapshot_time)
    past = scored[scored[C.TIME_COL] <= snapshot_time].sort_values([C.ID_COL, C.TIME_COL])
    latest = past.groupby(C.ID_COL).tail(1).set_index(C.ID_COL)

    prev_time = snapshot_time - pd.Timedelta(hours=24)
    prev = past[past[C.TIME_COL] <= prev_time].groupby(C.ID_COL).tail(1).set_index(C.ID_COL)

    out = pd.DataFrame(index=latest.index)
    out[C.TYPE_COL] = latest[C.TYPE_COL]
    if "area" in latest:
        out["area"] = latest["area"]
    out["last_reading"] = latest[C.TIME_COL]
    out["risk_score"] = latest[score_col].round(4)
    out["risk_score_24h_ago"] = prev[score_col].reindex(out.index).round(4)
    out["risk_score_change_24h"] = (out["risk_score"] - out["risk_score_24h_ago"]).round(4)
    out["risk_category"] = categorize(out["risk_score"], threshold_medium, threshold_high)
    out["relevant_sensor_trends"] = latest.apply(describe_sensor_trends, axis=1)
    out["hours_since_last_maintenance"] = latest["hours_since_last_maintenance"].round(0)

    stale = (snapshot_time - out["last_reading"]) > pd.Timedelta(hours=max_staleness_hours)
    out.loc[stale, "risk_category"] = "No current data"
    out["suggested_review_priority"] = out["risk_category"].map(REVIEW_PRIORITY).fillna(
        "Not ranked - no recent readings")

    out = out.reset_index()
    out["_stale"] = stale.to_numpy()
    out = out.sort_values(["_stale", "risk_score"], ascending=[True, False]).drop(columns="_stale")
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


def weekly_topk_backtest(scored: pd.DataFrame, k: int = C.TOP_K_REVIEW,
                         score_col: str = "score", hour: int = 6) -> pd.DataFrame:
    """Back-test of the ranking: every Monday 06:00, take the k highest-scoring units.

    For each snapshot, report how many of those k units had a failure logged in the
    following horizon, against the share expected from picking k units at random.
    """
    times = pd.date_range(scored[C.TIME_COL].min().ceil("D"), scored[C.TIME_COL].max(), freq="W-MON")
    rows = []
    for t in times + pd.Timedelta(hours=hour):
        snap = scored[scored[C.TIME_COL] == t]
        if len(snap) < k:
            continue
        top = snap.nlargest(k, score_col)
        rows.append({"snapshot": t, "units_scored": len(snap),
                     "units_with_failure_ahead": int(snap[C.TARGET].sum()),
                     "top_k_hits": int(top[C.TARGET].sum()),
                     "precision_at_k": top[C.TARGET].mean(),
                     "base_rate": snap[C.TARGET].mean()})
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame, k: int = C.TOP_K_REVIEW) -> dict:
    total_ahead = int(bt["units_with_failure_ahead"].sum())
    hits = int(bt["top_k_hits"].sum())
    reviewed = k * len(bt)
    base = total_ahead / int(bt["units_scored"].sum())
    return {"snapshots": int(len(bt)), "k": k, "units_reviewed": reviewed, "hits": hits,
            "precision_at_k": hits / reviewed, "base_rate": base,
            "lift_over_random": (hits / reviewed) / base if base else float("nan"),
            "share_of_upcoming_failures_in_top_k": hits / total_ahead if total_ahead else float("nan")}
