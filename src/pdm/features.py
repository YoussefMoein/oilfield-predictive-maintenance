"""Leakage-safe feature engineering and label construction.

Rule applied to every feature: at timestamp ``t`` a feature may only use
information recorded at or before ``t`` for that same unit.

* rolling windows are time-based and right-closed: (t - window, t]
* lags are joined on exact timestamps (t - 24 h), not on row offsets
* the unit baseline is an expanding statistic shifted by one row
* maintenance / failure history is joined with a backward as-of merge on the
  *completion* time of the work order
* the label looks forward, features never do

Two families of sensor features are built:

* **unit-normalised** (``*_z_*``, ``*_relstd_*``) - expressed in units of the
  equipment's own historical spread around its own historical median. These are
  what the models use.
* **absolute** (``*_abs_*`` and the raw readings) - kept only for the ablation
  study. Absolute levels differ between units (a pump at 2,400 kPa, a motor at
  240 kPa), so they act as equipment identifiers and invite memorisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ROLL_SENSORS = ["temperature_c", "vibration_mm_s", "pressure_kpa", "rotational_speed_rpm"]
BASELINE_MIN_HOURS = 72
Z_CLIP = 8.0                    # winsorise unit-normalised features at +/- 8 robust SDs
HOURS_SINCE_CAP = 120 * 24      # cap "hours since maintenance" at 120 days
FAILURE_RATE_WINDOW_DAYS = 90
NON_FEATURE_COLS = {C.ID_COL, C.TIME_COL, C.TYPE_COL, C.TARGET, "hours_to_failure",
                    "next_failure_mode", "area", "operating_hours"}
ABSOLUTE_RAW = set(ROLL_SENSORS)


def fit_condition_model(tele: pd.DataFrame, fit_end: str = C.TRAIN_END) -> pd.DataFrame:
    """Per-unit linear response of each sensor to load and ambient temperature.

    Bearing temperature rises on a hot day or at high load without anything being
    wrong. Fitting ``sensor ~ load + ambient`` per unit and subtracting that response
    leaves a condition-adjusted reading in which seasonal and duty effects are removed.

    Leakage control: coefficients are estimated only from rows at or before ``fit_end``
    (the end of the training period) and then applied unchanged to later data.
    """
    d = tele[tele[C.TIME_COL] <= pd.Timestamp(fit_end)].sort_values([C.ID_COL, C.TIME_COL])
    cols = ["load_pct", "ambient_temp_c", *ROLL_SENSORS]
    # High-pass filter: remove each unit's centred 7-day moving average from sensors and
    # drivers before regressing. Slow wear and the seasons move together over months, which
    # would bias a regression on raw levels; hour-to-hour and day-to-day variation does not
    # have that problem. (A centred window is fine here: this is model fitting inside the
    # training period, not a feature.)
    slow = d.groupby(C.ID_COL)[cols].transform(
        lambda x: x.rolling(168, center=True, min_periods=48).mean())
    fast = d[cols] - slow
    rows = []
    for unit, g in fast.groupby(d[C.ID_COL]):
        X = g[["load_pct", "ambient_temp_c"]].to_numpy()
        for s in ROLL_SENSORS:
            y = g[s].to_numpy()
            ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if ok.sum() < 200:
                beta = np.zeros(2)
            else:
                beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            rows.append({C.ID_COL: unit, "sensor": s, "beta_load": beta[0], "beta_ambient": beta[1],
                         "n_fit": int(ok.sum())})
    return pd.DataFrame(rows)


def apply_condition_model(tele: pd.DataFrame, coefs: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``tele`` whose sensor columns are condition-adjusted.

    Units without coefficients (unseen at fit time) are left unadjusted.
    """
    out = tele.copy()
    for s in ROLL_SENSORS:
        c = coefs[coefs["sensor"] == s].set_index(C.ID_COL)
        bl = out[C.ID_COL].map(c["beta_load"]).fillna(0.0)
        ba = out[C.ID_COL].map(c["beta_ambient"]).fillna(0.0)
        out[s] = out[s] - bl * (out["load_pct"] - 70.0) - ba * (out["ambient_temp_c"] - 5.0)
    return out


def _unit_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding median and robust spread (IQR / 1.349) per unit, using past rows only."""
    g = df.groupby(C.ID_COL, sort=False)
    centre, scale = {}, {}
    for s in ROLL_SENSORS:
        exp = g[s].expanding(min_periods=BASELINE_MIN_HOURS)
        med = exp.median().reset_index(level=0, drop=True)
        iqr = (exp.quantile(0.75) - exp.quantile(0.25)).reset_index(level=0, drop=True)
        # shift(1): the baseline at row t is built from rows strictly before t
        centre[s] = med.groupby(df[C.ID_COL], sort=False).shift(1)
        scale[s] = (iqr / 1.349).groupby(df[C.ID_COL], sort=False).shift(1).replace(0, np.nan)
    return pd.DataFrame(centre), pd.DataFrame(scale)


def add_sensor_features(df: pd.DataFrame, condition_model: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rolling means / standard deviations, trends and 24 h changes per unit.

    Unit-normalised (``_z_``, ``_relstd_``) features are computed on condition-adjusted
    readings when ``condition_model`` is given; ``_abs_`` features always use raw readings.
    """
    df = df.sort_values([C.ID_COL, C.TIME_COL]).reset_index(drop=True)
    adj = apply_condition_model(df, condition_model) if condition_model is not None else df
    centre, scale = _unit_baseline(adj)
    grouped = adj.set_index(C.TIME_COL).groupby(C.ID_COL, sort=False)
    raw_grouped = df.set_index(C.TIME_COL).groupby(C.ID_COL, sort=False)
    new: dict[str, np.ndarray] = {}

    for s in ROLL_SENSORS:
        new[f"{s}_z_now"] = ((adj[s] - centre[s]) / scale[s]).to_numpy()

    means: dict[tuple[str, int], np.ndarray] = {}
    for w in C.ROLLING_WINDOWS_H:
        mp = C.ROLLING_MIN_PERIODS[w]
        roll = grouped[ROLL_SENSORS + ["load_pct", "ambient_temp_c"]].rolling(f"{w}h", min_periods=mp)
        mean = roll.mean()
        for s in ROLL_SENSORS:
            means[s, w] = mean[s].to_numpy()
            new[f"{s}_z_mean_{w}h"] = (means[s, w] - centre[s].to_numpy()) / scale[s].to_numpy()
        if w == 24:
            new["load_pct_mean_24h"] = mean["load_pct"].to_numpy()
            new["ambient_temp_c_mean_24h"] = mean["ambient_temp_c"].to_numpy()
        if w in (24, 72):
            std = roll.std()
            for s in ROLL_SENSORS:
                new[f"{s}_relstd_{w}h"] = std[s].to_numpy() / scale[s].to_numpy()

    for w in (24, 168):  # absolute-level family (ablation only), from raw readings
        raw_roll = raw_grouped[ROLL_SENSORS].rolling(f"{w}h", min_periods=C.ROLLING_MIN_PERIODS[w])
        raw_mean = raw_roll.mean()
        for s in ROLL_SENSORS:
            new[f"{s}_abs_mean_{w}h"] = raw_mean[s].to_numpy()
        if w == 24:
            raw_std = raw_roll.std()
            for s in ROLL_SENSORS:
                new[f"{s}_abs_std_24h"] = raw_std[s].to_numpy()

    out = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # short-vs-long trend: positive means the last day sits above the last week
    for s in ROLL_SENSORS:
        out[f"{s}_z_trend_24h_vs_168h"] = out[f"{s}_z_mean_24h"] - out[f"{s}_z_mean_168h"]

    # change versus the same unit exactly 24 h earlier (exact timestamp join, gap-safe)
    lag = out[[C.ID_COL, C.TIME_COL]].copy()
    for s in ROLL_SENSORS:
        lag[f"{s}_lag24h"] = means[s, 6]
    lag[C.TIME_COL] = lag[C.TIME_COL] + pd.Timedelta(hours=24)
    out = out.merge(lag, on=[C.ID_COL, C.TIME_COL], how="left")
    for s in ROLL_SENSORS:
        out[f"{s}_z_change_24h"] = (means[s, 6] - out[f"{s}_lag24h"]) / scale[s].to_numpy()
        out = out.drop(columns=f"{s}_lag24h")
    z_cols = [c for c in out.columns if "_z_" in c or "_relstd_" in c]
    out[z_cols] = out[z_cols].clip(-Z_CLIP, Z_CLIP)
    return out


def add_history_features(df: pd.DataFrame, maintenance: pd.DataFrame,
                         failures: pd.DataFrame) -> pd.DataFrame:
    """Time since last completed maintenance and count of earlier failures."""
    df = df.sort_values([C.TIME_COL, C.ID_COL]).reset_index(drop=True)
    record_start = df.groupby(C.ID_COL)[C.TIME_COL].transform("min")

    def _since(events: pd.DataFrame, name: str) -> pd.Series:
        ev = events[[C.ID_COL, "end_time"]].sort_values("end_time").rename(
            columns={"end_time": "_event_time"})
        merged = pd.merge_asof(df[[C.ID_COL, C.TIME_COL]], ev, left_on=C.TIME_COL,
                               right_on="_event_time", by=C.ID_COL, direction="backward")
        return ((merged[C.TIME_COL] - merged["_event_time"]) / pd.Timedelta(hours=1)).rename(name)

    since_any = _since(maintenance, "hours_since_last_maintenance")
    since_pm = _since(maintenance[maintenance["maintenance_type"] == "preventive"],
                      "hours_since_last_preventive")
    # Before the first logged work order the true value is unknown (left-censored). Use the
    # time since the start of the unit's record as a lower bound and flag it.
    since_start = (df[C.TIME_COL] - record_start) / pd.Timedelta(hours=1)
    # Capped so that later periods never present values outside the range seen in training
    # (an ever-growing counter would force every model to extrapolate).
    df["maintenance_history_censored"] = since_any.isna().astype(int).to_numpy()
    df["hours_since_last_maintenance"] = since_any.fillna(since_start).clip(upper=HOURS_SINCE_CAP).to_numpy()
    df["hours_since_last_preventive"] = since_pm.fillna(since_start).clip(upper=HOURS_SINCE_CAP).to_numpy()

    # Failures logged for this unit in the trailing 90 days (a rate, so it does not drift
    # upwards with calendar time the way a cumulative count would).
    f = failures[[C.ID_COL, "failure_time"]].sort_values("failure_time").copy()
    f["_cum"] = f.groupby(C.ID_COL).cumcount() + 1

    def _cum_at(times: pd.Series) -> np.ndarray:
        left = df[[C.ID_COL]].assign(_t=times.to_numpy()).reset_index().sort_values("_t")
        m = pd.merge_asof(left, f, left_on="_t", right_on="failure_time", by=C.ID_COL,
                          direction="backward")
        return m.set_index("index")["_cum"].reindex(df.index).fillna(0).to_numpy()

    window = pd.Timedelta(days=FAILURE_RATE_WINDOW_DAYS)
    df["failures_last_90d"] = (_cum_at(df[C.TIME_COL]) - _cum_at(df[C.TIME_COL] - window)).astype(int)
    return df.sort_values([C.ID_COL, C.TIME_COL]).reset_index(drop=True)


def add_labels(df: pd.DataFrame, failures: pd.DataFrame,
               horizon_hours: int = C.HORIZON_HOURS) -> pd.DataFrame:
    """Binary target: is a failure logged for this unit in (t, t + horizon]?"""
    df = df.sort_values([C.TIME_COL, C.ID_COL]).reset_index(drop=True)
    f = failures[[C.ID_COL, "failure_time", "failure_mode"]].sort_values("failure_time")
    merged = pd.merge_asof(df[[C.ID_COL, C.TIME_COL]], f, left_on=C.TIME_COL,
                           right_on="failure_time", by=C.ID_COL, direction="forward",
                           allow_exact_matches=False)
    hours = (merged["failure_time"] - merged[C.TIME_COL]) / pd.Timedelta(hours=1)
    df["hours_to_failure"] = hours.to_numpy()
    df["next_failure_mode"] = merged["failure_mode"].to_numpy()
    df[C.TARGET] = (hours <= horizon_hours).astype(int).to_numpy()
    return df.sort_values([C.ID_COL, C.TIME_COL]).reset_index(drop=True)


def build_model_table(tele_clean: pd.DataFrame, maintenance: pd.DataFrame,
                      failures: pd.DataFrame, registry: pd.DataFrame,
                      condition_model: pd.DataFrame | None = None,
                      drop_censored: bool = True) -> pd.DataFrame:
    """Full feature + label table, one row per unit-hour that can be modelled."""
    df = add_sensor_features(tele_clean, condition_model)
    df = add_history_features(df, maintenance, failures)
    for t in C.EQUIPMENT_TYPES:
        df[f"type_{t}"] = (df[C.TYPE_COL] == t).astype(int)
    df = df.merge(registry[[C.ID_COL, "area"]], on=C.ID_COL, how="left")
    df = add_labels(df, failures)

    # warm-up: the unit baseline needs BASELINE_MIN_HOURS of history
    start = df.groupby(C.ID_COL)[C.TIME_COL].transform("min")
    df = df[(df[C.TIME_COL] - start) >= pd.Timedelta(hours=max(C.WARMUP_HOURS, BASELINE_MIN_HOURS))]
    if drop_censored:
        # the outcome of the final horizon is not observable yet
        cutoff = tele_clean[C.TIME_COL].max() - pd.Timedelta(hours=C.HORIZON_HOURS)
        df = df[df[C.TIME_COL] <= cutoff]
    return df.reset_index(drop=True)


def is_absolute_feature(col: str) -> bool:
    return col in ABSOLUTE_RAW or "_abs_" in col


def feature_columns(df: pd.DataFrame, include_absolute: bool = False) -> list[str]:
    """Columns the models are allowed to see.

    Identifiers, timestamps, labels, anything derived from the future and the
    simulator's hidden state are always excluded. Absolute sensor levels are
    excluded unless ``include_absolute`` is set (ablation only).
    """
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS and not c.startswith("latent_")]
    if not include_absolute:
        cols = [c for c in cols if not is_absolute_feature(c)]
    return cols


def label_stats(df: pd.DataFrame) -> dict:
    n, pos = len(df), int(df[C.TARGET].sum())
    return {"n_rows": n, "n_positive": pos, "positive_rate": pos / n,
            "neg_pos_ratio": (n - pos) / max(pos, 1)}
