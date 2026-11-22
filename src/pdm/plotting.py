"""All project figures. One function per figure; each returns the Matplotlib figure.

Colour is assigned by the job it does:
* identity (equipment type, model, failure mode) - fixed categorical order
* magnitude (confusion-matrix counts) - one blue ramp
* polarity (correlations, coefficient sign) - blue <-> red around a neutral grey
* state (risk category) - reserved status colours, always with a text label
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from . import config as C  # noqa: E402
from .modeling import MODEL_LABELS  # noqa: E402

# ------------------------------------------------------------------ palette
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED_POLE = "#d03b3b"

TYPE_COLORS = {"pump": BLUE, "motor": ORANGE, "compressor": AQUA}
MODE_COLORS = {"bearing_wear": BLUE, "overheating": ORANGE, "pressure_loss": AQUA,
               "sudden_no_precursor": YELLOW}
MODE_LABELS = {"bearing_wear": "Bearing wear", "overheating": "Overheating",
               "pressure_loss": "Pressure loss", "sudden_no_precursor": "Sudden (no precursor)"}
MODEL_COLORS = {"logistic_regression": BLUE, "random_forest": ORANGE, "gradient_boosting": AQUA,
                "sensor_rule_baseline": MUTED, "isolation_forest": "#b5b3ab",
                "prevalence_baseline": AXIS}
MODEL_STYLES = {"sensor_rule_baseline": (0, (4, 2)), "isolation_forest": (0, (1, 1.5))}
RISK_COLORS = {"High": "#d03b3b", "Medium": "#fab219", "Low": "#0ca30c", "No current data": AXIS}

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#e98a8a", "#b02a2a"])

SENSOR_TITLES = {"temperature_c": "Temperature (°C)", "vibration_mm_s": "Vibration (mm/s RMS)",
                 "pressure_kpa": "Pressure (kPa)", "rotational_speed_rpm": "Rotational speed (rpm)",
                 "load_pct": "Load (%)", "ambient_temp_c": "Ambient temperature (°C)",
                 "operating_hours": "Operating hours"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "figure.titlesize": 13, "figure.titleweight": "bold",
    })


def _finish(fig, title: str, subtitle: str | None, path: Path | None, top: float = 0.90):
    # Title and subtitle are placed in inches from the top edge so spacing is the same for
    # every figure height; tight_layout then fills the remaining area.
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.10 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.38 / h, subtitle, ha="left", va="top", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.72 if subtitle else 0.4) / h))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def _bar_labels(ax, bars, fmt="{:.2f}", horizontal=True, pad=3):
    for b in bars:
        if horizontal:
            ax.annotate(fmt.format(b.get_width()), (b.get_width(), b.get_y() + b.get_height() / 2),
                        xytext=(pad, 0), textcoords="offset points", va="center", fontsize=8.5,
                        color=INK2)
        else:
            ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, pad), textcoords="offset points", ha="center", fontsize=8.5,
                        color=INK2)


# ============================================================ exploratory figures
def fig_sensor_distributions(tele: pd.DataFrame, path=None):
    apply_style()
    sensors = ["temperature_c", "vibration_mm_s", "pressure_kpa", "load_pct"]
    fig, axes = plt.subplots(len(sensors), 3, figsize=(11, 8.6))
    for i, s in enumerate(sensors):
        for j, t in enumerate(C.EQUIPMENT_TYPES):
            ax = axes[i, j]
            x = tele.loc[tele[C.TYPE_COL] == t, s].dropna()
            lo, hi = x.quantile([0.001, 0.999])
            sns.histplot(x=x[(x >= lo) & (x <= hi)], bins=60, color=TYPE_COLORS[t], alpha=1,
                         linewidth=0, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.axvline(x.median(), color=INK, linewidth=1)
            ax.set_yticks([])
            ax.grid(False)
            ax.spines["left"].set_visible(False)
            if i == 0:
                ax.set_title(t.capitalize())
            if j == 0:
                ax.set_ylabel(SENSOR_TITLES[s], fontsize=9)
            left = s == "pressure_kpa"   # pressure is left-skewed, so the label goes on the left
            ax.annotate(f"median {x.median():,.1f}", (0.02 if left else 0.98, 0.9),
                        xycoords="axes fraction", ha="left" if left else "right", fontsize=8.5,
                        color=INK2)
    return _finish(fig, "Sensor distributions by equipment type",
                   "Cleaned hourly readings (central 99.8%). Black line marks the median; the long "
                   "tails on temperature, vibration and pressure are the degraded periods.", path, top=0.93)


def fig_failure_distribution(failures: pd.DataFrame, registry: pd.DataFrame, path=None):
    apply_style()
    f = failures.merge(registry[[C.ID_COL, C.TYPE_COL]], on=C.ID_COL)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.2, 1.3, 1]})

    ax = axes[0]
    by_month = f.groupby(f["failure_time"].dt.to_period("M")).size()
    bars = ax.bar([p.strftime("%b") for p in by_month.index], by_month.values, color=BLUE, width=0.7)
    _bar_labels(ax, bars, "{:.0f}", horizontal=False)
    ax.set_title("Failures logged per month")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    modes = list(MODE_COLORS)
    pivot = f.pivot_table(index=C.TYPE_COL, columns="failure_mode", values="failure_time",
                          aggfunc="count", fill_value=0).reindex(index=C.EQUIPMENT_TYPES, columns=modes,
                                                                  fill_value=0)
    width = 0.2
    for k, m in enumerate(modes):
        pos = np.arange(len(pivot)) + (k - 1.5) * width
        bars = ax.bar(pos, pivot[m].values, width=width * 0.9, color=MODE_COLORS[m], label=MODE_LABELS[m])
        _bar_labels(ax, bars, "{:.0f}", horizontal=False, pad=2)
    ax.set_xticks(np.arange(len(pivot)), [t.capitalize() for t in pivot.index])
    ax.set_title("Failures by equipment type and mode")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

    ax = axes[2]
    per_unit = f.groupby(C.ID_COL).size().reindex(registry[C.ID_COL]).fillna(0).astype(int)
    counts = per_unit.value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
    _bar_labels(ax, bars, "{:.0f}", horizontal=False)
    ax.set_xlabel("Failures in the year")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Number of units by failure count")
    ax.grid(axis="x", visible=False)
    return _finish(fig, "Failure distribution",
                   f"{len(f)} failure events across {registry[C.ID_COL].nunique()} simulated units, "
                   "one year. Failure mode is recorded after the event and is never a model input.",
                   path, top=0.86)


def pre_failure_profile(tele: pd.DataFrame, failures: pd.DataFrame, lookback_h: int = 240,
                        bin_h: int = 6) -> pd.DataFrame:
    """Average sensor deviation from the unit's median, aligned on hours before failure."""
    med = tele.groupby(C.ID_COL)[["temperature_c", "vibration_mm_s", "pressure_kpa"]].transform("median")
    dev = pd.DataFrame({
        C.ID_COL: tele[C.ID_COL], C.TIME_COL: tele[C.TIME_COL],
        "temperature_c": tele["temperature_c"] - med["temperature_c"],
        "vibration_mm_s": 100 * (tele["vibration_mm_s"] / med["vibration_mm_s"] - 1),
        "pressure_kpa": 100 * (tele["pressure_kpa"] / med["pressure_kpa"] - 1),
    }).sort_values(C.TIME_COL)
    f = failures.sort_values("failure_time")
    m = pd.merge_asof(dev, f, left_on=C.TIME_COL, right_on="failure_time", by=C.ID_COL,
                      direction="forward", allow_exact_matches=False)
    m["h"] = (m["failure_time"] - m[C.TIME_COL]) / pd.Timedelta(hours=1)
    m = m[m["h"] <= lookback_h]
    m["bin"] = -(np.ceil(m["h"] / bin_h) * bin_h - bin_h / 2)
    return m.groupby(["failure_mode", "bin"])[["temperature_c", "vibration_mm_s", "pressure_kpa"]].mean().reset_index()


def fig_pre_failure_trends(tele: pd.DataFrame, failures: pd.DataFrame, path=None):
    apply_style()
    prof = pre_failure_profile(tele, failures)
    panels = [("temperature_c", "Temperature vs unit median (°C)"),
              ("vibration_mm_s", "Vibration vs unit median (%)"),
              ("pressure_kpa", "Pressure vs unit median (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    for ax, (col, title) in zip(axes, panels):
        for mode, color in MODE_COLORS.items():
            d = prof[prof["failure_mode"] == mode]
            ax.plot(d["bin"], d[col], color=color, label=MODE_LABELS[mode])
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.axvspan(-C.HORIZON_HOURS, 0, color=GRID, alpha=0.5, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Hours before logged failure")
    axes[0].annotate("72 h label window", (-C.HORIZON_HOURS / 2, axes[0].get_ylim()[1]),
                     xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                     fontsize=8.5, color=INK2)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    return _finish(fig, "Average sensor behaviour in the 10 days before a failure",
                   "Mean deviation across all failures of each mode, 6-hour bins. Each mode leaves a "
                   "different signature; sudden failures leave none.", path, top=0.86)


def fig_unit_timeline(tele: pd.DataFrame, failures: pd.DataFrame, maintenance: pd.DataFrame,
                      unit: str, path=None):
    apply_style()
    d = tele[tele[C.ID_COL] == unit].set_index(C.TIME_COL)
    hourly = d[["temperature_c", "vibration_mm_s", "pressure_kpa"]].resample("h").mean()
    smooth = hourly.rolling(24, min_periods=6).mean()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fails = failures.loc[failures[C.ID_COL] == unit, "failure_time"]
    pms = maintenance.loc[(maintenance[C.ID_COL] == unit) &
                          (maintenance["maintenance_type"] == "preventive"), "start_time"]
    for ax, col in zip(axes, smooth.columns):
        ax.plot(smooth.index, smooth[col], color=BLUE, linewidth=1.4)
        for k, t in enumerate(fails):
            ax.axvline(t, color=RED_POLE, linewidth=1.4, label="Failure logged" if k == 0 else None)
        for k, t in enumerate(pms):
            ax.axvline(t, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                       label="Preventive maintenance" if k == 0 else None)
        ax.set_title(SENSOR_TITLES[col] + " - 24 h rolling mean")
    axes[0].legend(loc="upper left", ncol=2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    return _finish(fig, f"Sensor trends for one unit over the year ({unit})",
                   "Gaps are downtime. Readings drift before some failures and reset after "
                   "maintenance - the pattern the features are designed to capture.", path, top=0.91)


def fig_correlation(tele: pd.DataFrame, path=None):
    apply_style()
    cols = C.SENSORS + ["ambient_temp_c"]
    centred = tele[cols] - tele.groupby(C.ID_COL)[cols].transform("mean")
    corr = centred.corr(method="spearman")
    labels = ["Temperature", "Vibration", "Pressure", "Speed", "Load", "Ambient temp."]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    sns.heatmap(corr, ax=ax, cmap=DIVERGING, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"fontsize": 9}, linewidths=2, linecolor=SURFACE, square=True,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"fraction": 0.046, "pad": 0.04, "label": "Spearman correlation"})
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    ax.grid(False)
    return _finish(fig, "Relationships between sensor variables",
                   "Within-unit correlations (each unit's own mean removed first).", path, top=0.90)


def fig_equipment_comparison(tele: pd.DataFrame, failures: pd.DataFrame, registry: pd.DataFrame,
                             path=None):
    apply_style()
    stats = tele.groupby(C.ID_COL).agg(mean_load=("load_pct", "mean"), hours=("load_pct", "size"))
    stats["failures"] = failures.groupby(C.ID_COL).size().reindex(stats.index).fillna(0)
    stats["per_1000h"] = 1000 * stats["failures"] / stats["hours"]
    stats = stats.join(registry.set_index(C.ID_COL)[C.TYPE_COL])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    rng = np.random.default_rng(0)

    ax = axes[0]
    for k, t in enumerate(C.EQUIPMENT_TYPES):
        d = stats[stats[C.TYPE_COL] == t]
        ax.scatter(k + rng.uniform(-0.16, 0.16, len(d)), d["per_1000h"], s=42, color=TYPE_COLORS[t],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.hlines(d["per_1000h"].mean(), k - 0.3, k + 0.3, color=INK, linewidth=1.6, zorder=4)
        ax.annotate(f"mean {d['per_1000h'].mean():.2f}", (k + 0.32, d["per_1000h"].mean()),
                    va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(3), [t.capitalize() for t in C.EQUIPMENT_TYPES])
    ax.set_xlim(-0.5, 2.8)
    ax.set_ylabel("Failures per 1,000 running hours")
    ax.set_title("Failure rate per unit, by type")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    for t in C.EQUIPMENT_TYPES:
        d = stats[stats[C.TYPE_COL] == t]
        ax.scatter(d["mean_load"], d["failures"] + rng.uniform(-0.12, 0.12, len(d)), s=42,
                   color=TYPE_COLORS[t], edgecolor=SURFACE, linewidth=1.2, label=t.capitalize(), zorder=3)
    rho = stats[["mean_load", "failures"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("Mean load over the year (%)")
    ax.set_ylabel("Failures in the year")
    ax.set_title(f"Duty versus failures (Spearman ρ = {rho:.2f})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    return _finish(fig, "Equipment comparison",
                   "Each dot is one unit. Units differ widely even within a type, which is why "
                   "features are expressed relative to each unit's own baseline.", path, top=0.87)


def fig_class_balance(split_summary: pd.DataFrame, path=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(split_summary["split"].str.capitalize(), 100 * split_summary["positive_rate"],
                   color=BLUE, height=0.55)
    for b, (_, r) in zip(bars, split_summary.iterrows()):
        ax.annotate(f"{100 * r['positive_rate']:.2f}%  ({int(r['positive_rows']):,} of {int(r['rows']):,} rows)",
                    (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 100 * split_summary["positive_rate"].max() * 1.9)
    ax.invert_yaxis()
    ax.set_xlabel("Share of unit-hours followed by a failure within 72 h (%)")
    ax.grid(axis="y", visible=False)
    return _finish(fig, "Class imbalance in each split",
                   "Roughly 1 positive for every 40 negatives - accuracy would be misleading.",
                   path, top=0.82)


# ================================================================= model figures
def fig_model_comparison(metrics: pd.DataFrame, path=None):
    """``metrics``: one row per model with pr_auc, roc_auc, precision, recall, f1."""
    apply_style()
    panels = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("precision", "Precision"),
              ("recall", "Recall"), ("f1", "F1")]
    m = metrics.set_index("model")
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        vals = m[col]
        colors = [BLUE if name in ("logistic_regression", "random_forest", "gradient_boosting")
                  else AXIS for name in m.index]
        bars = ax.barh([MODEL_LABELS[n] for n in m.index], vals, color=colors, height=0.6)
        _bar_labels(ax, bars, "{:.2f}")
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    return _finish(fig, "Model comparison on the held-out test period",
                   "Precision, recall and F1 at each model's own threshold chosen on the validation "
                   "period. Grey bars are reference baselines.", path, top=0.80)


def fig_pr_roc(y_true, scores: dict[str, np.ndarray], path=None):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    prevalence = float(np.mean(y_true))
    order = ["gradient_boosting", "random_forest", "logistic_regression", "sensor_rule_baseline",
             "isolation_forest"]
    for name in [n for n in order if n in scores]:
        p, r, _ = precision_recall_curve(y_true, scores[name])
        fpr, tpr, _ = roc_curve(y_true, scores[name])
        kw = dict(color=MODEL_COLORS[name], linestyle=MODEL_STYLES.get(name, "-"),
                  linewidth=2 if name not in MODEL_STYLES else 1.5, label=MODEL_LABELS[name])
        axes[0].plot(r, p, **kw)
        axes[1].plot(fpr, tpr, **kw)
    axes[0].axhline(prevalence, color=AXIS, linewidth=1)
    axes[0].annotate(f"no-skill level = prevalence {prevalence:.1%}", (0.02, prevalence), xytext=(0, 4),
                     textcoords="offset points", ha="left", fontsize=8.5, color=INK2)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves",
                xlim=(0, 1), ylim=(0, 1.02))
    axes[1].plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves",
                xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(loc="upper right")
    return _finish(fig, "Precision-recall and ROC curves (test period)",
                   "With about 2-3% positives the precision-recall view is the informative one; "
                   "ROC looks flattering for every model.", path, top=0.88)


def fig_confusion_matrices(cms: dict[str, dict], path=None):
    """``cms``: model -> dict with tn, fp, fn, tp, threshold."""
    apply_style()
    fig, axes = plt.subplots(1, len(cms), figsize=(4.4 * len(cms), 4.4))
    for ax, (name, c) in zip(np.atleast_1d(axes), cms.items()):
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        share = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=SEQ_BLUE, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}\n{share[i, j]:.1%} of row", ha="center", va="center",
                        fontsize=10, color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["No alert", "Alert"])
        ax.set_yticks([0, 1], ["No failure\nwithin 72 h", "Failure\nwithin 72 h"])
        ax.set_xlabel("Model output")
        ax.set_title(f"{MODEL_LABELS[name]}\nthreshold {c['threshold']:.3f}", fontsize=10)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    return _finish(fig, "Confusion matrices on the test period (unit-hours)",
                   "Top-right = false alarms, bottom-left = missed pre-failure hours. Shading is the "
                   "share of each actual class.", path, top=0.84)


def fig_feature_importance(perm: pd.DataFrame, coefs: pd.DataFrame, model_label: str, path=None,
                           top: int = 15):
    """``perm``: feature, importance_mean, importance_std. ``coefs``: feature, coefficient."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    p = perm.sort_values("importance_mean", ascending=False).head(top).iloc[::-1]
    axes[0].barh(p["feature"], p["importance_mean"], xerr=p["importance_std"], color=BLUE,
                 height=0.62, error_kw={"ecolor": INK2, "elinewidth": 1})
    axes[0].set_xlabel("Drop in validation PR-AUC when the feature is shuffled")
    axes[0].set_title(f"{model_label}: permutation importance")
    axes[0].grid(axis="y", visible=False)

    c = coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    colors = [RED_POLE if v > 0 else "#256abf" for v in c["coefficient"]]
    axes[1].barh(c["feature"], c["coefficient"], color=colors, height=0.62)
    axes[1].axvline(0, color=AXIS, linewidth=0.8)
    axes[1].set_xlabel("Standardised coefficient (red raises the score, blue lowers it)")
    axes[1].set_title("Logistic regression: largest coefficients")
    axes[1].grid(axis="y", visible=False)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8.5)
    return _finish(fig, "Which variables the models rely on",
                   "Associations learned from the data, not causes. Correlated features share "
                   "importance, so individual bars understate groups of related features.",
                   path, top=0.89)


def fig_threshold_tradeoff(y_val, scores_val, t_medium: float, t_high: float, model_label: str,
                           path=None):
    apply_style()
    p, r, t = precision_recall_curve(y_val, scores_val)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(t, p[:-1], color=BLUE, label="Precision")
    ax.plot(t, r[:-1], color=ORANGE, label="Recall")
    for thr, name in ((t_medium, "Medium"), (t_high, "High")):
        ax.axvline(thr, color=INK2, linewidth=1)
        ax.annotate(f"{name} ≥ {thr:.3f}", (thr, 1.02), ha="center", fontsize=9, color=INK)
    ax.set(xlabel="Risk-score threshold", ylabel="Value on validation period", ylim=(0, 1.08),
           xlim=(0, 1))
    ax.legend(loc="center right")
    return _finish(fig, f"Choosing risk thresholds on the validation period ({model_label})",
                   "Lower thresholds catch more pre-failure hours at the price of more false alarms. "
                   "Both thresholds are fixed before the test period is scored.", path, top=0.85)


def fig_calibration(y_true, scores: dict[str, np.ndarray], path=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for name in ("logistic_regression", "random_forest", "gradient_boosting"):
        if name not in scores:
            continue
        frac, mean = calibration_curve(y_true, scores[name], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", markersize=5, color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.set(xlabel="Mean model score in bin", ylabel="Observed share of pre-failure hours",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    return _finish(fig, "Scores are rankings, not probabilities",
                   "Class weighting pushes scores far above observed frequencies (test period).",
                   path, top=0.87)


def fig_event_detection(events: pd.DataFrame, path=None):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    modes = [m for m in MODE_COLORS if m in set(events["failure_mode"])]
    g = events.groupby("failure_mode")["flagged"].agg(["sum", "count"]).reindex(modes)
    bars = ax.barh([MODE_LABELS[m] for m in modes], g["sum"] / g["count"],
                   color=[MODE_COLORS[m] for m in modes], height=0.55)
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.annotate(f"{int(r['sum'])} of {int(r['count'])}", (b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    ax.invert_yaxis()
    ax.set_title("Share of failures flagged beforehand, by mode")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    lead = events.loc[events["flagged"], "lead_time_hours"]
    ax.hist(lead, bins=np.arange(0, C.HORIZON_HOURS + 6, 6), color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(lead.median(), color=INK, linewidth=1.2)
    ax.annotate(f"median {lead.median():.0f} h", (lead.median(), ax.get_ylim()[1]), xytext=(-4, -4),
                textcoords="offset points", ha="right", va="top", fontsize=9, color=INK2)
    ax.set_xlabel("Hours between first alert in the window and the failure")
    ax.set_ylabel("Failures")
    ax.set_title("Lead time for flagged failures (capped at 72 h)")
    ax.grid(axis="x", visible=False)
    return _finish(fig, "Event-level view: which failures were flagged, and how early",
                   f"Test period, {len(events)} failures, alert = score at or above the High threshold "
                   "at any hour in the 72 h before the failure.", path, top=0.86)


def fig_risk_ranking(ranking: pd.DataFrame, t_medium: float, t_high: float, snapshot, path=None,
                     top: int = 20):
    apply_style()
    d = ranking[ranking["risk_category"] != "No current data"].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.33 * len(d) + 2.2))
    bars = ax.barh(d[C.ID_COL], d["risk_score"], color=[RISK_COLORS[c] for c in d["risk_category"]],
                   height=0.66, zorder=3)
    for b, (_, r) in zip(bars, d.iterrows()):
        ax.annotate(f"{r['risk_score']:.2f}  {r['risk_category']}",
                    (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
    for thr, name in ((t_medium, "Medium"), (t_high, "High")):
        ax.axvline(thr, color=MUTED, linewidth=1, zorder=1)
        ax.annotate(name, (thr, len(d) - 0.35), xytext=(3, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, va="bottom")
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.7, len(d) + 0.4)
    ax.set_xlabel("Model risk score (relative ranking, not a probability)")
    ax.grid(axis="y", visible=False)
    return _finish(fig, f"Equipment risk ranking - top {len(d)} units",
                   f"Snapshot {pd.Timestamp(snapshot):%Y-%m-%d %H:%M}. Analytical indicator to help "
                   "prioritise review by qualified personnel - not a safety assessment.", path,
                   top=0.90)


def fig_risk_timeline(scored: pd.DataFrame, failures: pd.DataFrame, units: list[str],
                      t_medium: float, t_high: float, path=None):
    apply_style()
    fig, axes = plt.subplots(len(units), 1, figsize=(12, 2.3 * len(units) + 1), sharex=True, sharey=True)
    for ax, unit in zip(np.atleast_1d(axes), units):
        d = scored[scored[C.ID_COL] == unit].set_index(C.TIME_COL)["score"].resample("h").mean()
        ax.plot(d.index, d.values, color=BLUE, linewidth=1.3, label="Risk score")
        ax.axhline(t_high, color=INK2, linewidth=0.9)
        ax.axhline(t_medium, color=AXIS, linewidth=0.9)
        fails = failures[(failures[C.ID_COL] == unit) & (failures["failure_time"] >= d.index.min())
                         & (failures["failure_time"] <= d.index.max() + pd.Timedelta(hours=1))]
        for k, t in enumerate(fails["failure_time"]):
            ax.axvline(t, color=RED_POLE, linewidth=1.5, label="Failure logged" if k == 0 else None)
        ax.set_title(unit, fontsize=10)
        ax.set_ylim(0, 1.03)
    first = np.atleast_1d(axes)[0]
    first.annotate("High", (first.get_xlim()[1], t_high), xytext=(3, 0), textcoords="offset points",
                   fontsize=8.5, color=INK2, va="center", annotation_clip=False)
    first.annotate("Medium", (first.get_xlim()[1], t_medium), xytext=(3, 0), textcoords="offset points",
                   fontsize=8.5, color=INK2, va="center", annotation_clip=False)
    first.legend(loc="upper left", ncol=2)
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    return _finish(fig, "Risk score over the test period for selected units",
                   "Hourly score with the two category thresholds. Shows hits, a late or missed "
                   "event and elevated periods that were not followed by a failure.", path, top=0.90)


def fig_score_vs_latent_wear(scored_with_truth: pd.DataFrame, path=None):
    apply_style()
    d = scored_with_truth.dropna(subset=["latent_wear"]).copy()
    bins = np.arange(0, 1.41, 0.1)
    d["bin"] = pd.cut(d["latent_wear"].clip(upper=1.399), bins, right=False)
    q = d.groupby("bin", observed=True)["score"].quantile([0.25, 0.5, 0.75]).unstack()
    x = [iv.left + 0.05 for iv in q.index]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(x, q[0.25], q[0.75], color="#cde2fb", linewidth=0, label="Interquartile range")
    ax.plot(x, q[0.5], color=BLUE, marker="o", markersize=5, label="Median risk score")
    ax.set(xlabel="Hidden simulator wear state (never shown to the model)", ylabel="Risk score",
           ylim=(0, 1.02))
    ax.legend(loc="upper left")
    return _finish(fig, "Sanity check available only because the data is simulated",
                   "Risk score against the simulator's hidden wear variable, test period.", path,
                   top=0.86)
