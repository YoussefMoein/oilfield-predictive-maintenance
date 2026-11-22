"""End-to-end pipeline stages. Used by ``scripts/run_pipeline.py`` and by the notebook.

Each stage takes the outputs of the previous ones and returns plain Python objects,
so stages can be run and inspected one at a time.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import data_generation, data_loading, evaluation, features, modeling, plotting, risk, splitting, validation

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _to_builtin(obj):
    """Make nested results JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


# =============================================================== 1. data + validation
@dataclass
class DataBundle:
    raw: dict
    telemetry: pd.DataFrame          # cleaned
    failures: pd.DataFrame           # de-duplicated
    maintenance: pd.DataFrame
    registry: pd.DataFrame
    findings: dict
    clean_log: dict


def stage_data(regenerate: bool = True) -> DataBundle:
    C.ensure_dirs()
    if regenerate or not C.TELEMETRY_FILE.exists():
        _log("Generating synthetic dataset ...")
        data_generation.write_dataset(data_generation.generate_dataset())
    raw = data_loading.load_raw()
    _log(f"Loaded {len(raw['telemetry']):,} telemetry rows; validating ...")
    findings = validation.validate_raw(raw["telemetry"], raw["registry"], raw["maintenance"],
                                       raw["failures"])
    tele, clean_log = validation.clean_telemetry(raw["telemetry"], raw["registry"])
    fails = validation.clean_failures(raw["failures"], raw["registry"])
    tele.to_parquet(C.CLEAN_TELEMETRY_FILE, index=False)
    return DataBundle(raw, tele, fails, raw["maintenance"], raw["registry"], findings, clean_log)


# ====================================================================== 2. features
@dataclass
class FeatureBundle:
    table: pd.DataFrame
    split: splitting.TemporalSplit
    features: list
    condition_model: pd.DataFrame
    label_stats: dict


def stage_features(data: DataBundle, write: bool = True) -> FeatureBundle:
    _log("Fitting condition compensation on the training period and building features ...")
    cond = features.fit_condition_model(data.telemetry)
    table = features.build_model_table(data.telemetry, data.maintenance, data.failures,
                                       data.registry, cond)
    split = splitting.temporal_split(table)
    stats = features.label_stats(table)
    if write:
        table.to_parquet(C.FEATURES_FILE, index=False)
        cond.to_csv(C.TABLES_DIR / "condition_model_coefficients.csv", index=False)
        split.summary().to_csv(C.TABLES_DIR / "split_summary.csv", index=False)
        injected = json.loads(C.INJECTED_ISSUES_FILE.read_text()) if C.INJECTED_ISSUES_FILE.exists() else None
        (C.REPORTS_DIR / "data_validation_report.md").write_text(
            validation.render_validation_report(data.findings, data.clean_log, stats, injected))
    _log(f"Model table: {len(table):,} rows, {len(features.feature_columns(table))} features, "
         f"positive rate {stats['positive_rate']:.2%}")
    return FeatureBundle(table, split, features.feature_columns(table), cond, stats)


# =========================================================================== 3. EDA
def stage_eda_figures(data: DataBundle, feats: FeatureBundle) -> None:
    _log("Drawing exploratory figures ...")
    F = C.FIGURES_DIR
    plotting.fig_sensor_distributions(data.telemetry, F / "01_sensor_distributions.png")
    plotting.fig_failure_distribution(data.failures, data.registry, F / "02_failure_distribution.png")
    plotting.fig_pre_failure_trends(data.telemetry, data.failures, F / "03_pre_failure_sensor_trends.png")
    unit = data.failures.groupby(C.ID_COL).size().idxmax()
    plotting.fig_unit_timeline(data.telemetry, data.failures, data.maintenance, unit,
                               F / "04_unit_sensor_timeline.png")
    plotting.fig_correlation(data.telemetry, F / "05_sensor_correlations.png")
    plotting.fig_equipment_comparison(data.telemetry, data.failures, data.registry,
                                      F / "06_equipment_comparison.png")
    plotting.fig_class_balance(feats.split.summary(), F / "07_class_balance.png")
    plt.close("all")


# ====================================================================== 4. training
def stage_train(feats: FeatureBundle, save: bool = True) -> modeling.TrainedModels:
    _log("Walk-forward tuning (3 chronological folds) ...")
    folds = splitting.walk_forward_folds(feats.table)
    trained = modeling.tune_and_fit(folds, feats.features)
    _log(f"Selected primary model: {trained.selected} "
         f"(mean walk-forward PR-AUC {trained.cv_scores[trained.selected]:.4f})")
    if save:
        pd.DataFrame(trained.tuning_log).to_csv(C.TABLES_DIR / "tuning_walk_forward.csv", index=False)
        for name, model in trained.models.items():
            joblib.dump({"model": model, "features": trained.features,
                         "params": trained.best_params[name]}, C.MODELS_DIR / f"{name}.joblib",
                        compress=3)
    return trained


# ==================================================================== 5. evaluation
@dataclass
class EvalBundle:
    scores_val: dict
    scores_test: dict
    thresholds: dict                 # model -> max-F1 validation threshold
    t_medium: float
    t_high: float
    metrics: dict
    tables: dict = field(default_factory=dict)
    scored_test: pd.DataFrame | None = None   # test rows + selected model score


def _all_scores(trained, feats: FeatureBundle, iso, part: pd.DataFrame) -> dict:
    s = modeling.predict_scores(trained, part)
    s["sensor_rule_baseline"] = modeling.sensor_rule_score(part)
    s["isolation_forest"] = modeling.isolation_score(iso, part[feats.features])
    return s


def stage_evaluate(data: DataBundle, feats: FeatureBundle, trained: modeling.TrainedModels,
                   write: bool = True, extras: bool = True) -> EvalBundle:
    sp = feats.split
    y_val, y_test = sp.val[C.TARGET].to_numpy(), sp.test[C.TARGET].to_numpy()
    _log("Scoring validation and test periods ...")
    iso = modeling.fit_isolation_forest(sp.train[feats.features])
    s_val = _all_scores(trained, feats, iso, sp.val)
    s_test = _all_scores(trained, feats, iso, sp.test)

    # ---- comparison table: thresholds are chosen on validation, metrics reported on test
    thresholds, rows = {}, []
    prev = float(y_test.mean())
    rows.append({"model": "prevalence_baseline", "pr_auc": prev, "roc_auc": 0.5, "threshold": np.nan,
                 "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 1 - prev, "alert_rate": 0.0,
                 "tn": int((y_test == 0).sum()), "fp": 0, "fn": int(y_test.sum()), "tp": 0})
    order = ["sensor_rule_baseline", "isolation_forest", *modeling.SUPERVISED]
    for name in order:
        thresholds[name] = evaluation.threshold_max_f1(y_val, s_val[name])
        r = evaluation.ranking_metrics(y_test, s_test[name])
        m = evaluation.threshold_metrics(y_test, s_test[name], thresholds[name])
        rows.append({"model": name, "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"], **m})
    comparison = pd.DataFrame(rows)
    comparison["validation_pr_auc"] = [np.nan] + [
        float(average_precision_score(y_val, s_val[n])) for n in order]

    # ---- primary model, risk thresholds
    best = trained.selected
    t_high = thresholds[best]
    t_medium = evaluation.threshold_for_recall(y_val, s_val[best], C.RECALL_TARGET_MEDIUM)
    medium_rule = f"highest threshold with validation recall >= {C.RECALL_TARGET_MEDIUM:.0%}"
    if not t_medium < t_high:  # documented fallback so the categories stay ordered
        t_medium, medium_rule = t_high / 2, "fallback: half of the High threshold"

    scored = sp.test.copy()
    scored["score"] = s_test[best]
    scored_val = sp.val.assign(score=s_val[best])

    metrics: dict = {
        "selected_model": best,
        "best_params": trained.best_params,
        "walk_forward_mean_pr_auc": trained.cv_scores,
        "n_features": len(feats.features),
        "label_stats": feats.label_stats,
        "thresholds": {"high": t_high, "medium": t_medium,
                       "high_rule": "threshold maximising F1 on the validation period",
                       "medium_rule": medium_rule},
        "test_prevalence": prev,
    }
    for tag, thr in (("high", t_high), ("medium", t_medium)):
        metrics[f"row_level_{tag}"] = evaluation.threshold_metrics(y_test, s_test[best], thr)
        ev = evaluation.event_level_metrics(scored, data.failures, thr)
        events = ev.pop("events")
        metrics[f"event_level_{tag}"] = ev
        if tag == "high":
            events_high = events
    metrics["validation_row_level_high"] = evaluation.threshold_metrics(y_val, s_val[best], t_high)

    # ---- uncertainty (resampling whole units)
    _log("Cluster bootstrap confidence intervals ...")
    ci = {}
    for name in modeling.SUPERVISED:
        d = sp.test[[C.ID_COL, C.TARGET]].assign(s=s_test[name])
        ci[name] = {"pr_auc": evaluation.cluster_bootstrap_ci(d, "s", average_precision_score),
                    "roc_auc": evaluation.cluster_bootstrap_ci(d, "s", roc_auc_score)}
    metrics["bootstrap_ci_95"] = ci

    # ---- where it works / fails
    by_type = evaluation.metrics_by_group(scored, "score", C.TYPE_COL, t_high)
    pos = scored[scored[C.TARGET] == 1].assign(alert=lambda d: d["score"] >= t_high)
    by_mode_rows = pos.groupby("next_failure_mode")["alert"].agg(
        positive_rows="size", flagged_rows="sum", row_recall="mean").reset_index()
    by_mode_events = events_high.groupby("failure_mode")["flagged"].agg(
        failures="size", failures_flagged="sum", event_recall="mean").reset_index()
    by_mode = by_mode_rows.merge(by_mode_events, left_on="next_failure_mode", right_on="failure_mode",
                                 how="outer").drop(columns="failure_mode")
    pos = pos.assign(ttf_bin=pd.cut(pos["hours_to_failure"], [0, 24, 48, 72],
                                    labels=["0-24 h", "24-48 h", "48-72 h"]))
    by_ttf = pos.groupby("ttf_bin", observed=True)["alert"].agg(
        positive_rows="size", row_recall="mean").reset_index()

    # false alarms seen through the simulator's hidden state (diagnostic only)
    truth = data_loading.load_ground_truth()
    scored_truth = scored.merge(truth, on=[C.ID_COL, C.TIME_COL], how="left")
    fa = scored_truth[(scored_truth["score"] >= t_high) & (scored_truth[C.TARGET] == 0)]
    neg = scored_truth[scored_truth[C.TARGET] == 0]
    onset = data_generation.HAZARD_WEAR_ONSET
    metrics["false_alarm_anatomy"] = {
        "false_alarm_rows": int(len(fa)),
        "share_with_failure_within_3_to_7_days": float(((fa["hours_to_failure"] > 72) &
                                                          (fa["hours_to_failure"] <= 168)).mean()),
        "share_with_latent_wear_above_hazard_onset": float((fa["latent_wear"] >= onset).mean()),
        "share_of_all_negative_rows_with_latent_wear_above_hazard_onset":
            float((neg["latent_wear"] >= onset).mean()),
        "median_latent_wear_false_alarms": float(fa["latent_wear"].median()),
        "median_latent_wear_all_negatives": float(neg["latent_wear"].median()),
        "top3_units_share_of_false_alarm_rows":
            float(fa[C.ID_COL].value_counts(normalize=True).head(3).sum()) if len(fa) else 0.0,
    }
    metrics["score_vs_latent_wear_spearman"] = float(
        scored_truth[["score", "latent_wear"]].corr(method="spearman").iloc[0, 1])

    tables = {"model_comparison": comparison, "by_equipment_type": by_type,
              "by_failure_mode": by_mode, "recall_by_time_to_failure": by_ttf,
              "events_test": events_high}

    if extras:
        # ---- interpretation
        _log("Permutation importance (validation sample) ...")
        sample = scored_val.sample(min(20_000, len(scored_val)), random_state=C.SEED)
        pi = permutation_importance(trained.models[best], sample[feats.features], sample[C.TARGET],
                                    scoring="average_precision", n_repeats=5, random_state=C.SEED,
                                    n_jobs=1)
        tables["permutation_importance"] = pd.DataFrame({
            "feature": feats.features, "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
        lr = trained.models["logistic_regression"]
        tables["logistic_coefficients"] = pd.DataFrame({
            "feature": feats.features, "coefficient": lr[-1].coef_[0]}).sort_values(
            "coefficient", key=np.abs, ascending=False)

        # ---- ablation: do absolute sensor levels help or hurt?
        _log("Ablation: adding absolute sensor levels ...")
        abs_feats = features.feature_columns(feats.table, include_absolute=True)
        rows = []
        for name in modeling.SUPERVISED:
            m = modeling.make_model(name, **trained.best_params[name]).fit(sp.train[abs_feats],
                                                                           sp.train[C.TARGET])
            rows.append({
                "model": name,
                "val_pr_auc_unit_normalised": float(average_precision_score(y_val, s_val[name])),
                "val_pr_auc_plus_absolute": float(average_precision_score(
                    y_val, m.predict_proba(sp.val[abs_feats])[:, 1])),
                "test_pr_auc_unit_normalised": float(average_precision_score(y_test, s_test[name])),
                "test_pr_auc_plus_absolute": float(average_precision_score(
                    y_test, m.predict_proba(sp.test[abs_feats])[:, 1]))})
        tables["ablation_absolute_features"] = pd.DataFrame(rows)

        # ---- grouped CV: transfer to equipment never seen in training
        _log("Equipment-grouped cross-validation ...")
        pre = pd.concat([sp.train, sp.val]).iloc[::3]
        rows = []
        for name in modeling.SUPERVISED:
            aps = []
            for tr_idx, te_idx in splitting.equipment_group_folds(pre):
                m = modeling.make_model(name, **trained.best_params[name]).fit(
                    pre.iloc[tr_idx][feats.features], pre.iloc[tr_idx][C.TARGET])
                aps.append(average_precision_score(
                    pre.iloc[te_idx][C.TARGET], m.predict_proba(pre.iloc[te_idx][feats.features])[:, 1]))
            rows.append({"model": name, "pr_auc_mean": float(np.mean(aps)),
                         "pr_auc_std": float(np.std(aps)), "pr_auc_min": float(np.min(aps)),
                         "pr_auc_max": float(np.max(aps)), "folds": len(aps)})
        tables["grouped_cv_by_equipment"] = pd.DataFrame(rows)

    if write:
        _log("Writing evaluation tables and figures ...")
        for name, tbl in tables.items():
            tbl.to_csv(C.TABLES_DIR / f"{name}.csv", index=False)
        scored[[C.ID_COL, C.TIME_COL, C.TYPE_COL, C.TARGET, "hours_to_failure", "next_failure_mode",
                "score"]].to_parquet(C.DATA_PROCESSED / "test_scores.parquet", index=False)
        F = C.FIGURES_DIR
        plotting.fig_model_comparison(comparison, F / "08_model_comparison.png")
        plotting.fig_pr_roc(y_test, s_test, F / "09_precision_recall_and_roc.png")
        cms = {n: {**{k: int(comparison.set_index("model").loc[n, k]) for k in ("tn", "fp", "fn", "tp")},
                   "threshold": thresholds[n]} for n in modeling.SUPERVISED}
        plotting.fig_confusion_matrices(cms, F / "10_confusion_matrices.png")
        label = modeling.MODEL_LABELS[best]
        if extras:
            plotting.fig_feature_importance(tables["permutation_importance"],
                                            tables["logistic_coefficients"], label,
                                            F / "11_feature_importance.png")
        plotting.fig_threshold_tradeoff(y_val, s_val[best], t_medium, t_high, label,
                                        F / "12_threshold_tradeoff.png")
        plotting.fig_calibration(y_test, s_test, F / "13_calibration.png")
        plotting.fig_event_detection(events_high, F / "14_event_detection.png")
        plotting.fig_score_vs_latent_wear(scored_truth, F / "17_score_vs_hidden_wear.png")
        plt.close("all")

    return EvalBundle(s_val, s_test, thresholds, t_medium, t_high, metrics, tables, scored)


# ================================================================== 6. risk ranking
def stage_risk(data: DataBundle, ev: EvalBundle, write: bool = True) -> dict:
    _log("Building risk ranking ...")
    scored = ev.scored_test
    snapshot = scored[C.TIME_COL].max()
    ranking = risk.build_risk_ranking(scored, snapshot, ev.t_medium, ev.t_high)

    # retrospective check - only possible because the following 72 h are in the record
    # (each unit's outcome is read from the same reading its score came from)
    latest = scored.sort_values(C.TIME_COL).groupby(C.ID_COL).tail(1).set_index(C.ID_COL)
    retro = ranking[["rank", C.ID_COL, "last_reading", "risk_score", "risk_category"]].copy()
    retro["failure_logged_within_72h_of_last_reading"] = retro[C.ID_COL].map(latest[C.TARGET]).map(
        {1: "yes", 0: "no"})
    retro["hours_from_last_reading_to_failure"] = retro[C.ID_COL].map(
        latest["hours_to_failure"].where(latest[C.TARGET] == 1))

    bt = risk.weekly_topk_backtest(scored)
    bt_summary = risk.summarize_backtest(bt)
    cat_counts = ranking["risk_category"].value_counts().to_dict()

    # category behaviour over the whole test period
    cats = risk.categorize(scored["score"], ev.t_medium, ev.t_high)
    by_cat = scored.assign(category=cats).groupby("category")[C.TARGET].agg(
        unit_hours="size", followed_by_failure_within_72h="mean").reindex(risk.CATEGORIES).reset_index()
    by_cat["share_of_unit_hours"] = by_cat["unit_hours"] / by_cat["unit_hours"].sum()

    out = {"snapshot": str(snapshot), "ranking": ranking, "retrospective": retro,
           "backtest": bt, "backtest_summary": bt_summary, "category_counts": cat_counts,
           "category_outcomes": by_cat}
    if write:
        ranking.to_csv(C.TABLES_DIR / "risk_ranking.csv", index=False)
        retro.to_csv(C.TABLES_DIR / "risk_ranking_retrospective.csv", index=False)
        bt.to_csv(C.TABLES_DIR / "weekly_top5_backtest.csv", index=False)
        by_cat.to_csv(C.TABLES_DIR / "risk_category_outcomes.csv", index=False)
        plotting.fig_risk_ranking(ranking, ev.t_medium, ev.t_high, snapshot,
                                  C.FIGURES_DIR / "15_risk_ranking.png")
        events = ev.tables["events_test"]
        units = _example_units(scored, events, ev.t_high)
        plotting.fig_risk_timeline(scored, data.failures, units, ev.t_medium, ev.t_high,
                                   C.FIGURES_DIR / "16_risk_score_timelines.png")
        plt.close("all")
    return out


def _example_units(scored: pd.DataFrame, events: pd.DataFrame, t_high: float) -> list[str]:
    """One clear hit, one missed failure, one unit with many alerts and no failure."""
    units: list[str] = []
    hit = events[events["flagged"]].sort_values("lead_time_hours", ascending=False)
    if len(hit):
        units.append(hit.iloc[0][C.ID_COL])
    miss = events[~events["flagged"] & ~events[C.ID_COL].isin(units)]
    if len(miss):
        units.append(miss.iloc[0][C.ID_COL])
    no_fail = scored[~scored[C.ID_COL].isin(events[C.ID_COL])]
    if len(no_fail):
        alerts = no_fail.assign(a=no_fail["score"] >= t_high).groupby(C.ID_COL)["a"].sum()
        units.append(alerts.idxmax())
    return units


# ================================================================== 7. replication
def stage_replication(best_params: dict, seeds=C.REPLICATION_SEEDS, write: bool = True) -> pd.DataFrame:
    """Re-run the frozen recipe on freshly simulated fleets (different random seeds).

    Nothing is tuned here: same features, same hyper-parameters, same threshold rule.
    These fleets were never looked at during development, so they are the cleanest
    estimate of how the recipe behaves on new data from the same simulator.
    """
    rows = []
    for seed in seeds:
        _log(f"Replication on a fresh simulated fleet (seed {seed}) ...")
        d = data_generation.generate_dataset(seed=seed)
        tele, _ = validation.clean_telemetry(d["telemetry"], d["registry"])
        fails = validation.clean_failures(d["failures"], d["registry"])
        cond = features.fit_condition_model(tele)
        table = features.build_model_table(tele, d["maintenance"], fails, d["registry"], cond)
        sp = splitting.temporal_split(table)
        cols = features.feature_columns(table)
        for name in modeling.SUPERVISED:
            m = modeling.make_model(name, **best_params[name]).fit(sp.train[cols], sp.train[C.TARGET])
            sv, st = m.predict_proba(sp.val[cols])[:, 1], m.predict_proba(sp.test[cols])[:, 1]
            thr = evaluation.threshold_max_f1(sp.val[C.TARGET], sv)
            tm = evaluation.threshold_metrics(sp.test[C.TARGET], st, thr)
            ev = evaluation.event_level_metrics(sp.test.assign(score=st), fails, thr)
            rows.append({"seed": seed, "model": name, "test_failures": ev["n_failures"],
                         "test_prevalence": float(sp.test[C.TARGET].mean()),
                         "pr_auc": float(average_precision_score(sp.test[C.TARGET], st)),
                         "roc_auc": float(roc_auc_score(sp.test[C.TARGET], st)),
                         "precision": tm["precision"], "recall": tm["recall"], "f1": tm["f1"],
                         "event_recall": ev["event_recall"],
                         "false_alert_days_per_unit_month": ev["false_alert_days_per_unit_month"]})
    out = pd.DataFrame(rows)
    if write:
        out.to_csv(C.TABLES_DIR / "replication_fresh_seeds.csv", index=False)
    return out


# ======================================================================= everything
def write_metrics(feats: FeatureBundle, ev: EvalBundle, risk_out: dict,
                  replication: pd.DataFrame | None) -> dict:
    m = dict(ev.metrics)
    m["split_summary"] = feats.split.summary().assign(
        start=lambda d: d["start"].astype(str), end=lambda d: d["end"].astype(str)).to_dict("records")
    m["model_comparison_test"] = ev.tables["model_comparison"].to_dict("records")
    for key in ("by_equipment_type", "by_failure_mode", "recall_by_time_to_failure",
                "ablation_absolute_features", "grouped_cv_by_equipment"):
        if key in ev.tables:
            m[key] = ev.tables[key].to_dict("records")
    m["risk_ranking"] = {"snapshot": risk_out["snapshot"],
                         "category_counts": risk_out["category_counts"],
                         "weekly_top_k_backtest": risk_out["backtest_summary"],
                         "category_outcomes": risk_out["category_outcomes"].to_dict("records")}
    if replication is not None:
        m["replication_fresh_seeds"] = replication.to_dict("records")
        m["replication_summary"] = replication.groupby("model")[
            ["pr_auc", "roc_auc", "precision", "recall", "f1", "event_recall"]].agg(
            ["mean", "min", "max"]).round(4).to_dict()
        m["replication_summary"] = {f"{a}_{b}": v for (a, b), v in m["replication_summary"].items()}
    m = _to_builtin(m)
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    return m


def run_all(regenerate: bool = True, replicate: bool = True) -> dict:
    t0 = time.time()
    data = stage_data(regenerate)
    feats = stage_features(data)
    stage_eda_figures(data, feats)
    trained = stage_train(feats)
    ev = stage_evaluate(data, feats, trained)
    risk_out = stage_risk(data, ev)
    rep = stage_replication(trained.best_params) if replicate else None
    metrics = write_metrics(feats, ev, risk_out, rep)
    _log(f"Pipeline finished in {time.time() - t0:.0f} s")
    return {"data": data, "features": feats, "trained": trained, "evaluation": ev,
            "risk": risk_out, "replication": rep, "metrics": metrics}
