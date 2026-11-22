"""Model definitions, baselines, validation-based tuning and training.

Protocol
--------
1. Choose hyper-parameters and the primary model by mean precision-recall AUC over
   chronological walk-forward folds (all inside the pre-test period).
2. Keep the estimator fitted on the final training period.
3. Choose alert thresholds on the validation period.
4. Score the test period with everything frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

MODEL_LABELS = {
    "prevalence_baseline": "Prevalence baseline",
    "sensor_rule_baseline": "Sensor-deviation rule",
    "isolation_forest": "Isolation forest (unsupervised)",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}
SUPERVISED = ["logistic_regression", "random_forest", "gradient_boosting"]

# Small, deliberately modest grids - the goal is a fair comparison, not a leaderboard.
PARAM_GRIDS: dict[str, list[dict]] = {
    "logistic_regression": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
    "random_forest": [{"min_samples_leaf": 25}, {"min_samples_leaf": 100}],
    "gradient_boosting": [
        {"max_depth": 3, "max_iter": 200},
        {"max_depth": 3, "max_iter": 400},
        {"max_depth": 6, "max_iter": 200},
    ],
}


def make_model(name: str, seed: int = C.SEED, **params):
    """Build an unfitted estimator. All models expose ``predict_proba``."""
    if name == "logistic_regression":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=params.get("C", 0.1), class_weight="balanced",
                                       max_iter=2000)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=params.get("min_samples_leaf", 50),
                max_features="sqrt", max_samples=0.25, class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed)),
        ])
    if name == "gradient_boosting":
        # Native missing-value support, so no imputer. Early stopping is disabled because
        # its internal random hold-out would mix adjacent (highly correlated) hours.
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=params.get("max_depth", 3),
            max_iter=params.get("max_iter", 200), l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def sensor_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Transparent non-ML baseline: the largest adverse deviation from the unit's own baseline.

    High temperature, high vibration or low pressure (in unit-baseline standard deviations).
    This is roughly what a fixed-alarm philosophy would look at.
    """
    parts = np.column_stack([
        df["temperature_c_z_mean_24h"].to_numpy(),
        df["vibration_mm_s_z_mean_24h"].to_numpy(),
        -df["pressure_kpa_z_mean_24h"].to_numpy(),
    ])
    return np.nan_to_num(np.nanmax(np.where(np.isnan(parts), -np.inf, parts), axis=1),
                         neginf=0.0)


def fit_isolation_forest(X_train: pd.DataFrame, seed: int = C.SEED) -> Pipeline:
    """Unsupervised reference: what could be done if failure labels were not trustworthy."""
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("iso", IsolationForest(n_estimators=200, max_samples=4096, random_state=seed, n_jobs=-1)),
    ])
    return model.fit(X_train)


def isolation_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous."""
    return -model.score_samples(X)


@dataclass
class TrainedModels:
    models: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    cv_scores: dict = field(default_factory=dict)      # mean walk-forward PR-AUC of best config
    tuning_log: list = field(default_factory=list)
    features: list = field(default_factory=list)
    selected: str = ""                                 # model with the best walk-forward score


def tune_and_fit(folds: list[tuple[pd.DataFrame, pd.DataFrame]], features: list[str],
                 names: list[str] | None = None, verbose: bool = True) -> TrainedModels:
    """Walk-forward tuning.

    ``folds`` is a chronological list of (train, validation) frames; the last fold's
    training frame is the final training period. For every configuration the model is
    fitted on each fold's training part and scored (PR-AUC) on its validation part. The
    configuration with the best *mean* score is kept, and the returned estimator is the
    one fitted on the final training period - no refit on validation data, because the
    alert thresholds are chosen there.
    """
    out = TrainedModels(features=list(features))
    for name in names or SUPERVISED:
        best_mean, best_model, best_params = -np.inf, None, None
        for params in PARAM_GRIDS[name]:
            scores, model = [], None
            for train, val in folds:
                model = make_model(name, **params).fit(train[features], train[C.TARGET])
                scores.append(average_precision_score(
                    val[C.TARGET], model.predict_proba(val[features])[:, 1]))
            mean = float(np.mean(scores))
            out.tuning_log.append({
                "model": name, "params": ", ".join(f"{k}={v}" for k, v in params.items()),
                **{f"fold_{i + 1}_pr_auc": sc for i, sc in enumerate(scores)},
                "mean_pr_auc": mean})
            if verbose:
                print(f"  {name:20s} {str(params):40s} folds "
                      + " ".join(f"{sc:.3f}" for sc in scores) + f" | mean {mean:.4f}")
            if mean > best_mean:
                best_mean, best_model, best_params = mean, model, params
        out.models[name] = best_model
        out.best_params[name] = best_params
        out.cv_scores[name] = best_mean
    out.selected = max(out.cv_scores, key=out.cv_scores.get)
    return out


def predict_scores(trained: TrainedModels, df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: m.predict_proba(df[trained.features])[:, 1] for name, m in trained.models.items()}
