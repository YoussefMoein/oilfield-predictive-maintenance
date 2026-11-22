import numpy as np
import pandas as pd
import pytest

from pdm import config as C
from pdm import modeling


@pytest.mark.parametrize("name", modeling.SUPERVISED)
def test_models_output_valid_probabilities(name, toy_classification):
    cols = [c for c in toy_classification.columns if c != C.TARGET]
    train, test = toy_classification.iloc[:1000], toy_classification.iloc[1000:]
    model = modeling.make_model(name).fit(train[cols], train[C.TARGET])
    proba = model.predict_proba(test[cols])
    assert proba.shape == (len(test), 2)
    assert np.isfinite(proba).all(), "missing inputs must be handled inside the pipeline"
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    # better than chance on a learnable problem
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(test[C.TARGET], proba[:, 1]) > 0.8


def test_models_are_reproducible(toy_classification):
    cols = [c for c in toy_classification.columns if c != C.TARGET]
    for name in ("random_forest", "gradient_boosting"):
        a = modeling.make_model(name).fit(toy_classification[cols], toy_classification[C.TARGET])
        b = modeling.make_model(name).fit(toy_classification[cols], toy_classification[C.TARGET])
        assert np.allclose(a.predict_proba(toy_classification[cols]),
                           b.predict_proba(toy_classification[cols]))


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        modeling.make_model("neural_net")


def test_sensor_rule_takes_worst_adverse_deviation():
    df = pd.DataFrame({"temperature_c_z_mean_24h": [0.5, 3.0, np.nan, np.nan],
                       "vibration_mm_s_z_mean_24h": [2.0, 1.0, np.nan, 0.2],
                       "pressure_kpa_z_mean_24h": [-4.0, 0.0, np.nan, 1.0]})
    # low pressure is adverse, so its sign is flipped
    assert modeling.sensor_rule_score(df).tolist() == [4.0, 3.0, 0.0, 0.2]


def test_walk_forward_tuning_picks_best_mean_and_keeps_last_fold_model(toy_classification, monkeypatch):
    cols = [c for c in toy_classification.columns if c != C.TARGET]
    monkeypatch.setattr(modeling, "PARAM_GRIDS", {
        "logistic_regression": [{"C": 0.01}, {"C": 1.0}],
        "gradient_boosting": [{"max_depth": 2, "max_iter": 20}]})
    d = toy_classification
    folds = [(d.iloc[:500], d.iloc[500:800]), (d.iloc[:800], d.iloc[800:1100])]
    trained = modeling.tune_and_fit(folds, cols, names=["logistic_regression", "gradient_boosting"],
                                    verbose=False)
    assert set(trained.models) == {"logistic_regression", "gradient_boosting"}
    assert len(trained.tuning_log) == 3
    log = pd.DataFrame(trained.tuning_log)
    for name in trained.models:
        assert np.isclose(trained.cv_scores[name], log.loc[log["model"] == name, "mean_pr_auc"].max())
    assert trained.selected == max(trained.cv_scores, key=trained.cv_scores.get)
    scores = modeling.predict_scores(trained, d.iloc[1100:])
    assert all(len(v) == 400 for v in scores.values())


def test_isolation_forest_scores_anomalies_higher(toy_classification):
    cols = [c for c in toy_classification.columns if c != C.TARGET]
    iso = modeling.fit_isolation_forest(toy_classification[cols])
    normal = toy_classification[cols].iloc[:50]
    weird = normal + 8.0
    assert modeling.isolation_score(iso, weird).mean() > modeling.isolation_score(iso, normal).mean()
