# Methodology and Decision Log

Every non-obvious choice in the project, with the reasoning and the alternative that was rejected.
Numbers are not repeated here; they live in `reports/metrics.json` and the reports.

## 1. Problem framing

| Decision | Reasoning | Rejected alternative |
|---|---|---|
| Binary classification: "is a failure logged for this unit within the next 72 hours?" | Matches how a planner works - a list for the coming days. 72 h is long enough to schedule a look, short enough that a flag is actionable. | Remaining-useful-life regression: needs run-to-failure trajectories and gives a number that looks more precise than it is. |
| One row per unit per running hour | Uses the data at its native resolution; lets lead time be measured. | Daily aggregation: simpler and less autocorrelated, but hides intra-day behaviour and coarsens lead time. |
| Output is a score, two thresholds and a ranked list - never a verdict | The asymmetry between a false alarm and a miss depends on criticality and redundancy, which the operator knows and the model does not. | A single yes/no alarm. |

## 2. Data

| Decision | Reasoning |
|---|---|
| Synthetic data with a documented simulator | No public dataset has the required equipment types, sensors, maintenance log and failure labels together. See `dataset_selection.md`. |
| Hidden wear state + hazard-based failure timing + random 24-144 h acceleration phase | Gives sensors real but imperfect information. A deterministic "fail when wear hits X" rule would make the problem trivially learnable and the results meaningless. |
| A small share of failures have no precursor at all | Real fleets have failures no sensor anticipates. It puts a ceiling on achievable recall and makes that ceiling visible in the results. |
| Deliberately injected data-quality problems | Validation code that has never met a problem has not been tested. The generator's own log lets detection be checked against known truth. |
| Failure rate set well above real-world levels | With realistic rates a 60-unit, one-year dataset would contain a handful of failures - not enough to train or evaluate anything. Disclosed as a limitation. |

## 3. Validation and cleaning

See `reports/data_validation_report.md` for measured counts. Principles:

* Out-of-range readings become missing rather than being clipped.
* Imputation is causal only: forward-fill within a unit, at most 6 hours. No back-fill, no
  interpolation.
* Downtime gaps stay as gaps. No rows are invented.
* The registry overrides telemetry for equipment type.

## 4. Feature engineering

| Decision | Reasoning | What went wrong first |
|---|---|---|
| **Condition adjustment**: subtract each unit's fitted response to load and ambient temperature | A bearing runs hotter on a hot day or at high load. Without adjustment the model confuses weather and duty with degradation. | - |
| Fit that response on **de-trended** (7-day high-pass) data | Over the training months the simulated fleet both aged and went from winter to summer, so wear and ambient temperature rose together. A regression on raw levels attributed part of the wear to ambient temperature, and the resulting over-correction pushed winter test-period features upward. Regressing on short-term variation removes the confounding; the fitted coefficients then match the simulator's true values. | The first version regressed on raw levels and recovered a visibly biased ambient coefficient. |
| Fit the response on the training period only | It is a fitted model, so fitting it on later data would be leakage. | - |
| **Unit-normalised features**: deviation from the unit's own expanding median, in units of its own robust spread | Units differ in normal level. "High for this machine" is the signal; "high in absolute terms" mostly identifies which machine it is. | The first version used absolute rolling means. Tree models scored clearly worse on validation than logistic regression, consistent with memorising unit-specific levels. (That run was on an earlier version of the simulator, so it is reported qualitatively; the controlled ablation in the evaluation report is the measured comparison.) |
| Robust baseline (median, IQR) rather than mean and standard deviation | The unit's history includes degraded periods; a mean and standard deviation would be dragged by them. | - |
| Time-based rolling windows, lags joined on exact timestamps | Row-offset windows silently change meaning across downtime gaps. | - |
| "Hours since maintenance" capped at 120 days; trailing 90-day failure count instead of a cumulative count | Counters that only grow take values in later periods that training never saw, forcing extrapolation. | An early version used a cumulative failure count; it received one of the larger logistic-regression coefficients even though its range in later periods lies outside anything seen in training. |
| Clip unit-normalised features at ±8 | Beyond that, magnitude adds little and single glitches dominate linear models. A fixed constant, so nothing is fitted. | - |
| Failure mode, hours-to-failure, hidden wear excluded | Not available at prediction time (or not available at all in reality). Enforced by `feature_columns` and unit tests. | - |

## 5. Splitting

| Decision | Reasoning |
|---|---|
| Chronological train / validation / test | Hourly rows from the same unit are near-duplicates; a random split would leak them across the boundary. |
| 72 h purge gap before each boundary | A row's label depends on the following 72 h. Without the gap, late training rows carry labels decided inside the next period. |
| Final 72 h of the record dropped | Their outcome is not observable (right-censored). |
| Walk-forward folds for model selection | A single two-month validation window turned out to be a noisy judge: logistic regression scored best there but worst on the following period. Averaging three chronological folds is more stable. |
| Equipment-grouped CV as a secondary check | Answers a different question - transfer to unseen units - and would reveal unit memorisation. |

Known approximation: the condition-adjustment coefficients are fitted on the whole training period
(to July), while the first two walk-forward folds validate on April-July. Those folds therefore use
adjustment coefficients estimated partly on their own validation months. No labels are involved and
only two coefficients per sensor per unit are affected, and the test period is untouched, but it is
not perfectly clean and is stated here for that reason.

## 6. Models

| Decision | Reasoning |
|---|---|
| Three baselines (prevalence, sensor rule, isolation forest) | A model is only interesting relative to what could be had without it. The sensor rule approximates a fixed-alarm philosophy; the isolation forest shows what is possible without labels. |
| Logistic regression, random forest, gradient boosting | Linear and interpretable; bagged trees; boosted trees. Standard, well understood, scikit-learn only. |
| Class weighting rather than resampling | Simpler, no synthetic rows, no risk of resampling across time boundaries. Consequence: scores are not calibrated probabilities (shown in the calibration figure). |
| Small hyper-parameter grids | The aim is a fair comparison under a clean protocol, not a leaderboard. Larger searches on so few failure events mostly fit noise. |
| No early stopping for gradient boosting | scikit-learn's early stopping holds out a random subset of rows, which for autocorrelated hourly data is a leak into the stopping decision. |
| No refit on train + validation | Thresholds are chosen on validation scores; refitting would invalidate them. |
| No SHAP dependency | Permutation importance and coefficients answer the interpretation questions here with scikit-learn alone. |

## 7. Evaluation

| Decision | Reasoning |
|---|---|
| PR-AUC as the selection metric | With roughly 2% positives, ROC-AUC is flattering for every model and accuracy is meaningless. PR-AUC rewards putting the rare positives at the top. |
| Thresholds chosen on validation, applied unchanged to test | Choosing them on test would overstate precision and recall. |
| Event-level metrics alongside row-level | A planner counts failures caught and wasted inspections, not unit-hours. |
| Bootstrap that resamples whole units | Rows within a unit are dependent; a row-level bootstrap would give falsely narrow intervals. |
| Fresh-seed replication | Test-period scores were inspected during development while debugging features. New simulated fleets, scored with everything frozen, give an estimate free of that. |
| No monetary values | None exist for simulated equipment. Costs are discussed qualitatively. |

## 8. Risk ranking

| Decision | Reasoning |
|---|---|
| High = validation max-F1 threshold; Medium = highest threshold reaching 80% validation recall | Two explicit, reproducible rules with different purposes: a short list that is often right, and a wider net that misses little. |
| Units with no reading in the last 24 h are listed as "No current data" | A stale score would be misleading. |
| Sensor-trend text from unit-baseline deviations | Gives the reviewer the evidence, in the reviewer's terms, without exposing model internals. |
| Wording limited to "suggest reviewing" | The tool supports a decision; it does not make one. Enforced by a unit test on the wording. |
