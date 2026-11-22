# LinkedIn Publication Package

Everything needed to publish this project on LinkedIn. All figures quoted are the actual results
in `reports/metrics.json`. Replace the placeholders in `<angle brackets>` before posting.

**GitHub link placeholder:** `https://github.com/<your-username>/oilfield-predictive-maintenance`

---

## 1. LinkedIn post (ready to paste)

> **I built a predictive maintenance model for oil field equipment. Its most useful result was
> finding out where it's wrong.**
>
> For my latest data science portfolio project I built an end-to-end system that reads sensor data
> from pumps, motors and compressors and ranks which units may deserve a closer look in the next
> 72 hours.
>
> Two things up front:
> 🔹 The data is synthetic. I wrote a simulator, because no public dataset combines these equipment
> types with sensor readings, maintenance history and failure labels. No real company or equipment
> data is involved.
> 🔹 The model does not decide whether equipment is failing. It produces a ranked indicator for
> qualified people to review.
>
> What I built:
> ▪️ 519,000 hourly readings from 60 simulated units over one year, with 153 failures
> ▪️ A validation stage that catches duplicates, impossible readings, sensor dropouts and
> inconsistent records
> ▪️ 47 leakage-safe features: rolling statistics and trends measured against each machine's own
> baseline, adjusted for load and ambient temperature
> ▪️ Logistic regression, random forest and gradient boosting, validated chronologically with purge
> gaps
> ▪️ A risk ranking with the sensor evidence behind every flag
>
> Results on the held-out quarter (random forest):
> ✅ PR-AUC 0.50, where random guessing scores 0.02
> ✅ 31 of 38 failures were flagged in the 72 hours beforehand, a median of 50 hours ahead
> ⚠️ Fewer than half of its strongest flags (46%) were followed by a failure within 72 hours
>
> What I learned:
> 1️⃣ Accuracy is useless here. A model that never raises an alert is 97.8% accurate - higher than
> every real model I trained.
> 2️⃣ "Degraded" is not "about to fail". Every false alarm landed on a machine the simulator
> considered genuinely worn, and 30% failed 3-7 days later. The alarms were mostly early, not random.
> 3️⃣ One validation window can fool you. Logistic regression looked best on my validation period
> and worst on the test period. Walk-forward validation and re-running on fresh simulated fleets
> gave a steadier answer.
> 4️⃣ My first feature pipeline had a subtle bias: equipment wear and summer temperatures rose
> together in the training months, so the model partly confused the two. Finding and fixing that
> taught me more than any metric.
>
> I'm not claiming cost savings or reduced downtime - those would need a real fleet and a
> controlled trial. What this project shows is the method: careful data preparation, honest
> evaluation and clear communication of limits.
>
> Code, notebook, tests and full write-up: `https://github.com/<your-username>/oilfield-predictive-maintenance`
>
> I'm a Data Science student looking for internship opportunities in Alberta's energy sector. If
> you work in reliability, maintenance analytics or industrial data, I'd value your feedback -
> especially on what real equipment data would break first.
>
> #DataScience #MachineLearning #PredictiveMaintenance #OilAndGas #Python

*(About 2,900 characters; LinkedIn's limit is 3,000, so remove a line before adding anything. Attach the images from section 9.)*

### Shorter alternative (about 800 characters)

> New portfolio project: predictive maintenance analytics for oil field pumps, motors and
> compressors.
>
> I simulated a year of hourly sensor data for 60 units (synthetic - no real company data), built
> leakage-safe features, and trained models to rank which units may deserve a look in the next 72
> hours.
>
> On the held-out quarter the random forest flagged 31 of 38 failures in advance (median 50 h
> ahead), with PR-AUC 0.50 against a 0.02 baseline. It was also wrong on more than half of its
> strongest flags - so it is a prioritisation aid for people, not an alarm.
>
> Biggest lesson: a model that never alerts is 97.8% accurate. Pick your metric before your model.
>
> Repo, notebook and write-up: `https://github.com/<your-username>/oilfield-predictive-maintenance`
>
> #DataScience #PredictiveMaintenance #MachineLearning #OilAndGas

---

## 2. Project summary

An end-to-end predictive maintenance analysis for oil field rotating equipment. A documented
simulator generates one year of hourly sensor data for 60 pumps, motors and compressors together
with maintenance and failure logs. The pipeline validates and cleans the data, engineers 47
leakage-safe features, trains and compares three supervised models against three baselines under
chronological validation, and produces an equipment risk ranking with supporting sensor evidence.
The project is reproducible from one command, covered by 57 automated tests, and documented with a
model evaluation report, a business analysis and a data dictionary. All data is synthetic and the
output is a decision-support indicator, not a determination that equipment is failing.

## 3. Problem statement

Oil and gas operations rely on pumps, motors and compressors. Unplanned failures cause downtime,
maintenance cost and operational disruption, and maintenance teams have more equipment than
inspection hours. The question: using sensor data that is already collected, which units show
patterns associated with elevated failure risk, so that qualified personnel can prioritise where
to look first?

## 4. Approach

1. **Data** - reviewed public datasets (NASA C-MAPSS, UCI AI4I 2020, Azure predictive-maintenance
   example data); none fit the problem, so built a transparent simulator with hidden wear,
   hazard-based failures, three degradation modes and deliberately injected data-quality problems.
2. **Validation** - measured missing values, duplicates, invalid readings, equipment-level
   inconsistencies, coverage gaps and label quality, then cleaned with documented decisions.
3. **Features** - condition-adjusted each sensor for load and ambient temperature; computed
   rolling means, variability, trends and changes relative to each unit's own baseline; added
   maintenance-history features. A unit test proves no feature uses future data.
4. **Modelling** - logistic regression, random forest and gradient boosting, compared with a
   prevalence baseline, a transparent sensor rule and an unsupervised isolation forest.
5. **Validation design** - chronological train / validation / test with 72-hour purge gaps,
   walk-forward model selection, equipment-grouped cross-validation.
6. **Evaluation** - PR-AUC, precision, recall, F1, confusion matrices, event-level recall and lead
   time, false-alert burden, unit-level bootstrap confidence intervals, and replication on three
   fresh simulated fleets.
7. **Risk ranking** - Low / Medium / High categories from explicit threshold rules, with sensor
   trends and a suggested review priority for each unit.

## 5. Technologies

Python, pandas, NumPy, SciPy, scikit-learn (LogisticRegression, RandomForestClassifier,
HistGradientBoostingClassifier, IsolationForest, permutation importance), Matplotlib, Seaborn,
Jupyter Notebook, pytest, Git.

## 6. Actual results

All on synthetic data; test period is the final quarter of the simulated year.

| Item | Result |
|---|---|
| Dataset | 519,474 hourly readings, 60 units, 153 failures, one year |
| Class balance | 2.2% positive unit-hours (1 in 45) |
| Primary model | Random forest (selected by walk-forward cross-validation) |
| Test PR-AUC | 0.50 (95% CI 0.41-0.62); no-skill level 0.022 |
| Test ROC-AUC | 0.93 |
| Precision / recall / F1 at the High threshold | 0.46 / 0.48 / 0.47 |
| Failures flagged in the prior 72 h | 31 of 38 (82%) |
| Median lead time | 50 hours |
| False-alert burden | about 0.7 false-alert days per unit per month |
| Gradient boosting test PR-AUC | 0.48 |
| Logistic regression test PR-AUC | 0.26 |
| Sensor-rule baseline test PR-AUC | 0.18 |
| Replication on three fresh simulated fleets (random forest) | PR-AUC 0.59-0.62; event recall 82-93% |
| Weekly top-5 ranking back-test | 10 of 60 surfaced units followed by a failure within 72 h, about 8x random |
| Automated tests | 57 passing |

## 7. Key insights

* **Accuracy is the wrong metric under class imbalance.** The never-alert baseline scored 97.8%.
* **Trends matter more than levels.** The top two features were the 24-hour-versus-7-day trends of
  vibration and temperature.
* **False alarms were mostly early alarms.** All were on units the simulator considered genuinely
  worn; 30% were followed by a failure 3-7 days later; about half came from three machines.
* **Some failures give no warning.** The three no-precursor failures were all missed; 31 of the
  other 35 were flagged.
* **Validation design changes conclusions.** Logistic regression won on a single validation window
  and lost on the test period.
* **Confounding hides in feature engineering.** Wear and seasonal temperature rose together in the
  training months and biased the first version of the temperature adjustment.
* **Comparing each machine with itself beats comparing machines with each other.** Adding absolute
  sensor levels did not help and lowered test PR-AUC for both tree models.

## 8. What to say if asked "is this real?"

Suggested wording for comments and interviews:

> "The data is simulated - I built the generator and documented its assumptions. That means the
> metrics describe how the method behaves on that simulator, not how it would perform on a real
> fleet. What I can demonstrate is the workflow: leakage-safe features, time-aware validation,
> imbalance-aware evaluation and honest reporting of where the model fails. The next step would be
> running the same pipeline on real time-stamped equipment data."

## 9. Suggested visualization / screenshot sequence

LinkedIn shows up to about four images well in a post; a document (PDF carousel) can carry more.

| Order | File | Why it is in the sequence | Suggested caption |
|---|---|---|---|
| 1 | `figures/15_risk_ranking.png` | The end product. Leads with what a user would see. | "The output: a ranked list for people to review - not an alarm." |
| 2 | `figures/03_pre_failure_sensor_trends.png` | Shows the signal exists and differs by failure mode. | "Each failure mode leaves a different sensor signature. Sudden failures leave none." |
| 3 | `figures/09_precision_recall_and_roc.png` | Model comparison, and why PR curves beat ROC under imbalance. | "ROC makes every model look good. Precision-recall tells them apart." |
| 4 | `figures/14_event_detection.png` | Results in maintenance terms. | "31 of 38 failures flagged ahead of time, median 50 hours." |
| 5 (carousel) | `figures/16_risk_score_timelines.png` | Honesty: a hit, a late call and an interrupted build-up. | "Where it works and where it doesn't." |
| 6 (carousel) | `figures/11_feature_importance.png` | Interpretation. | "Trends matter more than levels." |
| 7 (carousel) | `figures/10_confusion_matrices.png` | Classic evaluation view. | "False alarms and missed pre-failure hours." |
| 8 (carousel) | Screenshot of the README executive-summary table | Context and disclosure in one frame. | "Synthetic data, real method." |

Tip: put the words "synthetic data" in the first image's caption or alt text so the disclosure
travels with the image if it is reshared.

## 10. LinkedIn "Projects" section entry

**Project name:** Predictive Maintenance for Oil Field Equipment Using Machine Learning

**Dates:** `<month year>` - `<month year>`

**Description** (about 1,100 characters; the field allows 2,000):

> End-to-end data science project that ranks oil field pumps, motors and compressors by sensor
> patterns associated with failure in the next 72 hours, as a decision-support aid for
> maintenance prioritisation.
>
> • Built a documented simulator for one year of hourly sensor data from 60 units (519K rows, 153
> failures) after finding no public dataset that fit the problem. All data is synthetic.
> • Wrote a validation and cleaning stage for duplicates, invalid readings, sensor dropouts and
> inconsistent records.
> • Engineered 47 leakage-safe features (rolling statistics, trends, maintenance history) relative
> to each unit's own baseline, with a test proving no future information is used.
> • Compared logistic regression, random forest and gradient boosting using chronological
> validation; the random forest reached PR-AUC 0.50 against a 0.02 baseline and flagged 31 of 38
> test-period failures in advance.
> • Reported limits plainly: under half of the strongest flags preceded a failure within 72 hours.
>
> Skills: Python · pandas · scikit-learn · Feature Engineering · Time Series · Model Evaluation

**Project URL:** `https://github.com/<your-username>/oilfield-predictive-maintenance`

## 11. Resume bullets

Primary (one line):

> Built an end-to-end predictive maintenance pipeline in Python and scikit-learn on 519K simulated
> hourly sensor readings from 60 oil field units; a random forest with 47 leakage-safe features
> reached PR-AUC 0.50 versus a 0.02 baseline and flagged 31 of 38 held-out failures up to 72 hours
> ahead.

Alternatives, depending on the role:

> *Analytics emphasis:* Designed an equipment risk-ranking analysis for maintenance
> prioritisation on synthetic oil field sensor data, evaluating precision-recall trade-offs,
> false-alert burden and lead time, and documenting model limitations for non-technical readers.

> *Engineering emphasis:* Developed a reproducible, tested (57 pytest cases) ML pipeline covering
> data validation, time-aware feature engineering, walk-forward model selection and automated
> reporting; diagnosed and corrected a seasonal confounding bias in the feature pipeline.

Always keep the word "simulated" or "synthetic" in the bullet.

## 12. Suggested hashtags

Core (use 3-5 in the post):
`#DataScience` `#MachineLearning` `#PredictiveMaintenance` `#OilAndGas` `#Python`

Optional, pick by audience:
`#EnergyIndustry` `#Alberta` `#ReliabilityEngineering` `#IndustrialAnalytics` `#IIoT`
`#ScikitLearn` `#FeatureEngineering` `#TimeSeries` `#DataAnalytics` `#PortfolioProject`
`#Internship` `#OpenToWork`

## 13. Pre-publication checklist

- [ ] Push the repository and confirm the README renders with images.
- [ ] Replace every `<your-username>`, `<Your Name>` and `<your-handle>` placeholder.
- [ ] Confirm the numbers in the post still match `reports/metrics.json` if the pipeline was rerun.
- [ ] Keep the synthetic-data disclosure in the first few lines of the post.
- [ ] Do not add claims about cost savings, downtime reduction or real-world accuracy.
- [ ] Add alt text to each image.
