# Project Architecture

## Repository layout

```
oilfield-predictive-maintenance/
├── data/
│   ├── raw/                  # generated CSVs (committed; ~15 MB, reproducible from the seed)
│   └── processed/            # cleaned telemetry, model table, test scores (git-ignored)
├── documentation/
│   ├── data_dictionary.md
│   ├── dataset_selection.md  # public datasets considered, why synthetic
│   ├── methodology_decisions.md
│   ├── project_architecture.md
│   └── github_metadata.md    # repository description and topics
├── figures/                  # 17 PNG charts written by the pipeline
├── models/                   # fitted estimators (git-ignored, regenerated)
├── notebooks/
│   └── predictive_maintenance_analysis.ipynb   # executed narrative notebook
├── reports/
│   ├── data_validation_report.md   # auto-generated
│   ├── model_evaluation_report.md
│   ├── business_analysis.md
│   ├── linkedin_package.md
│   ├── metrics.json                # every reported number
│   └── tables/                     # CSV result tables incl. risk_ranking.csv
├── scripts/
│   ├── run_pipeline.py       # one command runs everything
│   └── build_notebook.py     # generates + executes the notebook
├── src/pdm/                  # the Python package
├── tests/                    # pytest suite
├── Makefile
├── pyproject.toml
├── pytest.ini
└── requirements.txt
```

## Package modules (`src/pdm/`)

| Module | Responsibility | Key functions |
|---|---|---|
| `config.py` | Paths, seeds, sensor ranges, horizon, split dates. The single place where choices that change results live. | - |
| `data_generation.py` | Synthetic fleet simulator and deliberate data-quality problems. | `generate_dataset`, `write_dataset` |
| `data_loading.py` | Typed loading with schema checks. | `load_raw`, `SchemaError` |
| `validation.py` | Measure data-quality problems; apply cleaning decisions; render the validation report. | `validate_raw`, `clean_telemetry`, `clean_failures` |
| `features.py` | Condition adjustment, unit-normalised rolling features, history features, labels. | `fit_condition_model`, `build_model_table`, `feature_columns` |
| `splitting.py` | Chronological split with purge gap, walk-forward folds, equipment-grouped folds. | `temporal_split`, `walk_forward_folds`, `equipment_group_folds` |
| `modeling.py` | Model definitions, baselines, walk-forward tuning. | `make_model`, `tune_and_fit`, `sensor_rule_score` |
| `evaluation.py` | Row-level metrics, threshold selection, event-level metrics, cluster bootstrap. | `threshold_metrics`, `event_level_metrics`, `cluster_bootstrap_ci` |
| `risk.py` | Risk categories, ranking table, weekly top-k back-test. | `categorize`, `build_risk_ranking`, `weekly_topk_backtest` |
| `plotting.py` | One function per figure, shared visual style. | `fig_*` |
| `pipeline.py` | Orchestrates the stages; used by the CLI script and by the notebook. | `run_all`, `stage_*` |

## Data flow

```
 generate_dataset(seed) ──► data/raw/*.csv
                                 │
                                 ▼
            validate_raw ──► findings ──► reports/data_validation_report.md
                                 │
                           clean_telemetry
                                 │
                                 ▼
   fit_condition_model (training period only) ──► per-unit load / ambient response
                                 │
                           build_model_table ──► features + 72 h label
                                 │
              ┌──────────────────┼────────────────────┐
              ▼                  ▼                    ▼
          train (Jan-Jul)   validation (Aug-Sep)   test (Oct-Dec)
              │                  │                    │
   walk-forward tuning    thresholds (High,       scored once with
   + model selection      Medium) chosen here     everything frozen
              │                  │                    │
              └────────► models ─┴──► evaluation ──► risk ranking, figures, metrics.json
                                                      │
                               fresh-seed replication ┘ (new simulated fleets, nothing re-tuned)
```

## Design principles

1. **One source of truth for numbers.** Reports quote `reports/metrics.json` and `reports/tables/`;
   the notebook computes everything by executing code.
2. **Measure, then decide.** Validation functions report problems without changing data; cleaning
   functions apply named decisions and return a log of what they did.
3. **Leakage is tested, not assumed.** A truncation-invariance test proves features at time *t* do
   not change when later data is removed.
4. **Separation of concerns for honesty.** The simulator's hidden state lives in its own file, is
   excluded from features by a naming rule, and that exclusion is unit-tested.
5. **Advisory language is enforced.** A test fails if risk-priority wording contains directive or
   diagnostic terms.
