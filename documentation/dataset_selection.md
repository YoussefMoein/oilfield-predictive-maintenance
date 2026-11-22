# Dataset Selection: Why Synthetic Data

## What the project needs

The business problem is about **pumps, motors and compressors** monitored by **temperature,
vibration, pressure, load and speed** sensors, with a **maintenance history** and **failure
labels** over many months for many units. The modelling questions (leakage-safe rolling features,
time-since-maintenance, equipment-level validation, risk ranking across a fleet) only make sense
with all of those pieces together.

## Public datasets considered

The descriptions below are from general knowledge of these well-known datasets. No file from any
of them was downloaded or used in this project.

| Dataset | What it is | Why it was not used here |
|---|---|---|
| **NASA C-MAPSS turbofan degradation** (NASA Prognostics Data Repository) | Simulated run-to-failure trajectories of aircraft turbofan engines with ~21 sensor channels. | Aircraft engines, not oil field equipment. Every unit runs to failure once, with no maintenance events, so maintenance-history features and repeated failure/repair cycles cannot be studied. It is itself simulation output. |
| **AI4I 2020 Predictive Maintenance** (UCI Machine Learning Repository) | A 10,000-row synthetic table for a milling process (temperatures, torque, rotational speed, tool wear, failure flags). | No timestamps and no equipment identity, so rolling features, temporal validation and per-unit ranking are impossible. Also synthetic. |
| **Microsoft Azure predictive-maintenance example data** | Hourly telemetry for 100 machines for one year (voltage, rotation, pressure, vibration) with error, maintenance and failure tables. | Closest in structure, but it has no temperature or load channel, the machines are generic rather than pumps / motors / compressors, and it is widely described as example data created for a tutorial rather than field records. Typically obtained through an account-gated platform, which would make the pipeline non-reproducible offline. |
| **NASA / IMS bearing run-to-failure data** | High-frequency vibration recordings of test-rig bearings. | A handful of bearings on a laboratory rig; raw waveforms rather than process sensors; no fleet, no maintenance log. |
| **Real oil and gas operator data** | - | Not publicly available. Operators treat equipment reliability data as confidential. |

## Decision

Generate a synthetic dataset designed around the business problem, and be explicit about it
everywhere results are shown.

Reasons:

1. **Fit to the problem.** The simulator produces exactly the tables a maintenance organisation
   has: telemetry, an equipment registry, a maintenance log and a failure log.
2. **Honesty about provenance.** The two most-used "public" tabular alternatives are synthetic as
   well. Using them would not make the results any more real; it would only make the scenario
   less relevant.
3. **Reproducibility.** `python scripts/run_pipeline.py` regenerates the identical dataset from a
   seed, with no downloads, accounts or licences involved.
4. **Verifiability.** Because the generator records its hidden state and the data-quality problems
   it injects, validation and modelling can be checked against known truth - something real data
   never allows.

## What this choice costs

* **Results do not transfer.** Every metric in this repository describes behaviour on simulated
  data. None of them is evidence about how the approach would perform on real equipment.
* **The simulator is simpler than reality.** Three degradation modes, instant sensor response, no
  sensor drift, no operating-mode changes, no human factors in the maintenance log.
* **Failures are far more frequent than in a real fleet** (about 2.5 per unit-year) so that there
  are enough events to learn from and evaluate on. Real class imbalance would be much more severe.
* **The modeller also wrote the simulator.** Feature ideas were inevitably informed by knowing how
  the data was built. The fresh-seed replication guards against over-fitting to one random draw,
  but not against this.

The natural next step is to rerun the same pipeline on a real, time-stamped fleet dataset (see
"Future work" in the README).
