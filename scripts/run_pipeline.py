#!/usr/bin/env python
"""Run the whole project: data -> validation -> features -> models -> evaluation -> risk ranking.

Usage
-----
    python scripts/run_pipeline.py                 # everything (about 7 minutes on a 10-core laptop)
    python scripts/run_pipeline.py --keep-data     # reuse data/raw instead of regenerating it
    python scripts/run_pipeline.py --no-replication  # skip the fresh-seed replication runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdm import pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--keep-data", action="store_true", help="do not regenerate data/raw")
    parser.add_argument("--no-replication", action="store_true", help="skip fresh-seed replication")
    args = parser.parse_args()
    out = pipeline.run_all(regenerate=not args.keep_data, replicate=not args.no_replication)
    m = out["metrics"]
    print("\nPrimary model:", m["selected_model"])
    print("Test PR-AUC  :", round(next(r["pr_auc"] for r in m["model_comparison_test"]
                                       if r["model"] == m["selected_model"]), 4))
    print("Outputs      : reports/metrics.json, reports/tables/, figures/")


if __name__ == "__main__":
    main()
