#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL6_DIR = REPO_ROOT / "model6"
PIPELINE = MODEL6_DIR / "figures" / "scripts" / "publication_pipeline.py"
BEST_METRICS = MODEL6_DIR / "results" / "locked_run" / "best_metrics.json"
EXT_SUMMARY = MODEL6_DIR / "results" / "evaluation" / "metrics" / "external_validation_summary.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the locked Model 6 branch and summarize external validation.")
    parser.add_argument("--recompute", action="store_true", help="Re-run the packaged evaluation stage.")
    parser.add_argument("--with-figures", action="store_true", help="Also rebuild the final figures.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["MODEL6_PUBLICATION_ROOT"] = str(REPO_ROOT)
    env["MODEL6_PUBLICATION_DATA_ROOT"] = env.get("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")

    if args.recompute:
        code = subprocess.call([sys.executable, str(PIPELINE), "evaluate"], cwd=str(REPO_ROOT), env=env)
        if code != 0:
            return code
    if args.with_figures:
        code = subprocess.call([sys.executable, str(PIPELINE), "figures"], cwd=str(REPO_ROOT), env=env)
        if code != 0:
            return code

    if BEST_METRICS.exists():
        metrics = json.loads(BEST_METRICS.read_text())
        print(pd.Series(metrics).to_string())
    if EXT_SUMMARY.exists():
        print("\nExternal validation summary:")
        print(pd.read_csv(EXT_SUMMARY).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

