#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "model6" / "figures" / "scripts" / "publication_pipeline.py"


def python_cmd() -> list[str]:
    override = os.environ.get("MODEL6_FIGURE_PYTHON")
    if override:
        return [override]
    if os.environ.get("CONDA_DEFAULT_ENV") == os.environ.get("MODEL6_FIGURE_ENV", "pytorch"):
        return [sys.executable]
    if shutil.which("conda"):
        return ["conda", "run", "-n", os.environ.get("MODEL6_FIGURE_ENV", "pytorch"), "python"]
    return [sys.executable]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full packaged Rohini-style replication sequence for locked Model 6."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["audit", "datasets", "evaluate", "figures", "sanity"],
        help="Ordered publication-pipeline stages to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    env["MODEL6_PUBLICATION_ROOT"] = str(REPO_ROOT)
    env.setdefault("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")
    for stage in args.stages:
        code = subprocess.call([*python_cmd(), str(PIPELINE), stage], cwd=str(REPO_ROOT), env=env)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
