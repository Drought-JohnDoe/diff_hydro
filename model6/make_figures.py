#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "model6" / "figures" / "scripts" / "make_all_figures.py"


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
    parser = argparse.ArgumentParser(description="Build the packaged Model 6 publication figures.")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Optional extra arguments forwarded to the underlying figure builder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = os.environ.copy()
    env["MODEL6_PUBLICATION_ROOT"] = str(REPO_ROOT)
    env.setdefault("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")
    return subprocess.call([*python_cmd(), str(SCRIPT), *args.script_args], cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
