#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "model6" / "figures" / "scripts" / "make_all_figures.py"


def main() -> int:
    env = os.environ.copy()
    env["MODEL6_PUBLICATION_ROOT"] = str(REPO_ROOT)
    env.setdefault("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")
    return subprocess.call([sys.executable, str(SCRIPT)], cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())

