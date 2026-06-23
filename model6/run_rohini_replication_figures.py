#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "model6" / "figures" / "scripts" / "publication_pipeline.py"


def main() -> int:
    env = os.environ.copy()
    env["MODEL6_PUBLICATION_ROOT"] = str(REPO_ROOT)
    env.setdefault("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")
    for stage in ["audit", "datasets", "evaluate", "figures", "sanity"]:
        code = subprocess.call([sys.executable, str(PIPELINE), stage], cwd=str(REPO_ROOT), env=env)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

