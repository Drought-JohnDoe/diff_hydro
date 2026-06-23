#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
if Path(__file__).resolve().parent.name == "figures":
    root = Path(__file__).resolve().parents[2]
script = root / "scripts" / "publication_pipeline.py"
raise SystemExit(subprocess.call([sys.executable, str(script), "figure8"]))
