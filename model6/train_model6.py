#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL6_DIR = REPO_ROOT / "model6"
DATA_ROOT = Path(os.environ.get("MODEL6_PUBLICATION_DATA_ROOT", "/home/mircore/Desktop/diff_hydro")).resolve()
SOURCE_RUNNER = MODEL6_DIR / "source" / "ECO_HYBRID" / "run_model6_closed_snow_asrz_simhyd_simple_laieco_full671.py"
LOCKED_CKPT = MODEL6_DIR / "checkpoints" / "best_model6_checkpoint.pt"
SUBSET455 = DATA_ROOT / "ECO_HYBRID" / "ECO_INPUTS_2020" / "notes" / "minimally_disturbed_subset" / "subset_full_1988_2007_455.csv"


def build_custom_subset_file(kind: str, custom_path: str | None) -> Path | None:
    tmp_dir = MODEL6_DIR / "results" / "train_runs" / "_subset_lists"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if kind == "455":
        return SUBSET455
    if kind == "custom":
        if not custom_path:
            raise SystemExit("--custom-basin-list is required when --subset custom is used.")
        return Path(custom_path).resolve()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or fine-tune the locked Model 6 LAIEco branch.")
    parser.add_argument("--subset", choices=["32", "455", "671", "custom"], default="671")
    parser.add_argument("--custom-basin-list", help="TXT or CSV basin list for --subset custom.")
    parser.add_argument("--epochs", type=int, default=0, help="Additional epochs to run. Use 0 for eval-only smoke test.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rho", type=int, default=365)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-name", default="publication_locked_model6_run")
    parser.add_argument("--warm-checkpoint", default=str(LOCKED_CKPT))
    args = parser.parse_args()

    env = os.environ.copy()
    env["MODEL6_PUBLICATION_ROOT"] = str(REPO_ROOT)
    env["MODEL6_PUBLICATION_DATA_ROOT"] = str(DATA_ROOT)
    env["MODEL6_LAIECO671_GPU_ID"] = str(args.gpu_id)
    env["MODEL6_LAIECO671_BATCH_SIZE"] = str(args.batch_size)
    env["MODEL6_LAIECO671_RHO"] = str(args.rho)
    env["MODEL6_LAIECO671_MAX_ITER"] = str(args.max_iter)
    env["MODEL6_LAIECO671_LR"] = str(args.learning_rate)
    env["MODEL6_LAIECO671_RUN_DIR"] = args.run_name
    env["MODEL6_LAIECO671_WARM_CKPT"] = str(Path(args.warm_checkpoint).resolve())

    if args.subset == "32":
        env["MODEL6_LAIECO671_BASIN_SET"] = "prototype32"
    elif args.subset == "671":
        env["MODEL6_LAIECO671_BASIN_SET"] = "all"
    else:
        env["MODEL6_LAIECO671_BASIN_SET"] = "all"
        env["MODEL6_LAIECO671_BASIN_LIST"] = str(build_custom_subset_file(args.subset, args.custom_basin_list))

    if args.epochs <= 0:
        env["MODEL6_LAIECO671_EVAL_ONLY"] = "1"
    else:
        env["MODEL6_LAIECO671_EPOCHS"] = str(args.epochs)

    cmd = [sys.executable, str(SOURCE_RUNNER)]
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())

