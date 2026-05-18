#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Resume full 671-basin Model 6 training")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", default=None)
    p.add_argument("--start-epoch", type=int, required=True)
    p.add_argument("--end-epoch", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--rho", type=int, default=365)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--nmul", type=int, default=4)
    p.add_argument("--max-iter-ep", type=int, default=200)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=111111)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    script = root / "code" / "dPLHBVrelease" / "hydroDL-dev" / "example" / "model_six" / "continueModelSix.py"
    env = os.environ.copy()
    env["DYNAMIC_SIMHYD_ROOT_DB"] = str(Path(args.data_root).resolve())
    if args.output_root:
        env["DYNAMIC_SIMHYD_ROOT_OUT"] = str(Path(args.output_root).resolve())
    cmd = [
        sys.executable,
        str(script),
        "--start-epoch",
        str(args.start_epoch),
        "--end-epoch",
        str(args.end_epoch),
        "--batch-size",
        str(args.batch_size),
        "--rho",
        str(args.rho),
        "--hidden-size",
        str(args.hidden_size),
        "--nmul",
        str(args.nmul),
        "--max-iter-ep",
        str(args.max_iter_ep),
        "--gpu-id",
        str(args.gpu_id),
        "--seed",
        str(args.seed),
        "--use-all-basins",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(script.parent), env=env, check=True)


if __name__ == "__main__":
    main()
