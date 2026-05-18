#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def release_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Run the full 671-basin Model 6 pipeline")
    p.add_argument("--data-root", required=True, help="Path to the prepared CAMELS data root")
    p.add_argument("--output-root", default=None, help="Optional custom output root")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--epoch-to-evaluate", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--rho", type=int, default=365)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--nmul", type=int, default=4)
    p.add_argument("--max-iter-ep", type=int, default=200)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=111111)
    args = p.parse_args()

    root = release_root()
    train_script = root / "scripts" / "train.py"
    eval_script = root / "scripts" / "evaluate.py"

    common = [
        "--data-root",
        args.data_root,
        "--epochs",
        str(args.epochs),
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
    ]
    if args.output_root:
        common += ["--output-root", args.output_root]

    run([sys.executable, str(train_script), "full671"] + common)

    eval_common = [
        "--data-root",
        args.data_root,
        "--epoch",
        str(args.epoch_to_evaluate),
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
    ]
    if args.output_root:
        eval_common += ["--output-root", args.output_root]

    run([sys.executable, str(eval_script), "full671", "test"] + eval_common)
    run([sys.executable, str(eval_script), "full671", "analyze"] + eval_common)


if __name__ == "__main__":
    main()
