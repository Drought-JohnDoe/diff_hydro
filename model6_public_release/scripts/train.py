#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from data_loader import build_demo_dataset
from model import build_model_six, ensure_hydrodl_on_path, release_root

ensure_hydrodl_on_path()
from hydroDL.model import crit, train as train_mod  # noqa: E402


def _run(cmd, cwd=None, env=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def train_demo(args):
    root = release_root()
    demo_root = root / "demo_data"
    out_dir = root / "outputs" / "demo_training" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_demo_dataset(
        demo_root,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        bufftime=args.bufftime,
    )
    ninv = dataset["z_train"].shape[-1] + dataset["attrs"].shape[-1]
    model = build_model_six(
        ninv=ninv,
        nattr=dataset["attrs"].shape[-1],
        hidden_size=args.hidden_size,
        nmul=args.nmul,
        inittime=args.bufftime,
        lgdyn=False,
    )
    loss_fun = crit.RmseLossComb(alpha=0.25)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu_id)

    x_tuple = (dataset["x_train"], dataset["z_train"])
    train_mod.trainModel(
        model,
        x_tuple,
        dataset["y_train"],
        dataset["attrs"],
        loss_fun,
        nEpoch=args.epochs,
        miniBatch=[min(args.batch_size, len(dataset["basin_ids"])), args.rho],
        saveEpoch=1,
        saveFolder=str(out_dir),
        bufftime=args.bufftime,
    )
    meta = {
        "basin_ids": dataset["basin_ids"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "rho": args.rho,
        "hidden_size": args.hidden_size,
        "bufftime": args.bufftime,
        "nmul": args.nmul,
    }
    with open(out_dir / "demo_training_config.json", "w") as fp:
        json.dump(meta, fp, indent=2)
    print("Demo training artifacts saved to", out_dir)


def train_full671(args):
    root = release_root()
    script = root / "code" / "dPLHBVrelease" / "hydroDL-dev" / "example" / "model_six" / "trainModelSix.py"
    env = os.environ.copy()
    if args.data_root:
        env["DYNAMIC_SIMHYD_ROOT_DB"] = str(Path(args.data_root).resolve())
    if args.output_root:
        env["DYNAMIC_SIMHYD_ROOT_OUT"] = str(Path(args.output_root).resolve())
    cmd = [
        sys.executable,
        str(script),
        "--epochs",
        str(args.epochs),
        "--save-epoch",
        str(args.save_epoch),
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
        "--forcing",
        args.forcing,
        "--gpu-id",
        str(args.gpu_id),
        "--seed",
        str(args.seed),
        "--use-all-basins",
    ]
    _run(cmd, cwd=str(script.parent), env=env)


def parse_args():
    p = argparse.ArgumentParser(description="Model 6 public-release training entry point")
    sub = p.add_subparsers(dest="mode")

    pdemo = sub.add_parser("demo", help="Train Model 6 on the included 5-basin demo data")
    pdemo.add_argument("--run-name", default="demo_5_basins")
    pdemo.add_argument("--epochs", type=int, default=2)
    pdemo.add_argument("--batch-size", type=int, default=5)
    pdemo.add_argument("--rho", type=int, default=365)
    pdemo.add_argument("--bufftime", type=int, default=365)
    pdemo.add_argument("--hidden-size", type=int, default=64)
    pdemo.add_argument("--nmul", type=int, default=4)
    pdemo.add_argument("--gpu-id", type=int, default=0)
    pdemo.add_argument("--seed", type=int, default=111111)
    pdemo.add_argument("--train-start", default="1980-10-01")
    pdemo.add_argument("--train-end", default="1983-10-01")
    pdemo.add_argument("--test-start", default="1983-10-01")
    pdemo.add_argument("--test-end", default="1984-10-01")

    pfull = sub.add_parser("full671", help="Launch the original full 671-basin training script")
    pfull.add_argument("--data-root", default=None)
    pfull.add_argument("--output-root", default=None)
    pfull.add_argument("--epochs", type=int, default=30)
    pfull.add_argument("--save-epoch", type=int, default=1)
    pfull.add_argument("--batch-size", type=int, default=32)
    pfull.add_argument("--rho", type=int, default=365)
    pfull.add_argument("--hidden-size", type=int, default=64)
    pfull.add_argument("--nmul", type=int, default=4)
    pfull.add_argument("--max-iter-ep", type=int, default=200)
    pfull.add_argument("--forcing", default="daymet")
    pfull.add_argument("--gpu-id", type=int, default=0)
    pfull.add_argument("--seed", type=int, default=111111)
    args = p.parse_args()
    if args.mode is None:
        p.error("a mode is required: demo or full671")
    return args


def main():
    args = parse_args()
    if args.mode == "demo":
        train_demo(args)
    else:
        train_full671(args)


if __name__ == "__main__":
    main()
