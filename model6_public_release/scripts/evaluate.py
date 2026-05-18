#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tempfile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import build_demo_dataset, build_demo_eval_inputs
from model import ensure_hydrodl_on_path, release_root

ensure_hydrodl_on_path()
from hydroDL.model import train as train_mod  # noqa: E402


def _run(cmd, cwd=None, env=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _nse(pred, obs):
    den = np.sum((obs - np.mean(obs)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - np.sum((pred - obs) ** 2) / den


def _kge(pred, obs):
    if len(pred) < 2:
        return np.nan
    r = np.corrcoef(pred, obs)[0, 1]
    alpha = np.std(pred) / max(np.std(obs), 1e-6)
    beta = np.mean(pred) / max(np.mean(obs), 1e-6)
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def evaluate_demo(args):
    root = release_root()
    demo_root = root / "demo_data"
    out_dir = root / "outputs" / "demo_evaluation" / args.run_name
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_demo_dataset(
        demo_root,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        bufftime=args.bufftime,
    )
    x_eval, z_eval, y_test = build_demo_eval_inputs(dataset, use_full_train_warmup=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = torch.load(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()
    model.inittime = dataset["x_train"].shape[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = Path(tmpdir) / "demo_q.csv"
        train_mod.testModel(
            model,
            (x_eval, z_eval),
            c=None,
            batchSize=len(dataset["basin_ids"]),
            filePathLst=[str(pred_path)],
        )
        pred = pd.read_csv(pred_path, dtype=float, header=None).values
    pred = pred.reshape(len(dataset["basin_ids"]), y_test.shape[1], 1)[:, :, 0]
    obs = y_test[:, :, 0]

    rows = []
    for i, basin_id in enumerate(dataset["basin_ids"]):
        p = pred[i]
        o = obs[i]
        mask = np.isfinite(p) & np.isfinite(o)
        cor = np.corrcoef(p[mask], o[mask])[0, 1] if mask.sum() > 1 else np.nan
        rows.append(
            {
                "basin_id": basin_id,
                "NSE": _nse(p[mask], o[mask]),
                "KGE": _kge(p[mask], o[mask]),
                "R2": cor ** 2 if np.isfinite(cor) else np.nan,
                "COR": cor,
            }
        )
        fig, ax = plt.subplots(figsize=(10, 3.5))
        short = slice(0, min(365, len(p)))
        ax.plot(dataset["test_dates"].iloc[short], o[short], color="black", lw=1.2, label="Observed")
        ax.plot(dataset["test_dates"].iloc[short], p[short], color="#1f77b4", lw=1.0, label="Predicted")
        ax.set_title(f"Demo basin {basin_id} first test year")
        ax.set_ylabel("Q (mm/day)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{basin_id}_demo_prediction.png", dpi=150)
        plt.close(fig)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "demo_metrics.csv", index=False)
    np.save(out_dir / "demo_pred.npy", pred)
    np.save(out_dir / "demo_obs.npy", obs)
    print("Demo evaluation results saved to", out_dir)
    print(metrics)


def evaluate_full671(args):
    root = release_root()
    base = root / "code" / "dPLHBVrelease" / "hydroDL-dev" / "example" / "model_six"
    env = os.environ.copy()
    if args.data_root:
        env["DYNAMIC_SIMHYD_ROOT_DB"] = str(Path(args.data_root).resolve())
    if args.output_root:
        env["DYNAMIC_SIMHYD_ROOT_OUT"] = str(Path(args.output_root).resolve())
    script = base / ("testModelSix.py" if args.action == "test" else "analyzeModelSix.py")
    cmd = [
        sys.executable,
        str(script),
        "--epoch",
        str(args.epoch),
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
    if args.action == "test":
        cmd += ["--test-batch", str(args.test_batch)]
    else:
        cmd += ["--chunk-size", str(args.chunk_size)]
    _run(cmd, cwd=str(base), env=env)


def parse_args():
    p = argparse.ArgumentParser(description="Model 6 public-release evaluation entry point")
    sub = p.add_subparsers(dest="mode")

    pdemo = sub.add_parser("demo", help="Evaluate a demo checkpoint on the included 5-basin data")
    pdemo.add_argument("--checkpoint", required=True)
    pdemo.add_argument("--run-name", default="demo_5_basins")
    pdemo.add_argument("--bufftime", type=int, default=365)
    pdemo.add_argument("--gpu-id", type=int, default=0)
    pdemo.add_argument("--cpu", action="store_true")
    pdemo.add_argument("--train-start", default="1980-10-01")
    pdemo.add_argument("--train-end", default="1983-10-01")
    pdemo.add_argument("--test-start", default="1983-10-01")
    pdemo.add_argument("--test-end", default="1984-10-01")

    pfull = sub.add_parser("full671", help="Run the original full 671-basin test or analysis script")
    pfull.add_argument("action", choices=["test", "analyze"])
    pfull.add_argument("--data-root", default=None)
    pfull.add_argument("--output-root", default=None)
    pfull.add_argument("--epoch", type=int, default=30)
    pfull.add_argument("--batch-size", type=int, default=32)
    pfull.add_argument("--rho", type=int, default=365)
    pfull.add_argument("--hidden-size", type=int, default=64)
    pfull.add_argument("--nmul", type=int, default=4)
    pfull.add_argument("--max-iter-ep", type=int, default=200)
    pfull.add_argument("--forcing", default="daymet")
    pfull.add_argument("--gpu-id", type=int, default=0)
    pfull.add_argument("--seed", type=int, default=111111)
    pfull.add_argument("--test-batch", type=int, default=64)
    pfull.add_argument("--chunk-size", type=int, default=64)
    args = p.parse_args()
    if args.mode is None:
        p.error("a mode is required: demo or full671")
    return args


def main():
    args = parse_args()
    if args.mode == "demo":
        evaluate_demo(args)
    else:
        evaluate_full671(args)


if __name__ == "__main__":
    main()
