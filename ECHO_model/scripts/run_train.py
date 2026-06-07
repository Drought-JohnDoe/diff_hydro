#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echo_model.config import DEFAULT_CONFIG, ExperimentConfig
from echo_model.data import build_demo_dataset, load_camels_dataset, save_dataset_summary
from echo_model.losses import RmseLossComb
from echo_model.rnn import MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple
from echo_model.train import train_model


def parse_args():
    p = argparse.ArgumentParser(description="Train standalone ECHO model")
    p.add_argument("--mode", choices=["demo", "camels"], default="demo")
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "demo_data"))
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "outputs" / "demo_run"))
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    p.add_argument("--rho", type=int, default=DEFAULT_CONFIG.rho)
    p.add_argument("--bufftime", type=int, default=DEFAULT_CONFIG.bufftime)
    p.add_argument("--max-iter-ep", type=int, default=DEFAULT_CONFIG.max_iter_ep)
    p.add_argument("--hidden-size", type=int, default=DEFAULT_CONFIG.hidden_size)
    p.add_argument("--nmul", type=int, default=DEFAULT_CONFIG.nmul)
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    p.add_argument("--gpu-id", type=int, default=DEFAULT_CONFIG.gpu_id)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.mode == "demo":
        dataset = build_demo_dataset(args.data_root, bufftime=args.bufftime)
    else:
        dataset = load_camels_dataset(args.data_root, bufftime=args.bufftime)
    save_dataset_summary(dataset, out_dir / "dataset_summary.json")
    model = MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple(
        ninv=dataset["z_train"].shape[-1] + dataset["attrs"].shape[-1],
        nmul=args.nmul,
        nattr=dataset["attrs"].shape[-1],
        hiddeninv=args.hidden_size,
        inittime=args.bufftime,
        routOpt=True,
        comprout=False,
        compwts=True,
        lgdyn=True,
        lgdynweight=0.6,
        dynamic_sq=True,
        dynamic_etgam=True,
        dynamic_partition=True,
        dynamic_cfmax_snow=True,
        dynamic_routing_scale=False,
        component_routing=True,
    )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    cfg = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        rho=args.rho,
        bufftime=args.bufftime,
        max_iter_ep=args.max_iter_ep,
        hidden_size=args.hidden_size,
        nmul=args.nmul,
        seed=args.seed,
        gpu_id=args.gpu_id,
        use_gpu=not args.cpu,
    )
    (out_dir / "train_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    train_model(
        model,
        dataset,
        RmseLossComb(alpha=0.25),
        epochs=args.epochs,
        batch_size=args.batch_size,
        rho=args.rho,
        bufftime=args.bufftime,
        max_iter_ep=args.max_iter_ep,
        out_dir=out_dir,
        seed=args.seed,
        use_gpu=not args.cpu,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
