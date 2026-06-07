#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echo_model.data import build_demo_dataset, load_camels_dataset
from echo_model.evaluate import evaluate_model
from echo_model.rnn import MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple
from echo_model.train_utils import get_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate standalone ECHO model")
    p.add_argument("--mode", choices=["demo", "camels"], default="demo")
    p.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "demo_data"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-path", default=str(Path(__file__).resolve().parents[1] / "outputs" / "eval_metrics.json"))
    p.add_argument("--bufftime", type=int, default=365)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = build_demo_dataset(args.data_root, bufftime=args.bufftime) if args.mode == "demo" else load_camels_dataset(args.data_root, bufftime=args.bufftime)
    model = MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple(
        ninv=dataset["z_train"].shape[-1] + dataset["attrs"].shape[-1],
        nmul=4,
        nattr=dataset["attrs"].shape[-1],
        hiddeninv=64,
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
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    rows, _ = evaluate_model(model, dataset, get_device(use_gpu=not args.cpu, gpu_id=args.gpu_id))
    Path(args.out_path).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
