#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SIX_DIR = ROOT / "code" / "dPLHBVrelease" / "hydroDL-dev" / "example" / "model_six"


def load_config(path):
    with open(path, "r") as fp:
        return json.load(fp)


def bool_flag(name, value):
    return [f"--{name.replace('_', '-')}" if value else f"--no-{name.replace('_', '-')}"]


def run_cmd(args, cwd):
    print("Running:", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def build_common_args(cfg):
    args = [
        "--gpu-id", str(cfg.get("gpu_id", 0)),
        "--batch-size", str(cfg.get("batch_size", 32)),
        "--rho", str(cfg.get("rho", 365)),
        "--hidden-size", str(cfg.get("hidden_size", 64)),
        "--max-iter-ep", str(cfg.get("max_iter_ep", 200)),
        "--nmul", str(cfg.get("nmul", 4)),
        "--forcing", cfg.get("forcing", "daymet"),
        "--seed", str(cfg.get("seed", 111111)),
    ]
    if cfg.get("use_all_basins", True):
        args += ["--use-all-basins"]
    else:
        args += ["--subset-limit", str(cfg.get("subset_limit", 64))]
    if cfg.get("exp_info_suffix"):
        args += ["--exp-info-suffix", cfg["exp_info_suffix"]]

    for key, default in [
        ("routing", True),
        ("comprout", False),
        ("compwts", True),
        ("lgdyn", True),
        ("dynamic_sq", True),
        ("dynamic_etgam", True),
        ("dynamic_partition", True),
        ("dynamic_cfmax_snow", True),
        ("dynamic_routing_scale", False),
        ("dynamic_all", False),
        ("component_routing", True),
        ("dry_channel_loss", True),
        ("zero_flow_gate", True),
    ]:
        args += bool_flag(key, cfg.get(key, default))

    if "lgdyn_weight" in cfg:
        args += ["--lgdyn-weight", str(cfg["lgdyn_weight"])]
    if "reg_amp_w" in cfg:
        args += ["--reg-amp-w", str(cfg["reg_amp_w"])]
    if "reg_smooth_w" in cfg:
        args += ["--reg-smooth-w", str(cfg["reg_smooth_w"])]
    if "reg_part_w" in cfg:
        args += ["--reg-part-w", str(cfg["reg_part_w"])]
    if "channel_loss_max" in cfg:
        args += ["--channel-loss-max", str(cfg["channel_loss_max"])]
    if cfg.get("zero_gate_hidden") is not None:
        args += ["--zero-gate-hidden", str(cfg["zero_gate_hidden"])]
    return args


def cmd_train(cfg):
    cmd = [sys.executable, "trainModelSix.py", "--epochs", str(cfg.get("epochs", 10)), "--save-epoch", str(cfg.get("save_epoch", 1))]
    cmd += build_common_args(cfg)
    run_cmd(cmd, MODEL_SIX_DIR)


def cmd_resume(cfg):
    cmd = [
        sys.executable, "continueModelSix.py",
        "--start-epoch", str(cfg["start_epoch"]),
        "--end-epoch", str(cfg["end_epoch"]),
    ]
    cmd += build_common_args(cfg)
    run_cmd(cmd, MODEL_SIX_DIR)


def cmd_test(cfg):
    cmd = [sys.executable, "testModelSix.py", "--epoch", str(cfg["epoch"]), "--test-batch", str(cfg.get("test_batch", 64))]
    cmd += build_common_args(cfg)
    run_cmd(cmd, MODEL_SIX_DIR)


def cmd_analyze(cfg):
    cmd = [sys.executable, "analyzeModelSix.py", "--epoch", str(cfg["epoch"]), "--chunk-size", str(cfg.get("chunk_size", 64))]
    cmd += build_common_args(cfg)
    run_cmd(cmd, MODEL_SIX_DIR)


def cmd_report(cfg):
    hbv_eva = cfg.get("hbv_eva_path", str(ROOT / "benchmarks" / "hbv_ep10" / "Eva10.npy"))
    result_suffix = cfg.get("result_suffix", "Train19801001_19951001Test19951001_20101001_ModelSixAll671_BS32_HS64_MaxIter200")
    analysis_dir = cfg.get("analysis_dir", str(ROOT / "outputs" / "rnnStreamflow" / "CAMELSMODELSIX" / "DynamicSimHydModelSix" / "AllBasins" / "daymet" / "111111" / f"analysis_ep{cfg['epoch']}"))
    out_dir = cfg.get("report_out_dir", str(ROOT / "outputs" / f"report_model_six_vs_hbv_ep{cfg['epoch']}"))
    subset_tag = cfg.get("subset_tag", "AllBasins")
    cmd = [
        sys.executable, str(ROOT / "scripts" / "report_vs_hbv.py"),
        "--epoch", str(cfg["epoch"]),
        "--result-suffix", result_suffix,
        "--analysis-dir", analysis_dir,
        "--out-dir", out_dir,
        "--subset-tag", subset_tag,
        "--hbv-eva-path", hbv_eva,
    ]
    run_cmd(cmd, ROOT)


def cmd_nse(cfg):
    import numpy as np
    eva_path = Path(cfg.get("eva_path", ROOT / "outputs" / "rnnStreamflow" / "CAMELSMODELSIX" / "DynamicSimHydModelSix" / "AllBasins" / "daymet" / "111111" / "Train19801001_19951001Test19951001_20101001_ModelSixAll671_BS32_HS64_MaxIter200" / f"Eva{cfg['epoch']}.npy"))
    eva = np.load(eva_path, allow_pickle=True)[0]
    print("Eva path:", eva_path)
    print("Median NSE:", float(np.nanmedian(eva["NSE"])))


def main():
    p = argparse.ArgumentParser(description="Configurable runner for final Model 6 experiments")
    p.add_argument("action", choices=["train", "resume", "test", "analyze", "report", "nse"])
    p.add_argument("--config", required=True, help="Path to JSON config file")
    args = p.parse_args()

    cfg = load_config(args.config)
    {
        "train": cmd_train,
        "resume": cmd_resume,
        "test": cmd_test,
        "analyze": cmd_analyze,
        "report": cmd_report,
        "nse": cmd_nse,
    }[args.action](cfg)


if __name__ == "__main__":
    main()
