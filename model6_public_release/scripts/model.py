#!/usr/bin/env python3

import sys
from pathlib import Path


def release_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_hydrodl_on_path():
    root = release_root()
    hydrodl_root = root / "code" / "dPLHBVrelease" / "hydroDL-dev"
    if str(hydrodl_root) not in sys.path:
        sys.path.insert(0, str(hydrodl_root))


ensure_hydrodl_on_path()

from hydroDL.model import rnn  # noqa: E402


def build_model_six(ninv: int, nattr: int, hidden_size: int = 64, nmul: int = 4, inittime: int = 365):
    return rnn.MultiInv_DynamicSimHydModelSix(
        ninv=ninv,
        nmul=nmul,
        nattr=nattr,
        hiddeninv=hidden_size,
        inittime=inittime,
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
        dynamic_all=False,
        reg_amp_w=1e-3,
        reg_smooth_w=1e-3,
        reg_part_w=1e-3,
        component_routing=True,
        dry_channel_loss=True,
        zero_flow_gate=True,
        channel_loss_max=0.60,
        zero_gate_hidden=None,
    )


MultiInv_DynamicSimHydModelSix = rnn.MultiInv_DynamicSimHydModelSix
