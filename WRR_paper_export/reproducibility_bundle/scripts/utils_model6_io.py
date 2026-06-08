from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_common import load_config


def load_main_closed_metrics(config_path: str | Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    df = pd.read_csv(cfg["model"]["ep100_metrics_csv"])
    return df.loc[df["model"] == "Model6Closed_Snow_aSrz_SIMHYD_Simple"].copy()


def load_main_closed_water_balance(config_path: str | Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    df = pd.read_csv(cfg["model"]["ep100_water_balance_csv"])
    return df.loc[df["model"] == "Model6Closed_Snow_aSrz_SIMHYD_Simple"].copy()


def load_daily_archive(config_path: str | Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    return pd.read_parquet(Path(cfg["outputs"]["diagnostics_dir"]) / "model6_ep100_daily_basin_day.parquet")


def load_monthly_archive(config_path: str | Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    return pd.read_parquet(Path(cfg["outputs"]["diagnostics_dir"]) / "model6_ep100_monthly_basin.parquet")

