#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd


META_COLUMNS = ["basin_id", "gage_name", "lat", "lon", "area_km2"]
DYNAMIC_COLUMNS = ["prcp_mm_day", "tmean_c", "pet_mm_day", "qobs_mm_day", "sin_doy", "cos_doy"]
SERIES_NORM_COLUMNS = ["prcp_mm_day", "tmean_c", "pet_mm_day"]


def load_demo_static(demo_root: Union[str, Path]) -> pd.DataFrame:
    demo_root = Path(demo_root)
    return pd.read_csv(demo_root / "static_attributes_5_basins.csv", dtype={"basin_id": str})


def load_demo_basin_timeseries(demo_root: Union[str, Path], basin_id) -> pd.DataFrame:
    demo_root = Path(demo_root)
    basin_id = str(basin_id)
    df = pd.read_csv(demo_root / "basins" / f"{basin_id}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_demo_basin_ids(demo_root: Union[str, Path]) -> List[str]:
    demo_root = Path(demo_root)
    with open(demo_root / "demo_5_basin_ids.json", "r") as fp:
        basin_ids = json.load(fp)
    return [str(x) for x in basin_ids]


def _safe_stats(arr: np.ndarray) -> Dict[str, np.ndarray]:
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def build_demo_dataset(
    demo_root,
    train_start: str = "1980-10-01",
    train_end: str = "1995-10-01",
    test_start: str = "1995-10-01",
    test_end: str = "2010-10-01",
    bufftime: int = 365,
):
    demo_root = Path(demo_root)
    static_df = load_demo_static(demo_root)
    basin_ids = static_df["basin_id"].astype(str).tolist()
    attr_columns = [c for c in static_df.columns if c not in META_COLUMNS]

    train_frames = []
    test_frames = []
    dynamic_frames = {}
    for basin_id in basin_ids:
        df = load_demo_basin_timeseries(demo_root, basin_id)
        dynamic_frames[basin_id] = df
        train_frames.append(df[(df["date"] >= train_start) & (df["date"] < train_end)].copy())
        test_frames.append(df[(df["date"] >= test_start) & (df["date"] < test_end)].copy())

    attr_vals = static_df[attr_columns].to_numpy(dtype=np.float32)
    attr_stats = _safe_stats(attr_vals)
    attrs_norm = (attr_vals - attr_stats["mean"]) / attr_stats["std"]
    attrs_norm[np.isnan(attrs_norm)] = 0.0

    snow_frac_raw = static_df["frac_snow"].to_numpy(dtype=np.float32).reshape(-1, 1)

    train_series = np.concatenate([df[SERIES_NORM_COLUMNS].to_numpy(dtype=np.float32) for df in train_frames], axis=0)
    series_stats = _safe_stats(train_series)

    def _pack(frames: List[pd.DataFrame], include_static: bool = False):
        x_list, z_list, y_list = [], [], []
        for i, df in enumerate(frames):
            series = df[SERIES_NORM_COLUMNS].to_numpy(dtype=np.float32)
            series_norm = (series - series_stats["mean"]) / series_stats["std"]
            series_norm[np.isnan(series_norm)] = 0.0
            x = df[["prcp_mm_day", "tmean_c", "pet_mm_day", "sin_doy", "cos_doy"]].to_numpy(dtype=np.float32)
            y = df[["qobs_mm_day"]].to_numpy(dtype=np.float32)
            snow_ts = np.repeat(snow_frac_raw[i : i + 1], len(df), axis=0)
            z = np.concatenate([series_norm, snow_ts], axis=1)
            if include_static:
                c_rep = np.repeat(attrs_norm[i : i + 1], len(df), axis=0)
                z = np.concatenate([z, c_rep], axis=1)
            x_list.append(x)
            z_list.append(z)
            y_list.append(y)
        return (
            np.stack(x_list, axis=0),
            np.stack(z_list, axis=0),
            np.stack(y_list, axis=0),
        )

    x_train, z_train, y_train = _pack(train_frames, include_static=False)
    x_test, z_test, y_test = _pack(test_frames, include_static=False)

    return {
        "basin_ids": basin_ids,
        "static_df": static_df,
        "dynamic_frames": dynamic_frames,
        "attr_columns": attr_columns,
        "series_norm_columns": SERIES_NORM_COLUMNS,
        "attrs": attrs_norm.astype(np.float32),
        "attr_stats": attr_stats,
        "series_stats": series_stats,
        "x_train": x_train.astype(np.float32),
        "z_train": z_train.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "z_test": z_test.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "train_dates": train_frames[0]["date"].reset_index(drop=True),
        "test_dates": test_frames[0]["date"].reset_index(drop=True),
        "bufftime": bufftime,
    }


def build_demo_eval_inputs(dataset: dict, use_full_train_warmup=True):
    if use_full_train_warmup:
        warm_len = dataset["x_train"].shape[1]
    else:
        warm_len = int(dataset["bufftime"])
    x_buff = dataset["x_train"][:, -warm_len:, :]
    z_buff = dataset["z_train"][:, -warm_len:, :]
    x_eval = np.concatenate([x_buff, dataset["x_test"]], axis=1)
    z_series = np.concatenate([z_buff, dataset["z_test"]], axis=1)
    c_rep = np.repeat(dataset["attrs"][:, None, :], z_series.shape[1], axis=1)
    z_eval = np.concatenate([z_series, c_rep], axis=2)
    return x_eval.astype(np.float32), z_eval.astype(np.float32), dataset["y_test"].astype(np.float32)
