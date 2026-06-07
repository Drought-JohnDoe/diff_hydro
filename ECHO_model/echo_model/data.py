import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ATTR_LST


META_COLUMNS = ["basin_id", "gage_name", "lat", "lon", "area_km2"]
DEMO_SERIES_NORM_COLUMNS = ["prcp_mm_day", "tmean_c", "pet_mm_day"]


def _safe_stats(arr: np.ndarray):
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _seasonal_features(date_index: pd.DatetimeIndex):
    doy = date_index.dayofyear.to_numpy(dtype=np.float32)
    ang = 2.0 * np.pi * (doy - 1.0) / 365.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def load_demo_static(demo_root):
    demo_root = Path(demo_root)
    return pd.read_csv(demo_root / "static_attributes_5_basins.csv", dtype={"basin_id": str})


def load_demo_basin_timeseries(demo_root, basin_id):
    demo_root = Path(demo_root)
    df = pd.read_csv(demo_root / "basins" / f"{str(basin_id)}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_demo_dataset(
    demo_root,
    train_start="1980-10-01",
    train_end="1995-10-01",
    test_start="1995-10-01",
    test_end="2010-10-01",
    bufftime=365,
):
    demo_root = Path(demo_root)
    static_df = load_demo_static(demo_root)
    basin_ids = static_df["basin_id"].astype(str).tolist()
    attr_columns = [c for c in static_df.columns if c not in META_COLUMNS]
    train_frames, test_frames, dynamic_frames = [], [], {}
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
    train_series = np.concatenate([df[DEMO_SERIES_NORM_COLUMNS].to_numpy(dtype=np.float32) for df in train_frames], axis=0)
    series_stats = _safe_stats(train_series)

    def _pack(frames, include_static=False):
        x_list, z_list, y_list = [], [], []
        for i, df in enumerate(frames):
            series = df[DEMO_SERIES_NORM_COLUMNS].to_numpy(dtype=np.float32)
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
        return np.stack(x_list, axis=0), np.stack(z_list, axis=0), np.stack(y_list, axis=0)

    x_train, z_train, y_train = _pack(train_frames, include_static=False)
    x_test, z_test, y_test = _pack(test_frames, include_static=False)
    x_hist = x_train[:, -bufftime:, :]
    z_hist = np.concatenate(
        [
            z_train[:, -bufftime:, :],
            np.repeat(attrs_norm[:, None, :], bufftime, axis=1),
        ],
        axis=2,
    )
    z_test_eval = np.concatenate(
        [z_test, np.repeat(attrs_norm[:, None, :], z_test.shape[1], axis=1)],
        axis=2,
    )
    return {
        "basin_ids": basin_ids,
        "static_df": static_df,
        "dynamic_frames": dynamic_frames,
        "attr_columns": attr_columns,
        "attrs": attrs_norm.astype(np.float32),
        "attr_stats": attr_stats,
        "series_stats": series_stats,
        "x_train": x_train.astype(np.float32),
        "z_train": z_train.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "z_test": z_test.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "x_eval": np.concatenate([x_hist, x_test], axis=1).astype(np.float32),
        "z_eval": np.concatenate([z_hist, z_test_eval], axis=1).astype(np.float32),
        "obs_test": y_test[:, :, 0].astype(np.float32),
        "train_dates": train_frames[0]["date"].reset_index(drop=True),
        "test_dates": test_frames[0]["date"].reset_index(drop=True),
        "bufftime": bufftime,
        "mode": "demo",
    }


def _load_camels_attributes(root_db: Path):
    attr_root = root_db / "camels_attributes_v2.0" / "camels_attributes_v2.0"
    tables = []
    for fname in sorted(attr_root.glob("camels_*.txt")):
        tables.append(pd.read_csv(fname, sep=";"))
    out = tables[0]
    for tbl in tables[1:]:
        out = out.merge(tbl, on="gauge_id", how="outer")
    out["gauge_id"] = out["gauge_id"].astype(str).str.zfill(8)
    return out


def _find_forcing_file(root_db: Path, forcing_name: str, basin_id: str):
    base = root_db / "basin_timeseries_v1p2_metForcing_obsFlow" / "basin_dataset_public_v1p2" / "basin_mean_forcing" / forcing_name
    matches = list(base.glob(f"*/{basin_id}_*forcing_leap.txt"))
    if not matches:
        raise FileNotFoundError(f"forcing file not found for {basin_id}")
    return matches[0]


def _find_streamflow_file(root_db: Path, basin_id: str):
    base = root_db / "basin_timeseries_v1p2_metForcing_obsFlow" / "basin_dataset_public_v1p2" / "usgs_streamflow"
    matches = list(base.glob(f"*/{basin_id}_streamflow_qc.txt"))
    if not matches:
        raise FileNotFoundError(f"streamflow file not found for {basin_id}")
    return matches[0]


def _read_forcing_file(path: Path):
    lines = path.read_text().splitlines()
    area = float(lines[2])
    df = pd.read_csv(path, sep="\\s+", skiprows=3)
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Mnth"], day=df["Day"]))
    df["tmean_c"] = ((df["tmax(C)"] + df["tmin(C)"]) / 2.0).astype(np.float32)
    df["prcp_mm_day"] = df["prcp(mm/day)"].astype(np.float32)
    return df[["date", "prcp_mm_day", "tmean_c"]], area


def _read_pet_file(path: Path):
    df = pd.read_csv(path)
    date_col = df.columns[0]
    val_col = df.columns[1]
    df["date"] = pd.to_datetime(df[date_col])
    df["pet_mm_day"] = df[val_col].astype(np.float32)
    return df[["date", "pet_mm_day"]]


def _read_streamflow_file(path: Path, area_km2: float):
    df = pd.read_csv(path, sep="\\s+", header=None, names=["basin_id", "year", "month", "day", "q_cfs", "flag"])
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    q = df["q_cfs"].astype(np.float32).to_numpy()
    q_mm_day = (q * 0.0283168 * 3600 * 24) / (area_km2 * 1e6) * 1e3
    return pd.DataFrame({"date": df["date"], "qobs_mm_day": q_mm_day.astype(np.float32)})


def _fit_attr_norm(attr_df: pd.DataFrame):
    attr_vals = attr_df[ATTR_LST].to_numpy(dtype=np.float32)
    stats = _safe_stats(attr_vals)
    norm = (attr_vals - stats["mean"]) / stats["std"]
    norm[np.isnan(norm)] = 0.0
    return norm.astype(np.float32), stats


def load_camels_dataset(
    root_db,
    basin_ids=None,
    train_start="1980-10-01",
    train_end="1995-10-01",
    test_start="1995-10-01",
    test_end="2010-10-01",
    forcing_name="daymet",
    bufftime=365,
):
    root_db = Path(root_db)
    attr_df = _load_camels_attributes(root_db)
    if basin_ids is None:
        basin_ids = attr_df["gauge_id"].dropna().astype(str).tolist()
    else:
        basin_ids = [str(x).zfill(8) for x in basin_ids]
        attr_df = attr_df[attr_df["gauge_id"].isin(basin_ids)].copy()
    attr_df = attr_df.sort_values("gauge_id").reset_index(drop=True)
    basin_ids = attr_df["gauge_id"].tolist()
    attrs_norm, attr_stats = _fit_attr_norm(attr_df)
    snow_frac_raw = attr_df[["frac_snow"]].to_numpy(dtype=np.float32)

    train_frames = []
    test_frames = []
    full_frames = {}
    for basin_id in basin_ids:
        forc_df, area_km2 = _read_forcing_file(_find_forcing_file(root_db, forcing_name, basin_id))
        pet_df = _read_pet_file(root_db / "pet_harg" / forcing_name / f"{int(basin_id)}.csv")
        q_df = _read_streamflow_file(_find_streamflow_file(root_db, basin_id), area_km2)
        df = forc_df.merge(pet_df, on="date", how="inner").merge(q_df, on="date", how="inner")
        season = _seasonal_features(pd.DatetimeIndex(df["date"]))
        df["sin_doy"] = season[:, 0]
        df["cos_doy"] = season[:, 1]
        full_frames[basin_id] = df
        train_frames.append(df[(df["date"] >= train_start) & (df["date"] < train_end)].copy())
        test_frames.append(df[(df["date"] >= test_start) & (df["date"] < test_end)].copy())
    train_series = np.concatenate(
        [df[["prcp_mm_day", "tmean_c", "pet_mm_day"]].to_numpy(dtype=np.float32) for df in train_frames],
        axis=0,
    )
    series_stats = _safe_stats(train_series)

    def _pack(frames, include_static=False):
        x_list, z_list, y_list = [], [], []
        for i, df in enumerate(frames):
            series = df[["prcp_mm_day", "tmean_c", "pet_mm_day"]].to_numpy(dtype=np.float32)
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
        return np.stack(x_list, axis=0), np.stack(z_list, axis=0), np.stack(y_list, axis=0)

    x_train, z_train, y_train = _pack(train_frames, include_static=False)
    x_test, z_test, y_test = _pack(test_frames, include_static=False)
    x_hist = x_train[:, -bufftime:, :]
    z_hist = np.concatenate(
        [
            z_train[:, -bufftime:, :],
            np.repeat(attrs_norm[:, None, :], bufftime, axis=1),
        ],
        axis=2,
    )
    z_test_eval = np.concatenate(
        [z_test, np.repeat(attrs_norm[:, None, :], z_test.shape[1], axis=1)],
        axis=2,
    )
    x_eval = np.concatenate([x_hist, x_test], axis=1).astype(np.float32)
    z_eval = np.concatenate([z_hist, z_test_eval], axis=1).astype(np.float32)
    meta = attr_df.rename(columns={"gauge_id": "basin_id"}).copy()
    if "area_gages2" in meta.columns and "area_km2" not in meta.columns:
        meta["area_km2"] = meta["area_gages2"]
    return {
        "basin_ids": basin_ids,
        "static_df": meta,
        "dynamic_frames": full_frames,
        "attr_columns": ATTR_LST,
        "attrs": attrs_norm.astype(np.float32),
        "attr_stats": attr_stats,
        "series_stats": series_stats,
        "x_train": x_train.astype(np.float32),
        "z_train": z_train.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "z_test": z_test.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "x_eval": x_eval,
        "z_eval": z_eval,
        "obs_test": y_test[:, :, 0].astype(np.float32),
        "train_dates": train_frames[0]["date"].reset_index(drop=True),
        "test_dates": test_frames[0]["date"].reset_index(drop=True),
        "bufftime": bufftime,
        "mode": "camels",
    }


def save_dataset_summary(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": dataset["mode"],
        "n_basins": len(dataset["basin_ids"]),
        "n_train_days": int(dataset["x_train"].shape[1]),
        "n_test_days": int(dataset["x_test"].shape[1]),
        "attr_columns": dataset["attr_columns"],
    }
    path.write_text(json.dumps(payload, indent=2))
