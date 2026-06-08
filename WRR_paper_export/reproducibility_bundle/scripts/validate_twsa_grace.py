from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_common import compute_kge, ensure_outputs, load_camels_attributes, load_config, save_basin_map, summarize_by_group
from utils_model6_io import load_monthly_archive


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 6:
        return np.nan
    obs = obs[mask]
    sim = sim[mask]
    return float(1.0 - np.sum((sim - obs) ** 2) / max(np.sum((obs - obs.mean()) ** 2), 1e-12))


def basin_metrics(group: pd.DataFrame) -> pd.Series:
    obs = group["grace_twsa_mm"].to_numpy(dtype=float)
    sim = group["TWSA_model"].to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 6:
        return pd.Series({"corr_regional": np.nan, "NSE_regional": np.nan, "KGE_regional": np.nan, "amplitude_ratio": np.nan, "phase_diff_month": np.nan})
    obs = obs[mask]
    sim = sim[mask]
    corr = np.corrcoef(obs, sim)[0, 1]
    amp = np.std(sim) / max(np.std(obs), 1e-12)
    kge_val = np.nan
    if abs(np.mean(obs)) > 1e-6 and abs(np.mean(sim)) > 1e-6:
        kge_val = compute_kge(obs, sim)
    return pd.Series(
        {
            "corr_regional": corr,
            "NSE_regional": nse(obs, sim),
            "KGE_regional": kge_val,
            "amplitude_ratio": amp,
            "phase_diff_month": float(np.argmax(sim) - np.argmax(obs)),
        }
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    ensure_outputs(cfg)

    monthly = load_monthly_archive(root / "configs" / "wrr_model6_config.yaml")
    monthly["date"] = pd.to_datetime(monthly["date"])
    ref_start = pd.Timestamp(cfg["dates"]["anomaly_reference_start"])
    ref_end = pd.Timestamp(cfg["dates"]["anomaly_reference_end"])
    grace = pd.read_parquet(cfg["independent_products"]["grace_jpl_basin_monthly_parquet"])
    grace["date"] = pd.to_datetime(grace["date"])
    metrics = pd.read_csv(cfg["independent_products"]["grace_jpl_metrics_csv"]).rename(
        columns={"R2": "R2_JPL", "NSE": "NSE_JPL", "KGE": "KGE_JPL"}
    )
    attrs = load_camels_attributes(cfg)

    tws = monthly[["basin_id", "date", "SNOWPACK", "MELTWATER", "Sa", "GW"]].copy()
    tws["TWS_model"] = tws[["SNOWPACK", "MELTWATER", "Sa", "GW"]].sum(axis=1)
    ref = tws.loc[(tws["date"] >= ref_start) & (tws["date"] <= ref_end)].groupby("basin_id")["TWS_model"].mean()
    tws["TWSA_model"] = tws["TWS_model"] - tws["basin_id"].map(ref)
    merged = tws.merge(grace, on=["basin_id", "date"], how="inner")

    basin_expl = metrics.merge(attrs, on="basin_id", how="left")
    basin_expl["area_threshold_flag"] = basin_expl["area_gages2"] >= cfg["basins"]["area_threshold_grace_km2"]
    basin_aux = merged.groupby("basin_id", dropna=False).apply(basin_metrics).reset_index()
    basin_expl = basin_expl.merge(basin_aux, on="basin_id", how="left")
    basin_expl.to_csv(Path(cfg["outputs"]["tables_dir"]) / "twsa_validation_by_basin_exploratory.csv", index=False)

    reg_join = merged.merge(attrs[["basin_id", "huc_02", "aridity", "frac_snow", "area_gages2"]], on="basin_id", how="left")
    regional_records = []
    for key, sub in reg_join.groupby("huc_02", dropna=False):
        w = sub["area_gages2"].fillna(1.0)
        agg = sub.groupby("date").apply(
            lambda x: pd.Series(
                {
                    "grace_twsa_mm": np.average(x["grace_twsa_mm"].fillna(0.0), weights=x["area_gages2"].fillna(1.0)),
                    "TWSA_model": np.average(x["TWSA_model"].fillna(0.0), weights=x["area_gages2"].fillna(1.0)),
                }
            )
        ).reset_index()
        m = basin_metrics(agg)
        regional_records.append({"huc_02": key, **m.to_dict(), "n_dates": len(agg)})
    regional = pd.DataFrame(regional_records)
    regional.to_csv(Path(cfg["outputs"]["tables_dir"]) / "twsa_validation_by_region.csv", index=False)

    comparison = pd.DataFrame(
        [
            {"product": "JPL_mascon", "status": "available_local", "n_basin_metrics": int(metrics["NSE_JPL"].notna().sum())},
            {"product": "CSR_mascon", "status": "missing_local_downloader_template_only", "n_basin_metrics": 0},
            {"product": "GSFC_mascon", "status": "missing_local_downloader_template_only", "n_basin_metrics": 0},
        ]
    )
    comparison.to_csv(Path(cfg["outputs"]["tables_dir"]) / "twsa_grace_product_comparison.csv", index=False)

    amp = merged.groupby("basin_id", dropna=False).agg(
        model_amplitude=("TWSA_model", "std"),
        grace_amplitude=("grace_twsa_mm", "std"),
    ).reset_index()
    amp["phase_difference"] = basin_aux["phase_diff_month"]
    amp["amplitude_ratio"] = amp["model_amplitude"] / amp["grace_amplitude"].replace(0, np.nan)

    for col, title in [
        ("model_amplitude", "Model TWSA amplitude"),
        ("grace_amplitude", "GRACE TWSA amplitude"),
        ("R2_JPL", "GRACE JPL exploratory R2"),
        ("corr_regional", "GRACE regional correlation proxy"),
        ("phase_difference", "TWSA phase difference"),
        ("amplitude_ratio", "TWSA amplitude ratio"),
    ]:
        src = amp if col in amp.columns else basin_expl
        save_basin_map(cfg, src[["basin_id", col]], col, Path(cfg["outputs"]["maps_dir"]) / f"map_{col}.png", title, col)

    # Regional time series figure for strongest regions.
    best = regional.sort_values("corr_regional", ascending=False).head(4)["huc_02"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    for ax, region_id in zip(axes.ravel(), best):
        sub = reg_join.loc[reg_join["huc_02"] == region_id].copy()
        if sub.empty:
            continue
        agg = sub.groupby("date").apply(
            lambda x: pd.Series(
                {
                    "grace_twsa_mm": np.average(x["grace_twsa_mm"].fillna(0.0), weights=x["area_gages2"].fillna(1.0)),
                    "TWSA_model": np.average(x["TWSA_model"].fillna(0.0), weights=x["area_gages2"].fillna(1.0)),
                }
            )
        ).reset_index()
        ax.plot(agg["date"], agg["grace_twsa_mm"], label="JPL GRACE")
        ax.plot(agg["date"], agg["TWSA_model"], label="Model")
        ax.set_title(f"HUC2 {region_id}")
    axes[0, 0].legend()
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(Path(cfg["outputs"]["figures_dir"]) / f"twsa_regional_timeseries.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
