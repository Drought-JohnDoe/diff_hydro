from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_common import (
    compute_kge,
    ensure_outputs,
    hydroclass_columns,
    load_camels_attributes,
    load_config,
    save_basin_map,
    summarize_by_group,
)
from utils_model6_io import load_daily_archive, load_main_closed_metrics


def compute_daily_metrics(group: pd.DataFrame) -> pd.Series:
    obs = group["Q_obs"].to_numpy(dtype=float)
    sim = group["Q_process"].to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 10:
        return pd.Series(
            {
                "corr_daily_aux": np.nan,
                "rmse_daily_aux": np.nan,
                "bias_daily_aux": np.nan,
                "logNSE_daily_aux": np.nan,
                "runoff_ratio_error_aux": np.nan,
                "seasonality_error_aux": np.nan,
                "fdc_error_aux": np.nan,
            }
        )
    obs = obs[mask]
    sim = sim[mask]
    corr = np.corrcoef(obs, sim)[0, 1]
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    bias = np.mean(sim - obs)
    log_obs = np.log1p(np.clip(obs, 0, None))
    log_sim = np.log1p(np.clip(sim, 0, None))
    log_nse = 1.0 - np.sum((log_sim - log_obs) ** 2) / max(np.sum((log_obs - log_obs.mean()) ** 2), 1e-12)
    runoff_ratio_error = sim.sum() / max(group["P"].sum(), 1e-12) - obs.sum() / max(group["P"].sum(), 1e-12)
    month = pd.to_datetime(group["date"]).dt.month
    seas = (
        pd.DataFrame({"month": month, "obs": group["Q_obs"], "sim": group["Q_process"]})
        .groupby("month")[["obs", "sim"]]
        .mean()
    )
    seasonality_error = float(np.mean(np.abs(seas["sim"] - seas["obs"])))
    q_obs = np.quantile(obs, np.linspace(0.05, 0.95, 10))
    q_sim = np.quantile(sim, np.linspace(0.05, 0.95, 10))
    fdc_error = float(np.mean(np.abs(q_sim - q_obs)))
    return pd.Series(
        {
            "corr_daily_aux": corr,
            "rmse_daily_aux": rmse,
            "bias_daily_aux": bias,
            "logNSE_daily_aux": log_nse,
            "runoff_ratio_error_aux": runoff_ratio_error,
            "seasonality_error_aux": seasonality_error,
            "fdc_error_aux": fdc_error,
        }
    )


def representative_basins(df: pd.DataFrame) -> list[int]:
    picks = []
    for _, sub in [
        ("humid", df[df["aridity"] < 0.8]),
        ("arid", df[df["aridity"] > 1.2]),
        ("snow", df[df["frac_snow"] > 0.35]),
        ("lowflow_fail", df.nsmallest(20, "low_flow_NSE")),
        ("strong", df.nlargest(20, "NSE")),
    ]:
        if len(sub) > 0:
            picks.append(int(sub.iloc[0]["basin_id"]))
    return list(dict.fromkeys(picks))[:5]


def plot_hydrographs(daily: pd.DataFrame, basin_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for basin_id in representative_basins(basin_df):
        sub = daily.loc[daily["basin_id"] == basin_id].copy()
        if sub.empty:
            continue
        sub["date"] = pd.to_datetime(sub["date"])
        year = sub["date"].dt.year.mode().iloc[0]
        window = sub.loc[sub["date"].dt.year == year].head(365)
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(window["date"], window["Q_obs"], label="Observed Q", lw=1.2)
        ax.plot(window["date"], window["Q_process"], label="Model Q (aux daily archive)", lw=1.2)
        ax.set_title(f"Basin {basin_id} hydrograph sample ({year})")
        ax.set_ylabel("mm/day")
        ax.legend()
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(out_dir / f"hydrograph_{basin_id}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    ensure_outputs(cfg)

    metrics = load_main_closed_metrics(root / "configs" / "wrr_model6_config.yaml")
    daily = load_daily_archive(root / "configs" / "wrr_model6_config.yaml")
    aux_metrics = daily.groupby("basin_id", dropna=False).apply(compute_daily_metrics).reset_index()
    attrs = hydroclass_columns(load_camels_attributes(cfg))
    basin = metrics.merge(aux_metrics, on="basin_id", how="left").merge(attrs, on="basin_id", how="left", suffixes=("", "_attr"))
    basin["metrics_source_main"] = "Ep100 benchmark"
    basin["metrics_source_aux_daily"] = "Ep60 auxiliary archive"
    out_basin = Path(cfg["outputs"]["tables_dir"]) / "streamflow_metrics_by_basin.csv"
    basin.to_csv(out_basin, index=False)

    region = summarize_by_group(basin, ["huc_02"], ["NSE", "KGE", "R2", "low_flow_NSE", "high_flow_NSE", "logNSE_daily_aux"])
    region.to_csv(Path(cfg["outputs"]["tables_dir"]) / "streamflow_metrics_by_region.csv", index=False)
    hydro = summarize_by_group(
        basin,
        ["aridity_class", "snow_class", "forest_class", "baseflow_class"],
        ["NSE", "KGE", "R2", "low_flow_NSE", "high_flow_NSE", "logNSE_daily_aux"],
    )
    hydro.to_csv(Path(cfg["outputs"]["tables_dir"]) / "streamflow_metrics_by_hydroclass.csv", index=False)

    for col, title in [
        ("NSE", "Model 6 Closed NSE"),
        ("KGE", "Model 6 Closed KGE"),
        ("logNSE_daily_aux", "Model 6 Closed log NSE (aux daily)"),
        ("low_flow_NSE", "Model 6 Closed low-flow NSE"),
        ("high_flow_NSE", "Model 6 Closed high-flow NSE"),
        ("FHV", "Model 6 Closed FHV"),
        ("FLV", "Model 6 Closed FLV"),
        ("bias_daily_aux", "Model 6 Closed Q bias (aux daily)"),
    ]:
        save_basin_map(cfg, basin[["basin_id", col]], col, Path(cfg["outputs"]["maps_dir"]) / f"map_{col}.png", title, col)

    plot_hydrographs(daily, basin, Path(cfg["outputs"]["figures_dir"]) / "hydrographs")


if __name__ == "__main__":
    main()

