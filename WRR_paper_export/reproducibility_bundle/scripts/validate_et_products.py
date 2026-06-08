from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_common import ensure_outputs, load_camels_attributes, load_config, save_basin_map, summarize_by_group
from utils_model6_io import load_monthly_archive


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    ensure_outputs(cfg)

    monthly = load_monthly_archive(root / "configs" / "wrr_model6_config.yaml")
    monthly["date"] = pd.to_datetime(monthly["date"])
    flux = pd.read_parquet(cfg["independent_products"]["fluxcom_monthly_parquet"])
    flux["date"] = pd.to_datetime(flux["date"])
    merged = monthly.merge(flux, on=["basin_id", "date"], how="left")
    merged["et_model_mm_month"] = merged["ET_model"]
    merged["et_bias_fluxcom_mm_month"] = merged["et_model_mm_month"] - merged["fluxcom_et_mm_month"]

    flux_metrics = pd.read_csv(cfg["independent_products"]["fluxcom_metrics_csv"]).rename(
        columns={"R2": "R2_FLUXCOM", "NSE": "NSE_FLUXCOM", "KGE": "KGE_FLUXCOM"}
    )
    mod16 = pd.read_csv(cfg["independent_products"]["mod16_annual_csv"]).rename(
        columns={"NSE": "NSE_MOD16", "KGE": "KGE_MOD16", "et_corr": "corr_MOD16"}
    )
    attrs = load_camels_attributes(cfg)

    basin_model = merged.groupby("basin_id", dropna=False).agg(
        model_mean_ET_mm_month=("et_model_mm_month", "mean"),
        model_mean_P_mm_month=("P", "mean"),
        fluxcom_mean_ET_mm_month=("fluxcom_et_mm_month", "mean"),
        fluxcom_valid_fraction=("valid_pixel_fraction", "mean"),
    ).reset_index()
    basin_model["model_ET_over_P"] = basin_model["model_mean_ET_mm_month"] / basin_model["model_mean_P_mm_month"]
    basin_model["ET_bias_FLUXCOM"] = basin_model["model_mean_ET_mm_month"] - basin_model["fluxcom_mean_ET_mm_month"]

    by_basin = basin_model.merge(flux_metrics, on="basin_id", how="left").merge(mod16, on="basin_id", how="left").merge(attrs, on="basin_id", how="left")
    by_basin["GLEAM_status"] = "missing_local_empty_product"
    by_basin.to_csv(Path(cfg["outputs"]["tables_dir"]) / "et_validation_by_basin.csv", index=False)

    summary_region = summarize_by_group(
        by_basin,
        ["huc_02"],
        ["model_ET_over_P", "ET_bias_FLUXCOM", "R2_FLUXCOM", "NSE_FLUXCOM", "KGE_FLUXCOM", "NSE_MOD16", "KGE_MOD16"],
    )
    summary_region.to_csv(Path(cfg["outputs"]["tables_dir"]) / "et_validation_summary_by_region.csv", index=False)

    uncertainty = by_basin[["basin_id", "fluxcom_mean_ET_mm_month", "model_mean_ET_mm_month"]].merge(
        mod16[["basin_id", "mod16_et_mean_mm_yr", "model6_et_mean_mm_yr"]], on="basin_id", how="left"
    )
    uncertainty["ET_product_uncertainty"] = (
        (uncertainty["fluxcom_mean_ET_mm_month"] * 12.0) - uncertainty["mod16_et_mean_mm_yr"]
    ).abs()

    for col, title in [
        ("model_mean_ET_mm_month", "Model mean ET"),
        ("model_ET_over_P", "Model ET/P"),
        ("ET_bias_FLUXCOM", "ET bias vs FLUXCOM"),
        ("R2_FLUXCOM", "ET R2 vs FLUXCOM"),
        ("ET_product_uncertainty", "ET product spread"),
    ]:
        source = uncertainty if col == "ET_product_uncertainty" else by_basin
        save_basin_map(cfg, source[["basin_id", col]], col, Path(cfg["outputs"]["maps_dir"]) / f"map_{col}.png", title, col)

    seasonal = merged.copy()
    seasonal["month"] = seasonal["date"].dt.month
    seasonal_cycle = seasonal.groupby("month")[["et_model_mm_month", "fluxcom_et_mm_month"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(seasonal_cycle["month"], seasonal_cycle["et_model_mm_month"], label="Model")
    axes[0].plot(seasonal_cycle["month"], seasonal_cycle["fluxcom_et_mm_month"], label="FLUXCOM")
    axes[0].set_title("Seasonal ET cycle")
    axes[0].legend()
    valid = merged[["et_model_mm_month", "fluxcom_et_mm_month"]].dropna()
    axes[1].scatter(valid["fluxcom_et_mm_month"], valid["et_model_mm_month"], s=5, alpha=0.3)
    axes[1].set_xlabel("FLUXCOM ET")
    axes[1].set_ylabel("Model ET")
    axes[1].set_title("Monthly ET scatter")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(Path(cfg["outputs"]["figures_dir"]) / f"et_validation_summary.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
