from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from paper_common import ensure_outputs, load_config


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    ensure_outputs(cfg)

    params = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "learned_parameters_by_basin.csv")
    gao = pd.read_csv(
        "/home/mircore/Desktop/diff_hydro/ECO_HYBRID/experiments/gao_SR_return_period/outputs/comparisons/model6_vs_gao_SR_comparison.csv"
    )
    df = params.merge(
        gao[
            [
                "basin_id",
                "theta_cap_w",
                "theta_wetpoint_w",
                "mean_Sa_w_x",
                "mean_GW_w",
                "aridity",
                "frac_forest",
                "lai_max",
                "root_depth_50",
                "soil_depth_pelletier",
                "base_matching_return_period_local" if "base_matching_return_period_local" in gao.columns else "best_matching_return_period_local",
            ]
        ],
        on="basin_id",
        how="left",
        suffixes=("", "_gao"),
    )
    # Use master table with static controls for soil/geology coherence.
    stocker = pd.read_csv("/home/mircore/Desktop/diff_hydro/ECO_HYBRID/theta_cap_vs_stocker_s0_671/theta_cap_vs_stocker_master_671.csv")
    for c in ["basin_id", "frac_forest", "lai_max", "geol_permeability", "geol_porostiy", "carbonate_rocks_frac"]:
        if c not in df.columns and c in stocker.columns:
            df = df.merge(stocker[["basin_id", c]], on="basin_id", how="left")

    corrs = []
    for col in ["aridity", "frac_forest", "lai_max", "root_depth_50", "soil_depth_pelletier", "runoff_ratio", "baseflow_index", "ET_over_P", "geol_permeability"]:
        if col in df.columns:
            corrs.append(
                {
                    "predictor": col,
                    "spearman_theta_cap": spearman(df["theta_cap_mean"], df[col]),
                    "spearman_aSrz_capacity": spearman(df["aSrz_capacity_mm"], df[col]),
                }
            )
    corr_df = pd.DataFrame(corrs)
    corr_df.to_csv(Path(cfg["outputs"]["tables_dir"]) / "theta_cap_interpretability_correlations.csv", index=False)

    X_cols = [c for c in ["aridity", "frac_forest", "lai_max", "root_depth_50", "soil_depth_pelletier", "geol_permeability", "geol_porostiy", "carbonate_rocks_frac"] if c in df.columns]
    rf_data = df[["theta_cap_mean"] + X_cols].dropna()
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(rf_data[X_cols], rf_data["theta_cap_mean"])
    rf_out = pd.DataFrame({"predictor": X_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    rf_out.to_csv(Path(cfg["outputs"]["tables_dir"]) / "theta_cap_random_forest_importance.csv", index=False)

    bound_df = pd.DataFrame(
        {
            "parameter": ["theta_wetpoint_weighted_ep60", "theta_ab_weighted", "theta_ak_weighted", "theta_efmax_weighted", "K_weighted"],
            "lower_bound": [0.3, 0.5, 1.0, 0.5, 0.003],
            "upper_bound": [0.9, 1.0, 10.0, 1.0, 0.3],
        }
    )
    records = []
    for _, row in bound_df.iterrows():
        series = df[row["parameter"]]
        records.append(
            {
                "parameter": row["parameter"],
                "frac_near_lower_1pct": float(((series - row["lower_bound"]).abs() <= 0.01 * (row["upper_bound"] - row["lower_bound"])).mean()),
                "frac_near_upper_1pct": float(((series - row["upper_bound"]).abs() <= 0.01 * (row["upper_bound"] - row["lower_bound"])).mean()),
                "median": float(series.median()),
            }
        )
    pd.DataFrame(records).to_csv(Path(cfg["outputs"]["tables_dir"]) / "parameter_bound_diagnostics.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    plots = [
        ("aridity", "theta_cap_mean"),
        ("frac_forest", "theta_cap_mean"),
        ("root_depth_50", "theta_cap_mean"),
        ("baseflow_index", "theta_cap_mean"),
        ("ET_over_P", "theta_cap_mean"),
        ("geol_permeability", "theta_cap_mean"),
    ]
    for ax, (x, y) in zip(axes.ravel(), plots):
        if x in df.columns:
            sub = df[[x, y]].dropna()
            ax.scatter(sub[x], sub[y], s=10, alpha=0.6)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{y} vs {x}")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(Path(cfg["outputs"]["figures_dir"]) / f"theta_cap_scatter_suite.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

