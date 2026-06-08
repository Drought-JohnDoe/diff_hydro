from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paper_common import ensure_outputs, hydroclass_columns, load_camels_attributes, load_config, save_basin_map, summarize_by_group
from utils_model6_io import load_main_closed_water_balance


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    ensure_outputs(cfg)

    wb = load_main_closed_water_balance(root / "configs" / "wrr_model6_config.yaml")
    attrs = hydroclass_columns(load_camels_attributes(cfg))
    basin = wb.merge(attrs, on="basin_id", how="left")
    basin["closure_source"] = "Ep100 benchmark basin summary"
    basin.to_csv(Path(cfg["outputs"]["tables_dir"]) / "water_balance_closure_by_basin.csv", index=False)

    for col, title in [
        ("mean_abs_daily_wb_residual_mm_day", "Mean abs daily water-balance residual"),
        ("cumulative_relative_wb_error", "Cumulative relative water-balance error"),
    ]:
        save_basin_map(
            cfg,
            basin[["basin_id", col]],
            col,
            Path(cfg["outputs"]["maps_dir"]) / f"map_{col}.png",
            title,
            col,
        )

    by_class = summarize_by_group(
        basin,
        ["aridity_class", "snow_class"],
        ["mean_abs_daily_wb_residual_mm_day", "cumulative_relative_wb_error", "external_loss_over_P"],
    )
    by_class.to_csv(Path(cfg["outputs"]["tables_dir"]) / "water_balance_closure_by_class.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    basin["mean_abs_daily_wb_residual_mm_day"].hist(ax=axes[0], bins=30)
    axes[0].set_title("Closure residual distribution")
    axes[0].set_xlabel("mm/day")
    by_class.plot(
        x="aridity_class",
        y="mean_abs_daily_wb_residual_mm_day_median",
        kind="bar",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Median closure residual by aridity class")
    axes[1].set_ylabel("mm/day")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(Path(cfg["outputs"]["figures_dir"]) / f"closure_summary.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

