from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_common import load_config, merge_metric_geometries, save_geodata


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    outputs = Path(cfg["outputs"]["maps_dir"]) / "geodata"
    outputs.mkdir(parents=True, exist_ok=True)

    basin_metrics = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "streamflow_metrics_by_basin.csv")
    et_metrics = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "et_validation_by_basin.csv")
    twsa_metrics = pd.read_csv(Path(cfg["outputs"]["tables_dir"]) / "twsa_validation_by_basin_exploratory.csv")

    save_geodata(merge_metric_geometries(cfg, basin_metrics), outputs / "model6_basin_metrics.gpkg")
    save_geodata(merge_metric_geometries(cfg, et_metrics), outputs / "model6_et_validation.gpkg")
    save_geodata(merge_metric_geometries(cfg, twsa_metrics), outputs / "model6_twsa_validation.gpkg")


if __name__ == "__main__":
    main()

