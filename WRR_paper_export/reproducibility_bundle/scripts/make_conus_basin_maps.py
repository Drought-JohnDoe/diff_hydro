from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_common import load_config, save_basin_map, merge_metric_geometries, save_geodata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "wrr_model6_config.yaml"))
    parser.add_argument("--metric_csv", required=True)
    parser.add_argument("--metric_column", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--cmap", default=None)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.metric_csv)
    save_basin_map(
        cfg,
        df,
        args.metric_column,
        args.output_path,
        args.title,
        args.units,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    gpkg_path = Path(cfg["outputs"]["maps_dir"]) / "geodata" / f"{Path(args.output_path).stem}.gpkg"
    gdf = merge_metric_geometries(cfg, df)
    save_geodata(gdf, gpkg_path)


if __name__ == "__main__":
    main()
