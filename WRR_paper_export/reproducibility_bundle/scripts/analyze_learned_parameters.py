from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_common import ensure_outputs, load_config, save_basin_map


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "wrr_model6_config.yaml")
    ensure_outputs(cfg)

    params = pd.read_csv(Path(cfg["outputs"]["diagnostics_dir"]) / "model6_ep100_learned_parameters.csv")
    params.to_csv(Path(cfg["outputs"]["tables_dir"]) / "learned_parameters_by_basin.csv", index=False)

    for col, title in [
        ("theta_cap_mean", "Model 6 theta_cap (Ep100 benchmark)"),
        ("theta_wetpoint_weighted_ep60", "Model 6 theta_wetpoint (Ep60 auxiliary)"),
        ("K_weighted", "Model 6 groundwater K (Ep60 auxiliary)"),
        ("component_entropy_ep60", "Component entropy (Ep60 auxiliary)"),
        ("mean_aSrz_mm", "Mean active root-zone storage"),
        ("aSrz_capacity_mm", "Active root-zone capacity"),
    ]:
        save_basin_map(cfg, params[["basin_id", col]], col, Path(cfg["outputs"]["maps_dir"]) / f"map_{col}.png", title, col)


if __name__ == "__main__":
    main()

