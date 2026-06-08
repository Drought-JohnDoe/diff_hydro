# Reproducibility Bundle

This bundle contains the material needed to understand and regenerate the WRR paper products from the saved local analysis workspace.

## Included

- `configs/`
- `data_inventory/`
- `scripts/`
- `tables/`
- `figures/`
- `maps/`
- `manuscript/`
- selected `diagnostics/`

## Included diagnostics

- `model6_ep100_monthly_basin.parquet`
- `model6_ep100_learned_parameters.csv`
- `model6_ep100_archive_manifest.json`

## Excluded large diagnostic

The full daily archive was not copied into this export:

- `model6_ep100_daily_basin_day.parquet`

That file remains available in the original local workspace:

- `/home/mircore/Desktop/diff_hydro/WRR_Model6_EndToEnd_Paper/diagnostics/model6_ep100_daily_basin_day.parquet`

## Purpose

This bundle is intended for:

- GitHub archiving
- rerunning figure/table generation with the included scripts
- inspecting the exact manuscript source and metadata
