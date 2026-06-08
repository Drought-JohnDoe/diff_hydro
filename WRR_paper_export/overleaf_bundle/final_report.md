# Final Report

## Files inspected
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/per_basin_metrics_compare.csv`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/water_balance_compare.csv`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep60/drought_tipping_theta_wetpoint/drought_tipping_daily_basin_day.parquet`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_clim.txt`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_geol.txt`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_hydro.txt`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_soil.txt`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_topo.txt`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/camels_us/camels_attributes_v2/camels_vege.txt`
- `/home/mircore/Desktop/diff_hydro/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/shapefiles/merge/basinset_gf_nhru.shp`
- `/home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/gleam_basin_daily.csv`
- `/home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/gleam_basin_monthly.csv`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/independent_validation_grace_fluxcom/fluxcom_et_basin_monthly.parquet`
- `/home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/MOD16A3GF_Model6_ET/model6_vs_mod16a3gf_annual_et_by_basin.csv`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/ECO_INPUTS_2020/grace_jpl/raw/GRCTellus.JPL.200204_202603.GLO.RL06.3M.MSCNv04CRI.nc`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/independent_validation_grace_fluxcom/grace_twsa_basin_monthly.parquet`
- `/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/independent_validation_grace_fluxcom/grace_twsa_per_basin_metrics.csv`
- `/home/mircore/Desktop/diff_hydro/results/validation/raw/esacci_sm`
- `/home/mircore/Desktop/diff_hydro/outputs/Model6_ExternalValidation/NSIDC0719_Model6_SWE`

## Files created
- `diagnostics/model6_ep100_daily_basin_day.parquet`
- `diagnostics/model6_ep100_monthly_basin.parquet`
- `diagnostics/model6_ep100_learned_parameters.csv`
- `tables/streamflow_metrics_by_basin.csv`
- `tables/water_balance_closure_by_basin.csv`
- `tables/learned_parameters_by_basin.csv`
- `tables/et_validation_by_basin.csv`
- `tables/twsa_validation_by_basin_exploratory.csv`
- `tables/model_comparison_summary.csv`
- `manuscript/wrr_model6_full_draft.md`

## Scripts created
- `scripts/export_model6_ep100_daily_archive.py`
- `scripts/analyze_streamflow_skill.py`
- `scripts/analyze_water_balance_closure.py`
- `scripts/analyze_learned_parameters.py`
- `scripts/analyze_interpretability_tests.py`
- `scripts/validate_et_products.py`
- `scripts/validate_twsa_grace.py`
- `scripts/build_model_comparison_table.py`
- `scripts/build_geodata_packages.py`
- `scripts/write_manuscript_package.py`
- downloader templates for GLEAM/CSR/GSFC

## Found vs missing products
- FLUXCOM monthly ET: found and used
- MOD16A3GF annual ET: found and used
- JPL GRACE mascon: found and used
- GLEAM ET: local CSV placeholders exist but are empty; no result claimed
- GRACE CSR/GSFC: not found locally; downloader templates only
- ESA CCI soil moisture: found locally but not used in the core WRR package
- NSIDC SWE validation: found locally as auxiliary context

## Whether ET maps were generated
- Yes. FLUXCOM-based ET maps and summary figures were generated.
- No GLEAM maps were generated because the local product is empty.

## Whether TWSA maps were generated
- Yes. JPL-based exploratory basin maps and regional figures were generated.
- CSR/GSFC comparison maps were not generated because those products are missing.

## Summary
- Streamflow: median NSE `0.685`, median KGE `0.643`, median low-flow NSE `-38.105`
- Closure: median daily residual `0.000183` mm/day, median cumulative error `0.000056`
- Storage: median theta_cap `227.398` mm, median aSrz capacity `166.087` mm
- ET: median FLUXCOM NSE `0.468`, median MOD16 annual NSE `0.709`
- TWSA: median JPL basin NSE `0.351`, median regional correlation `0.776`

## Key figures
- `figures/et_validation_summary.png`
- `figures/closure_summary.png`
- `figures/theta_cap_scatter_suite.png`
- `figures/twsa_regional_timeseries.png`
- `maps/map_NSE.png`
- `maps/map_ET_bias_FLUXCOM.png`
- `maps/map_R2_JPL.png`

## Map styling and LaTeX manuscript update
- Sequential colormaps used: `viridis`, `plasma`, `magma`, `cividis`, and `turbo`
- Diverging colormaps used: `RdBu_r`, `coolwarm`, and `BrBG`
- NSE/KGE were clipped only for visualization on fixed 0--1 maps; original values remain unchanged in the CSV tables
- Unclipped supplementary maps: `maps/supplementary/map_NSE_unclipped.png`, `maps/supplementary/map_KGE_unclipped.png`
- LaTeX engine available: `False`
- LaTeX class/template used: `article`
- PDF compilation status: `unavailable`
- LaTeX source: `manuscript/model6_wrr_paper.tex`
- Compiled PDF: `not generated`

## Remaining blockers
- No fully local usable GLEAM ET product
- No local CSR or GSFC GRACE product
- State-rich daily archive uses explicit `Ep60` auxiliary diagnostics for internal states/partitioning while `Ep100` remains the authoritative benchmark for streamflow and closure summaries
- GRACE small-basin results remain exploratory because of coarse product scale

## Exact rerun commands
```bash
cd /home/mircore/Desktop/diff_hydro/WRR_Model6_EndToEnd_Paper/scripts
python build_local_inventory.py
python export_model6_ep100_daily_archive.py
python analyze_streamflow_skill.py
python analyze_water_balance_closure.py
python analyze_learned_parameters.py
python analyze_interpretability_tests.py
python validate_et_products.py
python validate_twsa_grace.py
python build_model_comparison_table.py
python build_geodata_packages.py
python write_manuscript_package.py
```
