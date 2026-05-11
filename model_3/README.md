# Model 3: Snow-SIMHYD-MC-Heter

This package is the heterogeneity-aware differentiable HBV-SIMHYD style model used in the all-`671` CAMELS benchmark and evaluated here at `epoch 14`.

It is the follow-up to `model_2`, but packaged more explicitly for reuse:
- the exact model code
- the exact train/test scripts
- the saved `Ep14` checkpoint
- the HBV comparison summaries
- the internal state/flux diagnostic script and summary tables

## What This Model Is

`Snow-SIMHYD-MC-Heter` is a differentiable rainfall-runoff model with:
- a `SnowSIMHYD8Differentiable` process core
- `4` mixture components
- learned component weights
- gamma routing after mixing
- static basin-conditioned parameter prediction
- semi-dynamic groundwater loss `LG(t)`

The main architecture goal is to preserve process structure while avoiding the parameter-collapse issue seen in the earlier non-heterogeneous multi-component version.

## Core Code

- Model definition:
  `scripts/rnn.py`
  class `MultiInv_SnowSIMHYDMulTDHeterModel`
- Training script:
  `scripts/traindPLSnowSIMHYDMC_Heter.py`
- Testing script:
  `scripts/testdPLSnowSIMHYDMC_Heter.py`
- Internal state/flux diagnostics:
  `scripts/analyze_snowsimhydmc_heter_ep14_states_fluxes.py`

## Saved Checkpoint

- `checkpoints/model_Ep14.pt`

This is the checkpoint used for the packaged comparison and diagnostic reports.

## Main Inputs

Forcing input `x`:
- precipitation `P`
- temperature `T`
- PET

Inversion input `z`:
- normalized forcing history
- repeated static basin attributes

Static attributes used:
- `p_mean`
- `pet_mean`
- `p_seasonality`
- `frac_snow`
- `aridity`
- `high_prec_freq`
- `high_prec_dur`
- `low_prec_freq`
- `low_prec_dur`
- `elev_mean`
- `slope_mean`
- `area_gages2`
- `frac_forest`
- `lai_max`
- `lai_diff`
- `gvf_max`
- `gvf_diff`
- `dom_land_cover_frac`
- `dom_land_cover`
- `root_depth_50`
- `soil_depth_pelletier`
- `soil_depth_statsgo`
- `soil_porosity`
- `soil_conductivity`
- `max_water_content`
- `sand_frac`
- `silt_frac`
- `clay_frac`
- `geol_1st_class`
- `glim_1st_class_frac`
- `geol_2nd_class`
- `glim_2nd_class_frac`
- `carbonate_rocks_frac`
- `geol_porostiy`
- `geol_permeability`

## Process Parameters

Each component predicts `12` normalized parameters:
- `INSC`
- `COEF`
- `SQ`
- `SMSC`
- `SUB`
- `CRAK`
- `K`
- `LG`
- `TT`
- `CFMAX`
- `CFR`
- `CWH`

Routing parameters:
- `route_a`
- `route_b`

Mixture weights:
- `weight_c1`
- `weight_c2`
- `weight_c3`
- `weight_c4`

Dynamic loss head:
- time-varying `LG(t)` from a temporal head plus attribute-conditioned bias

## Benchmark Configuration

All-`671` practical benchmark:
- train period: `1980-10-01` to `1995-10-01`
- test period: `1995-10-01` to `2010-10-01`
- forcing: `daymet`
- batch size: `32`
- `rho`: `365`
- hidden size: `64`
- warmup / `inittime`: `365`
- random windows per epoch: `100`
- components: `4`
- routing: enabled
- component weights: enabled
- dynamic `LG`: enabled
- `lgdynweight`: `0.5`

## Packaged Results

Main comparison outputs:
- `results/model_progress_and_baselines.csv`
- `results/metric_summary_table.csv`
- `results/test_nse_bins_hbv_vs_snowsimhydmc_heter.csv`
- `results/learned_parameter_ranges.csv`
- `results/report.md`
- `results/run.csv`

Additional diagnostics:
- `diagnostics/per_basin_states_fluxes_metrics.csv`
- `diagnostics/parameter_attribute_correlation_matrix.csv`
- `diagnostics/parameter_identifiability_summary.csv`
- `diagnostics/collapsed_parameters.csv`
- `diagnostics/external_product_availability.csv`

## Ep14 Headline Performance

Against the saved HBV `10`-epoch all-`671` benchmark:
- HBV median test `NSE`: `0.6278`
- Model 3 median test `NSE`: `0.6464`
- basin wins on `NSE`: `387 / 671`

Other headline outcomes from the packaged report:
- better median `KGE`
- better median `RMSE`
- better median `MAE`
- better `FHV`, `FLV`, and `FMS` magnitude in the saved summary
- clear basin-to-basin heterogeneity in weighted-average parameters

## Why Model 3 Exists

The main reason for this model is structural:
- earlier multi-component Snow-SIMHYD could fit discharge reasonably well
- but the weighted-average parameters collapsed across basins
- this version predicts static parameters directly from basin attributes and breaks component symmetry
- that preserves heterogeneous parameter maps while staying competitive with, and at `Ep14` outperforming, the saved HBV benchmark on median `NSE`

## See Also

- `CONFIGURATION.md` for the exact architecture, parameter ranges, and run settings
