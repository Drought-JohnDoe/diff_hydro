# Model Five

Model Five is the selective-dynamic `DynamicSimHyd` package created after the earlier fully dynamic experiment.  
The goal of this version is to improve NSE while keeping the learned parameters more stable, interpretable, and regionally meaningful.

## What this package contains

- `checkpoints/model_Ep10.pt`
  - trained all-671-basin checkpoint used for the final benchmark below
- `scripts/`
  - `rnn.py`
  - `train.py`
  - `trainModelFive.py`
  - `testModelFive.py`
  - `analyzeModelFive.py`
  - `report_model_five_vs_hbv.py`
- `results/`
  - training log `run.csv`
  - comparison summary `summary.csv`
  - model note `report.txt`
  - NSE bin table `test_nse_bins_hbv_vs_model_five.csv`
  - key comparison figures in `results/comparison_maps/`
  - static parameter and flux maps in `results/analysis_maps/`
- `diagnostics/`
  - `per_basin_model_five_metrics.csv`
  - `per_basin_model_five_vs_hbv.csv`

## Train / Test setup

- Dataset:
  - CAMELS, all `671` basins
- Forcing:
  - `daymet`
- Training period:
  - `1980-10-01` to `1995-10-01`
- Testing period:
  - `1995-10-01` to `2010-10-01`
- Warmup period:
  - `365` days
- Sequence length `rho`:
  - `365`
- Batch size:
  - `32`
- Windows per epoch:
  - `200`
- Epochs in this saved benchmark:
  - `10`
- Evaluation date range:
  - predictions scored over `1995-10-01` to `2010-10-01`

## Model structure

Model Five keeps the multi-component heterogeneous structure:

- static basin attribute encoder predicts component-wise process parameters
- `4` process components run in parallel
- component outputs are mixed using static softmax mixture weights
- gamma routing is applied after mixing
- `LG(t)` remains dynamic

## Static vs dynamic parameters

### Static parameters

These are kept static for identifiability and physical consistency:

- `INSC`
- `SMSC`
- `TT`
- `CFR`
- `CWH`
- `route_a`
- component mixture weights

These are also treated as static basin/component baselines:

- `COEF`
- `K`
- `CFMAX`
- `SUB`
- `CRAK`
- `LG`
- `SG_CRIT`

### Dynamic parameters and controls

Only the following controls are dynamic in this package:

1. `SQ_t`

- `SQ_t = clamp(SQ_static * m_SQ_t, 0.0, 6.0)`
- `m_SQ_t` range: `0.5` to `2.0`

2. `ETGAM_t`

- controls nonlinear soil-moisture limitation on ET
- range: `0.25` to `4.0`

3. Dynamic runoff partition

Instead of letting `SUB_t` and `CRAK_t` vary independently, the model predicts:

- `f_surface_t`
- `f_interflow_t`
- `f_recharge_t`

through a softmax partition head, with static `SUB` and `CRAK` used as bias terms in the logits.

This enforces:

- `surface + interflow + recharge = available_water`

4. `LG_t`

- dynamic but more strongly bounded
- effective range constrained to `0.0` to `0.2`

5. `SG_CRIT`

- static groundwater disconnection threshold
- range: `0` to `300 mm`

6. Snow-basin-only `CFMAX_t`

- only activated when `frac_snow > 0.05`
- `CFMAX_t = CFMAX_static * m_CFMAX_t`
- `m_CFMAX_t` range: `0.7` to `1.5`

7. Dynamic routing scale

- implemented as an optional switch in code
- **not used** in the saved all-671 `Ep10` benchmark in this package

## Process equations used

### ET

`ET = PET * min(1, (soil_moisture / SMSC) ** ETGAM_t)`

### Groundwater disconnection

- `baseflow = K_static * softplus(groundwater_storage - SG_CRIT)`
- `groundwater_loss = LG_t * softplus(SG_CRIT - groundwater_storage)`

## Regularization

The model includes three light penalties so the dynamic controls do not become arbitrary:

- dynamic amplitude loss
- dynamic smoothness loss
- recharge dominance penalty

These are added during training through the model auxiliary loss hook.

## Diagnostics computed

The analysis pipeline exports and summarizes:

- NSE
- KGE
- logNSE
- low-flow NSE
- high-flow NSE
- FDC error
- BFI error
- ET/P
- Q/P
- water balance closure

It also stores separate groundwater and surface-water related summaries:

- groundwater storage
- baseflow
- groundwater loss
- recharge to groundwater
- surface runoff
- interflow
- ET

## Final benchmark result in this package

Compared against the saved `HBV Ep10` benchmark:

- Model Five `Ep10` median NSE: `0.6906`
- HBV `Ep10` median NSE: `0.6278`
- Model Five wins on NSE in `497 / 671` basins

Selected additional medians:

- KGE: `0.7240`
- logNSE: `0.5413`
- high-flow NSE: `0.5936`
- FDC error: `0.4603`
- BFI error: `0.1183`
- ET/P: `0.4624`
- Q/P: `0.3662`

## Important file references

- Training log:
  - `results/run.csv`
- Main benchmark summary:
  - `results/summary.csv`
- Basin-by-basin comparison:
  - `diagnostics/per_basin_model_five_vs_hbv.csv`
- Analysis metrics:
  - `diagnostics/per_basin_model_five_metrics.csv`
- Key comparison figures:
  - `results/comparison_maps/`
- Static parameter maps:
  - `results/analysis_maps/static_parameters/`
- Flux maps:
  - `results/analysis_maps/fluxes/`

## Date of this packaged benchmark

- Package prepared on `2026-05-14`
- Evaluation benchmark corresponds to the saved `Ep10` all-671-basin run
