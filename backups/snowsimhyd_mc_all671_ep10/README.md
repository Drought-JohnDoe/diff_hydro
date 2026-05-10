# Snow-SIMHYD-MC Backup

This folder is a code-and-model backup for the latest `Snow-SIMHYD-MC` experiment run from the `diff_hydro` workspace.

It intentionally excludes raw CAMELS forcings, PET source files, and raw observation datasets.

## What Is Included

- `checkpoints/model_Ep10.pt`
  Latest all-671-basin checkpoint from the matched practical benchmark.
- `scripts/rnn.py`
  Full source file containing the process-model and inversion-model classes.
- `scripts/train.py`
  Training and testing utility file, including dispatch support for `MultiInv_SnowSIMHYDMulTDModel`.
- `scripts/traindPLSnowSIMHYDMC.py`
  Training entrypoint for the multi-component Snow-SIMHYD experiment.
- `scripts/testdPLSnowSIMHYDMC.py`
  Testing and inference entrypoint for the same experiment.
- `scripts/report_snowsimhydmc_vs_hbv_detailed.py`
  Detailed comparison-report generator against the matched HBV benchmark.
- `scripts/snow_simhyd_mc_model_excerpt.py`
  Smaller reference file with the key helper functions and Snow-SIMHYD-MC classes extracted from `rnn.py`.
- `results/run.csv`
  Per-epoch training log for the all-671-basin `Ep10` run.
- `results/model_progress_and_baselines.csv`
  Summary comparison against HBV and plain SIMHYD.
- `results/metric_summary_table.csv`
  Aggregated metric comparison table.
- `results/test_nse_bins_hbv_vs_snowsimhydmc.csv`
  NSE bin comparison table.
- `results/report.md`
  Short written summary from the detailed report.

## Benchmark This Checkpoint Came From

- Basins: `671` CAMELS basins
- Forcing: `daymet`
- Train period: `1980-10-01` to `1995-10-01`
- Test period: `1995-10-01` to `2010-10-01`
- Batch size: `32`
- Sequence length `rho`: `365`
- Epochs: `10`
- Random windows per epoch: `100`
- Hidden size: `64`
- Components `nmul`: `4`
- Routing: `enabled`
- Component weights: `enabled`
- Semi-dynamic `LG`: `enabled`

Headline result from this checkpoint:

- Snow-SIMHYD-MC median test NSE: `0.6290`
- Matched HBV median test NSE: `0.6278`

## Model Structure

The model has two main parts:

1. `Inversion LSTM`
   Reads normalized meteorological inputs and basin attributes, then predicts:
   - per-component static Snow-SIMHYD parameters
   - routing parameters
   - optional component mixing weights
   - a semi-dynamic time series for groundwater loss `LG`

2. `Multi-component Snow-SIMHYD process model`
   Runs `4` Snow-SIMHYD components in parallel, mixes them, and routes the mixed discharge.

## Inputs

### Process-model input `x`

Shape:

```python
[time, basin, 3]
```

Variables:

- `prcp`
- `temperature` (`tmean` for `daymet`)
- `PET`

### Inversion input `z`

Built from:

- forcing time series used by the inversion branch
- PET
- static CAMELS attributes appended along the feature dimension

## Static Parameter Set Per Component

Each component predicts `12` normalized parameters, later scaled to physical ranges:

1. `INSC`: interception store capacity, `0.5` to `5.0`
2. `COEF`: infiltration capacity scale, `50.0` to `400.0`
3. `SQ`: wetness exponent in infiltration decline, `0.0` to `6.0`
4. `SMSC`: soil moisture storage capacity, `50.0` to `500.0`
5. `SUB`: interflow fraction coefficient, `0.0` to `1.0`
6. `CRAK`: recharge fraction coefficient, `0.0` to `1.0`
7. `K`: baseflow recession coefficient, `0.003` to `0.3`
8. `LG`: baseline groundwater loss term, `0.0` to `1.0`
9. `TT`: snow/rain threshold temperature, `-2.5` to `2.5`
10. `CFMAX`: melt factor, `0.5` to `10.0`
11. `CFR`: refreezing coefficient, `0.0` to `0.1`
12. `CWH`: snow water holding capacity fraction, `0.0` to `0.2`

## Semi-Dynamic Parameter

Only `LG` is semi-dynamic in this experiment.

- The LSTM predicts one `LG(t)` time series per component.
- During the main simulation period, the effective groundwater loss is:

```python
LG_eff(t) = (1 - w) * LG_static + w * LG_dynamic(t)
```

with `w = 0.5` in this run.

Warmup is run using only the static parameter set.

## Snow Process Representation

The snow module is HBV-style and includes:

- smooth rain/snow partitioning around `TT`
- snowpack accumulation
- degree-day melt via `CFMAX`
- refreezing via `CFR`
- liquid-water storage limit via `CWH`
- effective liquid input to the runoff model as:

```python
Peff = rain + release_from_snowpack
```

## Multi-Component Mixing

Each basin has `4` Snow-SIMHYD components.

- If component weights are enabled, the model learns softmax-normalized mixing weights.
- If disabled, components are simply averaged.

In this checkpoint, learned component weights are enabled.

## Routing

Routing is applied after mixing in this experiment.

The routing branch learns two normalized parameters which are mapped to HBV-style unit-hydrograph ranges:

- `rout_a`: `0.0` to `2.9`
- `rout_b`: `0.0` to `6.5`

These define a gamma unit hydrograph used to route discharge through `UH_gamma` and `UH_conv`.

## Main Files To Read First

- `scripts/snow_simhyd_mc_model_excerpt.py`
- `scripts/traindPLSnowSIMHYDMC.py`
- `scripts/testdPLSnowSIMHYDMC.py`
- `scripts/report_snowsimhydmc_vs_hbv_detailed.py`

## Notes

- This backup is meant for reproducibility and code recovery.
- Raw training/testing data are not included here.
- The full local workspace also contains additional reports and plots if needed later.
