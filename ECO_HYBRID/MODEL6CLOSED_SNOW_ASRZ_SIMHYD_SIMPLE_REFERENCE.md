# Model6Closed_Snow_aSrz_SIMHYD_Simple

This note documents the implemented `Model6Closed_Snow_aSrz_SIMHYD_Simple` used in:

- full `671`-basin `Ep30` run: [Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep30](/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep30)
- runner: [run_model6_closed_snow_asrz_simhyd_simple_full671.py](/home/mircore/Desktop/diff_hydro/ECO_HYBRID/run_model6_closed_snow_asrz_simhyd_simple_full671.py)
- core implementation: [rnn.py](/home/mircore/Desktop/diff_hydro/code/dPLHBVrelease/hydroDL-dev/hydroDL/model/rnn.py)

## Overview

`Model6Closed_Snow_aSrz_SIMHYD_Simple` is a deliberately simplified, fully closed differentiable hydrologic model. It keeps the **Model 6 regionalization architecture** but replaces the more permissive groundwater and loss structure with a compact closed process model:

1. `HBV`-style snow bucket
2. `aSrz` active root-zone storage
3. `SIMHYD`-style runoff partition
4. one simple groundwater store
5. optional post-process routing and component mixing

It has:

- `4` hydrologic components per basin
- `35` static CAMELS-US attributes
- static parameter regionalization from basin attributes
- dynamic daily modifiers for selected process controls
- no explicit external loss sinks besides evapotranspiration and streamflow

Allowed external outputs are only:

- interception evaporation `INT`
- root-zone ET `ET_a`
- process streamflow `Q_process`

There is no:

- `groundwater_loss`
- `channel_loss`
- zero-flow gate
- deep groundwater loss
- true leakage term

## Physical State Variables

Each component carries `4` physical stores:

1. `SNOWPACK`
   - solid snow storage
2. `MELTWATER`
   - liquid water retained in the snowpack
3. `Sa`
   - active root-zone storage accessible to vegetation ET
4. `GW`
   - lumped groundwater storage feeding baseflow

So each basin has `4 components x 4 stores = 16` process states, plus routing states after process closure.

## Inputs

### Daily forcing inputs

For each basin and day, the physical model uses:

1. `P`
   - precipitation
2. `T`
   - mean daily temperature
3. `PET`
   - potential evapotranspiration
4. `sin(doy)`
   - seasonal harmonic
5. `cos(doy)`
   - seasonal harmonic

### Static inputs

The outer regionalization model uses the same `35` CAMELS-US attributes as the original Model 6 setup:

1. `p_mean`
2. `pet_mean`
3. `p_seasonality`
4. `frac_snow`
5. `aridity`
6. `high_prec_freq`
7. `high_prec_dur`
8. `low_prec_freq`
9. `low_prec_dur`
10. `elev_mean`
11. `slope_mean`
12. `area_gages2`
13. `frac_forest`
14. `lai_max`
15. `lai_diff`
16. `gvf_max`
17. `gvf_diff`
18. `dom_land_cover_frac`
19. `dom_land_cover`
20. `root_depth_50`
21. `soil_depth_pelletier`
22. `soil_depth_statsgo`
23. `soil_porosity`
24. `soil_conductivity`
25. `max_water_content`
26. `sand_frac`
27. `silt_frac`
28. `clay_frac`
29. `geol_1st_class`
30. `glim_1st_class_frac`
31. `geol_2nd_class`
32. `glim_2nd_class_frac`
33. `carbonate_rocks_frac`
34. `geol_porostiy`
35. `geol_permeability`

## Neural Architecture

The model keeps the Model 6 multi-component architecture:

1. A static encoder maps the `35` basin attributes into a hidden embedding.
2. A linear head predicts:
   - component-specific static process parameters
   - component routing parameters
   - component mixture weights
3. A dynamic daily control head reads current forcing/state features and modulates selected physical parameters.
4. Each of the `4` components runs the same physical equations with different learned parameters.
5. Component discharges are routed.
6. Final basin discharge is the softmax-weighted mixture of routed component discharges.

### Static encoder

The outer class is:

- `MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple`

Key architecture settings used here:

- `hiddeninv = 64`
- `nmul = 4`
- `nattr = 35`
- `nfea = 18` static physical parameters per component

The static output head predicts:

- `18 x 4 = 72` process parameters
- routing parameters
- component weights

### Dynamic control head

The process model builds a daily dynamic feature vector from:

- `P / 20`
- `T / 20`
- `PET / 10`
- `Sa / 300`
- `Smoist`
- `GW / 300`
- `SNOWPACK / 300`
- `sin(doy)`
- `cos(doy)`

This daily dynamic head modulates:

1. `SQ_t`
2. `CFMAX_t`
3. runoff partition fractions

Notes:

- `dynamic_sq = True`
- `dynamic_partition = True`
- `dynamic_cfmax_snow = True`
- `dynamic_etgam` is enabled in the wrapper configuration for consistency with the broader codebase, but this specific simplified process model does **not** explicitly use an ETGAM term in its equations.
- `lgdyn` exists in the parent architecture, but this simplified closed model does **not** use an explicit groundwater-loss dynamic transfer in the forward physics.

## Static Physical Parameters

These are predicted per component from static basin attributes.

### Snow

1. `TT` in `[-2.5, 2.5] C`
   - temperature threshold separating snow and rain
   - higher `TT` means precipitation remains snow at warmer temperatures

2. `CFMAX_base` in `[0.5, 10.0] mm/day/C`
   - degree-day melt factor
   - higher values produce faster melt when `T > TT`

3. `CFR` in `[0, 0.1]`
   - refreezing coefficient
   - higher values return more liquid snow water back to ice when `T < TT`

4. `CWH` in `[0, 0.2]`
   - snow liquid-water holding fraction
   - controls how much liquid water the snowpack can retain before release

### Interception

5. `INSC` in `[0.5, 5.0] mm/day`
   - interception storage capacity / daily interception cap
   - higher values increase canopy interception evaporation before water reaches soil

### Legacy Model 6 inherited parameters still present

6. `COEF` in `[50, 400]`
   - retained from the broader Model 6 family
   - in this simplified implementation it is exported diagnostically but not a dominant control in the final simple equations

7. `SQ` in `[0, 6]`
   - static base for dynamic runoff-shape modulation
   - becomes `SQ_t`

### SIMHYD-style runoff partition

8. `SUB` in `[0, 1]`
   - baseline surface-runoff tendency
   - larger `SUB` shifts more partitioned water toward quick surface runoff

9. `CRAK` in `[0, 1]`
   - baseline recharge tendency within the non-surface fraction
   - larger `CRAK` shifts more remaining water into groundwater recharge rather than interflow

10. `K` in `[0.003, 0.3] day^-1`
   - groundwater recession coefficient
   - controls how quickly the simple `GW` store drains to baseflow

### aSrz root-zone parameters

11. `theta_ab` in `[0.5, 1.0]`
   - accessibility coefficient
   - scales how much post-interception water can enter the active root zone

12. `theta_ak` in `[1.0, 10.0]`
   - accessibility nonlinearity
   - controls how sharply active-soil access declines with increasing wetness

13. `theta_cap` in `[10, 1500] mm`
   - active root-zone capacity
   - maximum accessible root-zone storage before overflow

14. `theta_efmax` in `[0.5, 1.0]`
   - maximum ET efficiency multiplier
   - upper bound on how strongly PET can be realized as `ET_a`

15. `theta_wetpoint` in `[0.3, 0.9]`
   - ET stress wetness threshold
   - controls how quickly ET stress ramps up with `Sa / theta_cap`

### Routing and mixture

16. `route_a`
   - learned routing shape parameter
   - exported approximately in `[0, 2.9]`

17. `route_b`
   - learned routing scale parameter
   - exported approximately in `[0, 6.5]`

18. `component weight`
   - softmax mixture weight over the 4 components
   - nonnegative and sums to `1` across components

## Dynamic Parameters and Controls

These vary daily.

### 1. `SQ_t`

Computed from static `SQ` times a dynamic multiplier:

- multiplier range approximately `[0.5, 2.0]`
- clipped final `SQ_t` into `[0, 6]`

Physical meaning:

- provides a time-varying runoff-shape control linked to current wetness/forcing

### 2. `CFMAX_t`

Computed from static `CFMAX_base` times a dynamic multiplier:

- multiplier range approximately `[0.7, 1.5]`
- only active where snow fraction is meaningful

Physical meaning:

- allows daily snowmelt responsiveness to vary by state and season

### 3. Dynamic partition fractions

The model forms baseline partition logits from:

- `SUB`
- `(1 - SUB) * (1 - CRAK)`
- `(1 - SUB) * CRAK`

Then adds dynamic partition logits and applies softmax to get:

1. `f_surface`
2. `f_interflow`
3. `f_recharge`

These are:

- nonnegative
- sum to `1`

Physical meaning:

- dynamically allocates partitioned water between quick runoff, interflow, and groundwater recharge

## Physical Process Equations

## 1. Snow bucket

Rain–snow split:

```text
rain_frac = sigmoid(gain * (T - TT))
rainfall = P * rain_frac
snowfall = P * (1 - rain_frac)
```

Snow storage and melt:

```text
SNOWPACK1 = SNOWPACK + snowfall
snowmelt_pot = CFMAX_t * max(T - TT, 0)
snowmelt = min(snowmelt_pot, SNOWPACK1)
```

Refreezing:

```text
refreeze_pot = CFR * CFMAX_t * max(TT - T, 0)
refreezing = min(refreeze_pot, MELTWATER1)
```

Snow liquid-water holding and release:

```text
snow_holding = CWH * SNOWPACK
snow_release = min(max(MELTWATER - snow_holding, 0), MELTWATER)
```

Water reaching land surface:

```text
PL = rainfall + snow_release
```

## 2. Interception

```text
INT = min(INSC, PET, PL)
PL_after_int = max(PL - INT, 0)
POT = max(PET - INT, 0)
```

Physical meaning:

- evaporation first consumes intercepted water
- only remaining precipitation enters the root-zone/runoff system

## 3. aSrz active root-zone bucket

Relative root-zone wetness:

```text
Smoist = clamp(Sa / theta_cap, 0, 1)
```

Accessible fraction:

```text
alpha = theta_ab * (1 - Smoist)^theta_ak
alpha = clamp(alpha, 0, 1)
```

Partition of post-interception water:

```text
P_accessible = alpha * PL_after_int
P_inaccessible = (1 - alpha) * PL_after_int
```

Root-zone update:

```text
Sa_pre = Sa + P_accessible
ET_stress = clamp(Smoist / theta_wetpoint, 0, 1)
ET_a_pot = POT * theta_efmax * ET_stress
ET_a = min(ET_a_pot, Sa_pre)
Sa_after_ET = Sa_pre - ET_a
Sa_overflow = max(Sa_after_ET - theta_cap, 0)
Sa_next = Sa_after_ET - Sa_overflow
```

Physical meaning:

- `Sa` is the plant-accessible water pool
- ET comes **only** from `Sa`
- excess root-zone water spills to the runoff/recharge system

## 4. SIMHYD-style partition

Water available for partition:

```text
water_for_partition = P_inaccessible + Sa_overflow
```

Dynamic fractions:

```text
SRUN = f_surface * water_for_partition
IFLOW = f_interflow * water_for_partition
REC = f_recharge * water_for_partition
```

Physical meaning:

- inaccessible water and root-zone overflow are routed to fast runoff, interflow, or recharge

## 5. Simple groundwater store

```text
GW1 = GW + REC
BAS_raw = K * GW1
BAS = min(BAS_raw, GW1)
GW_next = GW1 - BAS
```

Physical meaning:

- a single linear recession reservoir
- recharge enters `GW`
- baseflow drains `GW`

## 6. Process discharge

```text
Q_process = SRUN + IFLOW + BAS
```

There are no discarded losses in this process model.

## Water Balance

Per component and day:

```text
residual =
P
- INT
- ET_a
- Q_process
- [ΔSNOWPACK + ΔMELTWATER + ΔSa + ΔGW]
```

There is no:

- groundwater loss term
- channel loss term
- gate loss term
- deep leakage term

## Routing and Mixing

Process closure is computed before routing.

After closure:

1. Each component discharge is routed with the existing gamma-style routing kernel using `route_a` and `route_b`.
2. The `4` routed component discharges are mixed using learned softmax component weights.

This means:

- process-level water balance uses `Q_process`
- final hydrograph skill metrics use routed and mixed discharge

## Static vs Dynamic Summary

### Static by component

- `INSC`
- `COEF`
- `SQ`
- `SUB`
- `CRAK`
- `K`
- `TT`
- `CFMAX_base`
- `CFR`
- `CWH`
- `theta_ab`
- `theta_ak`
- `theta_cap`
- `theta_efmax`
- `theta_wetpoint`
- `route_a`
- `route_b`
- component weight

### Dynamic by day

- `SQ_t`
- `CFMAX_t`
- `f_surface`
- `f_interflow`
- `f_recharge`
- state-dependent `alpha`
- state-dependent ET stress

### Present in the broader architecture but not physically used as a loss pathway here

- `lgdyn`
- `dynamic_etgam`
- explicit groundwater-loss pathway

## Training Loss

The training run used:

- primary discharge loss: `RmseLossComb(alpha=0.25)`

Plus auxiliary regularization terms from the process model:

1. `dynamic_amplitude_loss`
   - penalizes excessive dynamic multipliers
2. `dynamic_smoothness_loss`
   - penalizes rapid day-to-day swings in dynamic controls
3. `partition_entropy_loss`
   - discourages extreme recharge fractions
4. `storage_drift_loss`
   - penalizes end-of-window drift in `Sa`, `GW`, `SNOWPACK`, and `MELTWATER`

In the wrapper configuration:

- `reg_amp_w = 1e-3`
- `reg_smooth_w = 1e-3`
- `reg_part_w = 1e-3`

## Assumptions

1. `PET` is available and consistent with the forcing period.
2. The `35` CAMELS-US attributes are available.
3. Root-zone ET comes only from `Sa`.
4. No vegetation time series such as daily `LAI_t` is used.
5. No explicit transmission loss, gate loss, or deep leakage is allowed.
6. Routing is post-process and should not be used for process water balance.

## Diagnostics Exported

The model exports, among others:

- `SNOWPACK`
- `MELTWATER`
- `Sa`
- `GW`
- `precipitation`
- `rainfall`
- `snowfall`
- `snowmelt`
- `refreezing`
- `snow_release`
- `PL`
- `interception_evaporation`
- `actual_ET`
- `Smoist`
- `theta_cap`
- `alpha`
- `P_accessible`
- `P_inaccessible`
- `Sa_overflow`
- `surface_runoff`
- `interflow`
- `recharge_to_groundwater`
- `baseflow`
- `Q_process`
- local residual terms
- routed and mixed discharge outputs at the wrapper level

## Full 671-Basin Ep30 Metrics

From [summary_compare.csv](/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep30/summary_compare.csv):

- median `NSE = 0.6699`
- mean `NSE = 0.4006`
- median `KGE = 0.6477`
- median `R2 = 0.7075`
- median `FLV = 4.5544`
- median `FHV = -37.6968`
- median low-flow `NSE = -34.2094`
- median high-flow `NSE = 0.5781`
- median `ET/P = 0.6102`
- median `Q/P = 0.3876`
- median `alpha mean = 0.5831`
- median `mean aSrz = 72.50 mm`
- median `aSrz capacity = 169.85 mm`
- median process closure residual `= 0.000185 mm/day`
- median cumulative water-balance error `= 5.69e-05`
- basins `>1%` water-balance error `= 0`
- total external loss / `P = 0`

## Interpretation

This model is physically attractive because:

- it is closed
- external loss is exactly zero
- water balance is extremely tight
- it produces interpretable `aSrz` behavior

Its main weakness remains performance relative to the retained soft-gate Model 6:

- lower `NSE`
- lower `KGE`
- much weaker low-flow behavior
- more negative high-flow bias than the retained soft-gate baseline

So it is best viewed as:

- a clean closed benchmark
- a physically interpretable ecohydrology variant
- not yet the best predictive final model

## Main Files

- implementation: [rnn.py](/home/mircore/Desktop/diff_hydro/code/dPLHBVrelease/hydroDL-dev/hydroDL/model/rnn.py)
- full `Ep30` run: [Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep30](/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep30)
- full `Ep30` summary: [summary_compare.csv](/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep30/summary_compare.csv)
- full CONUS `aSrz` figure: [full671_asrz_mean_map_and_aridity.png](/home/mircore/Desktop/diff_hydro/ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep30/figures/full671_asrz_mean_map_and_aridity.png)
