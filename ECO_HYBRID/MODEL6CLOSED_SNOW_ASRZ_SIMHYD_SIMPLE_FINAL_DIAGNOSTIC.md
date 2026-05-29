# Model6Closed_Snow_aSrz_SIMHYD_Simple

This note documents the **current final closed physical Model 6 variant** used as the main ecohydrological reference in this workspace.

It is the fully closed, streamflow-trained baseline with:
- HBV-style snow
- active root-zone storage `Sa`
- SIMHYD-style runoff partition
- one groundwater store
- post-process routing and 4-component mixing

## Files

- model class: `code/dPLHBVrelease/hydroDL-dev/hydroDL/model/rnn.py`
  - `DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple`
  - `MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple`
- main 671-basin run: `ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep60`
- runner: `ECO_HYBRID/run_model6_closed_snow_asrz_simhyd_simple_full671.py`

## 1. Purpose

`Model6Closed_Snow_aSrz_SIMHYD_Simple` is a **streamflow-trained, differentiable, mass-conserving hydrologic model** that keeps the outer Model 6 regionalization architecture but simplifies the internal physics to:

1. HBV-style snow bucket
2. active root-zone storage `Sa`
3. simple SIMHYD-style runoff partition
4. one linear groundwater store
5. post-process routing and 4-component mixing

Water only leaves the system as:
- interception evaporation `INT`
- root-zone evapotranspiration `ET_a`
- streamflow `Q_process`

It does **not** allow any explicit discarded water sink.

## 2. What Is Trained

The trainable weights are **not direct basin-by-basin physical parameters**. The neural network learns how to map basin attributes into component-wise physical parameters.

### 2.1 Trainable neural weights

1. `staticFeat`
   - 2-layer MLP mapping `35` static attributes into a hidden embedding
   - structure:
     - `Linear(35 -> 64)`
     - `ReLU`
     - `Linear(64 -> 64)`
     - `ReLU`

2. `staticOut`
   - linear head mapping hidden embedding into:
     - physical parameters
     - routing parameters
     - component weights

3. `compStaticBias`
   - trainable bias that keeps the 4 components from collapsing to identical parameter sets

4. dynamic control head inside the physical model
   - `dynHead`
   - structure:
     - `Linear(9 -> 32)`
     - `ReLU`
     - `Linear(32 -> 32)`
     - `ReLU`
     - `Linear(32 -> n_dyn_outputs)`

5. routing parameters
   - predicted by `staticOut`
   - used by the gamma unit hydrograph after process closure

6. component mixture weights
   - predicted by `staticOut`
   - passed through softmax
   - determine how the 4 component discharges are mixed into final basin discharge

### 2.2 Effective physical parameters

The physical parameters are:
- static in time for each basin-component
- different across basins
- different across the 4 components within each basin
- learned indirectly from streamflow loss through the neural regionalization

## 3. Inputs

### 3.1 Daily meteorological inputs actually used

The physical core uses:
1. `P`
2. `T`
3. `PET`
4. `sin(doy)`
5. `cos(doy)`

### 3.2 Static basin inputs

The outer parameter-regionalization network uses the `35` CAMELS attributes:
`p_mean`, `pet_mean`, `p_seasonality`, `frac_snow`, `aridity`, `high_prec_freq`, `high_prec_dur`, `low_prec_freq`, `low_prec_dur`, `elev_mean`, `slope_mean`, `area_gages2`, `frac_forest`, `lai_max`, `lai_diff`, `gvf_max`, `gvf_diff`, `dom_land_cover_frac`, `dom_land_cover`, `root_depth_50`, `soil_depth_pelletier`, `soil_depth_statsgo`, `soil_porosity`, `soil_conductivity`, `max_water_content`, `sand_frac`, `silt_frac`, `clay_frac`, `geol_1st_class`, `glim_1st_class_frac`, `geol_2nd_class`, `glim_2nd_class_frac`, `carbonate_rocks_frac`, `geol_porostiy`, `geol_permeability`.

## 4. States

Each component has 4 process stores:
1. `SNOWPACK`
2. `MELTWATER`
3. `Sa`
4. `GW`

With `4` components per basin, this gives `16` component states before routing memory.

## 5. Static Physical Parameters and Their Meaning

The final closed-simple model uses `18` static outputs per component in the outer wrapper, but the physically active process parameters are:

### Snow and phase partition

1. `TT in [-2.5, 2.5] °C`
2. `CFMAX in [0.5, 10.0] mm d^-1 °C^-1`
3. `CFR in [0, 0.1]`
4. `CWH in [0, 0.2]`

### Interception

5. `INSC in [0.5, 5.0] mm d^-1`

### SIMHYD-style partition

6. `SUB in [0, 1]`
7. `CRAK in [0, 1]`

### Groundwater recession

8. `K in [0.003, 0.3] d^-1`

### Active root-zone storage

9. `theta_ab in [0.5, 1.0]`
10. `theta_ak in [1.0, 10.0]`
11. `theta_cap in [10, 1500] mm`
12. `theta_efmax in [0.5, 1.0]`
13. `theta_wetpoint in [0.3, 0.9]`

### Parameters inherited but not physically central here

14. `COEF in [50, 400]`
15. `SQ in [0, 6]`
16. `SMSC_legacy in [50, 500] mm`

### Routing and mixture

17. `route_a in [0, 2.9]`
18. `route_b in [0, 6.5]`

Component mixture weights are also learned and softmax-normalized across the 4 components.

## 6. Dynamic Controls

Daily dynamic controls are predicted from:
- `P / 20`
- `T / 20`
- `PET / 10`
- `Sa / 300`
- `Smoist`
- `GW / 300`
- `SNOWPACK / 300`
- `sin(doy)`
- `cos(doy)`

These modify:
1. `SQ_t`
   - `m_sq = 0.5 + 1.5 * sigmoid(raw_sq)`
   - `SQ_t = clamp(SQ * m_sq, 0, 6)`
2. `CFMAX_t`
   - `m_cf = 0.7 + 0.8 * sigmoid(raw_cfmax)`
   - `CFMAX_t = CFMAX * m_cf`
3. partition fractions
   - learned additive logits on the baseline partition, then softmax-normalized

Important:
- the final closed-simple model does **not** use dynamic `K_t`
- dynamic `K` exists only in the copied `Model6C_dynamicK` branch

## 7. End-to-End Physical Equations

All equations apply per basin, per component, per day.

### 7.1 Snow bucket

```text
frac_rain = sigmoid(gain * (T - TT))
rainfall = P * frac_rain
snowfall = P * (1 - frac_rain)

SNOWPACK1 = SNOWPACK + snowfall
snowmelt_pot = CFMAX_t * max(T - TT, 0)
snowmelt = min(snowmelt_pot, SNOWPACK1)
SNOWPACK2 = SNOWPACK1 - snowmelt
MELTWATER1 = MELTWATER + snowmelt

refreeze_pot = CFR * CFMAX_t * max(TT - T, 0)
refreezing = min(refreeze_pot, MELTWATER1)
MELTWATER2 = MELTWATER1 - refreezing
SNOWPACK3 = SNOWPACK2 + refreezing

snow_holding = CWH * SNOWPACK3
snow_release = min(max(MELTWATER2 - snow_holding, 0), MELTWATER2)
MELTWATER_next = MELTWATER2 - snow_release
SNOWPACK_next = SNOWPACK3
PL = rainfall + snow_release
```

### 7.2 Interception

```text
INT = min(INSC, PET, PL)
PL_after_int = max(PL - INT, 0)
POT = max(PET - INT, 0)
```

### 7.3 Active root-zone bucket

```text
Smoist = clamp(Sa / theta_cap, 0, 1)
alpha = theta_ab * (1 - Smoist) ^ theta_ak
alpha = clamp(alpha, 0, 1)

P_accessible = alpha * PL_after_int
P_inaccessible = (1 - alpha) * PL_after_int
Sa_pre = Sa + P_accessible

ET_stress = clamp(Smoist / theta_wetpoint, 0, 1)
ET_a_pot = POT * theta_efmax * ET_stress
ET_a = min(ET_a_pot, Sa_pre)
Sa_after_ET = Sa_pre - ET_a

Sa_overflow = max(Sa_after_ET - theta_cap, 0)
Sa_next = Sa_after_ET - Sa_overflow
```

### 7.4 SIMHYD-style partition

```text
water_for_partition = P_inaccessible + Sa_overflow

base_surface = SUB
base_recharge = (1 - SUB) * CRAK
base_interflow = (1 - SUB) * (1 - CRAK)
```

These are converted to logits, adjusted by the daily partition head, and softmax-normalized so:

```text
f_surface + f_interflow + f_recharge = 1
```

Final partition:

```text
SRUN = f_surface * water_for_partition
IFLOW = f_interflow * water_for_partition
REC = f_recharge * water_for_partition
```

### 7.5 Groundwater store

```text
GW1 = GW + REC
BAS_raw = K * GW1
BAS = min(BAS_raw, GW1)
GW_next = GW1 - BAS
```

### 7.6 Process discharge

```text
Q_process = SRUN + IFLOW + BAS
```

### 7.7 Routing and component mixing

After process closure:
1. each component `Q_process` is routed with a gamma unit hydrograph using `route_a`, `route_b`
2. routed component flows are combined using learned softmax component weights

Routing changes timing, not process-level mass accounting.

## 8. Water Balance Equation

Process closure is computed before routing:

```text
S_before = SNOWPACK + MELTWATER + Sa + GW
S_after  = SNOWPACK_next + MELTWATER_next + Sa_next + GW_next

residual = P - INT - ET_a - Q_process - (S_after - S_before)
```

There is no:
- groundwater loss
- channel loss
- gate loss
- deep leak
- true external sink

So:

```text
external_loss / P = 0
```

## 9. Loss Function Used for Training

Main loss:

```text
RmseLossComb(alpha = 0.25)
```

This is:

```text
Loss_main = (1 - alpha) * RMSE(Qsim, Qobs)
          + alpha * RMSE(log10(sqrt(Qsim + beta) + 0.1),
                         log10(sqrt(Qobs + beta) + 0.1))
```

with:
- `alpha = 0.25`
- `beta = 1e-6`

So training uses:
- `75%` ordinary RMSE
- `25%` log-sqrt discharge RMSE

### Auxiliary regularization losses

```text
Loss_total = Loss_main + Loss_aux
```

where

```text
Loss_aux =
  1e-3 * dynamic_amplitude_loss
+ 1e-3 * dynamic_smoothness_loss
+ 1e-3 * partition_entropy_loss
+ 1e-3 * storage_drift_loss
```

Meaning:
1. `dynamic_amplitude_loss`: discourages dynamic multipliers from moving too far from `1`
2. `dynamic_smoothness_loss`: discourages day-to-day jumps in dynamic controls
3. `partition_entropy_loss`: penalizes recharge fraction becoming too dominant
4. `storage_drift_loss`: penalizes end-of-window drift in `Sa`, `GW`, `SNOWPACK`, `MELTWATER`

## 10. What the Model Is Learning Physically

The model is learning:
1. how basin attributes map to snow behavior, root-zone capacity, ET stress threshold, runoff partition, groundwater recession, and routing
2. how daily forcing and current state modulate `SQ_t`, `CFMAX_t`, and the partition fractions
3. how the four internal components should differ and how much each should contribute

## 11. Explicit Assumptions

Main assumptions:
1. one simple linear groundwater store is enough
2. PET is used directly as atmospheric demand
3. root-zone ET stress depends only on relative `Sa / theta_cap`
4. active root-zone accessibility declines as soil gets wetter: `alpha = theta_ab * (1 - Smoist)^theta_ak`
5. no explicit anthropogenic water use
6. no explicit deep groundwater sink or return
7. routing is separate from process closure
8. training is streamflow-only

## 12. What We Are Not Using in This Final Model

### Not used as dynamic forcing
- daily LAI in the final baseline
- net radiation
- humidity or vapor pressure deficit
- wind speed
- soil moisture observations
- SWE observations
- TWSA observations

### Not used as training targets
- ET products like FLUXCOM or GLEAM
- SWE products
- GRACE TWSA
- ESA CCI soil moisture

Training is **streamflow only**.

### Not used as physical processes
- groundwater loss
- channel loss
- zero-flow gate
- deep groundwater reservoir
- true leakage
- dynamic groundwater recession `K_t`

## 13. Final 671-Basin Ep60 Metrics

- basins: `671`
- final continuation epochs: `31-60`
- median `NSE`: `0.6702`
- mean `NSE`: `0.4211`
- median `KGE`: `0.6557`
- median `R2`: `0.7133`
- median `FLV`: `5.2265`
- median `FHV`: `-34.9849`
- median low-flow `NSE`: `-31.2572`
- median high-flow `NSE`: `0.5814`
- median `ET/P`: `0.6007`
- median `Q/P`: `0.3958`
- median weighted process closure residual: `0.000184 mm/day`
- median weighted cumulative water-balance error: `0.000057`
- basins with `>1%` WB error: `0`
- total external loss / `P`: `0`
- median `alpha` mean: `0.5764`
- median mean `aSrz`: `71.86 mm`
- median `aSrz` capacity: `168.23 mm`

## 14. Main Diagnostic Weaknesses

1. low-flow skill is still weak
2. storage drift is still a concern, especially in `Sa` and `GW`
3. PET-only ET formulation may be too simple
4. ecohydrological realism is partial even though some meaningful spatial gradients were learned

## 15. Short Interpretation

This final model is best understood as:
- a **closed physical baseline**
- with **good water-balance behavior**
- with **moderate predictive skill**
- but still **too simple in groundwater and low-flow behavior**
