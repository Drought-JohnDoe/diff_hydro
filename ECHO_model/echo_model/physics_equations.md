# ECHO Model Physics

This package exports the closed Model 6 variant:

- `DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple`
- `MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple`

## Inputs

Daily dynamic forcings per basin:

- `P`: precipitation
- `T`: mean air temperature
- `PET`: potential evapotranspiration
- `sin(doy)`
- `cos(doy)`

Daily inversion features:

- normalized `P`
- normalized `T`
- normalized `PET`
- raw `frac_snow`

Static attributes:

- the 35 CAMELS attributes in `ATTR_LST`

## Learned Structure

## Static basin-component parameters

The static attribute encoder predicts one parameter set per component. The closed simple model uses:

- `INSC`: interception capacity
- `COEF`: infiltration-scale parameter
- `SQ`: runoff-infiltration nonlinearity
- `SUB`: baseline surface partition bias
- `CRAK`: baseline recharge bias
- `K`: linear groundwater recession coefficient
- `TT`: rain-snow threshold
- `CFMAX`: degree-day snowmelt factor
- `CFR`: refreezing factor
- `CWH`: snow liquid holding fraction
- `theta_ab`: accessibility coefficient
- `theta_ak`: accessibility exponent
- `theta_cap`: root-zone storage capacity
- `theta_efmax`: ET efficiency factor
- `theta_wetpoint`: wetness threshold for ET stress

## Dynamic quantities

The LSTM-based dynamic head predicts day-varying modifiers for:

- `SQ_t`
- `CFMAX_t`
- runoff partition logits

The routing branch is static in this exported baseline.

## States

- `SNOWPACK`
- `MELTWATER`
- `Sa`: active root-zone storage
- `GW`: groundwater storage

## Process Equations

### 1. Rain-snow partition

- `frac_rain = sigmoid(g * (T - TT))`
- `rainfall = P * frac_rain`
- `snowfall = P * (1 - frac_rain)`

### 2. Snow update

- `snowmelt_pot = CFMAX_t * relu(T - TT)`
- `snowmelt = min(snowmelt_pot, SNOWPACK + snowfall)`
- `refreezing = min(CFR * CFMAX_t * relu(TT - T), MELTWATER + snowmelt)`
- `snow_release = max(MELTWATER_after_refreeze - CWH * SNOWPACK_after_refreeze, 0)`

### 3. Interception

- `PL = rainfall + snow_release`
- `INT = min(INSC, min(PET, PL))`
- `PL_after_int = max(PL - INT, 0)`

### 4. Active root-zone access

- `Smoist = clamp(Sa / theta_cap, 0, 1)`
- `alpha = theta_ab * (1 - Smoist) ^ theta_ak`
- `P_accessible = alpha * PL_after_int`
- `P_inaccessible = (1 - alpha) * PL_after_int`

### 5. ET from active root zone

- `ET_stress = clamp(Smoist / theta_wetpoint, 0, 1)`
- `ET_a_pot = (PET - INT)+ * theta_efmax * ET_stress * LAI_scalar`
- `ET_a = min(ET_a_pot, Sa + P_accessible)`

### 6. Root-zone update

- `Sa_pre = Sa + P_accessible`
- `Sa_after_ET = max(Sa_pre - ET_a, 0)`
- `Sa_overflow = max(Sa_after_ET - theta_cap, 0)`
- `Sa_next = max(Sa_after_ET - Sa_overflow, 0)`

### 7. Partitioning of inaccessible water and overflow

- `water_for_partition = max(P_inaccessible + Sa_overflow, 0)`
- three logits define fractions:
  - `f_surface`
  - `f_interflow`
  - `f_recharge`
- `SRUN = f_surface * water_for_partition`
- `IFLOW = f_interflow * water_for_partition`
- `REC = f_recharge * water_for_partition`

### 8. Groundwater

- `GW1 = GW + REC`
- `BAS_raw = K * GW1`
- `BAS = min(BAS_raw, GW1)`
- `GW_next = max(GW1 - BAS, 0)`

### 9. Total discharge

- `Q_process = max(SRUN + IFLOW + BAS, 0)`

### 10. Routing and component mixing

Each component discharge is routed with a gamma unit hydrograph parameterized by two learned routing parameters. Final basin discharge is the weighted sum of routed component discharge.

## Closure

The physical intention of the closed model is:

- no groundwater loss
- no channel loss
- no gate loss

The process closure residual is:

- `residual = P - INT - ET_a - Q_process - (dSNOWPACK + dMELTWATER + dSa + dGW)`

In the closed exported baseline, external loss terms are identically zero.

## What is static vs dynamic

Static:

- parameter fields predicted from basin attributes
- routing parameters
- component weights

Dynamic:

- LSTM latent output
- `SQ_t`
- `CFMAX_t`
- partition fractions

## Explicit assumptions

- one active root-zone store
- one groundwater store
- ET stress begins around `theta_wetpoint`
- only a subset of physical parameters are allowed to vary daily
- no explicit deep leakage or channel loss in the exported closed baseline
- routing occurs after process closure
