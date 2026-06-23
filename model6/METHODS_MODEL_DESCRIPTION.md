# Methods

## Locked branch

The publication branch is the full-671-basin MLP-centered Model 6 eco-hybrid run:
`Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`.

It is a four-component, process-water-balance-closed hydrologic model with:

1. a smoothed HBV-style snow bucket,
2. interception,
3. an ecosystem-accessible active root-zone store `Sa`,
4. SIMHYD-style partition into surface runoff, interflow, and groundwater recharge,
5. one groundwater store `GW`,
6. gamma routing,
7. daily LAI modulation of ET and interception,
8. a four-member mixture wrapper.

## Inputs

Daily inputs:

- precipitation `prcp`
- mean air temperature `tmean`
- potential evapotranspiration `pet`
- `sin(doy)`
- `cos(doy)`
- daily gap-filled NOAA CDR LAI

Static inputs:

- 35 CAMELS-style basin attributes spanning climate, topography, vegetation, soils, and geology

LAI availability in the locked run:

- train exact-daily fraction: `0.948521`
- train climatology-fill fraction: `0.051479`
- test exact-daily fraction: `0.999635`
- test climatology-fill fraction: `0.000365`

## Parameter-generating network

High-level architecture of the locked branch:

```text
staticFeat: 35 -> 64 -> 64 MLP
staticOut: 64 -> 84 linear head
lstmdyn.linearIn: 39 -> 64
lstmdyn.lstm: 64 hidden units, 1 layer
lstmdyn.linearOut: 64 -> 4
lgAttr: 35 -> 4
dynHead inside process core: 9 -> 32 -> 32 -> 6
channelLossHead: 35 -> 4
zeroFlowGate: 5 -> 32 -> 4
```

Interpretation:

- `staticFeat` is the main basin-static parameter generator.
- `staticOut` emits raw component-wise parameter vectors that are mapped into bounded hydrologic parameters.
- `lstmdyn` exists in the wrapper, but this final branch is still mainly static-parameterized rather than a fully dHBV-style daily-parameter LSTM model.
- `dynHead` generates a small set of daily modifiers used inside the process equations.
- `lgAttr` and `component_weight_1..4` control the four-member mixture.

## Static versus dynamic controls

Static base parameters in the locked branch:

- `INSC`
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
- `component_weight_1..4`

Dynamic controls actually used every day:

1. `SQ_t`
2. `CFMAX_t`
3. runoff partition logits controlling daily surface-runoff, interflow, and recharge fractions
4. LAI-dependent ET and interception scaling

Important clarification:

- `K_t` appears in some diagnostics from earlier branches, but in this locked MLP publication branch the groundwater recession used in the process equations is effectively the static `K`.

## Parameter ranges

| Parameter | Range | Role |
| --- | --- | --- |
| `INSC` | `[0.5, 5.0] mm` | Interception storage capacity |
| `SUB` | `[0, 1]` | Baseline surface-runoff fraction |
| `CRAK` | `[0, 1]` | Recharge fraction of non-surface runoff |
| `K` | `[0.003, 0.3] d^-1` | Groundwater recession coefficient |
| `TT` | `[-2.5, 2.5] °C` | Snow-rain threshold temperature |
| `CFMAX_base` | `[0.5, 10] mm d^-1 °C^-1` | Base degree-day snowmelt factor |
| `CFR` | `[0, 0.1]` | Refreezing coefficient |
| `CWH` | `[0, 0.2]` | Liquid-water holding fraction in the snowpack |
| `theta_ab` | `[0.5, 1.0]` | Active-root-zone accessibility multiplier |
| `theta_ak` | `[1, 10]` | Active-root-zone accessibility exponent |
| `theta_cap` | `[10, 1500] mm` | Active-root-zone capacity |
| `theta_efmax` | `[0.5, 1.0]` | ET amplitude multiplier |
| `theta_wetpoint` | `[0.3, 0.9]` | ET moisture-stress threshold |
| `SQ_t` | `[0, 6]` after clamping | Daily runoff-shape control |
| `route_a` | `[0, 2.9]` | Gamma routing shape |
| `route_b` | `[0, 6.5]` | Gamma routing scale |
| `component_weight_1..4` | `[0, 1]`, sum to 1 | Four-component mixture weights |

## Core equations

### Snow

The locked branch keeps the smoothed HBV-style snow formulation:

```text
rain = P * f_rain(T, TT)
snow = P - rain
snowmelt_pot = CFMAX_t * max(T - TT, 0)
refreeze_pot = CFR * CFMAX_t * max(TT - T, 0)
```

Snowpack and retained meltwater are advanced with a `CWH` liquid-water holding limit before water is released to soil.

### Active root zone

```text
Smoist_prev = clamp(Sa / theta_cap, 0, 1)
alpha = clamp(theta_ab * (1 - Smoist_prev) ** theta_ak, 0, 1)
P_accessible = alpha * P_after_snow_and_interception
P_inaccessible = (1 - alpha) * P_after_snow_and_interception
Sa_pre = Sa + P_accessible
```

### PET-LAI evapotranspiration

```text
water_stress = clamp(Sa_pre / theta_cap / theta_wetpoint, 0, 1)
LAI_factor = f(LAI)
ET_a_pot = PET * theta_efmax * water_stress * LAI_factor
ET_a = min(ET_a_pot, Sa_pre)
Sa_after_ET = Sa_pre - ET_a
Sa_overflow = max(Sa_after_ET - theta_cap, 0)
Sa_next = Sa_after_ET - Sa_overflow
```

### SIMHYD partition and groundwater

```text
water_for_partition = P_inaccessible + Sa_overflow
water_for_partition -> SRUN, IFLOW, REC  using SUB, CRAK, SQ_t and daily partition logits
GW1 = GW + REC
BAS = min(K * GW1, GW1)
GW_next = GW1 - BAS
Q_process = SRUN + IFLOW + BAS
```

### Routing and storage diagnostics

Each of the four component members is routed with a gamma unit hydrograph using `route_a` and `route_b`, then the routed component discharges are mixed using `component_weight_1..4`.

Realized diagnostic active root-zone storage is:

```text
aSrz_t = Sa_t - min_tau(Sa_tau)
aSrz_capacity = max_t(aSrz_t)
```

### Process water balance

```text
S_before = SNOWPACK + MELTWATER + Sa + GW
S_after = SNOWPACK_next + MELTWATER_next + Sa_next + GW_next
residual = P - INT - ET_a - Q_process - (S_after - S_before)
```

The locked run median weighted daily residual is `0.000182 mm/day`, with zero basins exceeding `1%` cumulative relative process water-balance error.

## Training and validation

- loss: `RmseLossComb(alpha=0.25)`
- supervision during training: streamflow only
- train window: `1980-10-01` to `1995-10-01`
- test window: `1995-10-01` to `2010-10-01`
- evaluation window used in the publication bundle: `1995-10-01` to `2010-09-30`

Independent external validation products are evaluated only after training:

- MODIS MOD16 8-day ET
- GLEAM monthly ET
- GRACE/JPL TWSA
- SWE validation tables
- ESA CCI surface soil moisture

