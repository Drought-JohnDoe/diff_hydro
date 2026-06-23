# Methods: Locked Model 6 LAIEco

This package analyzes the locked Model 6 active-root-zone SIMHYD branch:
`Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`.

## States

- `SNOWPACK`: solid snow storage.
- `MELTWATER`: liquid water retained in snowpack.
- `Sa`: ecosystem-accessible active root-zone storage.
- `GW`: single groundwater/baseflow storage.

## Snow

The locked branch uses the Model 6 smoothed HBV-style snow module used during training:

- precipitation is smoothly partitioned into rain and snow around `TT`;
- snowmelt is controlled by a degree-day factor `CFMAX_t`;
- refreezing is controlled by `CFR`;
- liquid-water holding is controlled by `CWH`.

The exact hard HBV1.1 snow test was run separately and performed worse, so it is not the locked branch.

## Interception

Interception is active. Intercepted water contributes to `ET_total` as `INT + ET_a`.

## Active Root Zone

Soil wetness is computed from the previous active store:

`Smoist_prev = clamp(Sa / theta_cap, 0, 1)`

The accessible precipitation fraction is:

`alpha = clamp(theta_ab * (1 - Smoist_prev) ** theta_ak, 0, 1)`

`P_accessible = alpha * P_after_snow_and_interception`

`P_inaccessible = (1 - alpha) * P_after_snow_and_interception`

Active storage before ET is:

`Sa_pre = Sa + P_accessible`

The locked branch uses PET-based LAI-scaled ET:

`ET_a_pot = PET * theta_efmax * water_stress * LAI_et_scalar`

`ET_a = min(ET_a_pot, Sa_pre)`

Realized diagnostic storage is:

`aSrz_t = Sa_t - min_tau(Sa_tau)`

`aSrz_capacity = max_t(aSrz_t)`

## SIMHYD Runoff And Groundwater

Inaccessible water and active-store overflow are partitioned into surface runoff, interflow, and recharge using SIMHYD-style parameters `SUB`, `CRAK`, and `SQ_t`.

Groundwater release is:

`GW1 = GW + REC`

`BAS = min(K * GW1, GW1)`

`GW_next = GW1 - BAS`

`Q_process = SRUN + IFLOW + BAS`

Gamma routing turns process runoff into `Q_routed`.

## Water Balance

Before routing:

`S_before = SNOWPACK + MELTWATER + Sa + GW`

`S_after = SNOWPACK_next + MELTWATER_next + Sa_next + GW_next`

`residual = P - INT - ET_a - Q_process - (S_after - S_before)`

The locked run reports median weighted daily residual near zero.

## Parameter Ranges

- `INSC`: [0.5, 5.0] mm
- `SUB`: [0, 1]
- `CRAK`: [0, 1]
- `SQ_t`: [0, 6]
- `K`: [0.003, 0.3] d-1
- `TT`: [-2.5, 2.5] degC
- `CFMAX_t`: [0.5, 10] mm d-1 degC-1 nominal; learned dynamic multiplier in locked branch may soften the effective lower edge
- `CFR`: [0, 0.1]
- `CWH`: [0, 0.2]
- `theta_ab`: [0.5, 1.0]
- `theta_ak`: [1, 10]
- `theta_cap`: [10, 1500] mm
- `theta_efmax`: [0.5, 1.0]
- `theta_wetpoint`: [0.3, 0.9]
- routing `route_a`: [0, 2.9]
- routing `route_b`: [0, 6.5]

## Parameter Generator

The locked branch uses the existing Model 6 neural parameterization with static basin attributes, warm-started from the closed Model 6 checkpoint. It is not the later LSTM-parameter branch.

## Training And Evaluation

Training objective: `RmseLossComb(alpha=0.25)`.

Training supervision: streamflow only.

ET, SWE, TWSA, and soil moisture are independent validation products, not training losses in the locked branch.
