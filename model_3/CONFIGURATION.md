# Model 3 Configuration

This file records the detailed configuration for the heterogeneity-aware `Snow-SIMHYD-MC-Heter` model packaged in `model_3`.

## Class Names

- Process model:
  `SnowSIMHYD8Differentiable`
- Full model:
  `MultiInv_SnowSIMHYDMulTDHeterModel`

Defined in:
- `scripts/rnn.py`

## High-Level Architecture

The forward model is:

1. Static basin attributes are passed through a dedicated MLP encoder.
2. That encoder predicts:
   - per-component static process parameters
   - routing parameters
   - component mixing weights
3. A separate temporal LSTM predicts dynamic `LG(t)`.
4. An attribute-conditioned linear bias shifts the dynamic `LG(t)` head by basin.
5. The `4` Snow-SIMHYD components are run in parallel.
6. Their outputs are mixed with learned softmax weights.
7. Gamma routing is applied to the mixed discharge.

## Inputs

### Forcing input `x`

Shape:
- `[T, B, 3]`

Variables:
- `x[..., 0]`: precipitation `P`, `mm/day`
- `x[..., 1]`: temperature `T`, `deg C`
- `x[..., 2]`: PET, `mm/day`

### Inversion input `z`

Built from:
- normalized forcing history
- repeated static attributes

Nominal shape:
- `[T, B, ninv]`

## States

The underlying process model carries:
- `snowpack`
- `meltwater`
- `soil moisture`
- `groundwater`

During the saved diagnostic export we also reported:
- `interception_storage`

but note that interception is not a persistent prognostic state in this implementation.

## Parameters Per Component

Each component predicts 12 normalized parameters that are mapped to physical ranges.

| Parameter | Meaning | Range |
|---|---|---|
| `INSC` | interception capacity | `0.5` to `5.0` |
| `COEF` | infiltration capacity scale | `50.0` to `400.0` |
| `SQ` | infiltration decay / wetness sensitivity | `0.0` to `6.0` |
| `SMSC` | soil moisture storage capacity | `50.0` to `500.0` |
| `SUB` | interflow fraction coefficient | `0.0` to `1.0` |
| `CRAK` | recharge fraction coefficient | `0.0` to `1.0` |
| `K` | baseflow coefficient | `0.003` to `0.3` |
| `LG` | groundwater loss | `0.0` to `1.0` |
| `TT` | rain-snow threshold temperature | `-2.5` to `2.5` |
| `CFMAX` | snowmelt factor | `0.5` to `10.0` |
| `CFR` | refreezing coefficient | `0.0` to `0.1` |
| `CWH` | snow liquid water holding capacity | `0.0` to `0.2` |

## Routing Parameters

Routing is applied after component mixing.

| Parameter | Meaning | Range |
|---|---|---|
| `route_a` | gamma UH shape-related control | `0.0` to `2.9` |
| `route_b` | gamma UH scale-related control | `0.0` to `6.5` |

## Mixture Configuration

- number of components: `4`
- mixture weights: learned softmax over components
- component symmetry breaking:
  - `compStaticBias`
  - `compWeightBias`

These biases are important. Without them, the optimizer can settle into permutation-symmetric component templates and produce collapsed basin-average parameters.

## Dynamic LG Configuration

Dynamic `LG(t)` is enabled.

Implementation:
- sequence head: `self.lstmdyn`
- attribute bias: `self.lgAttr`
- output squashed by `sigmoid`

Final `LG` used in the process step:
- `LG_eff(t) = (1 - lgdynweight) * LG_static + lgdynweight * LG_dynamic(t)`

Saved setting:
- `lgdynweight = 0.5`

## Training Configuration

Saved benchmark settings:
- dataset: full CAMELS `671` basins
- forcing: `daymet`
- train period: `1980-10-01` to `1995-10-01`
- test period: `1995-10-01` to `2010-10-01`
- warmup / `inittime`: `365`
- batch size: `32`
- `rho`: `365`
- hidden size: `64`
- epochs for saved packaged checkpoint: `14`
- windows per epoch in the practical benchmark: `100`
- optimizer: `Adadelta`
- random seed: `111111`
- loss: `RmseLossComb(alpha=0.25)`

## Loss Function

The packaged train script uses:
- `hydroDL.model.crit.RmseLossComb`
- `alpha = 0.25`

This is not direct `NSE` optimization.

## Test Configuration

Saved test behavior:
- full test period evaluated after using the training period as warmup
- predictions saved to `.npy`
- basin metrics computed afterward from saved predictions and observations

## Outputs

Primary predicted output:
- routed discharge `Q`

The diagnostic script also reconstructs and exports:
- snow/rain partition
- snowmelt
- ET terms
- infiltration
- recharge
- surface runoff
- interflow
- baseflow
- groundwater loss
- water balance closure terms

## Key Files In This Package

- checkpoint:
  `checkpoints/model_Ep14.pt`
- model code:
  `scripts/rnn.py`
- train code:
  `scripts/traindPLSnowSIMHYDMC_Heter.py`
- test code:
  `scripts/testdPLSnowSIMHYDMC_Heter.py`
- state/flux export:
  `scripts/analyze_snowsimhydmc_heter_ep14_states_fluxes.py`

## Packaged Performance Snapshot

Saved `Ep14` headline:
- median test `NSE = 0.6464`
- saved HBV benchmark median test `NSE = 0.6278`

This package is therefore the first version in this project that:
- preserves parameter heterogeneity across basins
- stays competitive with HBV on the all-`671` benchmark
- provides a reusable path for parameter/flux/state diagnostics
