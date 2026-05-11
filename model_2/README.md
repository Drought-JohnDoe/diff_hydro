# Model 2: Snow-SIMHYD-MC-Heter

This is the heterogeneity-aware version of the multi-component Snow-SIMHYD model.

Main idea:
- keep the Snow-SIMHYD process structure with snow, soil moisture, groundwater, routing, and multi-component mixing
- fix the earlier basin-collapse issue by predicting static parameters directly from basin attributes
- keep dynamic `LG(t)` as a separate temporal head with an attribute-conditioned bias

Core code:
- model class:
  `code/dPLHBVrelease/hydroDL-dev/hydroDL/model/rnn.py`
  `MultiInv_SnowSIMHYDMulTDHeterModel`
- training script:
  `code/dPLHBVrelease/hydroDL-dev/example/dPLSnowSIMHYDMC_Heter/traindPLSnowSIMHYDMC_Heter.py`
- test script:
  `code/dPLHBVrelease/hydroDL-dev/example/dPLSnowSIMHYDMC_Heter/testdPLSnowSIMHYDMC_Heter.py`
- detailed HBV comparison report:
  `report_snowsimhydmc_heter_vs_hbv_detailed.py`

Inputs:
- forcing `x`:
  `[P, T, PET]`
- inversion input `z`:
  normalized forcing history plus repeated static basin attributes

Process parameters per component:
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

Architecture changes relative to the older Snow-SIMHYD-MC:
- static parameters come from a direct attribute encoder
- routing parameters come from the same attribute encoder
- component weights come from the same attribute encoder
- dynamic `LG(t)` comes from a separate sequence LSTM plus attribute bias
- component-specific learnable biases break permutation symmetry

Routing:
- HBV-style gamma routing after component mixing
- saved report maps include:
  `route_a`
  `route_b`

Mixture:
- 4 components
- learned softmax weights:
  `weight_c1` to `weight_c4`

Practical 671-basin benchmark used here:
- train:
  `1980-10-01` to `1995-10-01`
- test:
  `1995-10-01` to `2010-10-01`
- forcing:
  `daymet`
- batch size:
  `32`
- rho:
  `365`
- hidden size:
  `64`
- random windows per epoch:
  `100`

Ep14 result against saved HBV benchmark:
- HBV median NSE:
  `0.6278`
- Model 2 median NSE:
  `0.6464`
- Model 2 wins on NSE in:
  `387 / 671` basins

Key outputs from the saved Ep14 comparison:
- report folder:
  `outputs/report_snowsimhydmc_heter_vs_hbv_detailed_ep14`
- per-basin metrics:
  `per_basin_metrics_snowsimhydmc_heter_vs_hbv.csv`
- learned parameter ranges:
  `learned_parameter_ranges.csv`
- weighted-average parameter maps:
  `parameter_maps/weighted_average`
- routing and dynamic LG maps:
  `parameter_maps/routing_weights`

Why this model is called “Model 2”:
- it is the first version in this project where the learned basin parameters remain heterogeneous across basins instead of collapsing to near-shared values.
