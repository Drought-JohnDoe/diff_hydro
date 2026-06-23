# Assumptions And Limitations

- This is a Model 6-based Rohini-style diagnostic replication, not the original Rohini model.
- The locked model is Q-trained; ET, SWE, TWSA, and surface soil moisture are validation products.
- LAI is daily gap-filled NOAA AVHRR CDR LAI. Gap-filled early-period LAI is documented in the locked run manifest.
- PET-based LAI-scaled ET is used in the locked branch; this is not the Rohini net-radiation ET equation.
- GRACE TWSA has a spatial-resolution mismatch with individual CAMELS basins; only overlapping valid basin-months are used.
- SWE validation is weak and is reported honestly. The likely causes are the smoothed snow partition/melt formulation, lack of SWE supervision, basin-average forcing biases, and compensation from streamflow-only calibration.
- `theta_cap` is a structural upper bound. `aSrz_capacity` is realized capacity from the simulated `Sa` trajectory.
- Theta-cap structural ablations are configured but not retrained in this locked package.
- Observational ablations are not run because no completed multi-observation training branch is locked here.
