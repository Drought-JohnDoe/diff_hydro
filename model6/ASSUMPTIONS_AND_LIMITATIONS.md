# Assumptions And Limitations

- This is a Model 6-based Rohini-style diagnostic replication package, not the original Rohini model.
- The locked branch is Q-trained only. ET, SWE, GRACE/TWSA, and surface soil moisture are external validation products, not training constraints.
- The selected branch is intentionally the finalized LAIEco package, even though another summary table row in the old workspace reports a higher median NSE. That alternative branch was not kept as the publication source because it did not match this branch in completeness of diagnostics and reproducibility assets.
- LAI is daily gap-filled NOAA AVHRR CDR LAI. Early-period gaps were filled before training and are documented in the run manifest.
- ET in the locked branch is PET-driven with LAI scaling; it is not the later net-radiation ET branch.
- The snow module is the smoothed HBV-style snow partition/melt implementation used during the locked training run, not the later exact hard-threshold HBV1.1 snow experiment.
- `theta_cap` is a structural upper bound, while `aSrz_capacity` is a realized diagnostic derived from the simulated `Sa` trajectory.
- GRACE/TWSA support is limited by spatial overlap; the Rohini-style Figure 5 scatter against TWSA uses only the valid overlap subset.
- SWE skill is weak in this branch. The most likely causes are streamflow-only calibration, smoothed snow partition/melt dynamics, basin-mean forcing bias, and equifinality between snow and runoff-process compensation.

