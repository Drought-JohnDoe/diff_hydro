# Observational Ablation Status

The locked Model 6 LAIEco branch is trained with streamflow supervision only.
MODIS ET, GLEAM ET, SWE, GRACE TWSA, and ESA CCI surface soil moisture are
used for independent validation in this package.

Rohini-style observational ablations such as Q+ET+SWE+TWSA, no_ET, no_Q,
no_SWE, and no_TWSA require a separate multi-observation training branch.
They are not faked here.
