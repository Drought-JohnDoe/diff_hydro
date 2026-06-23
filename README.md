# diff_hydro Publication Package

This repository is a cleaned publication package centered on the locked full-671-basin Model 6 eco-hybrid branch:

- source run: `ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`
- source checkpoint: `model_best_state.pt`
- source publication assets: `ECO_HYBRID/PUBLICATION_MODEL6_ROHINI_REPLICATION`
- source evaluation bundle: `ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1_results_package`

## Repo layout

- `hbv_module/`: vendored `hydroDL` code used by the locked branch
- `hydro_ml/`: metrics/utilities required by the copied source and wrappers
- `model6/`: the locked Model 6 code, checkpoint, results, figures, wrappers, and docs
- `raw_data/`: manifest of retained raw-data roots

## Main commands

```bash
python model6/train_model6.py --subset 671 --epochs 1 --run-name smoke_test
python model6/evaluate_model6.py
python model6/make_figures.py
python model6/run_rohini_replication_figures.py
```

## Selected publication branch

This package intentionally locks the LAIEco branch as the source of truth. A different row in the original workspace summary tables reports a higher median NSE, but it is not the chosen publication branch because the locked LAIEco package is the one with complete diagnostics, external validation tables, `Sa`/`aSrz` outputs, and publication-ready figure support.

