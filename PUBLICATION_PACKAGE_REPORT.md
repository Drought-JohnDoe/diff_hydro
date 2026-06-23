# Publication Package Report

## Selected model

- chosen branch: `Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`
- reason: it is the best finalized full-671 Model 6 publication package with complete diagnostics, saved `Sa`/`aSrz`, independent validation tables, and reproducible Rohini-style figure support

## Important comparison note

- `summary_compare.csv` in the original workspace includes a `Model6PhysicalFix_B_soft_gate` row with a higher median NSE
- this package still locks the LAIEco branch as the publication source of truth because that alternative branch was not the finalized reproducible bundle requested here

## Kept in the clean repo

- `hbv_module/`
- `hydro_ml/`
- `model6/`
- `raw_data/`
- repo-level packaging files

## Retained raw data

Documented in [README_RAW_DATA.md](/home/mircore/Desktop/diff_hydro_publication/raw_data/README_RAW_DATA.md).

## Checkpoint

- [best_model6_checkpoint.pt](/home/mircore/Desktop/diff_hydro_publication/model6/checkpoints/best_model6_checkpoint.pt)

## Main commands

Train:

```bash
python model6/train_model6.py --subset 671 --epochs 1 --run-name smoke_test
```

Evaluate:

```bash
python model6/evaluate_model6.py
```

Rebuild figures:

```bash
python model6/make_figures.py
```

Full Rohini-style sequence:

```bash
python model6/run_rohini_replication_figures.py
```

## GitHub status

- target repo: `https://github.com/Drought-JohnDoe/diff_hydro`
- packaging branch recommended: `publication-package-2026-06-23`

## Remaining manual steps

- finish/archive verification in `CLEANUP_MANIFEST.csv`
- optionally replace the temporary `LICENSE` with the intended public license
- review the packaging branch on GitHub before merging to `main`

