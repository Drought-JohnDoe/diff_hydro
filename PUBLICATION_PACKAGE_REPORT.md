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
- the clean publication repo now lives in-place at `/home/mircore/Desktop/diff_hydro`

## Retained raw data

Documented in [README_RAW_DATA.md](/home/mircore/Desktop/diff_hydro/raw_data/README_RAW_DATA.md).

## Checkpoint

- [best_model6_checkpoint.pt](/home/mircore/Desktop/diff_hydro/model6/checkpoints/best_model6_checkpoint.pt)

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
- pushed branch: `publication-package-2026-06-23`
- remote PR URL: `https://github.com/Drought-JohnDoe/diff_hydro/pull/new/publication-package-2026-06-23`

## Sanity status

- `python model6/evaluate_model6.py`: passed and printed the locked metrics plus external validation summary
- `conda run -n pytorch python model6/make_figures.py`: passed and rebuilt the figure set under `model6/figures/final/`
- `conda run -n pytorch python model6/figures/scripts/publication_pipeline.py sanity`: passed
- note: the current base shell Python has a NumPy 2 / Matplotlib ABI mismatch, so figure regeneration should use the packaged environment or the existing `pytorch` environment

## Archive status

- archive target: `/mnt/nas/home_aman/Projects/WRR_repo_archive_2026-06-23/`
- archive mode: copy-first verification to NAS, followed by local removal only after the archive copy existed
- progress log: `model6/results/archive/archive_to_nas_2026-06-23.log`
- `CLEANUP_MANIFEST.csv` has been generated in the clean repo and refreshed against the NAS target
- verified copies already include key top-level non-kept content such as `Diagnosis.py`, `HBV11P_ECO_LSTM_CLEAN`, `Model_six_physical`, `code`, `data`, `data_processed`, `external_downloads`, and `results`
- additional workspace extras have also been archived, including the temporary staging repo `diff_hydro_publication.tar.gz` and temp/env tarballs under `root_cleanup_extras/`

## Remaining manual steps

- optionally replace the temporary `LICENSE` with the intended public license
- review the packaging branch on GitHub and merge when satisfied
