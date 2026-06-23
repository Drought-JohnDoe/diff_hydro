# Model 6

This folder packages the locked full-671-basin Model 6 eco-hybrid branch used for publication:
`Model6Closed_Snow_aSrz_SIMHYD_Simple_LAIEco_full671_target0732_train1`.

## What is here

- `train_model6.py`: public training/fine-tuning wrapper around the locked runner
- `evaluate_model6.py`: evaluation wrapper for streamflow plus MODIS, GLEAM, GRACE, SWE, and soil-moisture diagnostics
- `make_figures.py`: rebuilds the final figure set into `figures/final/`
- `run_rohini_replication_figures.py`: runs the full Rohini-style audit -> datasets -> evaluation -> figures -> sanity sequence
- `configs/best_model6_config.yaml`: normalized config snapshot for the locked branch
- `checkpoints/best_model6_checkpoint.pt`: copied best checkpoint
- `source/`: minimal copied source tree needed to run the locked branch
- `results/`: copied locked-run metrics, validation tables, and publication assets

## Quick start

Train or fine-tune on the locked branch:

```bash
python model6/train_model6.py --subset 671 --epochs 1 --run-name test_publication_run
```

Run the packaged evaluation summary:

```bash
python model6/evaluate_model6.py
```

Rebuild the final figure set:

```bash
python model6/make_figures.py
```

## Subset modes

- `32`: regime-stratified prototype subset
- `455`: minimally disturbed subset used in the Rohini-style comparison
- `671`: full locked basin set
- `custom`: pass `--custom-basin-list path/to/list.txt` or a CSV with `gauge_id` or `basin_id`

## Locked publication metrics

- Median NSE: `0.6914`
- Mean NSE: `0.4375`
- Median KGE: `0.6585`
- Median R²: `0.7310`
- Median `aSrz_capacity`: `171.43 mm`
- Median mean realized `aSrz`: `73.16 mm`
- Median weighted process water-balance residual: `1.82e-4 mm/day`

## Important note

`summary_compare.csv` in the original workspace contains a `Model6PhysicalFix_B_soft_gate` row with a higher median NSE, but this package deliberately locks the LAIEco branch because it is the best finalized, reproducible Model 6 package with complete diagnostics, figure support, and external validation assets.

