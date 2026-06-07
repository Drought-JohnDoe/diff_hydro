# ECHO_model

`ECHO_model` is a standalone export of the closed Model 6 hydrological model:

- process model: `DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple`
- wrapper: `MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple`

The package is self-contained and does not import from `hydroDL`, `ECO_HYBRID`, or other parent-workspace packages.

## Package layout

- `echo_model/rnn.py`
  - model classes and routing helpers
- `echo_model/losses.py`
  - `RmseLossComb(alpha=0.25)`
- `echo_model/train.py`
  - compact training loop with checkpointing each epoch
- `echo_model/train_utils.py`
  - batching, seeding, device helpers
- `echo_model/data.py`
  - demo-data loader and raw CAMELS loader
- `echo_model/evaluate.py`
  - rollout and summary metrics
- `echo_model/physics_equations.md`
  - end-to-end equations and assumptions
- `scripts/run_train.py`
  - command-line training entry point
- `scripts/run_evaluate.py`
  - command-line evaluation entry point
- `notebooks/train_ECHO_model_step_by_step.ipynb`
  - interactive walkthrough

## Quick start

From inside `ECHO_model/`:

```bash
PYTHONPATH=. python scripts/run_train.py --mode demo --epochs 1 --batch-size 2 --rho 60 --bufftime 30 --max-iter-ep 2
PYTHONPATH=. python scripts/run_evaluate.py --mode demo --checkpoint outputs/demo_run/model_Ep1.pt
```

## Data modes

### Demo mode

Uses the copied five-basin public-release dataset in:

- `demo_data/`

This is the fastest way to understand the package and verify training.

### CAMELS mode

Uses local raw CAMELS-style inputs:

- forcing
- Hargreaves PET
- observed streamflow
- static attributes

The loader normalizes dynamic forcings from the training split and normalizes static attributes across the selected basins.

## Results included

- `results/ep100_benchmark/`
  - saved benchmark outputs from `Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200`
- `results/independent_validation/`
  - copied independent GRACE TWSA and FLUXCOM ET summaries
- `results/independent_validation_grace_fluxcom/`
  - original fuller validation folder with PNG figures and per-basin tables

## What was intentionally excluded

- unrelated experimental branches
- dynamic-`K` variants
- LAI experiments
- nonessential analysis scripts
- HydroDL package-wide utilities not required by the closed simple model

## Verification status

This export was smoke-tested locally:

- package imports from inside `ECHO_model/`
- one short demo training run completed
- checkpoints and evaluation artifacts were written under `outputs/`
