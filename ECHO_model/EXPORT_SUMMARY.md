# ECHO_model Export Summary

## What was copied

- the closed simple process model equations and wrapper logic needed to define:
  - `DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple`
  - `MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple`
- the exact benchmark loss form:
  - `RmseLossComb(alpha=0.25)`
- a compact standalone training loop
- batching and utility helpers
- a standalone demo-data loader
- a standalone raw CAMELS loader
- a step-by-step notebook
- the `Ep100` benchmark artifacts
- independent `GRACE/FLUXCOM` validation summaries

## What was intentionally excluded

- unrelated experimental branches
- HydroDL-wide abstractions not required by the closed simple model
- LAI, dynamic-K, and deep-leakage variants
- analysis scripts unrelated to training, evaluation, or model inspection

## Verification completed

- imports work from inside `ECHO_model/`
- standalone `Ep100` checkpoint loads into the exported class with:
  - `missing keys = 0`
  - `unexpected keys = 0`
- a short demo training run completed:
  - `outputs/smoke_test/`
  - `outputs/script_smoke/`
- the standalone evaluation script completed:
  - `outputs/script_eval.json`
- the notebook executed top to bottom:
  - `notebooks/train_ECHO_model_step_by_step.executed.ipynb`

## How to train a new experiment

From inside `ECHO_model/`:

```bash
PYTHONPATH=. python scripts/run_train.py --mode demo --epochs 1 --batch-size 2 --rho 60 --bufftime 30 --max-iter-ep 2
```

For raw CAMELS-style inputs:

```bash
PYTHONPATH=. python scripts/run_train.py --mode camels --data-root /path/to/Camels --epochs 10 --batch-size 16 --rho 365 --bufftime 365 --max-iter-ep 50
```

To evaluate a checkpoint:

```bash
PYTHONPATH=. python scripts/run_evaluate.py --mode demo --checkpoint outputs/demo_run/model_Ep1.pt
```
