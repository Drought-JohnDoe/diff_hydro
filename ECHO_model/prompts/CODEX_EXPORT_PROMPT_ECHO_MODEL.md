# Codex Prompt: Export Current Closed Model 6 as Standalone `ECHO_model`

Create a **fully standalone hydrology package** named `ECHO_model` from the current trained model stack in this workspace.

## Goal
Export the current trained **closed Model 6 hydrology model** into a clean package that is easy to read, edit, and retrain without relying on the rest of this cluttered workspace.

The package should feel like a **minimal HydroDL-style package**, but only include the material actually required to train and evaluate this model.

## Authoritative source model
Use the saved checkpoint and benchmark artifacts from:

- `ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200`

Reference files that must be preserved or copied into the new package:

- `model_Ep100_state.pt`
- `summary_compare.csv`
- `per_basin_metrics_compare.csv`
- `water_balance_compare.csv`
- `final_result.txt`
- `run.csv`
- `basin_metadata.csv`

Also preserve the independent validation outputs from:

- `ECO_HYBRID/Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200/independent_validation_grace_fluxcom`

## Package name and structure
Create a new top-level folder:

- `ECHO_model/`

Inside it, create a clean standalone structure like:

- `ECHO_model/README.md`
- `ECHO_model/requirements.txt`
- `ECHO_model/environment.yml`
- `ECHO_model/setup.py` or `pyproject.toml`
- `ECHO_model/echo_model/__init__.py`
- `ECHO_model/echo_model/rnn.py`
- `ECHO_model/echo_model/losses.py`
- `ECHO_model/echo_model/train_utils.py`
- `ECHO_model/echo_model/data_utils.py`
- `ECHO_model/echo_model/camels_loader.py`
- `ECHO_model/echo_model/config.py`
- `ECHO_model/echo_model/evaluate.py`
- `ECHO_model/echo_model/plotting.py`
- `ECHO_model/scripts/train_model.py`
- `ECHO_model/scripts/evaluate_model.py`
- `ECHO_model/scripts/export_daily_fluxes.py`
- `ECHO_model/notebooks/train_step_by_step.ipynb`
- `ECHO_model/results/ep100_benchmark/...`
- `ECHO_model/results/independent_validation_grace_fluxcom/...`

## Strict export rules

1. **Do not rely on the old workspace after export**
   - Copy the required model classes, losses, utilities, and training helpers into the new package.
   - Do **not** leave imports pointing back to:
     - `hydroDL`
     - `Model_six_physical`
     - `ECO_HYBRID`
     - `Diagnosis.py`
     - any other workspace-only script

2. **Only keep the material required to train the model**
   - no unrelated experiments
   - no old benchmark clutter
   - no extra baselines
   - no obsolete model branches

3. **Preserve explicit physical equations**
   - keep the closed snow + aSrz + SIMHYD simple formulation
   - include comments explaining the physical meaning of each parameter and flux
   - keep the equations easy to read

4. **Make the package editable**
   I should be able to change easily:
   - number of components (`nmul`, default `4`)
   - training period
   - test period
   - epoch count
   - batch size
   - `rho`
   - `max_iter_ep`
   - hidden size
   - which dynamic variables are enabled
   - whether routing is on/off
   - whether component routing is on/off

5. **Make dynamic controls explicit**
   - isolate the dynamic head slices clearly
   - make it easy to turn dynamic variables on/off
   - keep parameter transforms explicit

## Model scope to export
Export the **closed simple Model 6** architecture corresponding to:

- `DynamicSimHydModelFiveDifferentiableClosedSnowaSrzSIMHYDSimple`
- `MultiInv_DynamicSimHydModelSix_Physical_ClosedSnowaSrzSIMHYDSimple`

If these classes currently depend on larger inheritance chains, refactor them into a **minimal local standalone implementation** in `echo_model/rnn.py`.

That file must include only the required helper classes and functions needed to run this model.

## Required training functionality
The new package must include:

### 1. Losses
At minimum:
- `RmseLossComb(alpha=0.25)`

If auxiliary regularization is used through `model.get_auxiliary_loss()`, preserve that behavior.

### 2. Training utilities
Include:
- batching
- sequence window selection
- save-every-epoch checkpointing
- training log CSV
- seed handling
- GPU/CPU handling
- resume-from-checkpoint support

### 3. Evaluation
Include utilities to compute:
- NSE
- KGE
- R2
- FLV
- FHV
- low-flow NSE
- high-flow NSE
- ET/P
- Q/P
- water balance residual
- cumulative water balance error

### 4. Flux/state export
Include a script that can export daily:
- Q_process
- P
- ET_a
- INT
- SRUN
- IFLOW
- BAS
- REC
- Sa
- theta_cap
- Smoist
- GW
- SNOWPACK
- MELTWATER

## Required data loading behavior
Support two clean modes:

### Mode A: CAMELS-US raw loader
Implement a minimal standalone CAMELS-US loader that reads directly from:
- CAMELS forcing text files
- CAMELS streamflow text files
- CAMELS static attribute tables

and builds:
- train arrays
- test arrays
- static attribute arrays
- normalized dynamic inversion arrays

### Mode B: prepared arrays
Allow training from prepared `.npz` or `.parquet` style arrays so that future experiments do not need the raw loader every time.

## Jupyter notebook requirement
Create:

- `ECHO_model/notebooks/train_step_by_step.ipynb`

The notebook must:

1. load config
2. load data
3. build model
4. print model summary / parameter counts
5. run one batch forward pass
6. compute loss
7. train for a few epochs step by step
8. print loss every epoch
9. save checkpoint
10. run evaluation
11. plot observed vs simulated discharge for a few basins

The notebook should be readable and tutorial-like, not just a dumped script.

## README requirements
The `README.md` must clearly explain:

1. what the model is
2. what equations it uses
3. what files control the architecture
4. what files control training
5. how to switch dynamic variables on/off
6. how to change the number of components
7. how to retrain on a new period
8. how to load the saved `Ep100` checkpoint
9. how to reproduce the included benchmark outputs

## Benchmark/results packaging
Copy the current saved benchmark outputs into:

- `ECHO_model/results/ep100_benchmark/`

Also copy the independent validation outputs into:

- `ECHO_model/results/independent_validation_grace_fluxcom/`

Add a short `RESULTS_README.md` describing what each file is.

## Keep it clean
Avoid unnecessary abstraction.

Preferred style:
- a few readable Python files
- explicit formulas
- small helper functions
- direct configuration through a small config object or YAML file

Do **not** leave the package as a thin wrapper around the old workspace.
It must be understandable on its own.

## Deliverables
When done, produce:

1. the full `ECHO_model` package
2. copied benchmark and validation result folders
3. the training notebook
4. a short migration note explaining what source files were copied from where
5. a list of any assumptions or simplifications made during the export

## Validation checks
Before finishing, verify:

- the package imports cleanly without old workspace imports
- the model can be instantiated
- one forward pass works
- one training step works
- the notebook opens and runs its first cells
- the `Ep100` checkpoint can be loaded
- the benchmark result folder is present

## Optional GitHub step
If a Git repository is active, commit the new `ECHO_model` folder and push it.
If no Git repository is active, stop cleanly and report:

- that the export is complete locally
- that GitHub shipping still requires a repo root or remote target

