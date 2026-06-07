# ECHO_model

`ECHO_model` is a clean export target for the trained closed Model 6 hydrology workflow.

Current contents:

- `results/ep100_benchmark/`
  - saved `Ep100` checkpoint and benchmark outputs from `Model6Closed_Snow_aSrz_SIMHYD_Simple_full671_ep100_iter200`
- `results/independent_validation_grace_fluxcom/`
  - independent GRACE TWSA and FLUXCOM ET validation outputs
- `prompts/`
  - Codex export prompt for building a fully standalone package
- `notebooks/`
  - reserved for the final step-by-step training notebook in the exported package

This folder is meant to be the landing zone for a future standalone package export named `ECHO_model`, with:

- a minimal `rnn.py`
- minimal losses and train utilities
- a clean CAMELS loader or prepared-array pipeline
- a single notebook for step-by-step training
- no dependency on the rest of the cluttered workspace beyond explicit copied files

Important note:

- The current workspace root is **not** a live Git repository, so nothing in this folder has been pushed automatically.
- The export prompt below is written so Codex can build the standalone package deterministically from the current trained model and benchmark artifacts.
